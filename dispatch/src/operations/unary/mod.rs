//! Unary operators: one input channel, one output channel.
//!
//! A unary operator reads items of type `I` from its input channel, applies a
//! transform, and writes items of type `O` to its output channel. Most query stages
//! are unary: filter, project, count, order-by-limit, group-by, and the entire
//! parquet decode pipeline (indexer, decompressor, decoder).
//!
//! # Key types
//!
//! - [`Unary<I, O>`] — The transform trait. Implementors define [`consume`](Unary::consume)
//!   (process one input item) and optionally [`finish`](Unary::finish) (emit final results
//!   after all input is drained, e.g. aggregation output).
//!
//! - [`UnaryOperator<I, O, U, R, S>`] — The [`Operator`] implementation that wires a
//!   [`Unary`] to a receiver and sender. Created during the factory build step on the
//!   worker thread.
//!
//! - [`UnaryFactory<I, O>`] — Trait for creating a [`Unary`] instance. Each factory is
//!   consumed once per worker to produce the unary transform for that worker.
//!
//! - [`UnaryOperatorFactory<I, O, UF, C, OP>`] — An [`OperatorFactory`](crate::api::OperatorFactory)
//!   that chains a head factory (`OP`) with a [`UnaryFactory`] (`UF`) and a
//!   [`ChannelFactory`](super::channels::ChannelFactory) (`C`). When built, it creates the
//!   channel, builds the head (passing it the channel's sender), and creates a
//!   `UnaryOperator` reading from the channel's receiver.
//!
//! # Finishing protocol
//!
//! Unary operators across workers coordinate finishing via a shared `siblings_left`
//! atomic counter. When a worker's input channel is empty, it decrements the counter.
//! Once all siblings have decremented (counter reaches zero), [`Unary::finish`] is called
//! — this is where aggregation operators (count, group-by, order-by-limit) emit their
//! final results.
//!
//! # Concrete unary transforms
//!
//! - [`FilterFactory`] — Keeps rows matching a boolean mask.
//! - [`MapFactory`] — Transforms each batch (column selection, computation).
//! - [`CountFactory`] — Counts rows, coordinating across workers for the total.
//! - [`OrderByLimitFactory`] — Top-N sort across workers.
//! - [`GroupFactory`] — Hash-based group-by with aggregation.

mod group;
pub use group::{
    AggregationKind, AggregationRowValueExtractor, AggregationSlot, Compiled, Count, GroupFactory,
    IntKeyExtractor, IntPairKeyExtractor, KeyExtractor, StringKeyExtractor, Sum, ValueExtractor,
};

#[cfg(test)]
pub(crate) mod test_utils;

mod factory;
pub use factory::*;

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use arrow_schema::ArrowError;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

use super::Operator;
use super::channels::{Receiver, Sender};
use crate::worker::worker_waker;

mod pipeline_breaker;
pub use pipeline_breaker::{Consumer, Outputter, PipelineBreaker};

mod count;
pub use count::CountFactory;

mod aggregate;
pub use aggregate::{AggKind, AggSpec, AggregateFactory};

mod filter;
pub use filter::FilterFactory;

mod map;
pub use map::MapFactory;

mod default_unary_factory;
pub use default_unary_factory::DefaultUnaryFactory;

pub mod parquet;

mod copy_out;
mod order_by_limit;

pub use copy_out::CopyOutFactory;
pub use order_by_limit::{DynamicFilterSlot, OrderBy, OrderByLimitFactory};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] super::channels::Error),
    #[error("{0}")]
    OrderByLimit(#[from] order_by_limit::Error),
    #[error("{0}")]
    Group(#[from] group::Error),
    #[error("{0}")]
    Decompressor(#[from] crate::operations::parquet::DecompressorError),
    #[error("{0}")]
    Parquet(#[from] crate::operations::parquet::types::thrift::parquet_thrift::ParquetError),
    #[error("{0}")]
    Decoder(#[from] crate::operations::parquet::RowGroupDecoderError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The core transform trait for unary operators.
///
/// `I` is the input type (read from the upstream channel), `O` is the output type
/// (written to the downstream channel). Most implementations are `RecordBatch -> RecordBatch`,
/// but the parquet pipeline uses other types (e.g. `CompressedPage -> DecompressedPage`).
pub trait Unary<I, O> {
    /// Process one input item, sending zero or more output items to `sender`.
    fn consume<S: Sender<O>>(&mut self, object: I, sender: &mut S) -> Result<()>;

    /// Return any pending IO requests (e.g. async disk reads for parquet pages).
    fn next_io_requests(&mut self) -> Result<Vec<IORequest>> {
        Ok(vec![])
    }

    /// Whether this unary is ready to accept another input. Returns `false` when
    /// backpressured (e.g. waiting for IO to complete before consuming more).
    fn ready_for_more_work(&mut self) -> bool {
        true
    }

    /// Handle a completed disk read.
    fn process_disk_response<S: Sender<O>>(
        &mut self,
        _sender: &mut S,
        _request: IORequest,
    ) -> Result<()> {
        unreachable!()
    }

    /// Called each iteration even when no input is available. Useful for operators
    /// that generate work independently of input (e.g. emitting buffered results).
    fn run<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    /// Called after all siblings have drained their input. Emit any final results
    /// (e.g. aggregation totals). Return `true` when fully done, `false` to be
    /// called again (e.g. if final output requires multiple steps).
    ///
    /// Once finish is called, `consume` is guaranteed never to be called again.
    fn finish<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<bool> {
        Ok(true)
    }
}

/// Wires a [`Unary`] transform to a receiver (input channel) and sender (output channel),
/// implementing the [`Operator`] trait so it can be driven by the worker's event loop.
///
/// Created during the factory build step — not constructed directly. See
/// [`UnaryOperatorFactory`] in the [`factory`] module.
pub struct UnaryOperator<I, O, U: Unary<I, O>, R: Receiver<I>, S: Sender<O>> {
    unary: U,
    receiver: R,
    sender: S,
    notified_finished: bool,
    siblings_left: Arc<AtomicUsize>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, U: Unary<I, O>, R: Receiver<I>, S: Sender<O>> UnaryOperator<I, O, U, R, S> {
    pub fn new(unary: U, receiver: R, sender: S, siblings_left: Arc<AtomicUsize>) -> Self {
        Self {
            unary,
            receiver,
            sender,
            notified_finished: false,
            siblings_left,
            _phantom: Default::default(),
        }
    }
}

impl<I, O, U: Unary<I, O>, IN: Receiver<I>, OUT: Sender<O>> Operator
    for UnaryOperator<I, O, U, IN, OUT>
{
    fn run_cpu_work(&mut self) -> super::Result<WorkStatus> {
        if !self.unary.ready_for_more_work() {
            return Ok(WorkStatus::Pending);
        }

        let item = match self.receiver.try_recv() {
            None => return Ok(self.unary.run(&mut self.sender)?),
            Some(t) => t,
        };

        self.unary.consume(item, &mut self.sender)?;

        Ok(WorkStatus::Ran)
    }

    fn next_io_requests(&mut self) -> super::Result<Vec<IORequest>> {
        Ok(self.unary.next_io_requests()?)
    }

    fn process_disk_response(&mut self, context: IORequest) -> super::Result<()> {
        Ok(self
            .unary
            .process_disk_response(&mut self.sender, context)?)
    }

    fn try_finish(&mut self) -> super::Result<bool> {
        if !self.receiver.is_empty() {
            return Ok(false);
        }

        let ready = if self.notified_finished {
            self.siblings_left.load(Ordering::Relaxed) == 0
        } else {
            self.notified_finished = true;
            let was_last = self.siblings_left.fetch_sub(1, Ordering::Relaxed) == 1;
            if was_last {
                // Sibling counter just hit 0: every worker's `try_finish` for this
                // operator can now run. Wake any peers parked on the waker so they
                // advance to their `finish` instead of sleeping out the timeout.
                worker_waker().notify();
            }
            was_last
        };

        if ready {
            // Race guard: a peer may have stolen work into our channel between
            // the is_empty() check above and the decrement. Re-check and back
            // off if new work appeared. Relaxed ordering suffices because the
            // channel ops themselves provide Acquire/Release on the data.
            if !self.receiver.is_empty() {
                self.notified_finished = false;
                self.siblings_left.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }

            let done = self.unary.finish(&mut self.sender)?;
            if done {
                // `finish` may have created new batches downstream (e.g.
                // OrderByLimit flushing), so wake any parked peers
                // to pick that work up rather than waiting out their park
                // timeout.
                worker_waker().notify();
            }
            return Ok(done);
        }
        Ok(false)
    }

    fn try_steal_work(&mut self) -> super::Result<WorkStatus> {
        if !self.unary.ready_for_more_work() {
            return Ok(WorkStatus::Pending);
        }
        match self.receiver.steal() {
            Some(s) => {
                self.unary.consume(s, &mut self.sender)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }
}
