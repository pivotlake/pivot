//! Unary operators: one input channel, one output channel.
//!
//! A unary operator reads items of type `I` from its input channel, applies a
//! transform, and writes items of type `O` to its output channel. Most query stages
//! are unary: filter, project, count, order-by-limit, and group-by. The Parquet
//! decode pipeline in `catalog` (indexer, decompressor, decoder) is built from
//! these same primitives.
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
//! - [`OrderByLimitFactory`] — Top-N sort across workers.
//! - [`GroupFactory`] — Hash-based group-by with aggregation.

mod group;
pub use group::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot, Distinct,
    Dynamic, F64Cell, Fold, GroupFactory, GroupLimit, HashOnlyIntKeyExtractor, IntCell,
    IntKeyExtractor, IntPairKeyExtractor, IntRead, IntStrKeyExtractor, KeyExtractor, Max, MaxSlot,
    Min, MinSlot, NoRead, OpTuple, Read, RowKeyExtractor, RowKeySchema, StrMax, StrMin, StrRead,
    StringKeyExtractor, Sum, SumSlot, WideSum,
};

#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;

mod factory;
pub use factory::*;

use crate::data_flow::WorkStatus;
use crate::io::{FsRequest, HttpRequest};
use arrow_schema::ArrowError;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use thiserror::Error;

use super::channels::{Receiver, Sender};
use super::{FinishStatus, Operator};
use crate::worker::worker_waker;

mod pipeline_breaker;
pub use pipeline_breaker::{Consumer, Outputter, PipelineBreaker};

mod aggregate;
pub use aggregate::AggregateFactory;

mod filter;
pub use filter::FilterFactory;

mod map;
pub use map::MapFactory;

mod async_map;
pub use async_map::AsyncMapFactory;

mod fan_in;
pub use fan_in::FanInFactory;

mod sink;
pub use sink::SinkFactory;

mod default_unary_factory;
pub use default_unary_factory::DefaultUnaryFactory;

mod copy_out;
mod limit;
mod order_by_limit;

pub use copy_out::CopyOutFactory;
pub use limit::LimitFactory;
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
    /// An error from an operator defined outside this crate (e.g. the Parquet
    /// reader, now in `catalog`). Such operators map their own error into this.
    #[error("{0}")]
    Operator(Box<dyn std::error::Error + Send + Sync>),
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

    /// Return any pending filesystem read requests (e.g. async disk reads for
    /// parquet pages). See [`Operator::next_fs_requests`].
    fn next_fs_requests(&mut self) -> Result<Vec<FsRequest>> {
        Ok(vec![])
    }

    /// Return any pending HTTP requests (reads of remote regions). See
    /// [`Operator::next_http_requests`].
    fn next_http_requests(&mut self) -> Result<Vec<HttpRequest>> {
        Ok(vec![])
    }

    /// Whether this unary is ready to accept another input. Returns `false` when
    /// backpressured (e.g. waiting for IO to complete before consuming more).
    fn ready_for_more_work(&mut self) -> bool {
        true
    }

    /// Whether this unary may enter finalization after its input has drained.
    ///
    /// An operator with independently-running work can return `false` until all
    /// completions have been re-entered through [`run`](Self::run). The worker
    /// may park while it waits; the completion source must wake it.
    fn ready_to_finish(&self) -> bool {
        true
    }

    /// Handle a completed filesystem read; its bytes are already committed to
    /// the cache slot.
    fn process_fs_response<S: Sender<O>>(
        &mut self,
        _sender: &mut S,
        _request: FsRequest,
    ) -> Result<()> {
        unreachable!()
    }

    /// Handle a completed HTTP read; its bytes are already committed to the
    /// cache slot.
    fn process_http_response<S: Sender<O>>(
        &mut self,
        _sender: &mut S,
        _request: HttpRequest,
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

    /// Whether this transform is done consuming *before* its input has run dry,
    /// e.g. a `LIMIT` that has already buffered enough rows. The default is
    /// `false`: an operator finishes only when its input channel drains.
    ///
    /// When `true`, [`UnaryOperator::try_finish`] proceeds to finalization even
    /// though items may remain in the input channel; those leftovers are simply
    /// never consumed (the matching scan is abandoned, see
    /// [`Operator::upstream_cancel_flag`]).
    fn finished_consuming(&self) -> bool {
        false
    }

    /// See [`Operator::upstream_cancel_flag`].
    /// Wired through the wrapping [`UnaryOperator`] so a transform can ask the
    /// dataflow to abandon its upstream.
    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        None
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
            return Ok(self.unary.run(&mut self.sender)?);
        }

        let item = match self.receiver.try_recv() {
            None => return Ok(self.unary.run(&mut self.sender)?),
            Some(t) => t,
        };

        self.unary.consume(item, &mut self.sender)?;

        Ok(WorkStatus::Ran)
    }

    fn next_fs_requests(&mut self) -> super::Result<Vec<FsRequest>> {
        Ok(self.unary.next_fs_requests()?)
    }

    fn next_http_requests(&mut self) -> super::Result<Vec<HttpRequest>> {
        Ok(self.unary.next_http_requests()?)
    }

    fn process_fs_response(&mut self, request: FsRequest) -> super::Result<()> {
        Ok(self.unary.process_fs_response(&mut self.sender, request)?)
    }

    fn process_http_response(&mut self, request: HttpRequest) -> super::Result<()> {
        Ok(self
            .unary
            .process_http_response(&mut self.sender, request)?)
    }

    fn try_finish(&mut self) -> super::Result<FinishStatus> {
        // Normally an operator finishes only once its input has drained. A
        // transform that has decided to stop early (e.g. a satisfied `LIMIT`)
        // signals `finished_consuming` so we proceed regardless: the unconsumed
        // tail belongs to a scan that's being abandoned anyway.
        let done_consuming = || self.receiver.is_empty() || self.unary.finished_consuming();
        if !done_consuming() {
            return Ok(FinishStatus::Pending);
        }

        // The input may be empty while independently-running work is still in
        // flight. Do not notify the sibling barrier until every completion has
        // re-entered the operator and been forwarded downstream.
        if !self.unary.ready_to_finish() {
            return Ok(FinishStatus::Pending);
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
            // the emptiness check above and the decrement. Re-check and back
            // off if new work appeared (unless we're finishing early, in which
            // case we don't care about leftover input). Relaxed ordering
            // suffices because the channel ops themselves provide Acquire/Release.
            if !done_consuming() {
                self.notified_finished = false;
                self.siblings_left.fetch_add(1, Ordering::Relaxed);
                return Ok(FinishStatus::Pending);
            }

            if self.unary.finish(&mut self.sender)? {
                // `finish` may have emitted final batches downstream, so wake
                // any parked peers to pick that work up.
                worker_waker().notify();
                return Ok(FinishStatus::Done);
            }
            // A pipeline breaker still draining its outputter. The worker re-
            // drives it through `run` next iteration; reporting `Working` keeps
            // the worker from parking in between (its finish pass doesn't set
            // `did_work`). No notify — the output's own `send`s wake peers.
            return Ok(FinishStatus::Working);
        }
        Ok(FinishStatus::Pending)
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

    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.unary.upstream_cancel_flag()
    }
}
