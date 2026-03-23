//! Binary operators: two input channels, one output channel.
//!
//! A binary operator reads items of type `L` from its left input channel and
//! type `R` from its right input channel, applies a transform, and writes
//! items of type `O` to its output channel.
//!
//! # Key types
//!
//! - [`Binary<L, R, O>`] — The transform trait. Implementors define
//!   [`consume_left`](Binary::consume_left) and [`consume_right`](Binary::consume_right)
//!   (process one item from each side) and optionally [`finish`](Binary::finish)
//!   (emit final results after both inputs are drained).
//!
//! - [`BinaryOperator<L, R, O, B, RL, RR, S>`] — The [`Operator`] implementation
//!   that wires a [`Binary`] to two receivers and one sender.
//!
//! - [`BinaryFactory<L, R, O>`] — Trait for creating a [`Binary`] instance.
//!   Each factory is consumed once per worker to produce the binary transform
//!   for that worker.
//!
//! # Finishing protocol
//!
//! Binary operators use a single shared `siblings_left` counter (same as
//! unary). A worker decrements only when *both* its left and right input
//! channels are empty. When the counter reaches zero, [`Binary::finish`]
//! is called.

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use crate::memory::ReadBuffer;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

use super::Operator;
use super::channels::{Receiver, Sender};

mod concat;

pub use concat::ConcatFactory;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Channel(#[from] super::channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The core transform trait for binary operators.
///
/// `L` is the left input type, `R` is the right input type, `O` is the
/// output type.
pub trait Binary<L, R, O> {
    /// Process one item from the left input.
    fn consume_left<S: Sender<O>>(&mut self, item: L, sender: &mut S) -> Result<()>;

    /// Process one item from the right input.
    fn consume_right<S: Sender<O>>(&mut self, item: R, sender: &mut S) -> Result<()>;

    /// Called after both inputs are drained across all workers. Return `true`
    /// when fully done.
    fn finish<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<bool> {
        Ok(true)
    }
}

/// Creates a [`Binary`] transform instance. Consumed once per worker during
/// the build step.
pub trait BinaryFactory<L, R, O>: Send + 'static {
    type Binary: Binary<L, R, O>;
    fn build_binary(self) -> Self::Binary;
}

/// Wires a [`Binary`] transform to two receivers (left and right input channels)
/// and one sender (output channel), implementing the [`Operator`] trait so it
/// can be driven by the worker's event loop.
pub struct BinaryOperator<
    L,
    R,
    O,
    B: Binary<L, R, O>,
    RL: Receiver<L>,
    RR: Receiver<R>,
    S: Sender<O>,
> {
    binary: B,
    left_receiver: RL,
    right_receiver: RR,
    sender: S,
    notified_finished: bool,
    siblings_left: Arc<AtomicUsize>,
    _phantom: PhantomData<(L, R, O)>,
}

impl<L, R, O, B: Binary<L, R, O>, RL: Receiver<L>, RR: Receiver<R>, S: Sender<O>>
    BinaryOperator<L, R, O, B, RL, RR, S>
{
    pub fn new(
        binary: B,
        left_receiver: RL,
        right_receiver: RR,
        sender: S,
        siblings_left: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            binary,
            left_receiver,
            right_receiver,
            sender,
            notified_finished: false,
            siblings_left,
            _phantom: PhantomData,
        }
    }
}

impl<L, R, O, B: Binary<L, R, O>, RL: Receiver<L>, RR: Receiver<R>, S: Sender<O>> Operator
    for BinaryOperator<L, R, O, B, RL, RR, S>
{
    fn run_cpu_work(&mut self) -> super::Result<WorkStatus> {
        if let Some(item) = self.left_receiver.try_recv() {
            self.binary.consume_left(item, &mut self.sender)?;
            return Ok(WorkStatus::Ran);
        }
        if let Some(item) = self.right_receiver.try_recv() {
            self.binary.consume_right(item, &mut self.sender)?;
            return Ok(WorkStatus::Ran);
        }
        Ok(WorkStatus::Pending)
    }

    fn next_io_requests(&mut self) -> super::Result<Vec<IORequest>> {
        Ok(vec![])
    }

    fn process_disk_response(
        &mut self,
        _buffer: ReadBuffer,
        _request: IORequest,
    ) -> super::Result<()> {
        unreachable!()
    }

    fn try_finish(&mut self) -> super::Result<bool> {
        if !self.left_receiver.is_empty() || !self.right_receiver.is_empty() {
            return Ok(false);
        }

        let ready = if self.notified_finished {
            self.siblings_left.load(Ordering::Relaxed) == 0
        } else {
            self.notified_finished = true;
            self.siblings_left.fetch_sub(1, Ordering::Relaxed) == 1
        };

        if ready {
            // Race guard: a peer may have stolen work into our channels
            // between the is_empty() checks and the decrement.
            if !self.left_receiver.is_empty() || !self.right_receiver.is_empty() {
                self.notified_finished = false;
                self.siblings_left.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
            return Ok(self.binary.finish(&mut self.sender)?);
        }
        Ok(false)
    }

    fn try_steal_work(&mut self) -> super::Result<WorkStatus> {
        if let Some(item) = self.left_receiver.steal() {
            self.binary.consume_left(item, &mut self.sender)?;
            return Ok(WorkStatus::Ran);
        }
        if let Some(item) = self.right_receiver.steal() {
            self.binary.consume_right(item, &mut self.sender)?;
            return Ok(WorkStatus::Ran);
        }
        Ok(WorkStatus::Pending)
    }
}
