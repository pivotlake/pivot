//! Channels that connect operators within a worker's dataflow.
//!
//! Each pair of adjacent operators in a dataflow is connected by a channel. The upstream
//! operator writes to a [`Sender`], and the downstream operator reads from a [`Receiver`].
//! Channels are created by [`ChannelFactory`] instances during the factory build step on
//! the worker thread.
//!
//! Three channel types are provided, each with different trade-offs:
//!
//! - **[`stealable`]** — Work-stealing deque. The local worker pushes/pops from its own
//!   deque (LIFO for cache locality), while idle workers can steal from peers. Used for
//!   most intermediate stages.
//!
//! - **[`mpsc`]** — Multi-producer, single-consumer. Used for the final output channel
//!   (collecting results) and internally by [`return_to_worker`].
//!
//! - **[`return_to_worker`]** — Routes each message back to a specific worker based on
//!   [`WorkerIdOutput::worker_id`]. Used when a message must be processed by the same
//!   worker that owns related state (e.g. decompressed pages returning to the worker
//!   that holds the corresponding row group decoder).

use arrow_schema::ArrowError;
use crossbeam_deque::Injector;
use thiserror::Error;

mod stealable;
pub use stealable::{StealableChannelFactory, stealable};

mod mpsc;
pub use mpsc::{MpscSender, mpsc_channel};

mod return_to_worker;
pub use return_to_worker::{ReturnToWorkerMpscFactory, WorkerIdOutput, return_to_worker_mpsc};

use crate::operations::unary::parquet::RowGroupRequest;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("Receiver dropped")]
    MpscSendError,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Factory that produces a sender/receiver pair for a channel between two operators.
///
/// Each [`ChannelFactory`] is consumed once during the build step to create the channel
/// endpoints. The factory itself is `Send` (shipped to worker threads), but the
/// produced sender/receiver may not be (e.g. `Rc<Worker<T>>`).
pub trait ChannelFactory<T>: Send {
    type Sender: Sender<T> + 'static;
    type Receiver: Receiver<T> + 'static;
    fn build(self) -> (Self::Sender, Self::Receiver);
}

/// Factory for the root (source) operator's input channel, which has no sender.
///
/// The root operator receives work from an external source (e.g. a shared injector
/// queue), so only a receiver is produced.
pub trait RootChannelFactory<T>: Send {
    type Receiver: Receiver<T> + 'static;
    fn build(self) -> Self::Receiver;
}

/// Writing end of a channel between operators.
pub trait Sender<O> {
    fn send(&mut self, item: O) -> Result<()>;
}

/// Reading end of a channel between operators.
pub trait Receiver<I> {
    /// Returns `true` if no messages are currently available.
    fn is_empty(&self) -> bool;
    /// Try to receive a message from this worker's local queue.
    fn try_recv(&self) -> Option<I>;
    /// Try to steal a message from a peer worker's queue.
    /// Returns `None` if stealing is not supported or no work is available.
    fn steal(&self) -> Option<I>;
}

/// No-op sender for operators whose output is discarded
pub struct VoidSender;

impl Sender<()> for VoidSender {
    fn send(&mut self, _item: ()) -> Result<()> {
        Ok(())
    }
}

impl Sender<RowGroupRequest> for Injector<RowGroupRequest> {
    fn send(&mut self, item: RowGroupRequest) -> Result<()> {
        self.push(item);
        Ok(())
    }
}
