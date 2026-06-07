//! Operators and channels: the parallel execution toolkit.
//!
//! This module contains everything that executes within a worker's
//! `DataFlow`:
//!
//! - **[`Operator`]** — The trait every node in a dataflow implements. An operator
//!   is a stateful object that can do CPU work, request disk IO, process IO completions,
//!   steal work from peers, and signal when it is finished.
//!
//! - **[`channels`]** — Sender/receiver pairs that connect adjacent operators in a
//!   dataflow. Three flavors: work-stealing (default), mpsc (final output), and
//!   return-to-worker (worker-affinity routing).
//!
//! - **[`nullary`]** — Source-like operators with no input channel that can emit
//!   output or perform side effects directly.
//!
//! - **[`unary`]** — The [`UnaryOperator`], which reads from one
//!   input channel and writes to one output channel, applying a [`Unary`]
//!   transform. Most query stages (filter, project, count, etc.) are built as unary
//!   operators. The [`UnaryOperatorFactory`] creates them during the factory build step
//!
//! # Operator lifecycle
//!
//! Operators are created during the factory build step on the worker thread (not before —
//! some hold `Rc` or other non-`Send` state). Once created, the worker's event loop
//! repeatedly calls:
//!
//! 1. [`run_cpu_work`](Operator::run_cpu_work) — does one 'unit' of CPU work, optionally.
//!    Returns [`WorkStatus::Ran`] if it did anything, [`WorkStatus::Pending`] if no work was available
//!    to do.
//!
//! 2. [`next_io_requests`](Operator::next_io_requests) — return any pending IO requests
//!    (e.g. read a parquet page from disk). The worker submits these asynchronously and
//!    delivers completions via [`process_disk_response`](Operator::process_disk_response).
//!
//! 3. [`try_finish`](Operator::try_finish) — called when the input channel is drained
//!    and all sibling operators (across workers) have also drained. The operator does
//!    any final work (e.g. emit aggregation results) and returns `true` when fully done.
//!
//! 4. [`try_steal_work`](Operator::try_steal_work) — called when the worker is
//!    idle. The operator attempts to steal from a peer worker's input channel.

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use thiserror::Error;

pub mod channels;

pub mod in_memory;
pub use in_memory::{Forward, InjectorSourceFactory};

pub mod nullary;
pub use nullary::*;

pub mod unary;
pub use unary::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Unary(#[from] unary::Error),
    #[error("{0}")]
    Nullary(#[from] nullary::Error),
    #[error("{0}")]
    Channel(#[from] channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single node in a `DataFlow`.
///
/// Each operator lives for the full lifetime of the dataflow on one worker thread.
/// Sibling operators (the same stage on different workers) share the same identifier,
/// enabling work stealing and coordinated finishing (e.g. a shared atomic counter for
/// `Count`).
///
/// CPU work and IO are intentionally separate: the worker can saturate the IO queue
/// while running CPU work from a different operator, hiding disk latency.
pub trait Operator {
    /// Try to consume one item from the input and produce output.
    /// Returns [`WorkStatus::Ran`] if work was done, [`Pending`](WorkStatus::Pending) otherwise.
    fn run_cpu_work(&mut self) -> Result<WorkStatus>;

    /// Return any pending IO requests (e.g. parquet page reads).
    /// The worker will submit them and later call [`process_disk_response`](Self::process_disk_response).
    fn next_io_requests(&mut self) -> Result<Vec<IORequest>>;

    /// Handle a completed disk read (its bytes are already committed to the
    /// cache). Called by the worker when IO finishes.
    fn process_disk_response(&mut self, request: IORequest) -> Result<()>;

    /// Attempt to finish, return whether the operator is ready to finish. Regardless of whether it
    /// is, this function may be called many times.
    fn try_finish(&mut self) -> Result<bool>;

    /// Try to steal work from a peer worker's channel. Default: no stealing.
    fn try_steal_work(&mut self) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }
}
