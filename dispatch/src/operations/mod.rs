mod materializer;

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use bytes::Bytes;
pub use materializer::{
    MaterializeJobGeneratorFactory, MaterializeRequest,
    MaterializerFactory,
};
use thiserror::Error;

mod channels;
pub use channels::{
    ChannelFactory, MpscSender, ReturnToWorkerMpscFactory, Sender,
    StealableChannelFactory, mpsc_channel, return_to_worker_mpsc, stealable,
};

pub mod unary;
pub use unary::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Unary(#[from] unary::Error),
    #[error("{0}")]
    Channel(#[from] channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// An `Operator` is a single node in a dataflow. The same instance of an `Operation` will live
/// for the entirety of the dataflow's lifetime, able to do CPU work or request IO. The same
/// instance of the operator is expected to live in *each* worker
///
/// CPU and IO are purposefully separated so that the worker can exhaust resources appropriately. For example, if
/// no IO is happening, the worker can poll all operators for outstanding IO, and while it's doing
/// said IO it can run any cpu work.
///
/// An `Operator` is meant to be parallelism-aware. Thus, it is canonical for an Operator to,
/// for example, send data to a sibling operator
pub trait Operator {
    fn run_cpu_work(&mut self) -> Result<WorkStatus>;
    fn next_io_request(&mut self) -> Result<Option<IORequest>>;
    fn process_disk_response(&mut self, buffer: Bytes, request: IORequest) -> Result<()>;
    fn try_finish(&mut self) -> Result<bool>;

    fn try_steal_cpu_work(&mut self) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    fn try_steal_io_request(&mut self) -> Result<Option<IORequest>> {
        Ok(None)
    }
}
