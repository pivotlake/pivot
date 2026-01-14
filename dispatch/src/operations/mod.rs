mod count;
mod filter;
mod group;
mod materializer;
mod order_by_limit;
mod output_operation;
mod project;
mod stdout;

use crate::io::OperationIOSubmitter;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
pub use count::Count;
pub use filter::Filter;
pub use group::*;
pub use materializer::Materializer;
pub use order_by_limit::{OrderBy, OrderByLimit};
pub use project::Project;
use std::any::Any;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Materializer(#[from] materializer::Error),
    #[error("{0}")]
    OrderByLimit(#[from] order_by_limit::Error),
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub enum ConsumeContext {
    IORequest(Box<dyn Any + Send>),
    Publisher,
}

pub trait Output: Send + Sync {
    fn write(&mut self, batch: RecordBatch);
    fn finish(&mut self) {}
}

/// An `Operation` is a single node in a pipeline. The same instance of an `Operation` will live
/// for the entirety of the pipeline's lifetime, continuously being able to be consume new record
/// batches
///
/// An `Operation` is meant to be parallelism aware. Thus, it is canonical for an Operation to,
/// for example, hold a barrier, synchronizing with other counterpart operations in sibling
/// pipelines
pub trait Operation: Send {
    /// Run on a new batch from one of the operations inputs (or another context, given by `context`),
    /// optionally returning a new record batch to be run on by its subscribers
    fn consume(
        &mut self,
        context: &ConsumeContext,
        io_submitter: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> Result<Option<RecordBatch>>;
}

/// A `PipelineBreaker` is a node in the pipeline that does not only consume, but outputs to
/// another place (outside the pipeline) once it's entire parent subtree is "finished".
/// It outputs not by returning RecordBatches, but by using an internal `Box<dyn Output>`.
/// Any node that needs to output once the pipeline is finished is a pipeline breaker.
///
/// A `PipelineBreaker` is so named because given the fact that it essentially "collects" data
/// until the pipeline finishes and outputs it to outside the pipeline,
/// it is a divider between multiple pipelines. For example, a Group By is a `PipelineBreaker`,
/// because it only sends on rows once it has collected all input rows to it
/// (since any intermediate result could be incorrect).
pub trait PipelineBreaker: Operation {
    /// "Finish" the pipeline-breaker - By consuming Self this essentially "promises" that no input
    /// will ever be delivered again, and the pipeline-breaker can safely do anything it wants at
    /// this point, such as outputting to another pipeline
    fn finish(self: Box<Self>) -> Result<()>;
}
