mod count;
mod filter;

use arrow_array::RecordBatch;

pub type Identifier = usize;

pub use count::Count;
pub use filter::Filter;

/// An `Operation` is a single node in a pipeline. The same instance of an `Operation` will live
/// for the entirety of the pipeline's lifetime, continuously being able to be "consumed" from (i.e.,
/// its output read) or run on new `RecordBatch`s.
///
/// An `Operation` is meant to be parallelism aware. Thus, it is canonical for an Operation to,
/// for example, hold a barrier, synchronizing with other counterpart operations in sibling
/// pipelines
pub trait Operation: Send + Sync {
    /// The unique identifier of the operation in the pipeline
    fn id(&self) -> Identifier;
    /// Attempt to consume an output batch from this Operation. Should no output batch exist, this
    /// function should return `None`.
    fn consume_output_batch(&mut self) -> Option<RecordBatch>;
    /// Run on a new batch from one of the operations inputs
    fn run(&mut self, batch: &RecordBatch);
    /// This function will only be called once upon completion of the pipeline, and is guaranteed
    /// to only be called after every input's finish has been called. The Operation may optionally
    /// return a last RecordBatch to be run on by it's outputs
    fn finish(&mut self) -> Option<RecordBatch>;
}
