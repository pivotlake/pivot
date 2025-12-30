use crate::table::RowGroupMetadataHandle;
use arrow_array::RecordBatch;
use parquetd::Projection;

/// An Input is the data entrypoint for a pipeline. An input can be built either for IO requests
/// or for in memory queues.
///
/// Conceptually, it is expected that an Input is above a large stream of *shared* data that other
/// sibling inputs (in other workers) are also above. Therefore, it is important not to pull more
/// data from an input that can be handled, as it can cause lack of work balance between inputs.
///
/// `poll_io` and `poll_record_batch` are set as different methods (instead of one `poll` returning
/// an enum) so that the controlling worker can specifically request io/memory. For
/// example, the worker's IO may be saturated, but the worker may have nothing to do; in this case,
/// it does NOT want to request more IO (even if it exists, as this will stop other workers that
/// perhaps have availability from pulling the IO) but may want to request record batches.
pub trait Input: Send + Sync {
    /// Is the input's source complete? If all inputs sources are finished and no other IO/work
    /// exists in the system, the pipeline will complete
    fn source_finished(&self) -> bool;

    /// Poll for any IO work available to read row groups
    fn poll_io(&self) -> Option<(RowGroupMetadataHandle, Option<Projection>)> {
        None
    }

    /// For for any in-memory work available
    fn poll_record_batch(&self) -> Option<RecordBatch> {
        None
    }
}
