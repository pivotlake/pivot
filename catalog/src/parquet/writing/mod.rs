//! The streaming, page-parallel Parquet **write** pipeline — the write-side
//! mirror of dispatch's read pipeline (indexer → decompressor → decoder), built
//! from the same operator/channel toolkit. Each pipeline stage is its own module
//! (a file, or a directory when it has private helpers of its own); the modules
//! genuinely shared across stages sit beside them: the message [`types`] and the
//! pipeline's [`WriteError`](error::WriteError) in [`error`].
//!
//! A flush's buffered items flow through four work-stealing stages and finished
//! Parquet files ([`EncodedFile`]) stream out the far end (consumed via
//! `execute()`), so nothing waits for the whole flush and only a bounded amount
//! sits in memory. After the first stage the unit of work is one **column chunk**
//! (a single column's values for one row group):
//!
//! 1. [`convert`] (`T → RecordBatch`) — each worker flattens the items it
//!    steals into Arrow batches.
//! 2. [`partition`] (`RecordBatch → ColumnChunkJob`) — split each batch by its
//!    `partition_by` tuple, buffer per partition, and once a partition reaches one
//!    file's worth, cut it into row groups and emit one job per column — stamping
//!    each with its file/partition provenance, `sort_bounds`, and sort-column
//!    statistics. The sub-file remainder is consolidated per partition on worker
//!    0. This is the pipeline's only pipeline-breaker, so every later stage is a
//!    plain parallel map or a gather. (Its partition-tuple and column-statistics
//!    helpers live in the `partition` directory.)
//! 3. [`encoder`] (`ColumnChunkJob → EncodedColumnChunk`) — encode each column
//!    chunk: dictionary-encode it where it pays, else PLAIN; cut into pages and
//!    snappy-compress. The one heavy stage; finished chunks route back to their
//!    file's owner worker. (Page cutting and index RLE live in the `encoder`
//!    directory.)
//! 4. [`assembler`] (`EncodedColumnChunk → EncodedFile`) — gather a file's column
//!    chunks, lay each out (the dictionary page, then the data pages) with
//!    sort-column footer statistics, and emit the finished file.
//!
//! There is one pipeline, not two: an unpartitioned, unsorted write is just the
//! degenerate case where [`partition`] makes a single group (key `None`) and the
//! per-file metadata is empty. [`encode_items`] feeds it a flush's items (via
//! [`convert`]); [`encode_record_batches`] feeds it an existing `RecordBatch`
//! dataflow (skipping [`convert`]), which is how compaction re-encodes a table
//! without its batches ever leaving the worker pool.

mod assembler;
mod convert;
mod encoder;
mod error;
mod partition;
mod types;

pub use types::EncodedFile;

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use dispatch::{
    DataFlowDispatcher, DataFlowHandle, OperatorFactory, OperatorSpec, RecordBatchFactoryBridge,
    RecordBatchOperatorSpec, return_to_worker_mpsc, stealable, values_input,
};

use types::{ColumnChunkJob, EncodedColumnChunk};

/// A buffered ingest item that flattens itself into one Arrow `RecordBatch`.
/// Conversion runs inside the pipeline (stage 1) on a worker, so the receive
/// path only ever buffers the cheap, unconverted item.
pub trait ToRecordBatch: Send + 'static {
    /// Output rows this item will produce — a cheap count (not the full
    /// conversion), used for the flush threshold.
    fn num_rows(&self) -> usize;
    /// Flatten into one batch, or `None` if it produced no rows.
    fn to_record_batch(self) -> Result<Option<arrow_array::RecordBatch>, ArrowError>;
}

/// Encode a flush's buffered `items` into Parquet files: convert them to Arrow
/// batches, then route each row to a file by its `partition_by` tuple, recording
/// the sort-key range (both specs may be empty — then it's one unpartitioned,
/// unsorted file stream). The returned handle streams [`EncodedFile`]s: iterate
/// it on a blocking thread and write each file (with its manifest metadata) as it
/// arrives. This is the ingest flush path.
pub fn encode_items<T: ToRecordBatch>(
    dispatcher: &DataFlowDispatcher,
    items: Vec<T>,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows: usize,
    target_row_groups: usize,
) -> DataFlowHandle<EncodedFile> {
    let workers = dispatcher.worker_count();
    let batches = values_input(dispatcher, items).chain(
        stealable::<T>(workers).into_iter().collect(),
        convert::factories::<T>(workers),
    );
    encode_stages(
        batches,
        workers,
        partition_by,
        sort_by,
        target_rows,
        target_row_groups,
    )
    .execute()
}

/// Encode an existing `RecordBatch` dataflow into Parquet files (the batches are
/// already Arrow, so this skips [`convert`]). Re-applies `partition_by`/`sort_by`,
/// so the output files carry the right partition tuple and recomputed sort bounds.
/// This is the compaction path: a table scan feeds the batches a worker decodes
/// straight back into the write pipeline without ever leaving the pool.
pub fn encode_record_batches(
    spec: RecordBatchOperatorSpec,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows: usize,
    target_row_groups: usize,
) -> DataFlowHandle<EncodedFile> {
    encode_record_batches_spec(spec, partition_by, sort_by, target_rows, target_row_groups)
        .execute()
}

/// Attach the Parquet encoding stages without executing the dataflow. Catalog
/// INSERT uses this to continue directly into asynchronous upload and commit
/// operators on the same workers.
pub(crate) fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows: usize,
    target_row_groups: usize,
) -> OperatorSpec<EncodedFile, impl OperatorFactory<EncodedFile> + Send + 'static> {
    let (dispatcher, heads) = spec.into_parts();
    let workers = heads.len();
    let batches = OperatorSpec::new(
        dispatcher,
        heads
            .into_iter()
            .map(RecordBatchFactoryBridge::new)
            .collect::<Vec<_>>(),
    );
    encode_stages(
        batches,
        workers,
        partition_by,
        sort_by,
        target_rows,
        target_row_groups,
    )
}

/// Chain the encode stages (partition onward) onto a `RecordBatch` dataflow and
/// run it, yielding finished [`EncodedFile`]s as they complete.
fn encode_stages<OF: OperatorFactory<RecordBatch> + Send + 'static>(
    batches: OperatorSpec<RecordBatch, OF>,
    workers: usize,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows: usize,
    target_row_groups: usize,
) -> OperatorSpec<EncodedFile, impl OperatorFactory<EncodedFile> + Send + 'static> {
    // One file's worth of rows; a partition flushes a file once it reaches this.
    let file_rows = target_rows.saturating_mul(target_row_groups).max(1);
    batches
        .chain(
            stealable::<RecordBatch>(workers).into_iter().collect(),
            partition::factories(partition_by, sort_by, file_rows, target_rows, workers),
        )
        .chain(
            stealable::<ColumnChunkJob>(workers).into_iter().collect(),
            encoder::factories(workers),
        )
        .chain(
            return_to_worker_mpsc::<EncodedColumnChunk>(workers)
                .into_iter()
                .collect(),
            assembler::factories(workers),
        )
}
