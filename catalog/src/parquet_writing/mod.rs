//! The streaming, page-parallel Parquet **write** pipeline - the write-side
//! mirror of dispatch's read pipeline (indexer → decompressor → decoder), built
//! from the same operator/channel toolkit. Each pipeline stage is its own module
//! (a file, or a directory when it has private helpers of its own); the modules
//! genuinely shared across stages sit beside them: the message [`types`] and the
//! pipeline's [`WriteError`](error::WriteError) in [`error`].
//!
//! A write's batches flow through three work-stealing stages and finished
//! Parquet files ([`EncodedFile`]) stream out the far end (consumed via
//! `execute()`), so nothing waits for the whole write and only a bounded amount
//! sits in memory. After the first stage the unit of work is one **column chunk**
//! (a single column's values for one row group):
//!
//! 1. [`partition`] (`RecordBatch → ColumnChunkJob`) - split each batch by its
//!    `partition_by` tuple, buffer per partition, and once a partition reaches one
//!    file's worth, cut it into row groups and emit one job per column - stamping
//!    each with its file/partition provenance, `sort_bounds`, and sort-column
//!    statistics. The sub-file remainder is consolidated per partition on worker
//!    0. This is the pipeline's only pipeline-breaker, so every later stage is a
//!    plain parallel map or a gather. (Its partition-tuple and column-statistics
//!    helpers live in the `partition` directory.)
//! 2. [`encoder`] (`ColumnChunkJob → EncodedColumnChunk`) - encode each column
//!    chunk: dictionary-encode it where it pays, else PLAIN; cut into pages and
//!    snappy-compress. The one heavy stage; finished chunks route back to their
//!    file's owner worker. (Page cutting and index RLE live in the `encoder`
//!    directory.)
//! 3. [`assembler`] (`EncodedColumnChunk → EncodedFile`) - gather a file's column
//!    chunks, lay each out (the dictionary page, then the data pages) with
//!    sort-column footer statistics, and emit the finished file.
//!
//! There is one pipeline for every writer: an unpartitioned, unsorted write is
//! just the degenerate case where [`partition`] makes a single group (key `None`)
//! and the per-file metadata is empty. [`encode_spec`] chains it onto an
//! existing `RecordBatch` dataflow: an `INSERT`'s source plan, or a compaction
//! scan whose decoded batches re-enter the pipeline without ever leaving the
//! worker pool.

mod assembler;
mod encoder;
mod error;
mod partition;
mod types;

pub use types::EncodedFile;

use std::sync::Arc;

use arrow_array::RecordBatch;
use dispatch::{
    OperatorFactory, OperatorSpec, RecordBatchFactoryBridge, RecordBatchOperatorSpec,
    return_to_worker_mpsc, stealable,
};

use types::{ColumnChunkJob, EncodedColumnChunk};

/// Rows per Parquet row group. A write's batches are cut into row groups of
/// about this many rows as they stream through the pipeline.
pub const ROW_GROUP_ROWS: usize = 128 * 1024;
/// Row groups per Parquet file. Once this many accumulate, a file is emitted and
/// written; the write's remainder lands in a final, smaller file.
pub const ROW_GROUPS_PER_FILE: usize = 8;

/// The encode stages (partition → encoder → assembler) chained onto a
/// `RecordBatch` dataflow, as an *unlaunched* spec streaming finished
/// [`EncodedFile`]s. Applies `partition_by`/`sort_by`, so the output files
/// carry the right partition tuple and recomputed sort bounds (both specs may
/// be empty, giving one unpartitioned, unsorted file stream). An `INSERT`
/// chains its write-commit sink on before executing; compaction executes it
/// directly and writes each file as it streams out.
pub fn encode_spec(
    spec: RecordBatchOperatorSpec,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows: usize,
    target_row_groups: usize,
) -> OperatorSpec<EncodedFile, impl OperatorFactory<EncodedFile> + 'static> {
    let (dispatcher, heads) = spec.into_parts();
    let workers = heads.len();
    let batches = OperatorSpec::new(
        dispatcher,
        heads
            .into_iter()
            .map(RecordBatchFactoryBridge::new)
            .collect::<Vec<_>>(),
    );
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
