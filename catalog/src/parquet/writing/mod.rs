//! The streaming, page-parallel Parquet **write** pipeline — the write-side
//! mirror of dispatch's read pipeline (indexer → decompressor → decoder), built
//! from the same operator/channel toolkit. Each pipeline stage is its own module
//! (a file, or a directory when it has private helpers of its own); the modules
//! genuinely shared across stages sit beside them: the message [`types`] and the
//! pipeline's [`WriteError`](error::WriteError) in [`error`].
//!
//! A `RecordBatch` dataflow flows through three work-stealing stages and
//! finished Parquet files ([`EncodedFile`]) stream out the far end (consumed via
//! `execute()`), so nothing waits for the whole input and only a bounded amount
//! sits in memory. After the first stage the unit of work is one **column chunk**
//! (a single column's values for one row group):
//!
//! 1. [`partition`] (`RecordBatch → ColumnChunkJob`) — split each batch by its
//!    `partition_by` tuple, buffer per partition, and once a partition reaches one
//!    file's worth, cut it into row groups and emit one job per column — stamping
//!    each with its file/partition provenance, `sort_bounds`, and sort-column
//!    statistics. This is also where each file's variant columns pick their
//!    [`shredding`], which is why that decision is per file: this stage is the
//!    only one that sees a whole file's rows at once. The sub-file remainder is
//!    consolidated per partition on worker 0. This is the pipeline's only
//!    pipeline-breaker, so every later stage is a plain parallel map or a gather.
//!    (Its partition-tuple and column-statistics helpers live in the `partition`
//!    directory.)
//! 2. [`encoder`] (`ColumnChunkJob → EncodedColumnChunk`) — encode each column
//!    chunk: flatten the column into the leaves Parquet stores, then for each,
//!    dictionary-encode it where it pays, else PLAIN; cut into pages and
//!    snappy-compress. The one heavy stage; finished chunks route back to their
//!    file's owner worker. (Leaf flattening, page cutting and index RLE live in
//!    the `encoder` directory.)
//! 3. [`assembler`] (`EncodedColumnChunk → EncodedFile`) — gather a file's column
//!    chunks, lay each out (the dictionary page, then the data pages) with
//!    sort-column footer statistics, and emit the finished file.
//!
//! An unpartitioned, unsorted write is just the degenerate case where
//! [`partition`] makes a single group (key `None`) and the per-file metadata is
//! empty. [`encode_record_batches`] feeds the pipeline an existing `RecordBatch`
//! dataflow, which is how compaction re-encodes a table without its batches ever
//! leaving the worker pool; INSERT continues into the upload operators via
//! [`encode_record_batches_spec`].

mod assembler;
pub(crate) mod encoder;
pub(crate) mod error;
mod partition;
mod shredding;
mod stats;
mod types;

pub(crate) use shredding::unshred_batch;
pub use types::{Compression, EncodedFile};

/// What every page the engine writes is compressed with. Held in one place so a
/// table's files are uniform however they were produced, by an INSERT or by a
/// compaction rewriting them.
pub const DEFAULT_COMPRESSION: Compression = Compression::Snappy;

use std::sync::Arc;

use arrow_array::RecordBatch;
use dispatch::{
    DataFlowHandle, OperatorFactory, OperatorSpec, RecordBatchFactoryBridge,
    RecordBatchOperatorSpec, return_to_worker_mpsc, stealable,
};

use types::{ColumnChunkJob, EncodedColumnChunk};

/// Encode an existing `RecordBatch` dataflow into Parquet files. Re-applies
/// `partition_by`/`sort_by`, so the output files carry the right partition tuple
/// and recomputed sort bounds.
/// This is the compaction path: a table scan feeds the batches a worker decodes
/// straight back into the write pipeline without ever leaving the pool.
pub fn encode_record_batches(
    spec: RecordBatchOperatorSpec,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_row_groups_per_file: usize,
    compression: Compression,
) -> DataFlowHandle<EncodedFile> {
    encode_record_batches_spec(
        spec,
        partition_by,
        sort_by,
        target_rows_per_group,
        target_row_groups_per_file,
        compression,
    )
    .execute()
}

/// Attach the Parquet encoding stages without executing the dataflow. Catalog
/// INSERT uses this to continue directly into asynchronous upload and commit
/// operators on the same workers.
pub(crate) fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_row_groups_per_file: usize,
    compression: Compression,
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
    encode_stages(
        batches,
        workers,
        partition_by,
        sort_by,
        target_rows_per_group,
        target_row_groups_per_file,
        compression,
    )
}

#[cfg(test)]
mod tests;

/// Chain the encode stages (partition onward) onto a `RecordBatch` dataflow and
/// run it, yielding finished [`EncodedFile`]s as they complete.
fn encode_stages<OF: OperatorFactory<RecordBatch> + Send + 'static>(
    batches: OperatorSpec<RecordBatch, OF>,
    workers: usize,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_row_groups_per_file: usize,
    compression: Compression,
) -> OperatorSpec<EncodedFile, impl OperatorFactory<EncodedFile> + 'static> {
    // One file's worth of rows; a partition flushes a file once it reaches this.
    let file_rows = target_rows_per_group
        .saturating_mul(target_row_groups_per_file)
        .max(1);
    let topology = batches.dispatcher().topology();
    batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            partition::factories(
                partition_by,
                sort_by,
                file_rows,
                target_rows_per_group,
                compression,
                workers,
            ),
        )
        .chain(
            stealable::<ColumnChunkJob>(topology).into_iter().collect(),
            encoder::factories(workers),
        )
        .chain(
            return_to_worker_mpsc::<EncodedColumnChunk>(workers)
                .into_iter()
                .collect(),
            assembler::factories(workers),
        )
}
