//! The streaming, page-parallel Parquet **write** pipeline — the write-side
//! mirror of dispatch's read pipeline (indexer → decompressor → decoder), built
//! from the same operator/channel toolkit. Each pipeline stage is its own module
//! (a file, or a directory when it has private helpers of its own); the modules
//! genuinely shared across stages sit beside them: the message [`types`] and the
//! pipeline's [`WriteError`](error::WriteError) in [`error`].
//!
//! A `RecordBatch` dataflow flows through these stages, and finished Parquet
//! files stream out the far end:
//!
//! 1. **ORDER BY** (dispatch's, present when the table is partitioned or
//!    sorted) — batches leave it holding one partition each, partitions
//!    grouped in tuple order, rows within a partition in key order.
//! 2. [`indexer`] (`RecordBatch → ColumnChunkJob`) — every batch funnels to
//!    one worker's indexer in creation order (the single-worker channel), so
//!    it sees the stream whole and ordered. It cuts one file per partition as
//!    each partition's batches end, emitting one job per column per
//!    `target_rows_per_group` window; a job carries a gather recipe over the
//!    stream's chunks, never row data. This is also where each file's variant
//!    columns pick their [`shredding`] — the only stage that sees a whole
//!    file's rows at once.
//! 3. [`encoder`] (`ColumnChunkJob → EncodedColumnChunk`) — materialize each
//!    job's rows by its gather, then encode: flatten the column into the
//!    leaves Parquet stores, dictionary-encode where it pays, else PLAIN; cut
//!    into pages and snappy-compress. The one heavy stage, and the only one
//!    that touches row data; finished chunks route back to their file's owner
//!    worker. (Leaf flattening, page cutting and index RLE live in the
//!    `encoder` directory.)
//! 4. [`assembler`] (`EncodedColumnChunk → AssembledFile`) — gather a file's
//!    column chunks, lay each out (the dictionary page, then the data pages)
//!    with sort-column footer statistics, and emit the finished file.
//!
//! An unpartitioned, unsorted write skips stage 1; the whole stream is the one
//! partition and becomes one file. The pipeline is reached by inserting into a
//! table: [`encode_record_batches_spec`] attaches these stages to a
//! `RecordBatch` dataflow and the caller continues into the upload operators,
//! which is how both INSERT and compaction write without a file's bytes ever
//! leaving the worker pool.

mod assembler;
mod compression;
pub(crate) mod encoder;
pub(crate) mod error;
mod indexer;
mod shredding;
mod stats;
pub(crate) use stats::aggregate_file_stats;
mod types;

pub(crate) use shredding::unshred_batch;
pub(crate) use types::AssembledFile;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use dispatch::{
    OperatorFactory, OperatorSpec, RecordBatchOperatorSpec, return_to_worker_mpsc, stealable,
    to_single_worker_mpsc,
};

use types::{ColumnChunkJob, EncodedColumnChunk};

/// Which worker hosts the next pipeline's [`indexer`]. Every batch of a write
/// funnels to one worker's indexer; rotating the host spreads concurrent
/// writes' serial stages across the pool instead of stacking them on worker 0.
static NEXT_INDEXER_HOST: AtomicUsize = AtomicUsize::new(0);

/// Attach the whole Parquet write pipeline to a `RecordBatch` dataflow
/// without executing it, finished files streaming out the far end. Catalog
/// INSERT and compaction use this to continue directly into asynchronous
/// upload and commit operators on the same workers.
///
/// A partitioned or sorted table's rows go through the grouped ORDER BY
/// first: batches leave it holding one partition each, partitions grouped in
/// tuple order, rows within a partition in key order. Everything then funnels
/// in that order to one worker's [`indexer`], which cuts the stream into one
/// file per partition. Variant columns fold back to their plain pair ahead of
/// all of it, so batches read back from differently-shredded files agree on a
/// schema before any of them regroup.
pub(crate) fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_file_bytes: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let spec = spec.project(|| |batch| unshred_batch(batch).expect("a variant column reassembles"));
    let spec = if partition_by.is_empty() && sort_by.is_empty() {
        spec
    } else {
        let partition_columns = partition_by
            .iter()
            .map(|name| {
                schema
                    .index_of(name)
                    .expect("partition columns name declared columns")
            })
            .collect();
        let sort_keys = sort_by
            .iter()
            .map(|name| {
                let column = schema
                    .index_of(name)
                    .expect("sort columns name declared columns");
                dispatch::OrderBy::new(column, false, true)
            })
            .collect();
        spec.order_by_per_partition(partition_columns, sort_keys)
    };

    let (dispatcher, heads) = spec.into_parts();
    let workers = heads.len();
    let indexer_host = NEXT_INDEXER_HOST.fetch_add(1, Ordering::Relaxed) % workers;
    let batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = batches.dispatcher().topology();
    batches
        .chain(
            to_single_worker_mpsc::<RecordBatch>(workers, indexer_host)
                .into_iter()
                .collect(),
            indexer::factories(
                partition_by,
                target_rows_per_group,
                target_file_bytes,
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod type_tests;
