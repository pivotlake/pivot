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
//! 1. **Partitioner** (dispatch's `PartitionerFactory`, all workers) — split
//!    each batch by its partition tuple, sort each piece by the sort keys
//!    into self-contained memory, and ship every piece the moment it is cut.
//!    The stage buffers nothing.
//! 2. [`collector`] (`SortedPiece → FileJob`) — every piece funnels to one
//!    worker's collector (the single-worker channel), which groups pieces by
//!    partition and cuts a file per `target_bytes_per_file` gathered
//!    (several files per partition, each internally sorted; the compacter
//!    merges them later). Cutting fixes the file's identity and plans its
//!    k-way merge as independent slices; no row data moves here.
//! 3. [`sorter`] (`FileJob → ColumnChunkJob`) — file jobs go into a shared
//!    injector queue every worker claims from, so the drain never waits on
//!    any one worker; the worker completing a file's last slice deals the sorted file
//!    into one job per column per `target_rows_per_group` window, each a
//!    gather recipe over the file's chunks, never row data. This is also
//!    where the file's variant columns pick their [`shredding`] — the only
//!    stage that sees a whole file's rows at once.
//! 4. [`encoder`] (`ColumnChunkJob → EncodedColumnChunk`) — materialize each
//!    job's rows by its gather, then encode: flatten the column into the
//!    leaves Parquet stores, dictionary-encode where it pays, else PLAIN; cut
//!    into pages and snappy-compress. The one heavy stage, and the only one
//!    that touches row data; finished chunks route back to their file's owner
//!    worker. (Leaf flattening, page cutting and index RLE live in the
//!    `encoder` directory.)
//! 5. [`assembler`] (`EncodedColumnChunk → AssembledFile`) — gather a file's
//!    column chunks, lay each out (the dictionary page, then the data pages)
//!    with sort-column footer statistics, and emit the finished file.
//!
//! An unpartitioned, unsorted write takes the same path; its batches form one
//! partition whose runs are the arrival order. The pipeline is reached by
//! inserting into a table: [`encode_record_batches_spec`] attaches these
//! stages to a `RecordBatch` dataflow and the caller continues into the
//! upload operators, which is how both INSERT and compaction write without a
//! file's bytes ever leaving the worker pool.

mod assembler;
mod collector;
pub(crate) mod encoder;
pub(crate) mod error;
mod shredding;
mod sorter;
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
    OperatorFactory, OperatorSpec, OrderBy, PartitionerFactory, RecordBatchOperatorSpec,
    SortedPiece, injector, return_to_worker_mpsc, stealable, to_single_worker_mpsc,
};

use types::{ColumnChunkJob, EncodedColumnChunk, FileJob};

/// Which worker hosts the next pipeline's [`collector`]. Every piece of a write
/// funnels to one worker's collector; rotating the host spreads concurrent
/// writes' serial stages across the pool instead of stacking them on worker 0.
static NEXT_COLLECTOR_HOST: AtomicUsize = AtomicUsize::new(0);

/// Attach the whole Parquet write pipeline to a `RecordBatch` dataflow
/// without executing it, finished files streaming out the far end. Catalog
/// INSERT and compaction use this to continue directly into asynchronous
/// upload and commit operators on the same workers.
///
/// Variant columns fold back to their plain pair ahead of everything, so
/// batches read back from differently-shredded files agree on a schema before
/// any of them regroup. `target_bytes_per_file` is where the collector cuts a
/// partition's gathered pieces into a file; INSERT passes a bounded target so
/// the pipeline never holds more than a file's worth of raw bytes per open
/// partition, while compaction passes `usize::MAX` to keep one globally
/// sorted file per partition.
pub(crate) fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_bytes_per_file: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let spec = spec.project(|| |batch| unshred_batch(batch).expect("a variant column reassembles"));
    let partition_columns: Vec<usize> = partition_by
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .expect("partition columns name declared columns")
        })
        .collect();
    let sort_keys: Vec<OrderBy> = sort_by
        .iter()
        .map(|name| {
            let column = schema
                .index_of(name)
                .expect("sort columns name declared columns");
            OrderBy::new(column, false, true)
        })
        .collect();

    let (dispatcher, heads) = spec.into_parts();
    let workers = heads.len();
    let collector_host = NEXT_COLLECTOR_HOST.fetch_add(1, Ordering::Relaxed) % workers;
    let batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = batches.dispatcher().topology();
    batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            PartitionerFactory::create_for_workers(partition_columns, sort_keys.clone(), workers),
        )
        .chain(
            to_single_worker_mpsc::<SortedPiece>(workers, collector_host)
                .into_iter()
                .collect(),
            collector::factories(
                partition_by,
                sort_keys.into(),
                target_rows_per_group,
                target_bytes_per_file,
                workers,
            ),
        )
        .chain(
            injector::<FileJob>(workers).into_iter().collect(),
            sorter::factories(workers),
        )
        .chain(
            injector::<ColumnChunkJob>(workers).into_iter().collect(),
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
