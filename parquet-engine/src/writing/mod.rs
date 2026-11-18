//! Streaming Parquet file construction for INSERT and compaction.
//!
//! INSERT prepares files as follows:
//!
//! 1. [`partition_sorter`] splits each input batch by table partition, sorts
//!    each piece, and copies it into independent memory. Each piece is one
//!    sorted run located on the NUMA node that produced it.
//! 2. [`file_collector`] chooses the runs belonging to each output file and
//!    groups them by source node. Every worker collects the runs it sorted
//!    into pending runs all workers share, and the byte target applies to the
//!    retained Arrow data each partition has pending.
//!
//! The remainder of an INSERT:
//!
//! 3. The local planner in [`file_merge`] reads the sort keys of one node's runs
//!    and divides their k-way merge into independent output slices. It does not
//!    copy rows. A node-local work queue keeps this planning beside the input
//!    data and distributes the resulting slice jobs among workers on that node.
//! 4. The local executors heap-merge each slice, gather every column, and
//!    produce one sorted run for the node.
//! 5. The global planner waits for all participating nodes, then divides the
//!    k-way merge of their node-level runs into global output slices. A shared
//!    queue is appropriate here because the inputs already span nodes.
//! 6. The global executors materialize those slices on their selected NUMA
//!    nodes. Together, in slice order, they are the fully sorted file input.
//! 7. [`row_group_planner`] divides the ordered file into column jobs.
//! 8. [`shredder`] splits each column job into its primitive leaves, shredding
//!    a variant column a record batch of rows per worker turn.
//! 9. [`encoder`] materializes and encodes each leaf as Parquet pages.
//! 10. [`assembler`] returns encoded leaf chunks to one worker per file and
//!     produces the final file bytes and footer metadata.
//!
//! In outline, the two merge levels are:
//!
//! ```text
//! node 0 runs ── local k-way merge ── node 0 run ─┐
//! node 1 runs ── local k-way merge ── node 1 run ─┼─ global k-way merge ─ file
//! node N runs ── local k-way merge ── node N run ─┘
//! ```
//!
//! A merge level with one run is a zero-copy identity. Files without sort
//! columns pass through all merge stages without planning or copying rows.
//!
//! An unsorted file skips the merge work. Files remain on dispatch workers
//! through assembly and upload because their bytes occupy worker-owned ring
//! memory.
//!
//! Compaction does not build whole files in memory. Its input files are
//! already sorted, so [`compaction`] finds the output's row order by merging the
//! files' sort columns alone, reads the rows as the output reaches them, and
//! hands encoded row groups on in file order for a writer to append to a
//! file as they come. It shares the shredder and encoder with INSERT.

mod assembler;
pub use assembler::{EncodedRowGroup, ParquetFileBuilder};
mod compaction;
mod compression;
pub(crate) mod encoder;
pub(crate) mod error;
mod file_collector;
mod file_merge;
mod leaves;
pub use compaction::{FileCompaction, MemoryReservation};
mod partition_sorter;
mod row_group_planner;
mod shredder;
mod shredding;
pub use shredding::FileShredding;
mod stats;
pub use stats::aggregate_file_stats;
mod types;
pub mod upload;

pub use types::{AssembledFile, WrittenFileMetadata};

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use dispatch::memory::SlabAllocator;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorFactory, OperatorSpec, OrderBy,
    RecordBatchOperatorSpec, node_work_queue, return_to_worker_mpsc, shared_work_queue, stealable,
    values_input,
};

use partition_sorter::{PartitionSorterFactory, SortedPartitionRun};
use types::{
    ColumnChunkJob, EncodedLeafChunk, FileOrderInput, GlobalMergeJob, LeafChunkJob, LocalMergeJob,
    LocalMergeResult, ReadyFile,
};

/// How an INSERT's rows are split into files: by the values of the
/// `key_columns`, in `key_columns` order. A file holds rows of one key, and
/// its partition values are the key column values its rows share, by column
/// name. The last `unwritten_trailing_columns` of the batch columns exist
/// only to key the partition (a value computed from a row's columns ahead of
/// the encoder, as a table format partitioned by a transform of a column
/// needs): they are read for the key and the partition values, and are not
/// written to the file.
#[derive(Clone, Debug, Default)]
pub struct Partitioning {
    pub key_columns: Arc<[String]>,
    pub unwritten_trailing_columns: usize,
}

impl Partitioning {
    /// Files keyed by the values of the named table columns, every one of
    /// them written.
    pub fn by_columns(columns: &[String]) -> Self {
        Self {
            key_columns: columns.into(),
            unwritten_trailing_columns: 0,
        }
    }
}

/// Appends Parquet construction to an existing record-batch dataflow.
///
/// The retained-memory target cuts each partition's INSERT rows into output
/// files. Callers first restore shredded variants with
/// [`unshred_batches_spec`]. `schema` is the batch schema, the unwritten
/// trailing columns of `partitioning` last.
pub fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partitioning: Partitioning,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let Partitioning {
        key_columns: partition_column_names,
        unwritten_trailing_columns,
    } = partitioning;
    assert!(
        unwritten_trailing_columns <= schema.fields().len(),
        "the unwritten trailing columns are among the batch columns"
    );
    let partition_column_indices: Vec<usize> = partition_column_names
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .expect("partition columns name declared columns")
        })
        .collect();
    let order_by = sort_order(&schema, &sort_column_names);

    let (dispatcher, heads) = spec.into_parts();
    let input_batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = input_batches.dispatcher().topology();
    let files = input_batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            PartitionSorterFactory::create_for_workers(
                partition_column_indices,
                order_by.clone(),
                topology,
            ),
        )
        // Each worker collects the runs it sorted: the collectors share their
        // pending runs, so no run queues behind a single collecting worker.
        .chain(
            stealable::<SortedPartitionRun>(topology)
                .into_iter()
                .collect(),
            // Every run was sorted by the stage above, so each file's merge
            // leaves it in the order of the sort columns.
            file_collector::factories(
                partition_column_names,
                unwritten_trailing_columns,
                order_by.into(),
                sort_column_names,
                target_rows_per_group,
                target_in_memory_bytes_per_file,
                None,
                topology,
            ),
        );
    encode_files_spec(files)
}

/// How many input row groups a compaction's shredding plan reads.
const PLAN_SAMPLE_ROW_GROUPS: usize = 8;

/// How many rows a batch of the sample holds. A batch is folded back to
/// plain variants in one call, which takes several microseconds a row and
/// keeps its worker from every other dataflow meanwhile, so the batches are
/// small.
const PLAN_SAMPLE_BATCH_ROWS: usize = 512;

/// Plan the shredding of a compaction's output before the merge writes its
/// first row group. Only variant columns are shredded, so a table without
/// any reads nothing. Otherwise the plan comes from the variant columns of a
/// few row groups spread evenly over the input. `output_schema` is the
/// schema the output is written under, before shredding: the input files'
/// own, or one that declares the same columns with more on them, such as
/// field ids.
pub fn plan_compaction_shredding(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<crate::ParquetTable>,
    output_schema: SchemaRef,
) -> Result<Arc<FileShredding>, dispatch::DataFlowError> {
    use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
    let schema = &output_schema;
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&column| crate::is_variant_field(schema.field(column)))
        .collect();
    let row_groups_with_rows: Vec<usize> = table
        .row_groups()
        .iter()
        .enumerate()
        .filter(|(_, metadata)| metadata.num_rows > 0)
        .map(|(group, _)| group)
        .collect();
    if variant_columns.is_empty() || row_groups_with_rows.is_empty() {
        return Ok(Arc::new(shredding::plan_unshredded(schema.clone())));
    }
    let sampled_count = PLAN_SAMPLE_ROW_GROUPS.min(row_groups_with_rows.len());
    let requests: Vec<QueryRowGroupMetadata> = (0..sampled_count)
        .map(|sampled| {
            let group = row_groups_with_rows[sampled * row_groups_with_rows.len() / sampled_count];
            QueryRowGroupMetadata::new(table, group, RowSelection::All).bypassing_caches()
        })
        .collect();
    let projection = crate::types::projection::Projection::columns(variant_columns.clone());
    let sample = crate::fetch_row_groups(
        values_input(dispatcher, requests),
        table,
        projection,
        PLAN_SAMPLE_BATCH_ROWS,
        false,
    );
    let batches = unshred_batches_spec(sample.record_batches()).collect()?;
    let shredding = shredding::plan_shredding_from_sample(schema, &variant_columns, &batches)
        .map_err(|error| {
            dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into())
        })?;
    Ok(Arc::new(shredding))
}

/// Merge the files of `table`, each already sorted by `sort_column_names`,
/// into one stream of encoded row groups of `target_rows_per_group` rows in
/// that order. The order is found by merging the files' sort columns alone,
/// and the rows are then read only as the output reaches them (see
/// [`compaction`]). `shredding` is the output's plan (see
/// [`plan_compaction_shredding`]). A table without sort columns has its
/// files' rows passed through in file order. Fails for a table with deleted
/// rows.
pub fn compact_files_spec(
    dispatcher: &DataFlowDispatcher,
    table: Arc<crate::ParquetTable>,
    shredding: Arc<FileShredding>,
    sort_column_names: &[String],
    target_rows_per_group: usize,
) -> Result<FileCompaction<impl OperatorFactory<EncodedRowGroup> + 'static>, dispatch::DataFlowError>
{
    compaction::compact_files_spec(
        dispatcher,
        table,
        shredding,
        sort_column_names.into(),
        target_rows_per_group,
    )
    .map_err(|error| dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into()))
}

pub fn unshred_batches_spec(spec: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
    spec.project(build_batch_unshredder)
}

/// One worker's transform of a batch read from a file into the batch with
/// its variant columns unshredded.
fn build_batch_unshredder()
-> impl FnMut(RecordBatch) -> dispatch::UnaryResult<RecordBatch> + Send + 'static {
    // Built on the first batch rather than here: `SlabAllocator::new` takes
    // a write buffer from the worker's memory context, which exists only
    // once the worker is running. One allocator per worker, reused across
    // batches so its part-filled slab carries over.
    let mut allocator: Option<SlabAllocator> = None;
    move |batch| {
        let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
        Ok(shredding::unshred_batch(batch, allocator)?)
    }
}

fn sort_order(schema: &SchemaRef, sort_column_names: &[String]) -> Vec<OrderBy> {
    sort_column_names
        .iter()
        .map(|name| {
            let column = schema
                .index_of(name)
                .expect("sort columns name declared columns");
            OrderBy::new(column, false, true)
        })
        .collect()
}

/// Shared half of INSERT and compaction: merge the already-sorted runs for each
/// output file, divide the result into row groups, encode its columns, and
/// assemble the Parquet bytes.
fn encode_files_spec<OF>(
    files: OperatorSpec<FileOrderInput, OF>,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static>
where
    OF: OperatorFactory<FileOrderInput> + 'static,
{
    let worker_count = files.dispatcher().worker_count();
    let topology = files.dispatcher().topology();
    files
        .chain(
            node_work_queue::<FileOrderInput>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::LocalMergePlanner>::create_for_workers(worker_count),
        )
        .chain(
            node_work_queue::<LocalMergeJob>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::LocalMergeExecutor>::create_for_workers(worker_count),
        )
        .chain(
            shared_work_queue::<LocalMergeResult>(worker_count)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::GlobalMergePlanner>::create_for_workers(worker_count),
        )
        .chain(
            node_work_queue::<GlobalMergeJob>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::GlobalMergeExecutor>::create_for_workers(
                worker_count,
            ),
        )
        .chain(
            node_work_queue::<ReadyFile>(topology).into_iter().collect(),
            row_group_planner::factories(topology),
        )
        .chain(
            node_work_queue::<ColumnChunkJob>(topology)
                .into_iter()
                .collect(),
            shredder::factories(worker_count),
        )
        .chain(
            node_work_queue::<LeafChunkJob>(topology)
                .into_iter()
                .collect(),
            encoder::factories(worker_count),
        )
        .chain(
            return_to_worker_mpsc::<EncodedLeafChunk>(worker_count)
                .into_iter()
                .collect(),
            assembler::factories(worker_count),
        )
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod type_tests;
