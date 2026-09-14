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
//! Compaction replaces those two stages: its selected files already share one
//! known partition, and every decoded batch retains its source file's sort
//! order. It wraps each batch as a run without partitioning, sorting, or copying,
//! then collects all selected runs into one file without a memory-size cutoff.
//!
//! Both paths share the remainder:
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

mod assembler;
mod compression;
pub(crate) mod encoder;
pub(crate) mod error;
mod file_collector;
mod file_merge;
mod leaves;
mod partition_sorter;
mod presorted_run;
mod reorder;
mod row_group_planner;
mod shredder;
mod shredding;
pub use shredding::FileShredding;
mod stats;
pub use stats::aggregate_file_stats;
mod streaming;
mod types;

pub use types::AssembledFile;

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use dispatch::memory::SlabAllocator;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorFactory, OperatorSpec, OrderBy,
    RecordBatchOperatorSpec, node_work_queue, return_to_worker_mpsc, shared_work_queue, stealable,
    to_single_worker_mpsc, values_input,
};

use partition_sorter::{PartitionSorterFactory, SortedPartitionRun};
use presorted_run::PresortedRun;
use types::{
    ColumnChunkJob, EncodedLeafChunk, FileOrderInput, GlobalMergeJob, LeafChunkJob, LocalMergeJob,
    LocalMergeResult, ReadyFile,
};

/// Appends Parquet construction to an existing record-batch dataflow.
///
/// The retained-memory target cuts each partition's INSERT rows into output
/// files. Callers first restore shredded variants with
/// [`unshred_batches_spec`].
pub fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition_column_names: Arc<[String]>,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
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
            file_collector::factories(
                partition_column_names,
                order_by.into(),
                target_rows_per_group,
                target_in_memory_bytes_per_file,
                None,
                topology,
            ),
        );
    encode_files_spec(files)
}

/// Appends Parquet construction to a compaction scan. Every input batch already
/// belongs to the selected partition and is ordered as it was in its source
/// file, so the INSERT-only partition/sort/copy stage is skipped. All batches
/// form one ordered output stream, split at target-sized row-group boundaries;
/// sorted tables still use the downstream k-way merge to combine
/// overlapping source-file runs.
pub fn encode_compaction_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition: Option<crate::PartitionValues>,
    partition_column_names: Arc<[String]>,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
    max_file_size: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let spec = unshred_batches_spec(spec);
    let order_by = sort_order(&schema, &sort_column_names);
    let (dispatcher, heads) = spec.into_parts();
    let worker_count = heads.len();
    let file_collector_worker = dispatcher.next_worker();
    let input_batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = input_batches.dispatcher().topology();
    let presorted_runs =
        PresortedRun::create_for_workers(partition.as_ref(), &partition_column_names, topology);
    let files = input_batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            presorted_runs,
        )
        .chain(
            to_single_worker_mpsc::<SortedPartitionRun>(worker_count, file_collector_worker)
                .into_iter()
                .collect(),
            file_collector::factories(
                partition_column_names,
                order_by.into(),
                target_rows_per_group,
                usize::MAX,
                Some(max_file_size),
                topology,
            ),
        );
    encode_files_spec(files)
}

/// How far ahead of the rows laid out so far a sorted compaction requests
/// its input row groups, in output row groups.
const LOOKAHEAD_ROW_GROUPS: usize = 4;

/// How many input row groups a compaction's shredding plan samples rows from,
/// and how many rows of each.
const PLAN_SAMPLE_ROW_GROUPS: usize = 32;
const PLAN_SAMPLE_ROWS_PER_GROUP: usize = 128;

/// Plan the shredding of a compaction's output from rows sampled evenly
/// across its input row groups, the spread a write over the whole file
/// samples, so the layout does not hinge on whichever rows sort first. Only
/// the variant columns are sampled, and a table without any reads nothing.
pub fn plan_compaction_shredding(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<crate::ParquetTable>,
) -> Result<Arc<FileShredding>, dispatch::DataFlowError> {
    use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
    let schema = table.schema();
    let fields = schema.fields().to_vec();
    let variant_columns: Vec<usize> = (0..fields.len())
        .filter(|&column| crate::is_variant_field(&fields[column]))
        .collect();
    if variant_columns.is_empty() {
        return Ok(Arc::new(FileShredding {
            schema: Arc::new(arrow_schema::Schema::new(fields)),
            column_shredding: vec![None; schema.fields().len()],
        }));
    }
    let projection = crate::types::projection::Projection::columns(variant_columns.clone());
    let groups = table.row_groups();
    let group_stride = groups.len().div_ceil(PLAN_SAMPLE_ROW_GROUPS).max(1);
    let requests: Vec<crate::RowGroupRequest> = (0..groups.len())
        .step_by(group_stride)
        .map(|group| {
            let rows = groups[group].num_rows as usize;
            let row_stride = rows.div_ceil(PLAN_SAMPLE_ROWS_PER_GROUP).max(1);
            let rows: Vec<u32> = (0..rows)
                .step_by(row_stride)
                .map(|row| row as u32)
                .collect();
            crate::RowGroupRequest::from(
                QueryRowGroupMetadata::new(table, group, RowSelection::Indices(rows.into())),
                &projection,
            )
        })
        .collect();
    let sample =
        crate::fetch_row_groups(values_input(dispatcher, requests), table, projection, false);
    let batches = unshred_batches_spec(sample).collect()?;
    let sampled_columns: Vec<(usize, usize)> = variant_columns
        .into_iter()
        .enumerate()
        .map(|(chunk_column, column)| (column, chunk_column))
        .collect();
    let shredding =
        shredding::plan_shredding(fields, Default::default(), &batches, &sampled_columns).map_err(
            |error| dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into()),
        )?;
    Ok(Arc::new(shredding))
}

/// Appends Parquet construction to a compaction scan of a sorted table's sort
/// columns alone: the keys are sorted with their `(row group, row)` metadata,
/// then every input row group is fetched whole in the order the sorted rows
/// first need it, and the rows are laid out and encoded a row group at a
/// time (see [`streaming`]), so the whole decoded input is never held at
/// once. `shredding` is the output's plan (see [`plan_compaction_shredding`]).
pub fn encode_sorted_by_keys_spec(
    keys: RecordBatchOperatorSpec,
    table: Arc<crate::ParquetTable>,
    shredding: Arc<FileShredding>,
    schema: SchemaRef,
    partition: Option<crate::PartitionValues>,
    sort_column_count: usize,
    target_rows_per_group: usize,
    max_file_size: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let order_by: Vec<OrderBy> = (0..sort_column_count)
        .map(|column| OrderBy::new(column, false, true))
        .collect();
    let sorted = keys
        .project(|| crate::plain_row_group_column)
        .order_by(order_by);
    let fetch = crate::OrderedFetch {
        order: Arc::new(OnceLock::new()),
        consumed: Arc::new(AtomicUsize::new(0)),
        lookahead_rows: LOOKAHEAD_ROW_GROUPS * target_rows_per_group,
    };
    let rows = crate::materialize_in_order(
        sorted,
        table,
        crate::types::projection::Projection::all(schema.fields().len()),
        fetch.clone(),
    );
    let (dispatcher, heads) = unshred_batches_spec(rows).into_parts();
    let worker_count = heads.len();
    let reorder_worker = dispatcher.next_worker();
    let jobs = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>()).chain(
        to_single_worker_mpsc::<RecordBatch>(worker_count, reorder_worker)
            .into_iter()
            .collect(),
        reorder::factories(fetch, worker_count),
    );
    streaming::encode_gathered_spec(
        jobs,
        shredding,
        partition,
        target_rows_per_group,
        max_file_size,
    )
}

pub fn unshred_batches_spec(spec: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
    spec.project(|| {
        // Built on the first batch rather than here: `SlabAllocator::new` takes
        // a write buffer from the worker's memory context, which exists only
        // once the worker is running. One allocator per worker, reused across
        // batches so its part-filled slab carries over.
        let mut allocator: Option<SlabAllocator> = None;
        move |batch| {
            let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
            shredding::unshred_batch(batch, allocator).expect("a variant column reassembles")
        }
    })
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
mod tests;

#[cfg(test)]
mod type_tests;
