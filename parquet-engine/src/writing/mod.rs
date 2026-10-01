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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{FieldRef, SchemaRef};
use arrow_select::interleave::interleave;
use dispatch::memory::SlabAllocator;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorFactory, OperatorSpec, OrderBy,
    RecordBatchOperatorSpec, node_work_queue, return_to_worker_mpsc, shared_work_queue, stealable,
    to_single_worker_mpsc, values_input,
};

use crate::reading::record_batch_metadata::{global_row_group, row_index};
use crate::types::metadata::{FileStatistics, QueryRowGroupMetadata, RowSelection};
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

/// Sort a compaction's rows by their sort columns alone. `keys` scans just the
/// sort columns with their row-group metadata; the result is every row's
/// `(row group, row)` in sorted order, which [`plan_compaction_shredding`] and
/// [`encode_sorted_rows_spec`] take.
pub fn sort_compaction_rows(
    keys: RecordBatchOperatorSpec,
    sort_column_count: usize,
) -> Result<Vec<(u32, u32)>, dispatch::DataFlowError> {
    let order_by: Vec<OrderBy> = (0..sort_column_count)
        .map(|column| OrderBy::new(column, false, true))
        .collect();
    let batches = keys
        .project(|| crate::plain_row_group_column)
        .order_by(order_by)
        .project(|| {
            |batch: RecordBatch| {
                let columns = batch.num_columns();
                batch
                    .project(&[columns - 2, columns - 1])
                    .expect("the metadata columns end the batch")
            }
        })
        .collect()?;
    let mut order = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    for batch in &batches {
        let groups = batch.column(0).as_primitive::<UInt32Type>();
        let rows = batch.column(1).as_primitive::<UInt32Type>();
        order.extend(
            groups
                .values()
                .iter()
                .copied()
                .zip(rows.values().iter().copied()),
        );
    }
    Ok(order)
}

/// Plan the shredding of a sorted compaction's output exactly as a write of
/// the whole sorted file plans it: from the rows at the positions a file's
/// shredding samples, found through `order` and decoded alone. Only the
/// variant columns are read, and a table without any reads nothing.
pub fn plan_compaction_shredding(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<crate::ParquetTable>,
    order: &[(u32, u32)],
) -> Result<Arc<FileShredding>, dispatch::DataFlowError> {
    let fields = table.schema().fields().to_vec();
    let variant_columns: Vec<usize> = (0..fields.len())
        .filter(|&column| crate::is_variant_field(&fields[column]))
        .collect();
    if variant_columns.is_empty() || order.is_empty() {
        return Ok(Arc::new(FileShredding {
            column_shredding: vec![None; fields.len()],
            schema: Arc::new(arrow_schema::Schema::new(fields)),
        }));
    }
    // The sampled rows by row group, each with its place in the sample, in
    // row order: an index selection decodes its rows in that order and
    // numbers them by their position in it.
    let positions: Vec<usize> = shredding::sampled_rows(order.len()).collect();
    let mut rows_by_group: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
    for (sample_index, &position) in positions.iter().enumerate() {
        let (group, row) = order[position];
        rows_by_group
            .entry(group)
            .or_default()
            .push((row, sample_index));
    }
    for rows in rows_by_group.values_mut() {
        rows.sort_unstable();
    }
    let projection = crate::types::projection::Projection::columns(variant_columns.clone());
    let requests: Vec<crate::RowGroupRequest> = rows_by_group
        .iter()
        .map(|(&group, rows)| {
            let indices: Vec<u32> = rows.iter().map(|&(row, _)| row).collect();
            crate::RowGroupRequest::from(
                QueryRowGroupMetadata::new(
                    table,
                    group as usize,
                    RowSelection::Indices(indices.into()),
                ),
                &projection,
            )
        })
        .collect();
    let sample =
        crate::fetch_row_groups(values_input(dispatcher, requests), table, projection, true);
    let batches = unshred_batches_spec(sample).collect()?;

    // Lay the sampled rows out in the sample's order, as the file holds them.
    let mut placement: Vec<Option<(usize, usize)>> = vec![None; positions.len()];
    for (batch_index, batch) in batches.iter().enumerate() {
        let selected = row_index(batch);
        let mut row_in_batch = 0;
        for (group, logical_end) in row_group_runs(batch) {
            while row_in_batch < logical_end {
                let (_, sample_index) =
                    rows_by_group[&group][selected.value(row_in_batch) as usize];
                placement[sample_index] = Some((batch_index, row_in_batch));
                row_in_batch += 1;
            }
        }
    }
    let placement: Vec<(usize, usize)> = placement
        .into_iter()
        .map(|slot| slot.expect("every sampled row was fetched"))
        .collect();
    let sampled_fields: Vec<FieldRef> =
        batches[0].schema().fields()[..variant_columns.len()].to_vec();
    let sampled_columns = (0..variant_columns.len())
        .map(|column| {
            let arrays: Vec<&dyn Array> = batches
                .iter()
                .map(|batch| batch.column(column).as_ref())
                .collect();
            interleave(&arrays, &placement)
        })
        .collect::<Result<Vec<ArrayRef>, _>>()
        .map_err(|error| {
            dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into())
        })?;
    let sample = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(sampled_fields)),
        sampled_columns,
    )
    .expect("the sampled columns match their fields");
    let variant_columns: Vec<(usize, usize)> = variant_columns
        .into_iter()
        .enumerate()
        .map(|(sample_column, column)| (column, sample_column))
        .collect();
    let shredding =
        shredding::plan_shredding(fields, Default::default(), &[sample], &variant_columns)
            .map_err(|error| {
                dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into())
            })?;
    Ok(Arc::new(shredding))
}

/// The batch's rows as runs of one row group: `(group, logical end)` pairs.
fn row_group_runs(batch: &RecordBatch) -> Vec<(u32, usize)> {
    let groups = global_row_group(batch);
    let run_ends = groups.run_ends();
    let physical_start = run_ends.get_start_physical_index();
    let group_values = groups.values().as_primitive::<UInt32Type>();
    run_ends
        .sliced_values()
        .enumerate()
        .map(|(run_offset, logical_end)| {
            (
                group_values.value(physical_start + run_offset),
                logical_end as usize,
            )
        })
        .collect()
}

/// Appends Parquet construction to a sorted compaction whose rows `order`
/// names (see [`sort_compaction_rows`]). Every input row group is fetched up
/// front, in the order the sorted rows first need it, and held undecoded.
/// Each is decoded once the output reaches the row group before it in its
/// file, so every input file has its current row group and the next one
/// decoded, and each is let go of after its last row. The rows are laid out
/// and encoded a row group at a time (see [`streaming`]). `shredding` is the
/// output's plan (see [`plan_compaction_shredding`]).
pub fn encode_sorted_rows_spec(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<crate::ParquetTable>,
    order: Vec<(u32, u32)>,
    shredding: Arc<FileShredding>,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let row_groups = table.row_groups();
    let mut first_row = vec![usize::MAX; row_groups.len()];
    for (position, &(group, _)) in order.iter().enumerate() {
        let first = &mut first_row[group as usize];
        if *first == usize::MAX {
            *first = position;
        }
    }
    let mut needed: Vec<usize> = (0..row_groups.len())
        .filter(|&group| first_row[group] != usize::MAX)
        .collect();
    needed.sort_unstable_by_key(|&group| first_row[group]);

    // A row group is released once the stretch being laid out reaches the
    // first row of the row group needed before it in the same file; a file's
    // first row group goes out at once. The row group the next row lies in is
    // always released by then, since the one before it in its file is needed
    // earlier still.
    let mut release_after = vec![0; row_groups.len()];
    let mut previous_in_file: HashMap<*const FileStatistics, usize> = HashMap::new();
    for &group in &needed {
        let file = Arc::as_ptr(&row_groups[group].statistics);
        if let Some(previous) = previous_in_file.insert(file, group) {
            release_after[group] = (first_row[previous] + 1).saturating_sub(reorder::STRETCH_ROWS);
        }
    }

    let projection = crate::types::projection::Projection::all(table.schema().fields().len());
    let requests: Vec<crate::RowGroupRequest> = needed
        .iter()
        .map(|&group| {
            crate::RowGroupRequest::from(
                QueryRowGroupMetadata::new(table, group, RowSelection::All),
                &projection,
            )
        })
        .collect();
    let consumed = Arc::new(AtomicUsize::new(0));
    let gate = crate::DecodeGate {
        consumed: consumed.clone(),
        release_after: Arc::new(release_after),
    };
    // The gate runs beside the stage that advances `consumed`, so the worker
    // that lays out a stretch is the one that next releases row groups.
    let reorder_worker = dispatcher.next_worker();
    let rows = crate::fetch_row_groups_gated(
        values_input(dispatcher, requests),
        projection,
        gate,
        reorder_worker,
    );
    let (dispatcher, heads) = unshred_batches_spec(rows).into_parts();
    let worker_count = heads.len();
    let jobs = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>()).chain(
        to_single_worker_mpsc::<RecordBatch>(worker_count, reorder_worker)
            .into_iter()
            .collect(),
        reorder::factories(Arc::new(order), consumed, worker_count),
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
