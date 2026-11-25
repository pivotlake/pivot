//! Plans a k-way merge and executes it as independent merge tasks.
//!
//! [`KWayMergePlan`] represents a complete merge level. Zero runs produce no
//! output, one run passes through without copying, and multiple runs produce
//! independent [`KWayMergeTask`]s.
//!
//! Each merge task produces one key range of the merged output. A boundary
//! between two neighbouring merge tasks is a place in every run: the rows
//! before it go to the earlier task. Every boundary therefore takes a binary
//! search in every run, and how the boundaries are found depends on the size
//! of the merge.
//!
//! A merge small enough for each merge task to produce a single record batch
//! finds its boundaries while the plan is built. A larger merge would keep
//! the one thread that builds the plan busy for longer than the merge itself
//! takes, so its plan only picks the boundaries: it reads a sample of the
//! rows, every so many rows of each run, sorts it, and takes rows at even
//! steps through the sorted sample. Each merge task then finds its own two
//! boundaries in the runs.
//!
//! The worker that runs a merge task cuts the task's rows into sort chunks,
//! and for each sort chunk orders the row positions and gathers each output
//! column once. The number of input runs does not add extra copies of the
//! row data.
//!
//! Cutting a sort chunk out of a merge task's rows takes a search in every
//! run as well. Merge tasks and sort chunks therefore hold more rows the
//! more runs there are, which keeps the number of searches below the number
//! of rows however large the merge is.
//!
//! Boundaries found while planning and sort chunks are cut the same way: by
//! repeatedly bisecting the largest remaining run. Its midpoint is the
//! pivot, and binary searches locate the corresponding boundary in every
//! other run. A sort chunk therefore gets no larger than asked even when
//! many runs contain the same key. Equal keys use run position only to make
//! the independently found boundaries agree with each other; callers must
//! not rely on that internal order.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::ArrowError;

use crate::arrays::take_chunked;
use crate::memory::SlabAllocator;
use crate::numa::dominant_node;
use crate::operations::channels::NodeIdOutput;
use crate::operations::unary::order_by_limit::OrderBy;

use super::ChunkedRowSorter;
use super::batch_sort::MERGE_SLICE_ROWS;
use super::keys::{KeyOrdering, RunRow};

/// How many merge tasks a large merge is divided into. Enough for every
/// worker to have several, so that the workers finish at about the same
/// time.
const TARGET_TASK_COUNT: usize = 256;

/// The most rows of output one merge task produces, unless the number of
/// runs asks for more (see [`TASK_ROWS_PER_RUN`]). A worker runs a merge
/// task to the end without returning to its loop, so this bounds how long
/// the worker is away from the rest of its work.
const MAX_TASK_ROWS: usize = 128 * MERGE_SLICE_ROWS;

/// The fewest rows a merge task produces for every run of the merge. Finding
/// the task's boundaries takes a binary search in every run. With this many
/// rows of every run to merge afterwards, the searches stay a small share of
/// the task.
const TASK_ROWS_PER_RUN: usize = 128;

/// The fewest rows a sort chunk holds for every run of the merge. Cutting a
/// sort chunk out of a merge task's rows takes a search in every run too,
/// but within the few rows of the run that the task holds.
const SORT_CHUNK_ROWS_PER_RUN: usize = 16;

/// How many sampled rows lie between two neighbouring sampled boundaries.
/// The sample only estimates how many rows a merge task gets, and more
/// samples to a task make the estimate closer.
const SAMPLES_PER_TASK: usize = 64;

/// Where each merge task's key range ends and the next task's begins.
enum MergeTaskBoundaries {
    /// Found while the plan was built: for each boundary, how many rows of
    /// each run sort before it.
    Planned(Vec<Vec<usize>>),
    /// Picked from a sample, for the merge tasks to find in the runs.
    Sampled(SampledBoundaries),
}

/// Boundaries between merge tasks that are rows of a sorted sample of the
/// runs.
struct SampledBoundaries {
    /// How many rows lie between two neighbouring sampled rows of a run.
    sample_gap: usize,
    /// Which row of each run is its first sampled row. A run shorter than
    /// this has no sampled row.
    first_sample_row_by_run: Vec<usize>,
    /// The ranks each run's sampled rows have in the sorted sample, lowest
    /// first, which is also the order of the rows in the run.
    sample_ranks_by_run: Vec<Vec<usize>>,
    boundaries: Vec<SampledBoundary>,
}

/// A sampled row that is the boundary between two merge tasks. The rows that
/// sort before it belong to the earlier task.
struct SampledBoundary {
    /// The sampled row's rank in the sorted sample.
    sample_rank: usize,
    /// The run the sampled row is in, and which row of that run it is.
    run: usize,
    run_row: usize,
    /// The sampled row's position in the plan's batches.
    position: RunRow,
    /// How many rows of each run sort before the sampled row. The first of
    /// the two neighbouring merge tasks to need them finds them.
    run_rows_before: OnceLock<Vec<usize>>,
}

/// An immutable plan for merging sorted runs.
struct ParallelKWayMerge {
    order_by: Arc<[OrderBy]>,
    /// Compares rows of `batches` with each other. It points into the
    /// batches' buffers, and is declared before them so that it is dropped
    /// first.
    ordering: KeyOrdering<'static>,
    batches: Vec<RecordBatch>,
    run_indexes: Vec<RunIndex>,
    /// The NUMA node each merge task runs on. There is one entry per merge
    /// task, in key order.
    task_nodes: Vec<usize>,
    /// The boundary between each merge task and the next, so one fewer than
    /// there are merge tasks.
    boundaries: MergeTaskBoundaries,
    /// The most rows that are ordered and gathered in one go.
    sort_chunk_rows: usize,
    row_count: usize,
}

impl ParallelKWayMerge {
    /// Plans the merge of `runs`. Every run may contain several batches, but
    /// its rows must already be ordered by `order_by`.
    #[cfg(test)]
    fn try_new(order_by: Arc<[OrderBy]>, runs: Vec<Vec<RecordBatch>>) -> Result<Self, ArrowError> {
        let run_count = runs.len();
        Self::try_new_on_nodes(order_by, runs, vec![0; run_count], 1)
    }

    fn try_new_on_nodes(
        order_by: Arc<[OrderBy]>,
        runs: Vec<Vec<RecordBatch>>,
        run_nodes: Vec<usize>,
        node_count: usize,
    ) -> Result<Self, ArrowError> {
        assert!(node_count > 0, "a merge needs at least one NUMA node");
        assert_eq!(runs.len(), run_nodes.len());
        assert!(run_nodes.iter().all(|&node| node < node_count));
        let run_indexes = build_run_indexes(&runs);
        let run_count = run_indexes.len();
        let row_count: usize = run_indexes.iter().map(|index| index.row_count).sum();
        let batches = runs.into_iter().flatten().collect::<Vec<_>>();
        let ordering = KeyOrdering::try_new(&order_by, &batches, &batches)?;

        // Each merge task of a small merge produces one record batch, so
        // that the merge still has a task for every worker. A large merge
        // gets `TARGET_TASK_COUNT` tasks, or more when those would run for
        // too long.
        let sort_chunk_rows = MERGE_SLICE_ROWS.max(SORT_CHUNK_ROWS_PER_RUN * run_count);
        let shortest_task_rows = sort_chunk_rows.max(TASK_ROWS_PER_RUN * run_count);
        let longest_task_rows = MAX_TASK_ROWS.max(shortest_task_rows);
        let task_rows =
            (row_count / TARGET_TASK_COUNT).clamp(shortest_task_rows, longest_task_rows);
        // Merge tasks of one record batch mean a merge of at most
        // `TARGET_TASK_COUNT` record batches in a few dozen runs. Finding
        // its boundaries here takes less than a sample does: a sampled
        // boundary costs `SAMPLES_PER_TASK` rows read and sorted, and still
        // a search in every run afterwards. It also fills the record
        // batches, where sampled merge tasks are aimed under one.
        let (task_nodes, boundaries) = if task_rows == MERGE_SLICE_ROWS {
            Self::plan_boundaries(&ordering, &run_indexes, &run_nodes, node_count, task_rows)
        } else {
            Self::sample_boundaries(
                &order_by,
                &batches,
                &run_indexes,
                &run_nodes,
                node_count,
                task_rows,
            )?
        };

        // SAFETY: the ordering points into the buffers of the batches'
        // arrays. Those are reference counted and stay where they are when
        // `batches` moves into the plan, and the plan holds the batches,
        // unchanged, for as long as it holds the ordering.
        let ordering =
            unsafe { std::mem::transmute::<KeyOrdering<'_>, KeyOrdering<'static>>(ordering) };
        Ok(Self {
            order_by,
            ordering,
            batches,
            run_indexes,
            task_nodes,
            boundaries,
            sort_chunk_rows,
            row_count,
        })
    }

    /// Divides the merge of the runs into merge tasks of at most `task_rows`
    /// rows, and returns each task's node with the boundaries between the
    /// tasks.
    fn plan_boundaries(
        ordering: &KeyOrdering,
        run_indexes: &[RunIndex],
        run_nodes: &[usize],
        node_count: usize,
        task_rows: usize,
    ) -> (Vec<usize>, MergeTaskBoundaries) {
        let all_run_rows = run_indexes
            .iter()
            .map(|index| 0..index.row_count)
            .collect::<Vec<_>>();
        let mut run_rows_by_task = Vec::new();
        split_into_key_ranges(
            ordering,
            run_indexes,
            all_run_rows,
            task_rows,
            &mut run_rows_by_task,
        );
        // A merge task goes to the node that holds most of its rows.
        let task_nodes = run_rows_by_task
            .iter()
            .map(|run_rows| {
                let mut rows_by_node = vec![0usize; node_count];
                for (rows, &node) in run_rows.iter().zip(run_nodes) {
                    rows_by_node[node] += rows.len();
                }
                dominant_node(&rows_by_node)
            })
            .collect();
        // A merge task starts in every run where the task before it ends.
        let run_rows_before_boundaries = run_rows_by_task
            .iter()
            .skip(1)
            .map(|run_rows| run_rows.iter().map(|rows| rows.start).collect())
            .collect();
        (
            task_nodes,
            MergeTaskBoundaries::Planned(run_rows_before_boundaries),
        )
    }

    /// Picks the boundaries between merge tasks of about `task_rows` rows
    /// from a sample of the runs' rows, and returns each task's node with
    /// the boundaries.
    fn sample_boundaries(
        order_by: &[OrderBy],
        batches: &[RecordBatch],
        run_indexes: &[RunIndex],
        run_nodes: &[usize],
        node_count: usize,
        task_rows: usize,
    ) -> Result<(Vec<usize>, MergeTaskBoundaries), ArrowError> {
        let row_count: usize = run_indexes.iter().map(|index| index.row_count).sum();
        // The sample only estimates how many rows a merge task gets. Tasks
        // are aimed an eighth under `task_rows`, so that few come out over
        // it.
        let task_count = row_count.div_ceil(task_rows - task_rows / 8);

        // Every `sample_gap`-th row of the runs laid end to end, as a
        // `(batch, row)` position. Each run's samples are in key order.
        let sample_gap = (row_count / (task_count * SAMPLES_PER_TASK)).max(1);
        let mut samples = Vec::new();
        let mut first_sample_row_by_run = Vec::with_capacity(run_indexes.len());
        let mut sample_count_by_run = Vec::with_capacity(run_indexes.len());
        let mut next_sample_row = sample_gap / 2;
        for run in run_indexes {
            first_sample_row_by_run.push(next_sample_row);
            let sample_count_before_run = samples.len();
            while next_sample_row < run.row_count {
                let position = run.locate(next_sample_row);
                samples.push((position.chunk as u32, position.row as u32));
                next_sample_row += sample_gap;
            }
            next_sample_row -= run.row_count;
            sample_count_by_run.push(samples.len() - sample_count_before_run);
        }
        // Samples with equal keys stay in run order, which is the order the
        // boundary searches give rows with equal keys.
        ChunkedRowSorter::try_new(order_by, batches)?
            .merge_sorted_runs(&mut samples, &sample_count_by_run);

        // The run each sample is in, in sorted order, and for each run the
        // ranks its samples got.
        let sample_runs: Vec<usize> = samples
            .iter()
            .map(|&(batch, _)| {
                run_indexes.partition_point(|run| run.first_batch <= batch as usize) - 1
            })
            .collect();
        let mut sample_ranks_by_run = vec![Vec::new(); run_indexes.len()];
        for (rank, &run) in sample_runs.iter().enumerate() {
            sample_ranks_by_run[run].push(rank);
        }

        // The first sample of every merge task but the first is the boundary
        // before that task.
        let find_first_sample_of_task = |task_index: usize| task_index * samples.len() / task_count;
        let boundaries = (1..task_count)
            .map(|task_index| {
                let sample_rank = find_first_sample_of_task(task_index);
                let (batch, row) = samples[sample_rank];
                let run = sample_runs[sample_rank];
                let batch_in_run = batch as usize - run_indexes[run].first_batch;
                SampledBoundary {
                    sample_rank,
                    run,
                    run_row: run_indexes[run].first_row_by_batch[batch_in_run] + row as usize,
                    position: RunRow {
                        chunk: batch as usize,
                        row: row as usize,
                    },
                    run_rows_before: OnceLock::new(),
                }
            })
            .collect();
        // A merge task goes to the node that holds most of its rows, which
        // is the node of most of its samples.
        let task_nodes = (0..task_count)
            .map(|task_index| {
                let task_samples = find_first_sample_of_task(task_index)
                    ..find_first_sample_of_task(task_index + 1);
                let mut samples_by_node = vec![0usize; node_count];
                for &run in &sample_runs[task_samples] {
                    samples_by_node[run_nodes[run]] += 1;
                }
                dominant_node(&samples_by_node)
            })
            .collect();
        let boundaries = MergeTaskBoundaries::Sampled(SampledBoundaries {
            sample_gap,
            first_sample_row_by_run,
            sample_ranks_by_run,
            boundaries,
        });
        Ok((task_nodes, boundaries))
    }

    /// Total rows across all input runs.
    fn row_count(&self) -> usize {
        self.row_count
    }

    /// How many rows of each run sort before the boundary between merge task
    /// `boundary_index` and the task after it.
    ///
    /// For a sampled boundary, the sorted sample says where to look. A run's
    /// sampled rows that rank below the boundary sort before it, and the rest
    /// sort after it, so the boundary falls between two neighbouring sampled
    /// rows of the run, and only the rows between those are searched.
    fn find_run_rows_before(&self, boundary_index: usize) -> &[usize] {
        let sampled = match &self.boundaries {
            MergeTaskBoundaries::Planned(run_rows_before_boundaries) => {
                return &run_rows_before_boundaries[boundary_index];
            }
            MergeTaskBoundaries::Sampled(sampled) => sampled,
        };
        let boundary = &sampled.boundaries[boundary_index];
        boundary.run_rows_before.get_or_init(|| {
            self.run_indexes
                .iter()
                .enumerate()
                .map(|(run_index, run)| {
                    if run_index == boundary.run {
                        return boundary.run_row;
                    }
                    let sample_ranks = &sampled.sample_ranks_by_run[run_index];
                    let samples_before =
                        sample_ranks.partition_point(|&rank| rank < boundary.sample_rank);
                    let first_sample_row = sampled.first_sample_row_by_run[run_index];
                    let rows_to_search = Range {
                        start: match samples_before {
                            0 => 0,
                            _ => first_sample_row + (samples_before - 1) * sampled.sample_gap + 1,
                        },
                        end: if samples_before == sample_ranks.len() {
                            run.row_count
                        } else {
                            first_sample_row + samples_before * sampled.sample_gap
                        },
                    };
                    // Run number gives equal keys one consistent side of
                    // every boundary, the side their samples were sorted to.
                    search_cut(
                        &self.ordering,
                        run,
                        &rows_to_search,
                        boundary.position,
                        run_index < boundary.run,
                    )
                })
                .collect()
        })
    }

    /// Merges the rows of merge task `task_index` and returns them in key
    /// order, as record batches of at most [`MERGE_SLICE_ROWS`] rows.
    ///
    /// The task's stretch of every run lies between the boundary before the
    /// task and the one after it. When that is more rows than a sort chunk
    /// holds, it is cut into sort chunks, which takes a search in every run
    /// for every cut. All of those searches are made here, by the worker
    /// that runs the task, so they are spread over the workers.
    fn merge_task_rows(
        &self,
        allocator: &mut SlabAllocator,
        task_index: usize,
    ) -> Result<Vec<RecordBatch>, ArrowError> {
        if task_index >= self.task_nodes.len() {
            return Err(ArrowError::ComputeError(format!(
                "k-way merge task {task_index} is outside a {}-task plan",
                self.task_nodes.len()
            )));
        }
        // The first merge task starts at the first row of every run, and the
        // last one ends at the last.
        let run_rows_before_task = task_index
            .checked_sub(1)
            .map(|boundary_index| self.find_run_rows_before(boundary_index));
        let run_rows_before_next_task =
            (task_index + 1 < self.task_nodes.len()).then(|| self.find_run_rows_before(task_index));
        let task_run_rows: Vec<Range<usize>> = self
            .run_indexes
            .iter()
            .enumerate()
            .map(|(run_index, run)| {
                let start = run_rows_before_task.map_or(0, |run_rows| run_rows[run_index]);
                let end =
                    run_rows_before_next_task.map_or(run.row_count, |run_rows| run_rows[run_index]);
                start..end
            })
            .collect();

        let task_stretches = RunStretches::narrow(&self.batches, &self.run_indexes, &task_run_rows);
        let task_rows: usize = task_stretches.run_rows.iter().map(Range::len).sum();
        if task_rows <= self.sort_chunk_rows {
            return task_stretches.merge_into_batches(&self.order_by, allocator);
        }
        let mut run_rows_by_sort_chunk = Vec::new();
        split_into_key_ranges(
            &KeyOrdering::try_new(
                &self.order_by,
                &task_stretches.batches,
                &task_stretches.batches,
            )?,
            &task_stretches.run_indexes,
            task_stretches.run_rows.clone(),
            self.sort_chunk_rows,
            &mut run_rows_by_sort_chunk,
        );
        let mut task_batches = Vec::with_capacity(run_rows_by_sort_chunk.len());
        for sort_chunk_run_rows in &run_rows_by_sort_chunk {
            let sort_chunk_stretches = RunStretches::narrow(
                &task_stretches.batches,
                &task_stretches.run_indexes,
                sort_chunk_run_rows,
            );
            task_batches
                .extend(sort_chunk_stretches.merge_into_batches(&self.order_by, allocator)?);
        }
        Ok(task_batches)
    }
}

/// A stretch of rows from each of several sorted runs, with only the batches
/// those stretches lie in.
///
/// A merge task reads a stretch of every run, and one of its sort chunks
/// only a few rows of every run. The key readers and the row copy are bound
/// to the batches of a `RunStretches`, so their cost does not grow with the
/// size of the whole merge.
///
/// The batches are borrowed from the plan. Cloning them would write to the
/// reference counts of every batch and column a sort chunk touches, and
/// those counts are shared with the workers on the plan's other merge tasks:
/// each write would take the count's cache line away from them.
struct RunStretches<'batches> {
    /// The batches the stretches lie in, one run's after another's.
    batches: Vec<&'batches RecordBatch>,
    /// Where each run's batches are in `batches`.
    run_indexes: Vec<RunIndex>,
    /// Each run's stretch, as rows of that run's batches in `batches`.
    run_rows: Vec<Range<usize>>,
}

impl<'batches> RunStretches<'batches> {
    /// The stretches `run_rows` of the runs that `run_indexes` locates in
    /// `batches`. A run whose stretch is empty stays in the result without
    /// batches, so that every run keeps its position among the runs.
    fn narrow<Batch: Borrow<RecordBatch>>(
        batches: &'batches [Batch],
        run_indexes: &[RunIndex],
        run_rows: &[Range<usize>],
    ) -> Self {
        let mut stretch_runs: Vec<&'batches [Batch]> = Vec::with_capacity(run_indexes.len());
        let mut stretch_rows = Vec::with_capacity(run_indexes.len());
        for (run, rows) in run_indexes.iter().zip(run_rows) {
            if rows.is_empty() {
                stretch_runs.push(&[]);
                stretch_rows.push(0..0);
                continue;
            }
            let first_batch = run
                .first_row_by_batch
                .partition_point(|&first_row| first_row <= rows.start)
                - 1;
            let end_batch = run
                .first_row_by_batch
                .partition_point(|&first_row| first_row < rows.end);
            let rows_before_first_batch = run.first_row_by_batch[first_batch];
            stretch_runs.push(&batches[run.first_batch + first_batch..run.first_batch + end_batch]);
            stretch_rows
                .push(rows.start - rows_before_first_batch..rows.end - rows_before_first_batch);
        }
        Self {
            run_indexes: build_run_indexes(&stretch_runs),
            batches: stretch_runs
                .into_iter()
                .flatten()
                .map(Borrow::borrow)
                .collect(),
            run_rows: stretch_rows,
        }
    }

    /// Merges the stretches into independently owned record batches of at
    /// most [`MERGE_SLICE_ROWS`] rows, in key order. The batches are of
    /// equal length, give or take a row.
    fn merge_into_batches(
        &self,
        order_by: &[OrderBy],
        allocator: &mut SlabAllocator,
    ) -> Result<Vec<RecordBatch>, ArrowError> {
        // The rows, one run's stretch after another. Each stretch is in key
        // order, so ordering all of them is a merge of the stretches.
        let mut mapping = Vec::with_capacity(self.run_rows.iter().map(Range::len).sum());
        let stretch_lengths: Vec<usize> = self.run_rows.iter().map(Range::len).collect();
        for (run, rows) in self.run_indexes.iter().zip(&self.run_rows) {
            if rows.is_empty() {
                continue;
            }
            let mut position = run.locate(rows.start);
            for _ in 0..rows.len() {
                mapping.push((position.chunk as u32, position.row as u32));
                position.row += 1;
                if position.row == run.rows_by_batch[position.chunk - run.first_batch] {
                    position.chunk += 1;
                    position.row = 0;
                }
            }
        }
        if mapping.is_empty() {
            return Ok(Vec::new());
        }
        ChunkedRowSorter::try_new(order_by, &self.batches)?
            .merge_sorted_runs(&mut mapping, &stretch_lengths);

        let schema = self.batches[0].schema();
        // The batches' columns regrouped per column, which is how a gather
        // across batches reads them.
        let batches_of_column: Vec<Vec<&ArrayRef>> = (0..schema.fields().len())
            .map(|column_index| {
                self.batches
                    .iter()
                    .map(|batch| batch.column(column_index))
                    .collect()
            })
            .collect();
        let output_batch_count = mapping.len().div_ceil(MERGE_SLICE_ROWS);
        mapping
            .chunks(mapping.len().div_ceil(output_batch_count))
            .map(|batch_mapping| {
                let output_columns = batches_of_column
                    .iter()
                    .map(|input_columns| take_chunked(allocator, input_columns, batch_mapping))
                    .collect::<Result<Vec<ArrayRef>, _>>()?;
                RecordBatch::try_new(schema.clone(), output_columns)
            })
            .collect()
    }
}

/// One input run and the NUMA node holding its batches.
pub struct MergeRun {
    batches: Vec<RecordBatch>,
    row_count: usize,
    node_id: usize,
}

impl MergeRun {
    pub fn new(mut batches: Vec<RecordBatch>, node_id: usize) -> Self {
        batches.retain(|batch| batch.num_rows() > 0);
        let row_count = batches.iter().map(RecordBatch::num_rows).sum();
        Self {
            batches,
            row_count,
            node_id,
        }
    }

    pub fn node_id(&self) -> usize {
        self.node_id
    }
}

/// A batch and the NUMA node on which its rows were materialized.
#[derive(Clone)]
pub struct LocatedBatch {
    batch: RecordBatch,
    node_id: usize,
}

impl LocatedBatch {
    pub fn new(batch: RecordBatch, node_id: usize) -> Self {
        Self { batch, node_id }
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }

    pub fn node_id(&self) -> usize {
        self.node_id
    }
}

/// The ordered batches produced by one merge stage.
#[derive(Clone)]
pub struct MergedOutput {
    batches: Vec<LocatedBatch>,
    row_count: usize,
}

impl MergedOutput {
    pub fn empty() -> Self {
        Self {
            batches: Vec::new(),
            row_count: 0,
        }
    }

    fn from_runs(runs: Vec<MergeRun>) -> Self {
        let row_count = runs.iter().map(|run| run.row_count).sum();
        Self {
            batches: runs
                .into_iter()
                .flat_map(|run| {
                    let node_id = run.node_id;
                    run.batches
                        .into_iter()
                        .map(move |batch| LocatedBatch::new(batch, node_id))
                })
                .collect(),
            row_count,
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn batches(&self) -> &[LocatedBatch] {
        &self.batches
    }

    pub fn into_batches(self) -> Vec<LocatedBatch> {
        self.batches
    }

    /// Turns node-local output into an input run for the next merge level.
    pub fn into_run(self, node_id: usize) -> MergeRun {
        debug_assert!(self.batches.iter().all(|batch| batch.node_id == node_id));
        MergeRun {
            batches: self
                .batches
                .into_iter()
                .map(LocatedBatch::into_batch)
                .collect(),
            row_count: self.row_count,
            node_id,
        }
    }
}

/// A merge level with explicit no-work cases.
///
/// Empty, one-run, and already concatenated levels return their inputs without
/// allocating or copying arrays. Other levels become independent tasks whose
/// outputs are assembled in key order by the final task.
pub enum KWayMergePlan {
    Empty,
    Identity(MergedOutput),
    Parallel(Vec<KWayMergeTask>),
}

impl KWayMergePlan {
    /// Plans one merge level over runs already sorted by `order_by`.
    /// `node_count` defines the valid node IDs and the available task targets.
    pub fn try_new(
        order_by: Arc<[OrderBy]>,
        mut runs: Vec<MergeRun>,
        node_count: usize,
    ) -> Result<Self, ArrowError> {
        assert!(node_count > 0, "a merge needs at least one NUMA node");
        assert!(runs.iter().all(|run| run.node_id < node_count));
        runs.retain(|run| run.row_count > 0);
        if runs.is_empty() {
            return Ok(Self::Empty);
        }
        if order_by.is_empty() || runs_concatenate_in_order(&order_by, &runs)? {
            return Ok(Self::Identity(MergedOutput::from_runs(runs)));
        }
        let run_nodes = runs.iter().map(|run| run.node_id).collect();
        let batches = runs.into_iter().map(|run| run.batches).collect();
        let plan = ParallelKWayMerge::try_new_on_nodes(order_by, batches, run_nodes, node_count)?;
        let task_count = plan.task_nodes.len();
        let stage = Arc::new(KWayMergeStage {
            output: (0..task_count).map(|_| OnceLock::new()).collect(),
            remaining: AtomicUsize::new(task_count),
            plan,
        });
        Ok(Self::Parallel(
            (0..task_count)
                .map(|task_index| KWayMergeTask {
                    stage: stage.clone(),
                    task_index,
                })
                .collect(),
        ))
    }
}

fn runs_concatenate_in_order(order_by: &[OrderBy], runs: &[MergeRun]) -> Result<bool, ArrowError> {
    for adjacent in runs.windows(2) {
        let left_batch = adjacent[0].batches.last().expect("empty runs were removed");
        let right_batch = adjacent[1]
            .batches
            .first()
            .expect("empty runs were removed");
        let left = std::slice::from_ref(left_batch);
        let right = std::slice::from_ref(right_batch);
        let left_row = RunRow {
            chunk: 0,
            row: left_batch.num_rows() - 1,
        };
        let right_row = RunRow { chunk: 0, row: 0 };
        let ordering = KeyOrdering::try_new(order_by, left, right)?;
        if ordering.compare(left_row, right_row) == Ordering::Greater {
            return Ok(false);
        }
    }
    Ok(true)
}

struct KWayMergeStage {
    plan: ParallelKWayMerge,
    /// The merged batches of each merge task, set when the task has run.
    output: Box<[OnceLock<Vec<LocatedBatch>>]>,
    /// How many merge tasks have not run yet.
    remaining: AtomicUsize,
}

/// The merge of one key range of a merge level's output, which runs
/// independently of the level's other merge tasks.
pub struct KWayMergeTask {
    stage: Arc<KWayMergeStage>,
    task_index: usize,
}

impl KWayMergeTask {
    pub fn execute(
        self,
        allocator: &mut SlabAllocator,
    ) -> Result<Option<MergedOutput>, ArrowError> {
        let node_id = self.node_id();
        let task_batches = self
            .stage
            .plan
            .merge_task_rows(allocator, self.task_index)?
            .into_iter()
            .map(|batch| LocatedBatch::new(batch, node_id))
            .collect();
        self.stage.output[self.task_index]
            .set(task_batches)
            .unwrap_or_else(|_| panic!("merge task {} executed twice", self.task_index));
        if self.stage.remaining.fetch_sub(1, AtomicOrdering::AcqRel) != 1 {
            return Ok(None);
        }

        let batches = self
            .stage
            .output
            .iter()
            .flat_map(|task_batches| {
                task_batches
                    .get()
                    .expect("the last merge task to finish sees the output of every task")
                    .iter()
                    .cloned()
            })
            .collect();
        Ok(Some(MergedOutput {
            batches,
            row_count: self.stage.plan.row_count(),
        }))
    }
}

impl NodeIdOutput for KWayMergeTask {
    fn node_id(&self) -> usize {
        self.stage.plan.task_nodes[self.task_index]
    }
}

/// Locates the rows of one run in the plan's flattened batch list.
struct RunIndex {
    first_batch: usize,
    first_row_by_batch: Vec<usize>,
    rows_by_batch: Vec<usize>,
    row_count: usize,
}

impl RunIndex {
    /// The combined-list position of the run's row `row`.
    fn locate(&self, row: usize) -> RunRow {
        let batch = self
            .first_row_by_batch
            .partition_point(|&first| first <= row)
            - 1;
        RunRow {
            chunk: self.first_batch + batch,
            row: row - self.first_row_by_batch[batch],
        }
    }
}

fn build_run_indexes<Batch: Borrow<RecordBatch>>(runs: &[impl AsRef<[Batch]>]) -> Vec<RunIndex> {
    let mut first_batch = 0;
    runs.iter()
        .map(|batches| {
            let batches = batches.as_ref();
            let rows_by_batch: Vec<usize> = batches
                .iter()
                .map(|batch| batch.borrow().num_rows())
                .collect();
            let mut first_row_by_batch = Vec::with_capacity(batches.len());
            let mut row_count = 0;
            for &batch_rows in &rows_by_batch {
                first_row_by_batch.push(row_count);
                row_count += batch_rows;
            }
            let index = RunIndex {
                first_batch,
                first_row_by_batch,
                rows_by_batch,
                row_count,
            };
            first_batch += batches.len();
            index
        })
        .collect()
}

/// Divides the stretches `run_rows` of the runs into key ranges of at most
/// `max_key_range_rows` rows of output, and appends each key range's
/// stretches to `run_rows_by_key_range`, in key order. The largest run is
/// halved on each step, so equal keys cannot prevent progress.
fn split_into_key_ranges(
    ordering: &KeyOrdering,
    run_indexes: &[RunIndex],
    mut run_rows: Vec<Range<usize>>,
    max_key_range_rows: usize,
    run_rows_by_key_range: &mut Vec<Vec<Range<usize>>>,
) {
    loop {
        let row_count: usize = run_rows.iter().map(Range::len).sum();
        if row_count == 0 {
            return;
        }
        if row_count <= max_key_range_rows {
            run_rows_by_key_range.push(run_rows);
            return;
        }

        let pivot_run = run_rows
            .iter()
            .enumerate()
            .max_by_key(|(_, rows)| rows.len())
            .expect("a nonempty merge has at least one run")
            .0;
        let pivot_row = run_rows[pivot_run].start + run_rows[pivot_run].len() / 2;
        let pivot = run_indexes[pivot_run].locate(pivot_row);

        let mut rows_before_pivot = Vec::with_capacity(run_rows.len());
        let mut rows_from_pivot = Vec::with_capacity(run_rows.len());
        for (run_index, rows) in run_rows.iter().enumerate() {
            // Run number gives equal keys one consistent side of every boundary.
            let cut = if run_index == pivot_run {
                pivot_row
            } else {
                search_cut(
                    ordering,
                    &run_indexes[run_index],
                    rows,
                    pivot,
                    run_index < pivot_run,
                )
            };
            rows_before_pivot.push(rows.start..cut);
            rows_from_pivot.push(cut..rows.end);
        }
        // A single-row pivot run puts nothing before its own pivot, and when
        // that pivot is also the smallest remaining key every other cut lands
        // at its window's start too. The split would then repeat with the
        // same input forever, so the pivot row itself becomes the left side:
        // it precedes every remaining row, and every equal key belongs to a
        // later run.
        if rows_before_pivot.iter().all(|rows| rows.is_empty()) {
            rows_before_pivot[pivot_run] = pivot_row..pivot_row + 1;
            rows_from_pivot[pivot_run] = pivot_row + 1..run_rows[pivot_run].end;
        }
        split_into_key_ranges(
            ordering,
            run_indexes,
            rows_before_pivot,
            max_key_range_rows,
            run_rows_by_key_range,
        );
        // Iterating on the second half keeps the stack bounded when a split
        // strips only a few rows.
        run_rows = rows_from_pivot;
    }
}

/// Finds the run-local boundary corresponding to `pivot`.
fn search_cut(
    ordering: &KeyOrdering,
    run: &RunIndex,
    rows: &Range<usize>,
    pivot: RunRow,
    equal_keys_stay_left: bool,
) -> usize {
    let mut low = rows.start;
    let mut high = rows.end;
    while low < high {
        let middle = low + (high - low) / 2;
        let row = run.locate(middle);
        let row_stays_left = if equal_keys_stay_left {
            ordering.compare(pivot, row) != Ordering::Less
        } else {
            ordering.compare(pivot, row) == Ordering::Greater
        };
        if row_stays_left {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::{Int64Array, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    fn int_batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    fn tagged_batch(keys: &[i64], tags: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("tag", DataType::Utf8View, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(StringViewArray::from(tags.to_vec())),
            ],
        )
        .unwrap()
    }

    fn ascending() -> Vec<OrderBy> {
        vec![OrderBy::new(0, false, true)]
    }

    fn merge_all(order_by: &[OrderBy], runs: &[Vec<RecordBatch>]) -> Vec<RecordBatch> {
        let plan = ParallelKWayMerge::try_new(order_by.to_vec().into(), runs.to_vec()).unwrap();
        let mut allocator = SlabAllocator::new(false);
        (0..plan.task_nodes.len())
            .flat_map(|task_index| plan.merge_task_rows(&mut allocator, task_index).unwrap())
            .collect()
    }

    fn int_column(batches: &[RecordBatch], column_index: usize) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(column_index)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[test]
    fn three_runs_merge_into_one_sorted_sequence() {
        init_test_free_pool(16);
        let runs = vec![
            vec![int_batch(&[1, 4, 9]), int_batch(&[12, 30])],
            vec![int_batch(&[2, 2, 25])],
            vec![int_batch(&[-3, 7])],
        ];

        let merged_batches = merge_all(&ascending(), &runs);

        assert_eq!(
            int_column(&merged_batches, 0),
            vec![-3, 1, 2, 2, 4, 7, 9, 12, 25, 30]
        );
    }

    #[test]
    fn equal_keys_all_survive_the_merge() {
        init_test_free_pool(16);
        let runs = vec![
            vec![tagged_batch(&[5, 5], &["first run a", "first run b"])],
            vec![tagged_batch(&[5, 5], &["second run a", "second run b"])],
        ];

        let merged_batches = merge_all(&ascending(), &runs);

        let tags: Vec<String> = merged_batches
            .iter()
            .flat_map(|batch| {
                let tags = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|row| tags.value(row).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut tags = tags;
        tags.sort();
        assert_eq!(
            tags,
            vec!["first run a", "first run b", "second run a", "second run b"]
        );
    }

    #[test]
    fn a_large_merge_comes_out_in_bounded_batches() {
        init_test_free_pool(64);
        let build_run = |offset: i64| {
            let values: Vec<i64> = (0..3)
                .map(|batch_index| batch_index * 4_000)
                .flat_map(|base| (0..4_000).map(move |row| offset + (base + row) * 3))
                .collect();
            values.chunks(4_000).map(int_batch).collect::<Vec<_>>()
        };
        let runs = vec![build_run(0), build_run(1), build_run(2)];

        let merged_batches = merge_all(&ascending(), &runs);

        assert!(merged_batches.len() > 1);
        assert!(
            merged_batches
                .iter()
                .all(|batch| batch.num_rows() <= MERGE_SLICE_ROWS)
        );
        let keys = int_column(&merged_batches, 0);
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(keys.len(), 36_000);
    }

    #[test]
    fn a_merge_task_longer_than_a_record_batch_is_merged_into_bounded_batches() {
        init_test_free_pool(64);
        let rows_per_run: i64 = 800_000;
        let runs: Vec<Vec<RecordBatch>> = (0..3)
            .map(|run| {
                let keys: Vec<i64> = (0..rows_per_run).map(|row| row * 3 + run).collect();
                vec![int_batch(&keys)]
            })
            .collect();

        let merged_batches = merge_all(&ascending(), &runs);

        assert!(
            merged_batches
                .iter()
                .all(|batch| batch.num_rows() <= MERGE_SLICE_ROWS)
        );
        let keys = int_column(&merged_batches, 0);
        assert_eq!(keys.len(), 3 * rows_per_run as usize);
        assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn descending_single_row_runs_still_merge_into_key_order() {
        init_test_free_pool(64);
        let run_count = MERGE_SLICE_ROWS + 1;
        let runs: Vec<Vec<RecordBatch>> = (0..run_count)
            .map(|run_index| vec![int_batch(&[(run_count - run_index) as i64])])
            .collect();

        let merged_batches = merge_all(&ascending(), &runs);

        let keys = int_column(&merged_batches, 0);
        assert_eq!(keys.len(), run_count);
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn one_key_flooding_every_run_still_yields_bounded_batches() {
        init_test_free_pool(64);
        let repeated_key_run = |row_count: usize| vec![int_batch(&vec![7; row_count])];
        let runs = vec![
            repeated_key_run(9_000),
            repeated_key_run(9_000),
            repeated_key_run(50),
        ];

        let merged_batches = merge_all(&ascending(), &runs);

        assert!(
            merged_batches
                .iter()
                .all(|batch| batch.num_rows() <= MERGE_SLICE_ROWS)
        );
        assert_eq!(int_column(&merged_batches, 0), vec![7; 18_050]);
    }

    #[test]
    fn many_short_runs_merge_into_bounded_batches_in_key_order() {
        init_test_free_pool(64);
        let run_count: i64 = 700;
        let rows_per_run: i64 = 40;
        let runs: Vec<Vec<RecordBatch>> = (0..run_count)
            .map(|run| {
                let keys: Vec<i64> = (0..rows_per_run)
                    .map(|row| row * run_count + (run * 37) % run_count)
                    .collect();
                vec![int_batch(&keys)]
            })
            .collect();

        let merged_batches = merge_all(&ascending(), &runs);

        assert!(
            merged_batches
                .iter()
                .all(|batch| batch.num_rows() <= MERGE_SLICE_ROWS)
        );
        let expected_keys: Vec<i64> = (0..run_count * rows_per_run).collect();
        assert_eq!(int_column(&merged_batches, 0), expected_keys);
    }

    #[test]
    fn a_million_rows_in_a_hundred_runs_merge_into_bounded_batches_in_key_order() {
        init_test_free_pool(64);
        let run_count: i64 = 100;
        let rows_per_run: i64 = 12_000;
        let runs: Vec<Vec<RecordBatch>> = (0..run_count)
            .map(|run| {
                let keys: Vec<i64> = (0..rows_per_run)
                    .map(|row| row * run_count + (run * 37) % run_count)
                    .collect();
                vec![int_batch(&keys)]
            })
            .collect();

        let merged_batches = merge_all(&ascending(), &runs);

        assert!(
            merged_batches
                .iter()
                .all(|batch| batch.num_rows() <= MERGE_SLICE_ROWS)
        );
        let expected_keys: Vec<i64> = (0..run_count * rows_per_run).collect();
        assert_eq!(int_column(&merged_batches, 0), expected_keys);
    }

    #[test]
    fn two_keys_repeated_over_a_million_rows_in_a_hundred_runs_all_survive_the_merge() {
        init_test_free_pool(64);
        let rows_per_key = 6_000;
        let runs: Vec<Vec<RecordBatch>> = (0..100)
            .map(|_| {
                let keys: Vec<i64> = (0..2 * rows_per_key)
                    .map(|row| (row / rows_per_key) as i64)
                    .collect();
                vec![int_batch(&keys)]
            })
            .collect();

        let merged_batches = merge_all(&ascending(), &runs);

        let mut expected_keys = vec![0i64; 100 * rows_per_key];
        expected_keys.extend(vec![1i64; 100 * rows_per_key]);
        assert_eq!(int_column(&merged_batches, 0), expected_keys);
    }

    #[test]
    fn an_empty_run_contributes_nothing() {
        init_test_free_pool(16);
        let runs = vec![vec![int_batch(&[3, 8])], vec![], vec![int_batch(&[5])]];

        let merged_batches = merge_all(&ascending(), &runs);

        assert_eq!(int_column(&merged_batches, 0), vec![3, 5, 8]);
    }

    #[test]
    fn a_one_run_level_is_a_zero_copy_identity() {
        let batch = int_batch(&[1, 2, 3]);
        let original_column = batch.column(0).clone();

        let plan =
            KWayMergePlan::try_new(ascending().into(), vec![MergeRun::new(vec![batch], 0)], 1)
                .unwrap();

        let KWayMergePlan::Identity(output) = plan else {
            panic!("one run must not create merge work");
        };
        assert_eq!(output.batches().len(), 1);
        assert!(Arc::ptr_eq(
            output.batches()[0].batch().column(0),
            &original_column
        ));
    }

    #[test]
    fn concatenated_runs_are_also_a_zero_copy_identity() {
        let first = int_batch(&[1, 2]);
        let second = int_batch(&[2, 4]);
        let first_column = first.column(0).clone();
        let second_column = second.column(0).clone();

        let plan = KWayMergePlan::try_new(
            ascending().into(),
            vec![
                MergeRun::new(vec![first], 0),
                MergeRun::new(vec![second], 1),
            ],
            2,
        )
        .unwrap();

        let KWayMergePlan::Identity(output) = plan else {
            panic!("already concatenated runs must not create merge work");
        };
        assert!(Arc::ptr_eq(
            output.batches()[0].batch().column(0),
            &first_column
        ));
        assert!(Arc::ptr_eq(
            output.batches()[1].batch().column(0),
            &second_column
        ));
    }
}
