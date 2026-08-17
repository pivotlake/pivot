//! Plans a k-way merge and executes independent parts of its output.
//!
//! [`KWayMergePlan`] represents a complete merge level. Zero runs produce no
//! output, one run passes through without copying, and multiple runs produce
//! independent [`KWayMergeTask`]s. Each task retains the exact inputs and slice
//! boundaries it was planned against.
//!
//! Planning repeatedly bisects the largest remaining run. Its midpoint is the
//! pivot, and binary searches locate the corresponding boundary in every other
//! run. A slice therefore contains at most one record batch of output even when
//! many runs contain the same key. Equal keys use run position only to make the
//! independently planned slices agree on their boundaries; callers must not
//! rely on that internal order.
//!
//! Execution merges the selected row positions and gathers each output column
//! once. The number of input runs does not add extra copies of the row data.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::ArrowError;

use crate::arrays::take::take_chunked;
use crate::memory::SlabAllocator;
use crate::operations::channels::NodeIdOutput;
use crate::operations::unary::order_by_limit::OrderBy;

use super::batch_sort::MERGE_SLICE_ROWS;
use super::keys::{
    KeyOrdering, RunRow, SelectedKeyOrdering, select_key_ordering, with_key_ordering,
};

/// The node contributing the most rows in a per-node tally, node 0 when the
/// tally is empty or all zero. Merge and write stages route their output work
/// through this one policy.
pub fn dominant_node(rows_by_node: &[usize]) -> usize {
    rows_by_node
        .iter()
        .enumerate()
        .max_by_key(|(_, rows)| **rows)
        .map_or(0, |(node_id, _)| node_id)
}

/// The range contributed by every input run to one independently executable
/// part of the merged output.
struct KWayMergeSlice {
    run_rows: Vec<Range<usize>>,
    target_node: usize,
}

impl KWayMergeSlice {
    #[cfg(test)]
    fn row_count(&self) -> usize {
        self.run_rows.iter().map(Range::len).sum()
    }
}

/// An immutable plan for merging sorted runs.
struct ParallelKWayMerge {
    order_by: Arc<[OrderBy]>,
    batches: Vec<RecordBatch>,
    run_indexes: Vec<RunIndex>,
    slices: Vec<KWayMergeSlice>,
    row_count: usize,
}

impl ParallelKWayMerge {
    /// Builds all slice boundaries for `runs`. Every run may contain several
    /// batches, but its rows must already be ordered by `order_by`.
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
        let row_count = run_indexes.iter().map(|index| index.row_count).sum();
        let batches = runs.into_iter().flatten().collect::<Vec<_>>();
        let all_run_rows = run_indexes
            .iter()
            .map(|index| 0..index.row_count)
            .collect::<Vec<_>>();
        let mut slices = Vec::new();

        if row_count <= MERGE_SLICE_ROWS {
            if row_count > 0 {
                slices.push(KWayMergeSlice {
                    run_rows: all_run_rows,
                    target_node: 0,
                });
            }
        } else {
            with_key_ordering!(
                select_key_ordering(&order_by, &batches, &batches)?,
                |ordering| split_into_slices(
                    &mut ordering,
                    &run_indexes,
                    all_run_rows,
                    &mut slices
                )
            )
        }

        for slice in &mut slices {
            let mut rows_by_node = vec![0usize; node_count];
            for (run_index, rows) in slice.run_rows.iter().enumerate() {
                rows_by_node[run_nodes[run_index]] += rows.len();
            }
            slice.target_node = dominant_node(&rows_by_node);
        }

        Ok(Self {
            order_by,
            batches,
            run_indexes,
            slices,
            row_count,
        })
    }

    /// Number of independently executable slices in this plan.
    fn slice_count(&self) -> usize {
        self.slices.len()
    }

    /// Total rows across all input runs.
    fn row_count(&self) -> usize {
        self.row_count
    }

    /// Executes one slice and returns its rows as one independently owned
    /// record batch.
    fn merge_slice(
        &self,
        allocator: &mut SlabAllocator,
        slice_index: usize,
    ) -> Result<RecordBatch, ArrowError> {
        let slice = self.slices.get(slice_index).ok_or_else(|| {
            ArrowError::ComputeError(format!(
                "k-way merge slice {slice_index} is outside a {}-slice plan",
                self.slices.len()
            ))
        })?;
        let mapping = with_key_ordering!(
            select_key_ordering(&self.order_by, &self.batches, &self.batches)?,
            |ordering| slice_mapping(&mut ordering, &self.run_indexes, &slice.run_rows)
        );

        let schema = self.batches[0].schema();
        let mut output_columns = Vec::with_capacity(schema.fields().len());
        for column_index in 0..schema.fields().len() {
            let input_columns: Vec<ArrayRef> = self
                .batches
                .iter()
                .map(|batch| batch.column(column_index).clone())
                .collect();
            output_columns.push(take_chunked(allocator, &input_columns, &mapping)?);
        }
        RecordBatch::try_new(schema, output_columns)
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
/// outputs are assembled in slice order by the final task.
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
        let slice_count = plan.slice_count();
        let stage = Arc::new(KWayMergeStage {
            output: (0..slice_count).map(|_| OnceLock::new()).collect(),
            remaining: AtomicUsize::new(slice_count),
            plan,
        });
        Ok(Self::Parallel(
            (0..slice_count)
                .map(|slice_index| KWayMergeTask {
                    stage: stage.clone(),
                    slice_index,
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
        let in_order =
            with_key_ordering!(select_key_ordering(order_by, left, right)?, |ordering| {
                ordering.compare(left_row, right_row) != Ordering::Greater
            });
        if !in_order {
            return Ok(false);
        }
    }
    Ok(true)
}

struct KWayMergeStage {
    plan: ParallelKWayMerge,
    output: Box<[OnceLock<LocatedBatch>]>,
    remaining: AtomicUsize,
}

/// One independently executable output slice of a merge level.
pub struct KWayMergeTask {
    stage: Arc<KWayMergeStage>,
    slice_index: usize,
}

impl KWayMergeTask {
    pub fn execute(
        self,
        allocator: &mut SlabAllocator,
    ) -> Result<Option<MergedOutput>, ArrowError> {
        let batch = self.stage.plan.merge_slice(allocator, self.slice_index)?;
        self.stage.output[self.slice_index]
            .set(LocatedBatch::new(batch, self.node_id()))
            .unwrap_or_else(|_| panic!("merge slice {} executed twice", self.slice_index));
        if self.stage.remaining.fetch_sub(1, AtomicOrdering::AcqRel) != 1 {
            return Ok(None);
        }

        let batches = self
            .stage
            .output
            .iter()
            .map(|batch| {
                batch
                    .get()
                    .expect("the final merge task observes every output slice")
                    .clone()
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
        self.stage.plan.slices[self.slice_index].target_node
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

fn build_run_indexes(runs: &[Vec<RecordBatch>]) -> Vec<RunIndex> {
    let mut first_batch = 0;
    runs.iter()
        .map(|batches| {
            let mut first_row_by_batch = Vec::with_capacity(batches.len());
            let mut row_count = 0;
            for batch in batches {
                first_row_by_batch.push(row_count);
                row_count += batch.num_rows();
            }
            let index = RunIndex {
                first_batch,
                first_row_by_batch,
                rows_by_batch: batches.iter().map(RecordBatch::num_rows).collect(),
                row_count,
            };
            first_batch += batches.len();
            index
        })
        .collect()
}

/// Recursively divides the output until every slice fits in one record batch.
/// The largest run is halved on each step, so equal keys cannot prevent
/// progress.
fn split_into_slices<K: KeyOrdering>(
    ordering: &mut K,
    run_indexes: &[RunIndex],
    mut run_rows: Vec<Range<usize>>,
    slices: &mut Vec<KWayMergeSlice>,
) {
    loop {
        let row_count: usize = run_rows.iter().map(Range::len).sum();
        if row_count == 0 {
            return;
        }
        if row_count <= MERGE_SLICE_ROWS {
            slices.push(KWayMergeSlice {
                run_rows,
                target_node: 0,
            });
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
        split_into_slices(ordering, run_indexes, rows_before_pivot, slices);
        // Iterating on the second half keeps the stack bounded when a split
        // strips only a few rows.
        run_rows = rows_from_pivot;
    }
}

/// Finds the run-local boundary corresponding to `pivot`.
fn search_cut<K: KeyOrdering>(
    ordering: &mut K,
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

/// Produces the flattened `(batch, row)` order for one output slice.
fn slice_mapping<K: KeyOrdering>(
    ordering: &mut K,
    run_indexes: &[RunIndex],
    run_rows: &[Range<usize>],
) -> Vec<(u32, u32)> {
    let row_count = run_rows.iter().map(Range::len).sum();
    let mut heap: Vec<MappingCursor> = run_rows
        .iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(run_index, rows)| MappingCursor::new(run_index, &run_indexes[run_index], rows))
        .collect();
    for parent in (0..heap.len() / 2).rev() {
        sift_down(ordering, &mut heap, parent);
    }

    let mut mapping = Vec::with_capacity(row_count);
    while !heap.is_empty() {
        let position = heap[0].position;
        mapping.push((position.chunk as u32, position.row as u32));
        if heap[0].advance(run_indexes) {
            sift_down(ordering, &mut heap, 0);
        } else {
            let last = heap.pop().expect("the heap has a root");
            if !heap.is_empty() {
                heap[0] = last;
                sift_down(ordering, &mut heap, 0);
            }
        }
    }
    mapping
}

struct MappingCursor {
    run_index: usize,
    position: RunRow,
    rows_remaining: usize,
}

impl MappingCursor {
    fn new(run_index: usize, run: &RunIndex, rows: &Range<usize>) -> Self {
        Self {
            run_index,
            position: run.locate(rows.start),
            rows_remaining: rows.len(),
        }
    }

    /// Advances within the run and returns whether this cursor remains live.
    fn advance(&mut self, run_indexes: &[RunIndex]) -> bool {
        self.rows_remaining -= 1;
        if self.rows_remaining == 0 {
            return false;
        }
        self.position.row += 1;
        let run = &run_indexes[self.run_index];
        let local_batch = self.position.chunk - run.first_batch;
        if self.position.row == run.rows_by_batch[local_batch] {
            self.position.chunk += 1;
            self.position.row = 0;
        }
        true
    }
}

fn sift_down<K: KeyOrdering>(ordering: &mut K, heap: &mut [MappingCursor], mut parent: usize) {
    loop {
        let left = parent * 2 + 1;
        if left >= heap.len() {
            return;
        }
        let right = left + 1;
        let mut first = left;
        if right < heap.len() && precedes(ordering, &heap[right], &heap[left]) {
            first = right;
        }
        if !precedes(ordering, &heap[first], &heap[parent]) {
            return;
        }
        heap.swap(parent, first);
        parent = first;
    }
}

fn precedes<K: KeyOrdering>(ordering: &mut K, left: &MappingCursor, right: &MappingCursor) -> bool {
    match ordering.compare(left.position, right.position) {
        Ordering::Less => true,
        Ordering::Equal => left.run_index < right.run_index,
        Ordering::Greater => false,
    }
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
        (0..plan.slice_count())
            .map(|slice_index| plan.merge_slice(&mut allocator, slice_index).unwrap())
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
    fn a_large_merge_splits_into_bounded_slices() {
        init_test_free_pool(64);
        let build_run = |offset: i64| {
            let values: Vec<i64> = (0..3)
                .map(|batch_index| batch_index * 4_000)
                .flat_map(|base| (0..4_000).map(move |row| offset + (base + row) * 3))
                .collect();
            values.chunks(4_000).map(int_batch).collect::<Vec<_>>()
        };
        let runs = vec![build_run(0), build_run(1), build_run(2)];

        let plan = ParallelKWayMerge::try_new(ascending().into(), runs.clone()).unwrap();

        assert!(plan.slice_count() > 1);
        assert!(
            plan.slices
                .iter()
                .all(|slice| slice.row_count() <= MERGE_SLICE_ROWS)
        );
        assert_eq!(plan.row_count(), 36_000);
        let merged_batches = merge_all(&ascending(), &runs);
        let keys = int_column(&merged_batches, 0);
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(keys.len(), 36_000);
    }

    #[test]
    fn descending_single_row_runs_still_split_into_bounded_slices() {
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
    fn one_key_flooding_every_run_still_yields_bounded_slices() {
        init_test_free_pool(64);
        let repeated_key_run = |row_count: usize| vec![int_batch(&vec![7; row_count])];
        let runs = vec![
            repeated_key_run(9_000),
            repeated_key_run(9_000),
            repeated_key_run(50),
        ];

        let plan = ParallelKWayMerge::try_new(ascending().into(), runs).unwrap();

        assert!(
            plan.slices
                .iter()
                .all(|slice| slice.row_count() <= MERGE_SLICE_ROWS)
        );
        assert_eq!(plan.row_count(), 18_050);
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
