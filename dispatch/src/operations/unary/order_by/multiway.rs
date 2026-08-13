//! A k-way merge of sorted runs, planned whole and executed in independent
//! slices.
//!
//! The two-way merge tree in [`super`] grows dynamically: completing a node
//! plans the parent merge it unblocked, which needs every worker polling
//! shared queues until the tree drains. A streaming consumer cannot poll like
//! that, so this module offers the other shape: given all of a sort's runs at
//! once, [`plan_multiway_merge_slices`] cuts the entire merge into independent
//! batch-sized slices up front, and [`merge_multiway_slice`] executes any one
//! of them on its own. The slices can then travel as ordinary channel
//! messages, claimed by whichever workers are free, with nothing left behind
//! to poll for.
//!
//! Planning is rayon's `split_for_merge` generalized to k runs: take the
//! midpoint of the largest run's window as the pivot and binary search every
//! other run for its cut, recursing until a piece is a batch's worth. Equal
//! keys cut left of the pivot in runs before the pivot's and right of it in
//! runs after, so every slice's rows sort strictly between its neighbours'
//! and a stable merge of the pieces is a stable merge of the whole. Splitting
//! by row midpoints (not key values) is what bounds a slice's size even when
//! one key floods every run.
//!
//! Executing a slice merges its k segments as position lists (adjacent pairs,
//! repeatedly, run order kept for stability) and gathers every column once
//! through the slab-backed chunked take, so the rows move exactly one time
//! however many runs feed the merge.

use std::cmp::Ordering;
use std::ops::Range;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::ArrowError;

use crate::arrays::take::take_chunked;
use crate::memory::SlabAllocator;
use crate::operations::unary::order_by_limit::OrderBy;

use super::keys::{KeyOrdering, RunRow, SelectedKeyOrdering, select_key_ordering};
use super::merge::SEQUENTIAL_MERGE_ROWS;

/// One independent piece of a k-way merge: for each run, the rows of it this
/// slice covers. The slice's output is the stable merge of those segments,
/// and the slices of one plan concatenate (in plan order) into the merge of
/// the whole.
pub struct MultiwayMergeSlice {
    /// One window per run, in run order; empty windows mean the run
    /// contributes nothing to this slice.
    pub run_rows: Vec<Range<usize>>,
}

impl MultiwayMergeSlice {
    pub fn rows(&self) -> usize {
        self.run_rows.iter().map(Range::len).sum()
    }
}

/// Cut the merge of `runs` (each sorted by `order_by`, chunked into record
/// batches) into independent [`MultiwayMergeSlice`]s, in output order, each at
/// most a batch's worth of combined rows. Runs merge stably in the order
/// given: rows with equal keys keep run order, and arrival order within a run.
pub fn plan_multiway_merge_slices(
    order_by: &[OrderBy],
    runs: &[Vec<RecordBatch>],
) -> Result<Vec<MultiwayMergeSlice>, ArrowError> {
    let indexes = run_indexes(runs);
    let whole: Vec<Range<usize>> = indexes.iter().map(|index| 0..index.rows).collect();
    let total: usize = indexes.iter().map(|index| index.rows).sum();
    let mut slices = Vec::new();
    if total == 0 {
        return Ok(slices);
    }
    if total <= SEQUENTIAL_MERGE_ROWS {
        slices.push(MultiwayMergeSlice { run_rows: whole });
        return Ok(slices);
    }

    let all_chunks = concatenated_chunks(runs);
    match select_key_ordering(order_by, &all_chunks, &all_chunks)? {
        SelectedKeyOrdering::FixedWidth(mut ordering) => {
            split_into_slices(&mut ordering, &indexes, whole, &mut slices)
        }
        SelectedKeyOrdering::ViewBytes(mut ordering) => {
            split_into_slices(&mut ordering, &indexes, whole, &mut slices)
        }
        SelectedKeyOrdering::General(mut ordering) => {
            split_into_slices(&mut ordering, &indexes, whole, &mut slices)
        }
    }
    Ok(slices)
}

/// Execute one planned slice: walk its segments into a gather mapping and
/// take every column through it, producing the slice's output rows as one
/// slab-backed batch. Must be called with the same `order_by` and `runs` the
/// slice was planned against.
pub fn merge_multiway_slice(
    allocator: &mut SlabAllocator,
    order_by: &[OrderBy],
    runs: &[Vec<RecordBatch>],
    slice: &MultiwayMergeSlice,
) -> Result<RecordBatch, ArrowError> {
    let indexes = run_indexes(runs);
    let all_chunks = concatenated_chunks(runs);
    let mapping = match select_key_ordering(order_by, &all_chunks, &all_chunks)? {
        SelectedKeyOrdering::FixedWidth(mut ordering) => {
            slice_mapping(&mut ordering, &indexes, &slice.run_rows)
        }
        SelectedKeyOrdering::ViewBytes(mut ordering) => {
            slice_mapping(&mut ordering, &indexes, &slice.run_rows)
        }
        SelectedKeyOrdering::General(mut ordering) => {
            slice_mapping(&mut ordering, &indexes, &slice.run_rows)
        }
    };

    let schema = all_chunks[0].schema();
    let mut columns = Vec::with_capacity(schema.fields().len());
    for column in 0..schema.fields().len() {
        let chunks: Vec<ArrayRef> = all_chunks
            .iter()
            .map(|chunk| chunk.column(column).clone())
            .collect();
        columns.push(take_chunked(allocator, &chunks, &mapping)?);
    }
    RecordBatch::try_new(schema, columns)
}

/// Where one run's rows live within the combined chunk list: the runs' chunks
/// concatenate in run order, and a position addresses a chunk by its index in
/// that combined list.
struct RunIndex {
    first_chunk: usize,
    /// The run-local row each of the run's chunks starts at.
    first_row_of_chunk: Vec<usize>,
    chunk_rows: Vec<usize>,
    rows: usize,
}

impl RunIndex {
    /// The combined-list position of the run's row `row`.
    fn locate(&self, row: usize) -> RunRow {
        let chunk = self
            .first_row_of_chunk
            .partition_point(|&first| first <= row)
            - 1;
        RunRow {
            chunk: self.first_chunk + chunk,
            row: row - self.first_row_of_chunk[chunk],
        }
    }

    /// The combined-list positions of the run's rows in `window`, in order.
    fn positions(&self, window: &Range<usize>) -> Vec<(u32, u32)> {
        let mut positions = Vec::with_capacity(window.len());
        let start = self.locate(window.start);
        let mut chunk = start.chunk - self.first_chunk;
        let mut row = start.row;
        for _ in window.clone() {
            positions.push(((self.first_chunk + chunk) as u32, row as u32));
            row += 1;
            while chunk < self.chunk_rows.len() && row == self.chunk_rows[chunk] {
                chunk += 1;
                row = 0;
            }
        }
        positions
    }
}

fn run_indexes(runs: &[Vec<RecordBatch>]) -> Vec<RunIndex> {
    let mut first_chunk = 0;
    runs.iter()
        .map(|chunks| {
            let mut first_row_of_chunk = Vec::with_capacity(chunks.len());
            let mut rows = 0;
            for chunk in chunks {
                first_row_of_chunk.push(rows);
                rows += chunk.num_rows();
            }
            let index = RunIndex {
                first_chunk,
                first_row_of_chunk,
                chunk_rows: chunks.iter().map(RecordBatch::num_rows).collect(),
                rows,
            };
            first_chunk += chunks.len();
            index
        })
        .collect()
}

fn concatenated_chunks(runs: &[Vec<RecordBatch>]) -> Vec<RecordBatch> {
    runs.iter().flatten().cloned().collect()
}

/// The recursive halving: split the current per-run windows at a pivot until
/// a piece is small enough to be one slice. The pivot run always halves, so
/// the recursion terminates however the keys collide.
fn split_into_slices<K: KeyOrdering>(
    ordering: &mut K,
    indexes: &[RunIndex],
    windows: Vec<Range<usize>>,
    slices: &mut Vec<MultiwayMergeSlice>,
) {
    let total: usize = windows.iter().map(Range::len).sum();
    if total == 0 {
        return;
    }
    if total <= SEQUENTIAL_MERGE_ROWS {
        slices.push(MultiwayMergeSlice { run_rows: windows });
        return;
    }

    let pivot_run = windows
        .iter()
        .enumerate()
        .max_by_key(|(_, window)| window.len())
        .expect("a split has runs")
        .0;
    let pivot_row = windows[pivot_run].start + windows[pivot_run].len() / 2;
    let pivot = indexes[pivot_run].locate(pivot_row);

    let mut left = Vec::with_capacity(windows.len());
    let mut right = Vec::with_capacity(windows.len());
    for (run, window) in windows.iter().enumerate() {
        // Stability: the merged order ties equal keys by run, then by row.
        // So a run before the pivot's keeps its equal keys left of the cut,
        // a run after it sends them right, and the pivot run cuts at the
        // pivot row itself.
        let cut = if run == pivot_run {
            pivot_row
        } else {
            search_cut(ordering, &indexes[run], window, pivot, run < pivot_run)
        };
        left.push(window.start..cut);
        right.push(cut..window.end);
    }
    split_into_slices(ordering, indexes, left, slices);
    split_into_slices(ordering, indexes, right, slices);
}

/// The row of `window` the pivot cuts `run` at: the first row whose key sorts
/// strictly after the pivot's when `equal_keys_stay_left`, and the first at
/// or after it otherwise.
fn search_cut<K: KeyOrdering>(
    ordering: &mut K,
    run: &RunIndex,
    window: &Range<usize>,
    pivot: RunRow,
    equal_keys_stay_left: bool,
) -> usize {
    let mut low = window.start;
    let mut high = window.end;
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

/// One slice's gather mapping: each run's segment as a position list, lists
/// merged two at a time (adjacent, run order kept) until one remains. Every
/// pairwise merge is stable, so the reduction is.
fn slice_mapping<K: KeyOrdering>(
    ordering: &mut K,
    indexes: &[RunIndex],
    run_rows: &[Range<usize>],
) -> Vec<(u32, u32)> {
    let mut lists: Vec<Vec<(u32, u32)>> = run_rows
        .iter()
        .enumerate()
        .filter(|(_, window)| !window.is_empty())
        .map(|(run, window)| indexes[run].positions(window))
        .collect();
    if lists.is_empty() {
        return Vec::new();
    }
    while lists.len() > 1 {
        let mut merged = Vec::with_capacity(lists.len().div_ceil(2));
        let mut pairs = lists.into_iter();
        while let Some(first) = pairs.next() {
            match pairs.next() {
                Some(second) => merged.push(merge_position_lists(ordering, &first, &second)),
                None => merged.push(first),
            }
        }
        lists = merged;
    }
    lists.pop().expect("the reduction leaves one list")
}

fn merge_position_lists<K: KeyOrdering>(
    ordering: &mut K,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let as_run_row = |(chunk, row): (u32, u32)| RunRow {
        chunk: chunk as usize,
        row: row as usize,
    };
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if ordering.compare(as_run_row(left[i]), as_run_row(right[j])) == Ordering::Greater {
            out.push(right[j]);
            j += 1;
        } else {
            out.push(left[i]);
            i += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::{Int64Array, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

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
        let slices = plan_multiway_merge_slices(order_by, runs).unwrap();
        let mut allocator = SlabAllocator::new(false);
        slices
            .iter()
            .map(|slice| merge_multiway_slice(&mut allocator, order_by, runs, slice).unwrap())
            .collect()
    }

    fn int_column(batches: &[RecordBatch], column: usize) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(column)
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

        let merged = merge_all(&ascending(), &runs);

        assert_eq!(
            int_column(&merged, 0),
            vec![-3, 1, 2, 2, 4, 7, 9, 12, 25, 30]
        );
    }

    #[test]
    fn equal_keys_keep_run_order() {
        init_test_free_pool(16);
        let runs = vec![
            vec![tagged_batch(&[5, 5], &["first run a", "first run b"])],
            vec![tagged_batch(&[5, 5], &["second run a", "second run b"])],
        ];

        let merged = merge_all(&ascending(), &runs);

        let tags: Vec<String> = merged
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
        assert_eq!(
            tags,
            vec!["first run a", "first run b", "second run a", "second run b"]
        );
    }

    #[test]
    fn a_large_merge_splits_into_bounded_slices() {
        init_test_free_pool(64);
        let run_of = |offset: i64| {
            let values: Vec<i64> = (0..3)
                .map(|batch| batch * 4_000)
                .flat_map(|base| (0..4_000).map(move |row| offset + (base + row) * 3))
                .collect();
            values
                .chunks(4_000)
                .map(|chunk| int_batch(chunk))
                .collect::<Vec<_>>()
        };
        let runs = vec![run_of(0), run_of(1), run_of(2)];

        let slices = plan_multiway_merge_slices(&ascending(), &runs).unwrap();

        assert!(slices.len() > 1);
        assert!(
            slices
                .iter()
                .all(|slice| slice.rows() <= SEQUENTIAL_MERGE_ROWS)
        );
        assert_eq!(
            slices.iter().map(MultiwayMergeSlice::rows).sum::<usize>(),
            36_000
        );
        let merged = merge_all(&ascending(), &runs);
        let keys = int_column(&merged, 0);
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(keys.len(), 36_000);
    }

    #[test]
    fn one_key_flooding_every_run_still_yields_bounded_slices() {
        init_test_free_pool(64);
        let flooded = |rows: usize| vec![int_batch(&vec![7; rows])];
        let runs = vec![flooded(9_000), flooded(9_000), flooded(50)];

        let slices = plan_multiway_merge_slices(&ascending(), &runs).unwrap();

        assert!(
            slices
                .iter()
                .all(|slice| slice.rows() <= SEQUENTIAL_MERGE_ROWS)
        );
        assert_eq!(
            slices.iter().map(MultiwayMergeSlice::rows).sum::<usize>(),
            18_050
        );
    }

    #[test]
    fn an_empty_run_contributes_nothing() {
        init_test_free_pool(16);
        let runs = vec![vec![int_batch(&[3, 8])], vec![], vec![int_batch(&[5])]];

        let merged = merge_all(&ascending(), &runs);

        assert_eq!(int_column(&merged, 0), vec![3, 5, 8]);
    }
}
