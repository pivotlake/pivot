//! The sort and merge arithmetic behind ORDER BY, translated from rayon's
//! parallel merge sort (`rayon::slice::mergesort`): sorting one batch,
//! recognizing rows that already arrive sorted, splitting a merge of two
//! sorted runs into independent slices, and walking one slice's merge.
//!
//! Everything here is generic over [`KeyOrdering`], so each key shape's
//! comparison stays monomorphized into the loops, and none of it touches
//! column values: the output of a merge walk is the row *mapping* the gather
//! machinery applies per column.

use std::cmp::Ordering;
use std::ops::Range;

use super::SortedRun;
use super::keys::{KeyOrdering, RunRow};
use crate::RECORD_BATCH_SIZE;

/// A merge below this many combined rows runs as one sequential walk;
/// anything larger splits ([`plan_merge_slices`]). One batch's worth, so a
/// slice's output is a well-sized chunk. (Rayon's equivalent, chosen to hide
/// its scheduling overhead, is the same order of magnitude: 5000.)
const SEQUENTIAL_MERGE_ROWS: usize = RECORD_BATCH_SIZE;

/// One independent piece of a merge: these rows of the left run merge with
/// these rows of the right run, and nothing outside them lands in between.
pub(super) struct MergeSliceRows {
    pub(super) left_rows: Range<usize>,
    pub(super) right_rows: Range<usize>,
}

/// Whether `rows` of one batch already arrive in key order (no row sorts
/// before its predecessor). The ordering must have the batch as both sides.
pub(super) fn batch_arrives_sorted<K: KeyOrdering>(ordering: &mut K, rows: usize) -> bool {
    (1..rows).all(|row| {
        let previous = RunRow {
            chunk: 0,
            row: row - 1,
        };
        let current = RunRow { chunk: 0, row };
        ordering.compare(previous, current) != Ordering::Greater
    })
}

/// The order `rows` of one batch sort into: row indices from lowest key to
/// highest, ties keeping their arrival order. The ordering must have the
/// batch as both sides.
pub(super) fn sorted_row_indices<K: KeyOrdering>(ordering: &mut K, rows: usize) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..rows as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        let a_row = RunRow {
            chunk: 0,
            row: a as usize,
        };
        let b_row = RunRow {
            chunk: 0,
            row: b as usize,
        };
        // The index tiebreak keeps equal keys in arrival order, which is what
        // makes the whole sort stable.
        ordering.compare(a_row, b_row).then(a.cmp(&b))
    });
    indices
}

/// Split a merge of `left_rows` x `right_rows` so its halves can merge
/// independently: returns `(left_split, right_split)` such that every row of
/// `left_rows[..left_split]` and `right_rows[..right_split]` sorts at or
/// before every row of the two remainders.
///
/// Rayon's `split_for_merge`: take the midpoint of the larger side and binary
/// search its key in the smaller side. The asymmetry of the two searches
/// (first `>=` the key on one side, first `>` it on the other) is what keeps
/// equal keys stable — the left run's copies all land before the right run's.
fn split_for_parallel_merge<K: KeyOrdering>(
    ordering: &mut K,
    left: &SortedRun,
    right: &SortedRun,
    left_rows: &Range<usize>,
    right_rows: &Range<usize>,
) -> (usize, usize) {
    if left_rows.len() >= right_rows.len() {
        let left_mid = left_rows.start + left_rows.len() / 2;
        let left_mid_row = left.locate(left_mid);

        // The first right row whose key is `>=` the left midpoint's.
        let mut low = right_rows.start;
        let mut high = right_rows.end;
        while low < high {
            let middle = low + (high - low) / 2;
            if ordering.compare(left_mid_row, right.locate(middle)) == Ordering::Greater {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (left_mid - left_rows.start, low - right_rows.start)
    } else {
        let right_mid = right_rows.start + right_rows.len() / 2;
        let right_mid_row = right.locate(right_mid);

        // The first left row whose key is `>` the right midpoint's.
        let mut low = left_rows.start;
        let mut high = left_rows.end;
        while low < high {
            let middle = low + (high - low) / 2;
            if ordering.compare(left.locate(middle), right_mid_row) == Ordering::Greater {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        (low - left_rows.start, right_mid - right_rows.start)
    }
}

/// Cut the merge of two whole runs into independent [`MergeSliceRows`], in
/// output order, each at most [`SEQUENTIAL_MERGE_ROWS`] combined rows.
///
/// This is rayon's `par_merge` recursion with the two recursive calls turned
/// into list entries: where rayon forks the two sub-merges, the slices here
/// become stealable tasks for whichever workers are free.
pub(super) fn plan_merge_slices<K: KeyOrdering>(
    ordering: &mut K,
    left: &SortedRun,
    right: &SortedRun,
) -> Vec<MergeSliceRows> {
    let mut slices = Vec::new();
    split_into_slices(
        ordering,
        left,
        right,
        0..left.rows(),
        0..right.rows(),
        &mut slices,
    );
    slices
}

fn split_into_slices<K: KeyOrdering>(
    ordering: &mut K,
    left: &SortedRun,
    right: &SortedRun,
    left_rows: Range<usize>,
    right_rows: Range<usize>,
    slices: &mut Vec<MergeSliceRows>,
) {
    if left_rows.len() + right_rows.len() <= SEQUENTIAL_MERGE_ROWS {
        slices.push(MergeSliceRows {
            left_rows,
            right_rows,
        });
        return;
    }
    let (left_split, right_split) =
        split_for_parallel_merge(ordering, left, right, &left_rows, &right_rows);
    let left_mid = left_rows.start + left_split;
    let right_mid = right_rows.start + right_split;
    split_into_slices(
        ordering,
        left,
        right,
        left_rows.start..left_mid,
        right_rows.start..right_mid,
        slices,
    );
    split_into_slices(
        ordering,
        left,
        right,
        left_mid..left_rows.end,
        right_mid..right_rows.end,
        slices,
    );
}

/// Walk one slice's merge and produce the gather mapping of its output rows:
/// `(chunk, row)` pairs over the combined chunk list, the left run's chunks
/// first and the right run's after them. Equal keys take the left run's row
/// first, which keeps the merge stable.
pub(super) fn merge_slice_mapping<K: KeyOrdering>(
    ordering: &mut K,
    left: &SortedRun,
    right: &SortedRun,
    slice: &MergeSliceRows,
) -> Vec<(u32, u32)> {
    let right_chunks_start = left.chunks().len() as u32;
    let mut mapping = Vec::with_capacity(slice.left_rows.len() + slice.right_rows.len());

    let mut left_cursor = RunCursor::at(left, slice.left_rows.clone());
    let mut right_cursor = RunCursor::at(right, slice.right_rows.clone());
    while let (Some(left_row), Some(right_row)) = (left_cursor.position(), right_cursor.position())
    {
        if ordering.compare(left_row, right_row) == Ordering::Greater {
            mapping.push((
                right_chunks_start + right_row.chunk as u32,
                right_row.row as u32,
            ));
            right_cursor.advance();
        } else {
            mapping.push((left_row.chunk as u32, left_row.row as u32));
            left_cursor.advance();
        }
    }
    while let Some(left_row) = left_cursor.position() {
        mapping.push((left_row.chunk as u32, left_row.row as u32));
        left_cursor.advance();
    }
    while let Some(right_row) = right_cursor.position() {
        mapping.push((
            right_chunks_start + right_row.chunk as u32,
            right_row.row as u32,
        ));
        right_cursor.advance();
    }
    mapping
}

/// Walks a window of a run's rows in order, keeping the `(chunk, row)`
/// position current so the merge loop never re-derives it from a logical row.
struct RunCursor {
    position: RunRow,
    /// Rows of the current chunk; `position.row` reaching it moves to the
    /// next chunk.
    current_chunk_rows: usize,
    remaining: usize,
    chunk_rows: Vec<usize>,
}

impl RunCursor {
    fn at(run: &SortedRun, rows: Range<usize>) -> Self {
        let chunk_rows: Vec<usize> = run.chunks().iter().map(|chunk| chunk.num_rows()).collect();
        let position = if rows.is_empty() {
            RunRow { chunk: 0, row: 0 }
        } else {
            run.locate(rows.start)
        };
        Self {
            position,
            current_chunk_rows: chunk_rows.get(position.chunk).copied().unwrap_or(0),
            remaining: rows.len(),
            chunk_rows,
        }
    }

    fn position(&self) -> Option<RunRow> {
        (self.remaining > 0).then_some(self.position)
    }

    fn advance(&mut self) {
        self.remaining -= 1;
        self.position.row += 1;
        if self.position.row == self.current_chunk_rows && self.remaining > 0 {
            self.position.chunk += 1;
            self.position.row = 0;
            self.current_chunk_rows = self.chunk_rows[self.position.chunk];
        }
    }
}
