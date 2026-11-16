//! Batch-level ordering helpers used before runs enter the k-way merge.
//!
//! The helpers are generic over [`KeyOrdering`], keeping each key shape's
//! comparison monomorphized in the sort loop.

use std::cmp::Ordering;

use super::keys::{KeyOrdering, PackedRow, RunRow};
use crate::RECORD_BATCH_SIZE;

/// A merge output slice is capped at one normal record batch.
pub(super) const MERGE_SLICE_ROWS: usize = RECORD_BATCH_SIZE;

/// Whether one batch already arrives in key order. The ordering must have the
/// batch as both its left and right input.
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

/// The order `rows` of one batch sort into, from lowest key to highest. Equal
/// keys may appear in any order allowed by SQL.
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
        ordering.compare(a_row, b_row)
    });
    indices
}

/// Sorts `rows`, each a `(chunk, row)` position into the ordering's chunks,
/// from lowest key to highest. The ordering must have those chunks as both its
/// left and right input.
///
/// With `rows_are_sorted_runs`, `rows` is a sequence of stretches that are
/// each already in key order. The sort then merges those stretches, which
/// takes fewer comparisons than sorting from scratch, and rows with equal
/// keys stay in the order they were given. Without it, equal keys may appear
/// in any order allowed by SQL.
pub(super) fn sort_chunked_row_positions<K: KeyOrdering>(
    ordering: &mut K,
    rows: &mut [(u32, u32)],
    rows_are_sorted_runs: bool,
) {
    let compare = |&(left_chunk, left_row): &(u32, u32), &(right_chunk, right_row): &(u32, u32)| {
        let left = RunRow {
            chunk: left_chunk as usize,
            row: left_row as usize,
        };
        let right = RunRow {
            chunk: right_chunk as usize,
            row: right_row as usize,
        };
        ordering.compare(left, right)
    };
    if rows_are_sorted_runs {
        rows.sort_by(compare);
    } else {
        rows.sort_unstable_by(compare);
    }
}

/// Sorts `packed_rows` from lowest key to highest by their packed keys.
/// With `compare_ties_on_full_key`, the packed key holds only the leading
/// key columns, and rows whose packed keys are equal are ordered by
/// `ordering`, which must have the rows' chunks as both its inputs.
/// `rows_are_sorted_runs` is as for [`sort_chunked_row_positions`].
pub(super) fn sort_packed_rows<K: KeyOrdering>(
    ordering: &mut K,
    packed_rows: &mut [PackedRow],
    compare_ties_on_full_key: bool,
    rows_are_sorted_runs: bool,
) {
    // Comparing the packed keys alone is a separate case so that its sort
    // loop holds no call into `ordering`.
    if !compare_ties_on_full_key {
        if rows_are_sorted_runs {
            packed_rows.sort_by_key(|packed_row| packed_row.key);
        } else {
            packed_rows.sort_unstable_by_key(|packed_row| packed_row.key);
        }
        return;
    }
    let compare = |left: &PackedRow, right: &PackedRow| {
        left.key.cmp(&right.key).then_with(|| {
            let left = RunRow {
                chunk: left.chunk as usize,
                row: left.row as usize,
            };
            let right = RunRow {
                chunk: right.chunk as usize,
                row: right.row as usize,
            };
            ordering.compare(left, right)
        })
    };
    if rows_are_sorted_runs {
        packed_rows.sort_by(compare);
    } else {
        packed_rows.sort_unstable_by(compare);
    }
}
