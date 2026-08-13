//! Batch-level ordering helpers used before runs enter the k-way merge.
//!
//! The helpers are generic over [`KeyOrdering`], keeping each key shape's
//! comparison monomorphized in the sort loop.

use std::cmp::Ordering;

use super::keys::{KeyOrdering, RunRow};
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
