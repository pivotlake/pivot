//! Batch-level ordering helpers used before runs enter the k-way merge.
//!
//! A helper that compares through the key columns takes the
//! [`KeyOrdering`] bound to the batches its positions name.

use std::cmp::Ordering;

use super::keys::{KeyOrdering, KeyedRow, PackedKeyPrefix, RunRow, SortKey};
use crate::RECORD_BATCH_SIZE;

/// A merge output slice is capped at one normal record batch.
pub(super) const MERGE_SLICE_ROWS: usize = RECORD_BATCH_SIZE;

/// Whether one batch already arrives in key order. The ordering must have the
/// batch as both its left and right input.
pub(super) fn batch_arrives_sorted(ordering: &KeyOrdering, rows: usize) -> bool {
    (1..rows).all(|row| {
        let previous = RunRow {
            chunk: 0,
            row: row - 1,
        };
        let current = RunRow { chunk: 0, row };
        ordering.compare(previous, current) != Ordering::Greater
    })
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
pub(super) fn sort_chunked_row_positions(
    ordering: &KeyOrdering,
    rows: &mut [(u32, u32)],
    rows_are_sorted_runs: bool,
) {
    let compare =
        |&left: &(u32, u32), &right: &(u32, u32)| ordering.compare(left.into(), right.into());
    if rows_are_sorted_runs {
        rows.sort_by(compare);
    } else {
        rows.sort_unstable_by(compare);
    }
}

impl<Key: SortKey> KeyedRow<Key> {
    /// Reorders `rows`, each a `(chunk, row)` position into the chunks that
    /// `packed_key_prefix` and `ordering` are bound to, from lowest key to
    /// highest. `scratch` is space reused from call to call.
    ///
    /// The rows' leading key columns are packed into `Key`s, each column in
    /// the bits `column_bits` gives it, which must be what
    /// [`PackedKeyPrefix::plan_column_bits`] returned for these `rows`. The
    /// sort compares those keys. When they do not order the rows by
    /// themselves, rows whose packed keys are equal are ordered by
    /// `ordering`.
    ///
    /// With `sorted_run_lengths`, `rows` is a sequence of stretches of those
    /// lengths that are each already in key order, and rows with equal keys
    /// stay in the order they were given. Without it, equal keys may appear
    /// in any order allowed by SQL.
    pub(super) fn sort_positions(
        scratch: &mut Vec<Self>,
        packed_key_prefix: &PackedKeyPrefix,
        column_bits: &[u32],
        ordering: &KeyOrdering,
        rows: &mut [(u32, u32)],
        sorted_run_lengths: Option<&[usize]>,
    ) {
        let packed_keys_order_the_rows = packed_key_prefix.pack_rows(column_bits, rows, scratch);
        if packed_keys_order_the_rows {
            // Comparing two packed keys is so cheap that the standard sort,
            // whose loops have no branches to mispredict, orders even many
            // short sorted stretches faster than merging them does.
            if sorted_run_lengths.is_some() {
                scratch.sort_by_key(|keyed_row| keyed_row.key);
            } else {
                scratch.sort_unstable_by_key(|keyed_row| keyed_row.key);
            }
        } else {
            let compare = |left: &Self, right: &Self| {
                left.key.cmp(&right.key).then_with(|| {
                    ordering.compare(
                        (left.chunk, left.row).into(),
                        (right.chunk, right.row).into(),
                    )
                })
            };
            // Here a comparison may have to read the key columns, so the
            // number of comparisons is what counts, and merging the
            // stretches makes the fewest.
            match sorted_run_lengths {
                Some(run_lengths) => Self::merge_adjacent_runs(scratch, run_lengths, compare),
                None => scratch.sort_unstable_by(compare),
            }
        }
        for (position, keyed_row) in rows.iter_mut().zip(scratch.iter()) {
            *position = (keyed_row.chunk, keyed_row.row);
        }
    }

    /// Merges `keyed_rows`, a sequence of stretches of `run_lengths` rows
    /// that are each in the order of `compare`, into that order. Rows that
    /// compare equal stay in the order they were given.
    ///
    /// Neighbouring stretches are merged in pairs, then the merged pairs,
    /// and so on until one stretch is left. Every row is moved once per
    /// round and the number of rounds is the logarithm of the number of
    /// stretches, however short the stretches are. A general sort that looks
    /// for sorted stretches by itself only uses long ones, and sorts from
    /// scratch when many short stretches make up the rows.
    fn merge_adjacent_runs(
        keyed_rows: &mut [Self],
        run_lengths: &[usize],
        mut compare: impl FnMut(&Self, &Self) -> Ordering,
    ) {
        // Where each stretch ends. Empty stretches are left out.
        let mut run_ends = Vec::with_capacity(run_lengths.len());
        let mut row_count = 0;
        for &run_length in run_lengths {
            if run_length > 0 {
                row_count += run_length;
                run_ends.push(row_count);
            }
        }
        assert_eq!(row_count, keyed_rows.len());

        // Each round reads the stretches from one of two buffers and writes
        // the merged stretches to the other.
        let mut other_buffer = keyed_rows.to_vec();
        let mut source: &mut [Self] = keyed_rows;
        let mut target: &mut [Self] = &mut other_buffer;
        let mut rows_are_in_other_buffer = false;
        let mut merged_run_ends = Vec::with_capacity(run_ends.len().div_ceil(2));
        while run_ends.len() > 1 {
            merged_run_ends.clear();
            let mut start = 0;
            for pair in run_ends.chunks(2) {
                let end = pair[pair.len() - 1];
                let middle = pair[0];
                let (mut left, mut right, mut output) = (start, middle, start);
                while left < middle && right < end {
                    // The right row goes first only when it is strictly
                    // lower, which keeps equal rows in the order they were
                    // given.
                    if compare(&source[right], &source[left]) == Ordering::Less {
                        target[output] = source[right];
                        right += 1;
                    } else {
                        target[output] = source[left];
                        left += 1;
                    }
                    output += 1;
                }
                let left_rest = middle - left;
                target[output..output + left_rest].copy_from_slice(&source[left..middle]);
                target[output + left_rest..end].copy_from_slice(&source[right..end]);
                merged_run_ends.push(end);
                start = end;
            }
            std::mem::swap(&mut run_ends, &mut merged_run_ends);
            std::mem::swap(&mut source, &mut target);
            rows_are_in_other_buffer = !rows_are_in_other_buffer;
        }
        if rows_are_in_other_buffer {
            target.copy_from_slice(source);
        }
    }
}
