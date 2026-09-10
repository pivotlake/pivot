//! Run-length encoded boolean mask that marks which rows within a page to keep.
//!
//! [`FilterMask`] translates a slice of a row groups global indices into a sort of "bitmask"
//! [`RunArray`] scoped to a single page's `[start_index, end_index)` range: `true` runs
//! are rows to decode, `false` runs are rows to skip.
//!
//! [`RunningFilterMask`] wraps a [`FilterMask`] with a cursor so decoders can consume
//! runs one at a time.

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{BooleanArray, Int32Array, RunArray};
use std::cmp::min;
use std::ops::Range;

/// A run-length encoded boolean mask over the rows of a single page.
///
/// Each run is either `true` (keep / decode these rows) or `false` (skip).
/// Internally backed by an Arrow [`RunArray<Int32Type>`] whose values are booleans.
#[derive(Clone)]
pub struct FilterMask {
    total_rows: usize,
    filters: RunArray<Int32Type>,
}

impl FilterMask {
    /// Build a mask for the page spanning `[start_index, end_index)`.
    ///
    /// `global_filtered_indexes` must be sorted. Only indexes that fall within the page
    /// range are considered; the rest are ignored. Consecutive kept indexes are merged
    /// into a single `true` run.
    pub fn new(start_index: u32, end_index: u32, global_filtered_indexes: &[u32]) -> Self {
        let len = end_index - start_index;
        let lo = global_filtered_indexes.partition_point(|&x| x < start_index);
        let hi = global_filtered_indexes.partition_point(|&x| x < end_index);
        let relevant = &global_filtered_indexes[lo..hi];

        if relevant.is_empty() {
            return Self {
                total_rows: 0,
                filters: RunArray::try_new(
                    &Int32Array::from(vec![len as i32]),
                    &BooleanArray::from(vec![false]),
                )
                .unwrap(),
            };
        }

        let mut run_ends: Vec<i32> = Vec::new();
        let mut values = Vec::new();
        let mut pos = start_index;

        for &idx in relevant {
            if idx > pos {
                run_ends.push((idx - start_index) as i32);
                values.push(false);
            }
            pos = idx + 1;
            if values.last() == Some(&true) {
                *run_ends.last_mut().unwrap() = (pos - start_index) as i32;
            } else {
                run_ends.push((pos - start_index) as i32);
                values.push(true);
            }
        }

        if pos < end_index {
            run_ends.push(len as i32);
            values.push(false);
        }

        Self {
            total_rows: relevant.len(),
            filters: RunArray::try_new(&Int32Array::from(run_ends), &BooleanArray::from(values))
                .unwrap(),
        }
    }

    /// Build a mask for the page spanning `[start_index, end_index)` that keeps
    /// the rows inside `row_range` (row group row indices): at most one kept
    /// run, between a skipped head and a skipped tail.
    pub fn for_row_range(start_index: u32, end_index: u32, row_range: &Range<u32>) -> Self {
        let len = end_index - start_index;
        let kept_start = row_range.start.max(start_index);
        let kept_end = row_range.end.min(end_index);
        let mut run_ends: Vec<i32> = Vec::with_capacity(3);
        let mut values = Vec::with_capacity(3);
        if kept_start >= kept_end {
            run_ends.push(len as i32);
            values.push(false);
        } else {
            if kept_start > start_index {
                run_ends.push((kept_start - start_index) as i32);
                values.push(false);
            }
            run_ends.push((kept_end - start_index) as i32);
            values.push(true);
            if kept_end < end_index {
                run_ends.push(len as i32);
                values.push(false);
            }
        }
        Self {
            total_rows: kept_end.saturating_sub(kept_start) as usize,
            filters: RunArray::try_new(&Int32Array::from(run_ends), &BooleanArray::from(values))
                .unwrap(),
        }
    }

    /// Returns `true` when every run is `false` (no rows to keep).
    pub fn all_false(&self) -> bool {
        self.filters
            .values()
            .as_boolean()
            .iter()
            .all(|v| v == Some(false))
    }

    /// Number of rows marked as kept (`true`) in the mask.
    pub fn rows(&self) -> usize {
        self.total_rows
    }

    /// Returns the run at `physical_index` as `(is_kept, cumulative_run_end)`.
    ///
    /// `cumulative_run_end` follows Arrow's run-end convention: it is the exclusive
    /// end offset relative to the start of the page, not the length of the run.
    /// To get the run length, subtract the previous run end (or 0 for the first run).
    pub fn get_run_type_and_run_end(&self, physical_index: usize) -> (bool, usize) {
        let run = self.filters.run_ends().values()[physical_index];
        let values = self
            .filters
            .values()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        (
            unsafe { values.value_unchecked(physical_index) },
            run as usize,
        )
    }
}

/// Stateful cursor over a [`FilterMask`] that yields one run at a time.
///
/// `false` runs are always returned in full. `true` runs can be split into
/// chunks no larger than `max_size` (passed to [`next_run`](Self::next_run)),
/// which lets decoders process kept rows in bounded batches.
pub struct RunningFilterMask {
    filter_mask: FilterMask,
    previous_run_end: usize,
    /// How many values are left in the current `true` run.
    next_true_run_size: usize,
    /// Physical run index into the underlying [`FilterMask`].
    physical_index: usize,
}

impl RunningFilterMask {
    pub fn new(mask: FilterMask) -> Self {
        Self {
            filter_mask: mask,
            previous_run_end: 0,
            next_true_run_size: 0,
            physical_index: 0,
        }
    }
    /// Advance to the next run, returning `(is_kept, count)`.
    ///
    /// `false` runs return their full length. `true` runs are capped at `max_size`;
    /// call again to consume the remainder.
    pub fn next_run(&mut self, max_true_size: usize) -> (bool, usize) {
        if self.next_true_run_size == 0 {
            let physical_index = self.physical_index;
            self.physical_index += 1;
            match self.filter_mask.get_run_type_and_run_end(physical_index) {
                (true, end) => {
                    self.next_true_run_size = end - self.previous_run_end;
                    self.previous_run_end = end;
                }
                (false, end) => {
                    let count = end - self.previous_run_end;
                    self.previous_run_end = end;
                    return (false, count);
                }
            }
        }

        let next_run = min(max_true_size, self.next_true_run_size);
        self.next_true_run_size -= next_run;
        (true, next_run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FilterMask::new ──

    #[test]
    fn test_new_first_index_at_start() {
        let mask = FilterMask::new(0, 5, &[0, 2, 4]);

        assert_eq!(mask.rows(), 3);
    }

    #[test]
    fn test_new_consecutive_from_start() {
        let mask = FilterMask::new(0, 4, &[0, 1, 2, 3]);

        assert_eq!(mask.rows(), 4);
    }

    #[test]
    fn test_new_single_kept_in_middle() {
        let mask = FilterMask::new(0, 5, &[2]);

        assert_eq!(mask.rows(), 1);
        assert!(!mask.all_false());
    }

    #[test]
    fn test_new_no_relevant_indices() {
        let mask = FilterMask::new(0, 5, &[]);

        assert_eq!(mask.rows(), 0);
        assert!(mask.all_false());
    }

    #[test]
    fn test_new_global_indices_outside_range_ignored() {
        let mask = FilterMask::new(10, 15, &[3, 7, 12, 20]);

        assert_eq!(mask.rows(), 1);
    }

    // ── FilterMask::for_row_range ──

    /// A range strictly inside the page keeps one run between two skipped runs.
    #[test]
    fn row_range_inside_page_keeps_the_middle() {
        let mask = FilterMask::for_row_range(100, 110, &(103..107));

        assert_eq!(mask.rows(), 4);
        assert_eq!(mask.get_run_type_and_run_end(0), (false, 3));
        assert_eq!(mask.get_run_type_and_run_end(1), (true, 7));
        assert_eq!(mask.get_run_type_and_run_end(2), (false, 10));
    }

    /// A range covering the whole page keeps everything in a single run.
    #[test]
    fn row_range_covering_page_keeps_all() {
        let mask = FilterMask::for_row_range(100, 110, &(50..200));

        assert_eq!(mask.rows(), 10);
        assert_eq!(mask.get_run_type_and_run_end(0), (true, 10));
    }

    /// A range that ends before the page or starts after it keeps nothing.
    #[test]
    fn row_range_outside_page_keeps_nothing() {
        let before = FilterMask::for_row_range(100, 110, &(0..100));
        let after = FilterMask::for_row_range(100, 110, &(110..300));

        assert!(before.all_false());
        assert_eq!(before.rows(), 0);
        assert!(after.all_false());
        assert_eq!(after.rows(), 0);
    }

    /// A range starting inside the page and running past it skips only the head.
    #[test]
    fn row_range_past_page_end_skips_only_the_head() {
        let mask = FilterMask::for_row_range(100, 110, &(108..500));

        assert_eq!(mask.rows(), 2);
        assert_eq!(mask.get_run_type_and_run_end(0), (false, 8));
        assert_eq!(mask.get_run_type_and_run_end(1), (true, 10));
    }

    // ── FilterMask::get_run_type_and_run_end ──

    #[test]
    fn test_run_at_skip_keep_skip() {
        let mask = FilterMask::new(0, 10, &[3]);

        let (keep0, len0) = mask.get_run_type_and_run_end(0);
        assert_eq!((keep0, len0), (false, 3));

        let (keep1, len1) = mask.get_run_type_and_run_end(1);
        assert_eq!((keep1, len1), (true, 4));

        let (keep2, len2) = mask.get_run_type_and_run_end(2);
        assert_eq!((keep2, len2), (false, 10));
    }

    /// Two kept indices produce: skip 2, keep 1, skip 2, keep 1, skip 1
    #[test]
    fn test_run_at_two_kept() {
        let mask = FilterMask::new(0, 7, &[2, 5]);

        assert_eq!(mask.get_run_type_and_run_end(0), (false, 2));
        assert_eq!(mask.get_run_type_and_run_end(1), (true, 3));
        assert_eq!(mask.get_run_type_and_run_end(2), (false, 5));
        assert_eq!(mask.get_run_type_and_run_end(3), (true, 6));
        assert_eq!(mask.get_run_type_and_run_end(4), (false, 7));
    }

    /// Consecutive kept indices merge into one run: skip 1, keep 3, skip 1
    #[test]
    fn test_run_at_consecutive_kept() {
        let mask = FilterMask::new(0, 5, &[1, 2, 3]);

        assert_eq!(mask.get_run_type_and_run_end(0), (false, 1));
        assert_eq!(mask.get_run_type_and_run_end(1), (true, 4));
        assert_eq!(mask.get_run_type_and_run_end(2), (false, 5));
    }

    // ── RunningFilterMask ──

    /// Walk a mask with interleaved skip/keep runs and verify every run.
    #[test]
    fn test_running_walk_all_runs() {
        // [0,6) keeping [1, 4] → skip 1, keep 1, skip 2, keep 1, skip 1
        let mask = FilterMask::new(0, 6, &[1, 4]);
        let mut running = RunningFilterMask::new(mask);

        assert_eq!(running.next_run(100), (false, 1));
        assert_eq!(running.next_run(100), (true, 1));
        assert_eq!(running.next_run(100), (false, 2));
        assert_eq!(running.next_run(100), (true, 1));
        assert_eq!(running.next_run(100), (false, 1));
    }

    /// True runs must be clamped to max_size; false runs are returned whole.
    #[test]
    fn test_running_true_run_clamped() {
        // [0,8) keeping [2,3,4,5] → skip 2, keep 4, skip 2
        let mask = FilterMask::new(0, 8, &[2, 3, 4, 5]);
        let mut running = RunningFilterMask::new(mask);

        assert_eq!(running.next_run(100), (false, 2));
        assert_eq!(running.next_run(2), (true, 2)); // first 2 of the 4
        assert_eq!(running.next_run(2), (true, 2)); // remaining 2
        assert_eq!(running.next_run(100), (false, 2));
    }

    /// Mask that keeps nothing — only false runs.
    #[test]
    fn test_running_all_skipped() {
        let mask = FilterMask::new(0, 5, &[]);
        let mut running = RunningFilterMask::new(mask);

        assert_eq!(running.next_run(100), (false, 5));
    }

    /// Mask that keeps everything from start — only true runs.
    #[test]
    fn test_running_all_kept() {
        let mask = FilterMask::new(0, 3, &[0, 1, 2]);
        let mut running = RunningFilterMask::new(mask);

        assert_eq!(running.next_run(100), (true, 3));
    }
}
