//! Work-stealing source that feeds row groups into the fetching pipeline.
//!
//! [`RowGroupInjectorFactory`] pre-loads every row group from a
//! [`ParquetTable`] into a shared [`Injector`] queue. Each worker gets its own
//! [`RowGroupInjector`] (via [`RootChannelFactory`]) that steals row groups on
//! demand, wrapping them in a [`RowGroupRequest`] with the target projection.

use crate::parquet::RowGroupRequest;
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::projection::Projection;
use crate::parquet::types::table::ParquetTable;
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::parquet::{RowGroupFilter, ScanOrder};

/// Whether scalar `a` orders strictly before `b` (`a < b`). Used to build a
/// total order over per-row-group stat bounds; a null or type-mismatched
/// comparison is treated as "not less" so the sort stays well-defined.
fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    matches!(
        arrow_ord::cmp::lt(a as &dyn Datum, b as &dyn Datum),
        Ok(arr) if arr.len() == 1 && arr.is_valid(0) && arr.value(0)
    )
}

/// Steal order for the row groups of `table` under a Top-N's [`ScanOrder`]:
/// indices sorted so the most-promising row group (smallest `min` ascending,
/// largest `max` descending) is handed out first, making the dynamic boundary
/// tighten before the bulk of the data is touched. Row groups lacking the stat
/// can't be pruned (missing stats are never eliminated), so they must be read
/// regardless — they go last. Reordering is purely an optimization: any order is
/// correct (the Top-N re-sorts), this just minimizes how much gets read.
fn steal_order(table: &ParquetTable, order: ScanOrder) -> Vec<usize> {
    let n = table.row_groups.len();
    let mut keyed: Vec<(usize, Scalar<ArrayRef>)> = Vec::with_capacity(n);
    let mut unkeyed: Vec<usize> = Vec::new();
    for idx in 0..n {
        let stat = table.row_groups[idx]
            .column_statistics(order.column_idx)
            .and_then(|s| {
                if order.descending {
                    s.max.clone()
                } else {
                    s.min.clone()
                }
            });
        match stat {
            Some(scalar) => keyed.push((idx, scalar)),
            None => unkeyed.push(idx),
        }
    }
    keyed.sort_by(|(_, a), (_, b)| {
        // Smallest `min` first when ascending; largest `max` first when
        // descending (reverse the ascending comparison).
        let ord = if scalar_lt(a, b) {
            CmpOrdering::Less
        } else if scalar_lt(b, a) {
            CmpOrdering::Greater
        } else {
            CmpOrdering::Equal
        };
        if order.descending { ord.reverse() } else { ord }
    });
    let mut result: Vec<usize> = keyed.into_iter().map(|(idx, _)| idx).collect();
    result.extend(unkeyed);
    result
}

/// Steal order for a plain scan: row groups with the most pages still resident
/// in the decompressed cache go first, so the scan consumes them before its own
/// churn can evict them - without this, a repeated scan asks for each row group
/// exactly when its cached pages are the oldest thing in the ring, and hits
/// nothing. The sort is stable, so ties (a fully cold table in particular) keep
/// file order.
fn cache_first_order(table: &ParquetTable) -> Vec<usize> {
    let mut order: Vec<usize> = (0..table.row_groups.len()).collect();
    // sort_by_cached_key: the counters move concurrently (other queries insert
    // and evict pages mid-sort), so each must be read exactly once - a
    // per-comparison re-read hands the sort an inconsistent ordering.
    order.sort_by_cached_key(|&idx| {
        Reverse(
            table.row_groups[idx]
                .live_decompressed_pages
                .load(Ordering::Relaxed),
        )
    });
    order
}

/// Factory that populates a shared [`Injector`] with every row group in a
/// table and produces [`RowGroupInjector`] receivers for each worker.
#[derive(Clone)]
pub struct RowGroupInjectorFactory {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl RowGroupInjectorFactory {
    /// Creates a new factory, pushing all row groups from `table` into the
    /// shared work-stealing queue. When `filter` is set, each row group is
    /// offered to it on steal and skipped if it returns `false`. When
    /// `scan_order` is set (a Top-N boundary on one key), row groups are pushed
    /// in that key's order so the most-promising are stolen first and the
    /// boundary prunes the rest; otherwise decompressed-cache-covered row groups
    /// go first (see [`cache_first_order`]), falling back to file order.
    ///
    /// `cached_positions`, when present, is aligned to `table.row_groups`: a
    /// `Some(positions)` entry means those row positions are already known to
    /// survive the query's filter, so the scan reads only them. A row group
    /// whose known positions are empty is not enqueued at all (nothing to
    /// fetch); one whose positions cover every row scans plain (cheaper than an
    /// all-true mask).
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
        scan_order: Option<ScanOrder>,
        cached_positions: Option<Vec<Option<Arc<Vec<u32>>>>>,
    ) -> Self {
        let injector = Arc::new(Injector::new());
        let order = match scan_order {
            Some(order) => steal_order(table, order),
            None => cache_first_order(table),
        };
        for row_group_idx in order {
            let positions = cached_positions
                .as_ref()
                .and_then(|cached| cached[row_group_idx].clone());
            let filtered_indices = match positions {
                Some(positions) if positions.is_empty() => continue,
                Some(positions)
                    if positions.len() == table.row_groups[row_group_idx].num_rows as usize =>
                {
                    None
                }
                other => other,
            };
            injector.push(QueryRowGroupMetadata::new(
                table,
                row_group_idx,
                filtered_indices,
            ));
        }
        Self {
            row_groups: injector,
            projection,
            filter,
        }
    }
}

impl RootChannelFactory<RowGroupRequest> for RowGroupInjectorFactory {
    type Receiver = RowGroupInjector;

    fn build(self) -> Self::Receiver {
        RowGroupInjector {
            row_groups: self.row_groups,
            projection: self.projection,
            filter: self.filter,
        }
    }
}

/// A [`Receiver`] that steals row groups from the shared [`Injector`] queue.
///
/// Work is only consumed through [`steal`](Self::steal); `try_recv` always
/// returns `None`.
pub struct RowGroupInjector {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl Receiver<RowGroupRequest> for RowGroupInjector {
    fn is_empty(&self) -> bool {
        self.row_groups.is_empty()
    }

    /// Eager-path pull. The injector is the *only* source of row groups, so it
    /// must be reachable from `run_cpu_work` (which calls `try_recv`) — otherwise
    /// row groups are stolen only in the worker's idle branch, one at a time,
    /// and a high-latency remote scan never builds read-ahead (in-flight reads
    /// peak at 1 per worker). A single non-spinning attempt keeps the hot loop
    /// from spinning on `Steal::Retry`; the next iteration retries. A pruned row
    /// group also returns `None` (it has been removed; the next call advances).
    fn try_recv(&self) -> Option<RowGroupRequest> {
        match self.row_groups.steal() {
            Steal::Success(s) => self.admit(s),
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<RowGroupRequest> {
        loop {
            match self.row_groups.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                // Skip row groups the filter prunes; keep stealing rather than
                // handing the worker a no-op.
                Steal::Success(s) => {
                    if let Some(req) = self.admit(s) {
                        return Some(req);
                    }
                }
            }
        }
    }
}

impl RowGroupInjector {
    /// Apply the row-group filter and wrap the survivor in a [`RowGroupRequest`];
    /// `None` if the filter proves it holds no matching row.
    fn admit(&self, s: QueryRowGroupMetadata) -> Option<RowGroupRequest> {
        if let Some(filter) = &self.filter
            && !filter(s.get_metadata())
        {
            return None;
        }
        Some(RowGroupRequest::from(s, &self.projection))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::test_utils::dummy_row_group;

    fn table_of(count: usize) -> Arc<ParquetTable> {
        Arc::new(ParquetTable::new(
            (0..count).map(|_| dummy_row_group()).collect(),
        ))
    }

    #[test]
    fn cache_covered_row_groups_are_fed_first() {
        let table = table_of(3);
        table.row_groups[1]
            .live_decompressed_pages
            .store(5, Ordering::Relaxed);
        table.row_groups[2]
            .live_decompressed_pages
            .store(2, Ordering::Relaxed);

        let order = cache_first_order(&table);

        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn a_cold_table_keeps_file_order() {
        let table = table_of(4);

        let order = cache_first_order(&table);

        assert_eq!(order, vec![0, 1, 2, 3]);
    }
}
