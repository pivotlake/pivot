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
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;

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
    /// boundary prunes the rest; otherwise they're pushed in file order.
    ///
    /// When `condition_replay` is set, this is a replay scan: only the row groups
    /// a previous run recorded survivors for are pushed, each seeded with those
    /// surviving row indices so the decoder produces just those rows. A row group
    /// absent from the cache had no matches on the populate run (the cache records
    /// every group with at least one survivor), so it is skipped entirely. This
    /// mirrors late materialization's survivor-only re-fetch rather than scanning
    /// every row group, which is both the win and what keeps the working set
    /// bounded. The `Filter` above still runs (a no-op over already-surviving rows).
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
        scan_order: Option<ScanOrder>,
        condition_replay: Option<(Arc<crate::QueryConditionCache>, u64)>,
    ) -> Self {
        let injector = Arc::new(Injector::new());
        let order = match scan_order {
            Some(order) => steal_order(table, order),
            None => (0..table.row_groups.len()).collect(),
        };
        for row_group_idx in order {
            match &condition_replay {
                Some((cache, filter_id)) => match cache.lookup(*filter_id, row_group_idx as u32) {
                    Some(survivors) => injector.push(QueryRowGroupMetadata::new(
                        table,
                        row_group_idx,
                        Some(survivors.to_vec()),
                    )),
                    None => continue, // no survivors recorded for this group: skip it
                },
                None => injector.push(QueryRowGroupMetadata::new(table, row_group_idx, None)),
            }
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
