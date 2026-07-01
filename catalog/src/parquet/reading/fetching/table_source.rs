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

/// Factory that populates **one [`Injector`] per NUMA node** with the table's row
/// groups and produces a [`RowGroupInjector`] receiver for each worker, bound to
/// that worker's node queue.
///
/// Row groups are dealt round-robin across the node queues (preserving any
/// [`ScanOrder`] within each), so each node scans and caches roughly its own
/// disjoint share in its own memory domain — node-local scan/decode. A worker
/// whose node queue drains falls back to stealing from other nodes' queues.
#[derive(Clone)]
pub struct RowGroupInjectorFactory {
    row_groups: Vec<Arc<Injector<QueryRowGroupMetadata>>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl RowGroupInjectorFactory {
    /// Creates a new factory, dealing all row groups from `table` round-robin
    /// across `node_count` per-node work-stealing queues. When `filter` is set,
    /// each row group is offered to it on steal and skipped if it returns `false`.
    /// When `scan_order` is set (a Top-N boundary on one key), row groups are dealt
    /// in that key's order so each node steals its most-promising first and the
    /// boundary prunes the rest; otherwise they're dealt in file order.
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
        scan_order: Option<ScanOrder>,
        node_count: usize,
    ) -> Self {
        let node_count = node_count.max(1);
        let injectors: Vec<Arc<Injector<QueryRowGroupMetadata>>> =
            (0..node_count).map(|_| Arc::new(Injector::new())).collect();
        let order = match scan_order {
            Some(order) => steal_order(table, order),
            None => (0..table.row_groups.len()).collect(),
        };
        for (i, row_group_idx) in order.into_iter().enumerate() {
            injectors[i % node_count].push(QueryRowGroupMetadata::new(table, row_group_idx, None));
        }
        Self {
            row_groups: injectors,
            projection,
            filter,
        }
    }
}

impl RootChannelFactory<RowGroupRequest> for RowGroupInjectorFactory {
    type Receiver = RowGroupInjector;

    fn build(self) -> Self::Receiver {
        // Run on the worker thread, so `current_node` is this worker's node. Pull
        // from that node's queue first; the other nodes' queues are the fallback
        // once this node's share is exhausted.
        let node = dispatch::worker::current_node();
        let node = if node < self.row_groups.len() {
            node
        } else {
            0
        };
        let own = self.row_groups[node].clone();
        let others = self
            .row_groups
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != node)
            .map(|(_, inj)| inj.clone())
            .collect();
        RowGroupInjector {
            own,
            others,
            projection: self.projection,
            filter: self.filter,
        }
    }
}

/// A [`Receiver`] that steals row groups from its node's [`Injector`] queue, and
/// from other nodes' queues once its own is drained.
pub struct RowGroupInjector {
    own: Arc<Injector<QueryRowGroupMetadata>>,
    others: Vec<Arc<Injector<QueryRowGroupMetadata>>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl Receiver<RowGroupRequest> for RowGroupInjector {
    fn is_empty(&self) -> bool {
        self.own.is_empty() && self.others.iter().all(|i| i.is_empty())
    }

    /// Eager-path pull from **this node's** queue only, so read-ahead is built
    /// over node-local row groups. The injector is the *only* source of row
    /// groups, so it must be reachable from `run_cpu_work` (which calls
    /// `try_recv`) — otherwise row groups are stolen only in the worker's idle
    /// branch, one at a time, and a high-latency remote scan never builds
    /// read-ahead. A single non-spinning attempt keeps the hot loop from spinning
    /// on `Steal::Retry`; the next iteration retries. A pruned row group also
    /// returns `None`.
    fn try_recv(&self) -> Option<RowGroupRequest> {
        match self.own.steal() {
            Steal::Success(s) => self.admit(s),
            Steal::Empty | Steal::Retry => None,
        }
    }

    /// Idle-path steal: drain this node's queue, then fall back to other nodes so
    /// a node that finishes its share still helps drain the rest.
    fn steal(&self) -> Option<RowGroupRequest> {
        for injector in std::iter::once(&self.own).chain(self.others.iter()) {
            loop {
                match injector.steal() {
                    Steal::Empty => break,
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
        None
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
