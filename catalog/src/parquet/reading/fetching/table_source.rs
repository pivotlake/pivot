//! Work-stealing source that feeds row groups into the fetching pipeline.
//!
//! [`RowGroupInjectorFactory`] pre-loads every row group from a
//! [`ParquetTable`] into one shared [`Injector`] queue per NUMA node. Each
//! worker gets its own [`RowGroupInjector`] (via [`RootChannelFactory`]) that
//! claims row groups from its node's queue on demand, wrapping them in a
//! [`RowGroupRequest`] with the target projection.
//!
//! Row groups are assigned to nodes by a stable hash of their identity (file +
//! index within the file), so the *same* row group lands on the *same* node
//! query after query. Everything a claim produces (the read into the
//! compressed cache, decompression, decode) happens on the claiming worker,
//! so sticky assignment keeps a row group's cached bytes on one node and its
//! next scan node-local. A worker whose node's queue runs dry claims from
//! other nodes' queues rather than idle (the stolen row group is still read
//! and decoded entirely on the claiming node; only its cached bytes end up
//! remote for a later query).

use crate::parquet::RowGroupRequest;
use crate::parquet::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata};
use crate::parquet::types::projection::Projection;
use crate::parquet::types::table::ParquetTable;
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::parquet::{RowGroupFilter, ScanOrder};

/// The NUMA node a row group is assigned to, by a hash of its stable identity
/// (its file and its index within that file). Deterministic across queries so
/// assignment is sticky for as long as the file stays open, and uniform enough
/// to split any table's row groups evenly across nodes.
fn affinity_node(row_group: &RowGroupMetadata, node_count: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    row_group.location.hash(&mut hasher);
    row_group.file_row_group_idx.hash(&mut hasher);
    (hasher.finish() % node_count as u64) as usize
}

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

/// Factory that populates one [`Injector`] per NUMA node with the table's row
/// groups (split by [`affinity_node`]) and produces [`RowGroupInjector`]
/// receivers for each worker.
#[derive(Clone)]
pub struct RowGroupInjectorFactory {
    node_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl RowGroupInjectorFactory {
    /// Creates a new factory, pushing each of `table`'s row groups into its
    /// affinity node's queue (`node_count` queues in total). When `filter` is
    /// set, each row group is offered to it on claim and skipped if it returns
    /// `false`. When `scan_order` is set (a Top-N boundary on one key), row
    /// groups are pushed in that key's order so the most-promising are claimed
    /// first and the boundary prunes the rest; otherwise decompressed-cache-covered
    /// row groups go first (see [`cache_first_order`]), falling back to file
    /// order.
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
        scan_order: Option<ScanOrder>,
        node_count: usize,
    ) -> Self {
        let node_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>> =
            Arc::new((0..node_count).map(|_| Injector::new()).collect());
        let order = match scan_order {
            Some(order) => steal_order(table, order),
            None => cache_first_order(table),
        };
        for row_group_idx in order {
            let node = affinity_node(&table.row_groups[row_group_idx], node_count);
            node_queues[node].push(QueryRowGroupMetadata::new(table, row_group_idx, None));
        }
        Self {
            node_queues,
            projection,
            filter,
        }
    }
}

impl RootChannelFactory<RowGroupRequest> for RowGroupInjectorFactory {
    type Receiver = RowGroupInjector;

    /// Runs on the worker thread, so the receiver can snapshot which node's
    /// queue is local to it.
    fn build(self) -> Self::Receiver {
        RowGroupInjector {
            node: dispatch::worker::current_node(),
            node_queues: self.node_queues,
            projection: self.projection,
            filter: self.filter,
        }
    }
}

/// A [`Receiver`] that claims row groups from the per-node [`Injector`]
/// queues: its own node's queue on the hot path, other nodes' queues only
/// when its own runs dry.
pub struct RowGroupInjector {
    node_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>>,
    /// Index of this worker's node's queue in `node_queues`.
    node: usize,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl Receiver<RowGroupRequest> for RowGroupInjector {
    fn is_empty(&self) -> bool {
        self.node_queues.iter().all(|queue| queue.is_empty())
    }

    /// Pull eagerly from this node's own queue. The injector is the *only*
    /// source of row groups, so it must be reachable from `run_cpu_work` (which
    /// calls `try_recv`), otherwise row groups are claimed only in the worker's
    /// idle branch, one at a time, and a high-latency remote scan never builds
    /// read-ahead (in-flight reads peak at 1 per worker). A single non-spinning
    /// attempt keeps the hot loop from spinning on `Steal::Retry`; the next
    /// iteration retries. A pruned row group also returns `None` (it has been
    /// removed; the next call advances).
    fn try_recv(&self) -> Option<RowGroupRequest> {
        // This pre-check is a plain load; once the queue drains (every pass
        // for the rest of the query) it avoids an epoch-pinning steal per
        // call.
        let queue = &self.node_queues[self.node];
        if queue.is_empty() {
            return None;
        }
        match queue.steal() {
            Steal::Success(s) => self.admit(s),
            Steal::Empty | Steal::Retry => None,
        }
    }

    /// Pull on the idle path: drain this node's queue first, then fall back
    /// to other nodes' queues so a node that finished its share keeps its
    /// CPUs busy instead of waiting out the tail.
    fn steal(&self) -> Option<RowGroupRequest> {
        let queue_count = self.node_queues.len();
        for offset in 0..queue_count {
            let queue = &self.node_queues[(self.node + offset) % queue_count];
            while !queue.is_empty() {
                match queue.steal() {
                    Steal::Empty => break,
                    Steal::Retry => continue,
                    // Skip row groups the filter prunes; keep claiming rather
                    // than handing the worker a no-op.
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
