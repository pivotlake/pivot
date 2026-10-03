//! Dynamic row-group pruning and scan order.
//!
//! A Top-N's boundary tightens while the scan runs. Each row group is checked
//! against the current boundary as it is pulled, through the same statistics
//! as static filter pushdown, so every path reasons about them identically.

use std::sync::Arc;

use pruning::Predicate;

use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use crate::pushdown::row_group_statistics;
use crate::types::metadata::RowGroupMetadata;

/// Turn the planner's logical [`DynamicScanPredicate`]s into a [`RowGroupFilter`]:
/// for each, read the Top-N's live boundary from its slot and eliminate any row
/// group whose statistics prove it can't match. An empty list means no filter.
/// This is the bridge a Parquet-backed `BoundTable` uses to honour dynamic filters.
pub fn row_group_filter_from(predicates: Vec<DynamicScanPredicate>) -> Option<RowGroupFilter> {
    if predicates.is_empty() {
        return None;
    }
    Some(Arc::new(move |row_group: &RowGroupMetadata| -> bool {
        // Dynamic predicates only target top-level columns. DuckDB does not
        // push them down for JSON paths.
        let current: Vec<Predicate> = predicates
            .iter()
            .filter_map(|pred| {
                Some(Predicate {
                    column: pred.column_idx.into(),
                    comparison: pred.compare_type.into(),
                    value: pred.slot.boundary()?,
                })
            })
            .collect();
        row_group_statistics(std::slice::from_ref(row_group), &current)
            .prune(&current)
            .value(0)
    }))
}

/// A predicate that prunes a row group from its statistics, consulted as each
/// row group is pulled (e.g. a Top-N or other producer above the scan pruning
/// against a live boundary). Returning `false` skips the row group; ignoring it
/// is always correct, just without the optimization.
pub type RowGroupFilter = Arc<dyn Fn(&RowGroupMetadata) -> bool + Send + Sync>;

/// The order in which a scan should hand out row groups when a Top-N publishes a
/// dynamic boundary on `column_idx`. Stealing the most-promising row groups
/// first (smallest `min` for an ascending Top-N, largest `max` for a descending
/// one) makes the boundary tighten after the very first group, so the rest get
/// pruned instead of decoded — turning a scan-order race into near-ideal
/// pruning. Purely an optimization: any steal order is correct (the Top-N
/// re-sorts), this just minimizes how much gets read.
#[derive(Clone, Copy, Debug)]
pub struct ScanOrder {
    pub column_idx: usize,
    /// `true` for a descending Top-N (order by `max` desc); `false` ascending
    /// (order by `min` asc).
    pub descending: bool,
}

/// Derive a [`ScanOrder`] from dynamic predicates: a single predicate is a Top-N
/// boundary on one key, so order by it. The compare direction tells us which:
/// `col < / <= boundary` keeps the smallest (ascending), `col > / >=` the
/// largest (descending). With zero or multiple predicates there's no single key
/// to order by, so keep file order (`None`).
pub fn scan_order_from(predicates: &[DynamicScanPredicate]) -> Option<ScanOrder> {
    let [pred] = predicates else {
        return None;
    };
    let descending = matches!(
        pred.compare_type,
        CompareType::Greater | CompareType::GreaterEqual
    );
    Some(ScanOrder {
        column_idx: pred.column_idx,
        descending,
    })
}
