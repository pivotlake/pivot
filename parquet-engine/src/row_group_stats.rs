//! Stats-based row-group elimination.
//!
//! Given a single-column predicate `col <op> constant`, decide whether a row
//! group is guaranteed to contain no matching row from its min/max statistics.
//! Shared by static filter pushdown and dynamic-filter pruning at scan time, so
//! every path reasons about statistics identically.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Scalar};
use arrow_schema::ArrowError;

use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use crate::pruning::ColumnBounds;
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
        for pred in &predicates {
            let Some(constant) = pred.slot.boundary() else {
                continue;
            };
            // Dynamic predicates only target top-level columns. DuckDB does
            // not push them down for JSON paths.
            let leaf = crate::types::leaves::first_leaf(row_group.schema.fields(), pred.column_idx);
            if let Ok(true) = row_group_eliminated(row_group, leaf, pred.compare_type, &constant) {
                return false;
            }
        }
        true
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

/// Returns whether min/max statistics prove that the row group cannot match.
///
/// `leaf` is a column-chunk index in this row group's schema. Missing or
/// incompatible statistics cannot eliminate the row group.
pub fn row_group_eliminated(
    row_group: &RowGroupMetadata,
    leaf: usize,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    let Some(stats) = row_group.leaf_statistics(leaf) else {
        return Ok(false);
    };
    let bounds = ColumnBounds {
        lower: stats.min().map(Scalar::into_inner),
        upper: stats.max().map(Scalar::into_inner),
        all_null: Some(BooleanArray::from(vec![
            stats.null_count == Some(row_group.num_rows),
        ])),
        nan_free: Some(BooleanArray::from(vec![stats.nan_free])),
    };
    Ok(!bounds.may_match(1, compare_type, constant)?.value(0))
}

/// Returns whether the inclusive range `[min, max]` proves that
/// `col <compare_type> constant` matches no value in it. Shared by row-group
/// elimination and file-level table-format
/// stats pruning, so both reason about bounds identically. `min`/`max` are
/// single-value scalars typed as the physical column. Floating-point bounds
/// without proof that NaNs are absent can still exclude equality with a
/// non-NaN constant.
pub fn bounds_eliminate(
    min: &Scalar<ArrayRef>,
    max: &Scalar<ArrayRef>,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    let bounds = ColumnBounds {
        lower: Some(min.clone().into_inner()),
        upper: Some(max.clone().into_inner()),
        ..Default::default()
    };
    Ok(!bounds.may_match(1, compare_type, constant)?.value(0))
}
