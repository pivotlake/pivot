//! Stats-based row-group elimination.
//!
//! Given a single-column predicate `col <op> constant`, decide whether a row
//! group is guaranteed to contain no matching row from its min/max statistics.
//! Shared by static filter pushdown (in `TableBinding`) and by
//! dynamic-filter pruning at scan time, so both reason about stats identically.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::ArrowError;

use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use crate::parquet::types::metadata::RowGroupMetadata;

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
            let leaf = crate::parquet::types::leaves::first_leaf(
                row_group.schema.fields(),
                pred.column_idx,
            );
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
    let (Some(min), Some(max)) = (stats.min.as_ref(), stats.max.as_ref()) else {
        return Ok(false);
    };
    bounds_eliminate(min, max, compare_type, constant)
}

/// Returns whether the inclusive range `[min, max]` proves that
/// `col <compare_type> constant` matches no value in it. Shared by row-group
/// elimination and file-level ([`DeltaFileEntry`](crate::manifest::DeltaFileEntry))
/// stats pruning, so both reason about bounds identically. `min`/`max` are
/// single-value scalars typed as the physical column.
pub fn bounds_eliminate(
    min: &Scalar<ArrayRef>,
    max: &Scalar<ArrayRef>,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    // Stats come back typed as the physical parquet column (e.g. a DATE stored
    // as UInt16), while the constant carries the logical type (e.g. Date32).
    // The comparison kernels need matching types, so when they differ we can't
    // prune — keep the range (always safe, just no pruning).
    if min.get().0.data_type() != constant.get().0.data_type() {
        return Ok(false);
    }

    Ok(match compare_type {
        // `col <> k` is true on every row unless every row equals `k` —
        // provable only when min == max == k.
        CompareType::NotEqual => {
            bool_kernel(min, constant, arrow_ord::cmp::eq)?
                && bool_kernel(max, constant, arrow_ord::cmp::eq)?
        }
        // `col = k` can never match when k is strictly outside [min, max].
        CompareType::Equal => {
            bool_kernel(constant, min, arrow_ord::cmp::lt)?
                || bool_kernel(constant, max, arrow_ord::cmp::gt)?
        }
        // `col < k` matches nothing when every value is >= k, i.e. min >= k.
        CompareType::Less => bool_kernel(min, constant, arrow_ord::cmp::gt_eq)?,
        // `col > k` matches nothing when every value is <= k, i.e. max <= k.
        CompareType::Greater => bool_kernel(max, constant, arrow_ord::cmp::lt_eq)?,
        // `col <= k` matches nothing when every value is > k, i.e. min > k.
        CompareType::LessEqual => bool_kernel(min, constant, arrow_ord::cmp::gt)?,
        // `col >= k` matches nothing when every value is < k, i.e. max < k.
        CompareType::GreaterEqual => bool_kernel(max, constant, arrow_ord::cmp::lt)?,
    })
}

/// Run an Arrow comparison kernel on two scalars and read its single boolean.
fn bool_kernel(
    a: &Scalar<ArrayRef>,
    b: &Scalar<ArrayRef>,
    kernel: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>,
) -> Result<bool, ArrowError> {
    let result = kernel(a as &dyn Datum, b as &dyn Datum)?;
    Ok(result.len() == 1 && result.value(0))
}
