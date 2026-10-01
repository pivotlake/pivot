//! Stats-based row-group elimination.
//!
//! Given a single-column predicate `col <op> constant`, decide whether a row
//! group is guaranteed to contain no matching row from its min/max statistics.
//! Shared by static filter pushdown and dynamic-filter pruning at scan time, so
//! every path reasons about statistics identically.

use std::sync::Arc;

use arrow_arith::boolean::{and, or};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq};
use arrow_schema::ArrowError;

use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

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
    // Every ordinary comparison against SQL NULL is unknown and therefore
    // cannot pass a WHERE filter. This also handles shredded typed leaves whose
    // only fallbacks are absent values or JSON nulls.
    if stats.null_count == Some(row_group.num_rows) {
        return Ok(true);
    }
    let (Some(min), Some(max)) = (stats.min(), stats.max()) else {
        return Ok(false);
    };
    bounds_eliminate(&min, &max, compare_type, constant)
}

/// Returns whether the inclusive range `[min, max]` proves that
/// `col <compare_type> constant` matches no value in it. `min`/`max` are
/// single-value scalars typed as the physical column. The single-range form
/// of [`bounds_exclude`], which a row group's statistics come in.
pub fn bounds_eliminate(
    min: &Scalar<ArrayRef>,
    max: &Scalar<ArrayRef>,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    let excluded = bounds_exclude(min, max, compare_type, constant)?;
    Ok(excluded.is_some_and(|mask| mask.len() == 1 && mask.is_valid(0) && mask.value(0)))
}

/// Which of the inclusive ranges `[min[i], max[i]]` prove that
/// `col <compare_type> constant` matches no value in them: a mask that is
/// `true` at every range that excludes the comparison. A null mask element is
/// a range with no bound, which proves nothing; read the mask as excluded only
/// where it is valid and true. `min` and `max` are same-typed arrays with one
/// range per element (a null in one is a null in the other), or one-element
/// scalars for a single range. `None` when the bounds are not of the
/// constant's type, in which case nothing is proved either: statistics come
/// typed as the physical column (a DATE stored as UInt16), while the constant
/// carries the logical type (Date32), and the kernels need them to match.
/// Shared by row-group elimination and file-level table-format stats pruning,
/// so both reason about bounds identically, and a table format's files are
/// pruned in one comparison over all of them.
pub fn bounds_exclude(
    min: &dyn Datum,
    max: &dyn Datum,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<Option<BooleanArray>, ArrowError> {
    if min.get().0.data_type() != constant.get().0.data_type() {
        return Ok(None);
    }
    let mask = match compare_type {
        // `col <> k` is true on every row unless every row equals `k`,
        // provable only when min == max == k.
        CompareType::NotEqual => and(&eq(min, constant)?, &eq(max, constant)?)?,
        // `col = k` can never match when k is strictly outside [min, max].
        CompareType::Equal => or(&lt(constant, min)?, &gt(constant, max)?)?,
        // `col < k` matches nothing when every value is >= k, i.e. min >= k.
        CompareType::Less => gt_eq(min, constant)?,
        // `col > k` matches nothing when every value is <= k, i.e. max <= k.
        CompareType::Greater => lt_eq(max, constant)?,
        // `col <= k` matches nothing when every value is > k, i.e. min > k.
        CompareType::LessEqual => gt(min, constant)?,
        // `col >= k` matches nothing when every value is < k, i.e. max < k.
        CompareType::GreaterEqual => lt(max, constant)?,
    };
    Ok(Some(mask))
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;

    use super::*;

    fn constant(value: i64) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
    }

    /// The ranges `[1, 3]`, `[5, 7]`, `[9, 11]` and one with no bound, and
    /// which of them `compare` against `value` excludes.
    fn excluded(compare: CompareType, value: i64) -> Vec<bool> {
        let min = Int64Array::from(vec![Some(1), Some(5), Some(9), None]);
        let max = Int64Array::from(vec![Some(3), Some(7), Some(11), None]);
        let mask = bounds_exclude(&min, &max, compare, &constant(value))
            .unwrap()
            .expect("same-typed bounds are compared");
        (0..mask.len())
            .map(|range| mask.is_valid(range) && mask.value(range))
            .collect()
    }

    #[test]
    fn every_range_is_judged_in_one_comparison() {
        assert_eq!(excluded(CompareType::Equal, 6), [true, false, true, false]);
        assert_eq!(
            excluded(CompareType::NotEqual, 6),
            [false, false, false, false]
        );
        assert_eq!(excluded(CompareType::Less, 5), [false, true, true, false]);
        assert_eq!(
            excluded(CompareType::Greater, 7),
            [true, true, false, false]
        );
        assert_eq!(
            excluded(CompareType::LessEqual, 4),
            [false, true, true, false]
        );
        assert_eq!(
            excluded(CompareType::GreaterEqual, 8),
            [true, true, false, false]
        );
    }

    #[test]
    fn bounds_of_another_type_prove_nothing() {
        let min = arrow_array::Int32Array::from(vec![1]);
        let max = arrow_array::Int32Array::from(vec![3]);

        let mask = bounds_exclude(&min, &max, CompareType::Equal, &constant(6)).unwrap();

        assert!(mask.is_none());
    }
}
