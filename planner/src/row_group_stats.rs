//! Stats-based row-group elimination.
//!
//! Given a single-column predicate of the form `col <op> constant`, decide
//! whether a row group is guaranteed to contain no matching rows based on
//! its min/max statistics. Used by both static filter pushdown (in
//! `catalog::ParquetCatalogTable`) and dynamic-filter pruning at scan time.

use arrow_array::{ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::ArrowError;
use dispatch::RowGroupMetadata;

use crate::expression::CompareType;

/// Returns `Ok(true)` when min/max statistics prove no row in `row_group`
/// can satisfy `column_idx <op> constant`. Returns `Ok(false)` when the
/// row group must still be scanned (including when stats are missing —
/// absence of a bound is not proof of absence).
pub fn row_group_eliminated(
    row_group: &RowGroupMetadata,
    column_idx: usize,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    let Some(stats) = row_group
        .columns
        .get(column_idx)
        .and_then(|c| c.statistics.as_ref())
    else {
        return Ok(false);
    };
    let (Some(min), Some(max)) = (stats.min.as_ref(), stats.max.as_ref()) else {
        return Ok(false);
    };

    Ok(match compare_type {
        // `col <> k` is true on every row unless every row in this group
        // equals `k` — provable only when min == max == k.
        CompareType::NotEqual => {
            bool_kernel(min, constant, arrow_ord::cmp::eq)?
                && bool_kernel(max, constant, arrow_ord::cmp::eq)?
        }
        // `col = k` can never match when k is strictly outside [min, max].
        CompareType::Equal => {
            bool_kernel(constant, min, arrow_ord::cmp::lt)?
                || bool_kernel(constant, max, arrow_ord::cmp::gt)?
        }
        // `col < k` can never match when every value is >= k, i.e. min >= k.
        CompareType::LessThan => bool_kernel(min, constant, arrow_ord::cmp::gt_eq)?,
        // `col <= k` can never match when min > k.
        CompareType::LessThanOrEqual => bool_kernel(min, constant, arrow_ord::cmp::gt)?,
        // `col > k` can never match when max <= k.
        CompareType::GreaterThan => bool_kernel(max, constant, arrow_ord::cmp::lt_eq)?,
        // `col >= k` can never match when max < k.
        CompareType::GreaterThanOrEqual => bool_kernel(max, constant, arrow_ord::cmp::lt)?,
    })
}

fn bool_kernel(
    a: &Scalar<ArrayRef>,
    b: &Scalar<ArrayRef>,
    kernel: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>,
) -> Result<bool, ArrowError> {
    let result = kernel(a as &dyn Datum, b as &dyn Datum)?;
    Ok(result.len() == 1 && result.value(0))
}