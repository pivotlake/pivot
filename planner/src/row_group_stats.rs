//! Stats-based row-group elimination.
//!
//! Given a single-column predicate `col <op> constant`, decide whether a row
//! group is guaranteed to contain no matching row from its min/max statistics.
//! Shared by static filter pushdown (in `catalog::ParquetCatalogTable`) and by
//! dynamic-filter pruning at scan time, so both reason about stats identically.

use arrow_array::{ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::ArrowError;
use dispatch::RowGroupMetadata;

use crate::expression::CompareType;

/// Returns `Ok(true)` when min/max statistics prove no row in `row_group` can
/// satisfy `column_idx <compare_type> constant`, so the row group can be
/// skipped. Returns `Ok(false)` when it must still be scanned — including when
/// stats are missing (absence of a bound is not proof of absence) or when the
/// stats and constant are different Arrow types (no safe comparison).
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

    // Stats come back typed as the physical parquet column (e.g. a DATE stored
    // as UInt16), while the constant carries the logical type (e.g. Date32).
    // The comparison kernels need matching types, so when they differ we can't
    // prune — keep the row group (always safe, just no pruning).
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
