//! Translate SQL filter shapes into the storage-independent pruning interface.
//! The original filter remains above the scan to evaluate surviving rows.

use super::{CompareType, Expression, Function, JsonPath, TableFilter};
use crate::types::{Type, UTC_TIMEZONE, physical_arrow_type};
use arrow_array::{ArrayRef, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use pruning::ColumnPredicate;
use std::sync::Arc;

impl TableFilter {
    /// Translate supported column comparisons into metadata pruning predicates.
    /// Unsupported expressions stay with the SQL filter above the scan.
    pub fn pruning_predicates(&self) -> Vec<ColumnPredicate> {
        let TableFilter::Expression(expr) = self else {
            return Vec::new();
        };
        match expr.as_ref() {
            Expression::Compare(compare) => {
                // DuckDB normally canonicalizes the column to the left. Keep
                // constant-left comparisons upstream instead of risking an
                // incorrect direction during metadata pruning.
                let Expression::Constant(constant) = compare.right.as_ref() else {
                    return Vec::new();
                };
                from_bound(&compare.left, compare.compare_type, constant)
                    .into_iter()
                    .collect()
            }
            // DuckDB's filter combiner folds a lower and an upper bound on one
            // column into a single BETWEEN before offering it for pushdown, so
            // a two-sided range always arrives as one expression. Each bound
            // prunes on its own.
            Expression::Between(between) => {
                let (Expression::Constant(lower), Expression::Constant(upper)) =
                    (between.lower.as_ref(), between.upper.as_ref())
                else {
                    return Vec::new();
                };
                let lower_compare = if between.lower_inclusive {
                    CompareType::GreaterEqual
                } else {
                    CompareType::Greater
                };
                let upper_compare = if between.upper_inclusive {
                    CompareType::LessEqual
                } else {
                    CompareType::Less
                };
                [
                    from_bound(&between.input, lower_compare, lower),
                    from_bound(&between.input, upper_compare, upper),
                ]
                .into_iter()
                .flatten()
                .collect()
            }
            _ => Vec::new(),
        }
    }
}

/// A single `column <compare_type> constant` bound, when the column side
/// is one whose statistics can prune.
fn from_bound(
    column_expr: &Expression,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Option<ColumnPredicate> {
    // DuckDB casts a TIMESTAMP column to TIMESTAMPTZ for a mixed-type
    // comparison. In UTC the cast preserves the microsecond value, so
    // retag the constant as TIMESTAMP for statistics pruning.
    let (column, value) = match column_expr {
        Expression::Cast(cast)
            if cast.target == Type::TimestampTz
                && matches!(
                    cast.source(),
                    Expression::Ref(reference) if reference.return_type == Type::Timestamp
                ) =>
        {
            (
                prunable_column_and_json_path(cast.source())?,
                as_timestamp_constant(constant)?,
            )
        }
        column => (prunable_column_and_json_path(column)?, constant.clone()),
    };
    Some(ColumnPredicate {
        column_idx: column.column_idx,
        path: column.path,
        as_type: column.as_type,
        compare_type: compare_type.into(),
        value,
    })
}

/// A top-level column and the optional variant path whose statistics can prune.
struct PrunableColumn {
    column_idx: usize,
    path: JsonPath,
    as_type: Option<DataType>,
}

/// Returns the column and optional variant path that can use row-group stats.
fn prunable_column_and_json_path(expr: &Expression) -> Option<PrunableColumn> {
    match expr {
        Expression::Ref(reference) => Some(PrunableColumn {
            column_idx: reference.column_idx,
            path: Vec::new(),
            as_type: None,
        }),
        Expression::Function(Function::VariantGet(read)) if read.as_type.is_some() => {
            match read.input.as_ref() {
                Expression::Ref(reference) => Some(PrunableColumn {
                    column_idx: reference.column_idx,
                    path: read.path.clone(),
                    as_type: read.as_type.as_ref().map(physical_arrow_type),
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Remove the UTC timezone annotation from a one-value TIMESTAMPTZ constant
/// without touching its value buffer. This mirrors DuckDB's UTC
/// TIMESTAMP-to-TIMESTAMPTZ cast in the other direction for statistics only.
fn as_timestamp_constant(value: &Scalar<ArrayRef>) -> Option<Scalar<ArrayRef>> {
    let array = value.clone().into_inner();
    let DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) = array.data_type() else {
        return None;
    };
    if timezone.as_ref() != UTC_TIMEZONE {
        return None;
    }
    let timestamp = array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()?
        .clone()
        .with_timezone_opt(None::<Arc<str>>);
    Some(Scalar::new(Arc::new(timestamp) as ArrayRef))
}
