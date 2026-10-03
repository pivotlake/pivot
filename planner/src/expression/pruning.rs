//! Translate SQL filter shapes into the storage-independent pruning interface.
//! The original filter remains above the scan to evaluate surviving rows.

use super::{CompareType, Expression, Function, TableFilter};
use crate::types::{Type, UTC_TIMEZONE, physical_arrow_type};
use arrow_array::{Array, ArrayRef, Datum, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use pruning::{ColumnPath, Predicate};
use std::sync::Arc;

impl TableFilter {
    /// Translate supported column comparisons into metadata pruning predicates.
    /// Unsupported expressions stay with the SQL filter above the scan.
    pub fn pruning_predicates(&self) -> Vec<Predicate> {
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
) -> Option<Predicate> {
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
                prunable_column(cast.source(), constant)?,
                as_timestamp_constant(constant)?,
            )
        }
        column => (prunable_column(column, constant)?, constant.clone()),
    };
    Some(Predicate {
        column,
        comparison: compare_type.into(),
        value,
    })
}

/// The column, or the field of a VARIANT column, whose statistics can prune a
/// comparison with `constant`.
fn prunable_column(expr: &Expression, constant: &Scalar<ArrayRef>) -> Option<ColumnPath> {
    match expr {
        Expression::Ref(reference) => Some(reference.column_idx.into()),
        Expression::Function(Function::VariantGet(read)) => {
            let Expression::Ref(reference) = read.input.as_ref() else {
                return None;
            };
            // Statistics describe a field as one type, and answer a constant
            // of that type: the constant must be of the type the field is
            // cast to.
            let cast_type = physical_arrow_type(read.as_type.as_ref()?);
            (constant.get().0.data_type() == &cast_type).then(|| ColumnPath {
                column_idx: reference.column_idx,
                path: read.path.clone(),
            })
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
