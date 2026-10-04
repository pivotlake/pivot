//! The one shape of filter that metadata can answer on its own: a column, or a
//! field cast out of a VARIANT column, compared with a constant.

use arrow_array::{Array, ArrayRef, Datum, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use planner::expression::{Between, Compare, CompareType, Expression, Function, JsonPath};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};
use std::sync::Arc;

/// A column of a table, or the field at `path` inside it. Only VARIANT
/// columns have fields; a plain column has an empty path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColumnPath {
    /// Index in the full table schema, before scan projection.
    pub column_idx: usize,
    pub path: JsonPath,
}

impl From<usize> for ColumnPath {
    fn from(column_idx: usize) -> Self {
        Self {
            column_idx,
            path: Vec::new(),
        }
    }
}

/// `column <compare_type> constant`. For a VARIANT field the constant has the
/// type the field is cast to.
#[derive(Clone, Debug)]
pub struct ConstantComparison {
    pub column: ColumnPath,
    pub compare_type: CompareType,
    pub constant: Scalar<ArrayRef>,
}

impl ConstantComparison {
    /// `column_expr <compare_type> constant` as a comparison of a column, when
    /// `column_expr` reads one: a column, a column DuckDB cast to compare it
    /// with a TIMESTAMPTZ, or a cast VARIANT field.
    pub fn of(
        column_expr: &Expression,
        compare_type: CompareType,
        constant: &Scalar<ArrayRef>,
    ) -> Option<Self> {
        // DuckDB casts a TIMESTAMP column to TIMESTAMPTZ for a mixed-type
        // comparison. In UTC the cast preserves the microsecond value, so
        // the comparison is the column's with the constant as a TIMESTAMP.
        let (column, constant) = match column_expr {
            Expression::Cast(cast)
                if cast.target == Type::TimestampTz
                    && matches!(
                        cast.source(),
                        Expression::Ref(reference) if reference.return_type == Type::Timestamp
                    ) =>
            {
                (
                    column_path(cast.source(), constant)?,
                    as_timestamp_constant(constant)?,
                )
            }
            column => (column_path(column, constant)?, constant.clone()),
        };
        Some(Self {
            column,
            compare_type,
            constant,
        })
    }

    /// `compare` as a comparison of a column with a constant. DuckDB normally
    /// puts the column on the left; a constant on the left is not one.
    pub fn from_compare(compare: &Compare) -> Option<Self> {
        let Expression::Constant(constant) = compare.right.as_ref() else {
            return None;
        };
        Self::of(&compare.left, compare.compare_type, constant)
    }

    /// The comparisons with the lower and upper bound of `between`, when its
    /// input is a column and both bounds are constants.
    pub fn from_between(between: &Between) -> Option<[Self; 2]> {
        let (Expression::Constant(lower), Expression::Constant(upper)) =
            (between.lower.as_ref(), between.upper.as_ref())
        else {
            return None;
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
        Some([
            Self::of(&between.input, lower_compare, lower)?,
            Self::of(&between.input, upper_compare, upper)?,
        ])
    }
}

/// The column, or the field of a VARIANT column, that `expr` reads and
/// compares with `constant`.
fn column_path(expr: &Expression, constant: &Scalar<ArrayRef>) -> Option<ColumnPath> {
    match expr {
        Expression::Ref(reference) => Some(reference.column_idx.into()),
        Expression::Function(Function::VariantGet(read)) => {
            let Expression::Ref(reference) = read.input.as_ref() else {
                return None;
            };
            // Metadata describes a field as one type, and answers a constant
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
/// TIMESTAMP-to-TIMESTAMPTZ cast in the other direction.
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
