//! The filters pushed into a scan, as the Iceberg predicate the crate plans
//! the scan with.
//!
//! The filters stay above the scan, so the predicate only ever skips work,
//! and everything it cannot express is left true: an expression with no
//! Iceberg form, a constant of a type Iceberg columns do not hold.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{ArrayRef, Datum as _, Scalar};
use arrow_schema::{DataType, TimeUnit};
use iceberg::expr::{BinaryExpression, Bind, Predicate, PredicateOperator, Reference};
use iceberg::spec::{Datum, PrimitiveType, Schema, SchemaRef};
use planner::expression::{CompareType, ConjunctionOp, Expression};
use planner::types::Type;

/// Each value of an `IN` list is compared with every object's bounds, so a
/// longer list is left to the filter above the scan.
const MAX_IN_LIST_VALUES: usize = 200;

/// What `filters` require of a row of `schema`, as one predicate: true
/// wherever every filter may be. A filter whose predicate `schema` does not
/// bind (a constant of another type than its column's) is left true.
pub(crate) fn filters_to_predicate(schema: &SchemaRef, filters: &[Expression]) -> Predicate {
    let predicates = filters
        .iter()
        .map(|filter| {
            let predicate = to_predicate(schema, filter);
            match predicate.bind(schema.clone(), true) {
                Ok(_) => predicate,
                Err(_) => Predicate::AlwaysTrue,
            }
        })
        .collect();
    combine(predicates, Predicate::AlwaysTrue, Predicate::and)
}

/// `expression` as an Iceberg predicate: true wherever the expression may be.
fn to_predicate(schema: &Schema, expression: &Expression) -> Predicate {
    match expression {
        Expression::Compare(compare) => compare_with_constant(
            schema,
            &compare.left,
            to_operator(compare.compare_type),
            &compare.right,
        ),
        Expression::Between(between) => {
            let (lower, upper) = match (between.lower_inclusive, between.upper_inclusive) {
                (true, true) => (CompareType::GreaterEqual, CompareType::LessEqual),
                (true, false) => (CompareType::GreaterEqual, CompareType::Less),
                (false, true) => (CompareType::Greater, CompareType::LessEqual),
                (false, false) => (CompareType::Greater, CompareType::Less),
            };
            let input = &between.input;
            compare_with_constant(schema, input, to_operator(lower), &between.lower).and(
                compare_with_constant(schema, input, to_operator(upper), &between.upper),
            )
        }
        Expression::Conjunction(conjunction) => {
            let children = conjunction
                .children
                .iter()
                .map(|child| to_predicate(schema, child))
                .collect();
            match conjunction.op {
                ConjunctionOp::And => combine(children, Predicate::AlwaysTrue, Predicate::and),
                ConjunctionOp::Or => combine(children, Predicate::AlwaysFalse, Predicate::or),
            }
        }
        // A row is in the list when it equals one of its values.
        Expression::InList(list) if list.values.len() <= MAX_IN_LIST_VALUES => {
            let equalities = list
                .values
                .iter()
                .map(|value| {
                    compare_with_constant(schema, &list.input, PredicateOperator::Eq, value)
                })
                .collect();
            combine(equalities, Predicate::AlwaysFalse, Predicate::or)
        }
        _ => Predicate::AlwaysTrue,
    }
}

/// `predicates` joined pairwise by `join` into one tree, `of_none` when there
/// are none. The tree is as shallow as their number allows: a chain as long as
/// a wide `OR` would overflow the stack of whatever walks it.
fn combine(
    mut predicates: Vec<Predicate>,
    of_none: Predicate,
    join: fn(Predicate, Predicate) -> Predicate,
) -> Predicate {
    while predicates.len() > 1 {
        let mut joined = Vec::with_capacity(predicates.len().div_ceil(2));
        let mut remaining = predicates.into_iter();
        while let Some(left) = remaining.next() {
            joined.push(match remaining.next() {
                Some(right) => join(left, right),
                None => left,
            });
        }
        predicates = joined;
    }
    predicates.pop().unwrap_or(of_none)
}

/// `column <operator> constant`, when `column` reads a column of the table and
/// `constant` is one. DuckDB normally puts the column on the left; a constant
/// on the left is not one.
fn compare_with_constant(
    schema: &Schema,
    column: &Expression,
    operator: PredicateOperator,
    constant: &Expression,
) -> Predicate {
    // DuckDB casts a TIMESTAMP column to TIMESTAMPTZ to compare it with one.
    // In UTC the cast keeps the microsecond value, so the comparison is the
    // column's own, and binding reads the constant as a TIMESTAMP.
    let column = match column {
        Expression::Cast(cast)
            if cast.target == Type::TimestampTz
                && matches!(
                    cast.source(),
                    Expression::Ref(reference) if reference.return_type == Type::Timestamp
                ) =>
        {
            cast.source()
        }
        column => column,
    };
    let (Expression::Ref(reference), Expression::Constant(constant)) = (column, constant) else {
        return Predicate::AlwaysTrue;
    };
    // A comparison with NULL is never true.
    if constant.get().0.is_null(0) {
        return Predicate::AlwaysFalse;
    }
    let Some(constant) = constant_to_datum(constant) else {
        return Predicate::AlwaysTrue;
    };
    let name = &schema.as_struct().fields()[reference.column_idx].name;
    Predicate::Binary(BinaryExpression::new(
        operator,
        Reference::new(name),
        constant,
    ))
}

fn to_operator(compare_type: CompareType) -> PredicateOperator {
    match compare_type {
        CompareType::Equal => PredicateOperator::Eq,
        CompareType::NotEqual => PredicateOperator::NotEq,
        CompareType::Less => PredicateOperator::LessThan,
        CompareType::LessEqual => PredicateOperator::LessThanOrEq,
        CompareType::Greater => PredicateOperator::GreaterThan,
        CompareType::GreaterEqual => PredicateOperator::GreaterThanOrEq,
    }
}

/// A non-null constant as an Iceberg value of the type it has, or `None` for a
/// type Iceberg columns do not hold.
fn constant_to_datum(constant: &Scalar<ArrayRef>) -> Option<Datum> {
    let array = constant.get().0;
    Some(match array.data_type() {
        DataType::Boolean => Datum::bool(array.as_boolean().value(0)),
        DataType::Int32 => Datum::int(array.as_primitive::<Int32Type>().value(0)),
        DataType::Int64 => Datum::long(array.as_primitive::<Int64Type>().value(0)),
        DataType::Float32 => Datum::float(array.as_primitive::<Float32Type>().value(0)),
        DataType::Float64 => Datum::double(array.as_primitive::<Float64Type>().value(0)),
        DataType::Date32 => Datum::date(array.as_primitive::<Date32Type>().value(0)),
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
            let micros = array.as_primitive::<TimestampMicrosecondType>().value(0);
            match timezone {
                Some(_) => Datum::timestamptz_micros(micros),
                None => Datum::timestamp_micros(micros),
            }
        }
        DataType::Decimal64(precision, scale) => {
            let unscaled = array.as_primitive::<Decimal64Type>().value(0);
            decimal_to_datum(i128::from(unscaled), *precision, *scale)?
        }
        DataType::Decimal128(precision, scale) => {
            let unscaled = array.as_primitive::<Decimal128Type>().value(0);
            decimal_to_datum(unscaled, *precision, *scale)?
        }
        DataType::Utf8View => Datum::string(array.as_string_view().value(0)),
        _ => return None,
    })
}

fn decimal_to_datum(unscaled: i128, precision: u8, scale: i8) -> Option<Datum> {
    let decimal = PrimitiveType::Decimal {
        precision: u32::from(precision),
        scale: u32::try_from(scale).ok()?,
    };
    Datum::try_from_bytes(&unscaled.to_be_bytes(), decimal).ok()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, Scalar, StringViewArray};
    use iceberg::spec::{NestedField, PrimitiveType, Type as IcebergType};
    use planner::expression::{Between, Compare, InList, IsNull, Ref};
    use planner::types::Type as PivotType;

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::optional(
                    1,
                    "id",
                    IcebergType::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .unwrap(),
        )
    }

    fn column(column_idx: usize, return_type: PivotType) -> Box<Expression> {
        Box::new(Expression::Ref(Ref {
            column_idx,
            return_type,
            name: None,
        }))
    }

    fn constant(array: ArrayRef) -> Expression {
        Expression::Constant(Scalar::new(array))
    }

    fn long(value: Option<i64>) -> Expression {
        constant(Arc::new(Int64Array::from(vec![value])))
    }

    fn compare(left: Box<Expression>, compare_type: CompareType, right: Expression) -> Expression {
        Expression::Compare(Compare {
            left,
            right: Box::new(right),
            compare_type,
            return_type: PivotType::Boolean,
        })
    }

    fn id_column() -> Box<Expression> {
        column(0, PivotType::Int64)
    }

    #[test]
    fn a_comparison_names_the_column_and_keeps_its_operator() {
        let filters = [compare(id_column(), CompareType::Less, long(Some(5)))];

        let predicate = filters_to_predicate(&schema(), &filters);

        assert_eq!(predicate.to_string(), "id < 5");
    }

    #[test]
    fn filters_are_joined_by_and_and_a_between_becomes_two_comparisons() {
        let filters = [
            compare(id_column(), CompareType::NotEqual, long(Some(7))),
            Expression::Between(Between {
                input: id_column(),
                lower: Box::new(long(Some(1))),
                upper: Box::new(long(Some(9))),
                lower_inclusive: true,
                upper_inclusive: false,
            }),
        ];

        let predicate = filters_to_predicate(&schema(), &filters);

        assert_eq!(
            predicate.to_string(),
            "(id != 7) AND ((id >= 1) AND (id < 9))"
        );
    }

    #[test]
    fn an_in_list_within_the_limit_is_the_or_of_its_values() {
        let filters = [Expression::InList(InList {
            input: id_column(),
            values: vec![long(Some(1)), long(Some(2)), long(Some(3))],
        })];

        let predicate = filters_to_predicate(&schema(), &filters);

        assert_eq!(predicate.to_string(), "((id = 1) OR (id = 2)) OR (id = 3)");
    }

    #[test]
    fn what_has_no_iceberg_form_or_does_not_bind_is_left_true() {
        let filters = [
            Expression::InList(InList {
                input: id_column(),
                values: (0..=MAX_IN_LIST_VALUES as i64)
                    .map(|value| long(Some(value)))
                    .collect(),
            }),
            Expression::IsNull(IsNull {
                negated: false,
                input: id_column(),
            }),
            compare(
                id_column(),
                CompareType::Equal,
                constant(Arc::new(StringViewArray::from(vec!["seven"]))),
            ),
        ];

        let predicate = filters_to_predicate(&schema(), &filters);

        assert_eq!(predicate, Predicate::AlwaysTrue);
    }

    #[test]
    fn a_comparison_with_null_is_never_true() {
        let filters = [compare(id_column(), CompareType::Equal, long(None))];

        let predicate = filters_to_predicate(&schema(), &filters);

        assert_eq!(predicate, Predicate::AlwaysFalse);
    }
}
