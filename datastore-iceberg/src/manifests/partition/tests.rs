use super::*;
use crate::pruning::Statistics;
use arrow_array::Datum as _;
use arrow_array::{ArrayRef, BooleanArray, Date32Array, Scalar, TimestampMicrosecondArray};
use iceberg::spec::{NestedField, PrimitiveLiteral, Schema, Transform as IcebergTransform, Type};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::{Type as PivotType, type_from_physical};
use std::sync::Arc;

use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate as IcebergPredicate, PredicateOperator,
    Reference,
};
use iceberg::transform::create_transform_function;

const COMPARISONS: [(CompareType, PredicateOperator); 6] = [
    (CompareType::Equal, PredicateOperator::Eq),
    (CompareType::NotEqual, PredicateOperator::NotEq),
    (CompareType::Less, PredicateOperator::LessThan),
    (CompareType::LessEqual, PredicateOperator::LessThanOrEq),
    (CompareType::Greater, PredicateOperator::GreaterThan),
    (
        CompareType::GreaterEqual,
        PredicateOperator::GreaterThanOrEq,
    ),
];

/// The table column the partition field is derived from.
const COLUMN: usize = 0;
/// A stand-in column whose own values are the partition values.
const PARTITION: usize = 1;

fn scalar(datum: &Datum) -> Scalar<ArrayRef> {
    let physical =
        physical_arrow_type(&to_pivot_type(&Type::Primitive(datum.data_type().clone())).unwrap());
    Scalar::new(build_array(&physical, [Some(datum.literal())].into_iter()))
}

fn compare(column_idx: usize, compare_type: CompareType, constant: Scalar<ArrayRef>) -> Expression {
    let constant_type = type_from_physical(constant.get().0.data_type()).unwrap();
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx,
            return_type: constant_type,
            name: None,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: PivotType::Boolean,
    })
}

fn partition_predicate(
    comparison: CompareType,
    literal: &PrimitiveLiteral,
    data_type: &DataType,
) -> Expression {
    let constant = Scalar::new(build_array(data_type, [Some(literal)].into_iter()));
    compare(PARTITION, comparison, constant)
}

/// What iceberg-rs's projection of a column predicate requires of the
/// partition values, or `None` when it rules every partition out.
fn required_of_partition(
    projected: Option<IcebergPredicate>,
    data_type: &DataType,
) -> Option<Vec<Expression>> {
    match projected {
        None | Some(IcebergPredicate::AlwaysTrue) => Some(Vec::new()),
        Some(IcebergPredicate::AlwaysFalse) => None,
        Some(IcebergPredicate::Binary(expression)) => {
            let (comparison, _) = COMPARISONS
                .into_iter()
                .find(|(_, op)| *op == expression.op())?;
            let literal = expression.literal().literal();
            Some(vec![partition_predicate(comparison, literal, data_type)])
        }
        // iceberg-rs keeps a pre-epoch partition and the one after it as a
        // set of two adjacent values, which is the range between them.
        Some(IcebergPredicate::Set(expression)) if expression.op() == PredicateOperator::In => {
            let lowest = expression
                .literals()
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())?;
            let highest = expression
                .literals()
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())?;
            Some(vec![
                partition_predicate(CompareType::GreaterEqual, lowest.literal(), data_type),
                partition_predicate(CompareType::LessEqual, highest.literal(), data_type),
            ])
        }
        Some(_) => Some(Vec::new()),
    }
}

/// Prune one object per value, partitioned by iceberg-rs, and check that
/// exactly the objects iceberg-rs's own projection keeps survive.
fn assert_matches_iceberg(
    column_type: PrimitiveType,
    transform: IcebergTransform,
    values: Vec<Datum>,
    constants: Vec<Datum>,
) {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                7,
                "value",
                Type::Primitive(column_type.clone()),
            ))])
            .build()
            .unwrap(),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("value", "partition", transform)
        .unwrap()
        .build()
        .unwrap();
    let field = fields(&schema, &spec).pop().unwrap();
    let function = create_transform_function(&transform).unwrap();
    let partitions: Vec<_> = values
        .iter()
        .map(|value| function.transform_literal(value).unwrap())
        .collect();
    let partition_values = build_array(
        &field.data_type,
        partitions
            .iter()
            .map(|value| value.as_ref().map(Datum::literal)),
    );
    let bounds = Bounds {
        lower: partition_values.clone(),
        upper: partition_values,
        all_null: BooleanArray::from(vec![false; values.len()]),
    };
    let statistics = Statistics::new(
        values.len(),
        vec![
            Statistic {
                column: COLUMN.into(),
                transform: field.transform,
                bounds: bounds.clone(),
            },
            Statistic {
                column: PARTITION.into(),
                transform: Transform::Identity,
                bounds,
            },
        ],
    );

    for (comparison, op) in COMPARISONS {
        for constant in &constants {
            let bound = IcebergPredicate::Binary(BinaryExpression::new(
                op,
                Reference::new("value"),
                constant.clone(),
            ))
            .bind(schema.clone(), true)
            .unwrap();
            let projected = match bound {
                BoundPredicate::AlwaysFalse => Some(IcebergPredicate::AlwaysFalse),
                bound => spec.fields()[0]
                    .transform
                    .project("partition", &bound)
                    .unwrap(),
            };
            let expected = match required_of_partition(projected, &field.data_type) {
                Some(required) => statistics.prune(&required),
                None => BooleanArray::from(vec![false; values.len()]),
            };

            let keep = statistics.prune(&[compare(COLUMN, comparison, scalar(constant))]);

            if transform == IcebergTransform::Day
                && matches!(
                    column_type,
                    PrimitiveType::Timestamp | PrimitiveType::Timestamptz
                )
                && matches!(constant.literal(), PrimitiveLiteral::Long(value) if *value <= 0)
            {
                // iceberg-rs 0.10's subsecond day conversion is not always a
                // floor before the epoch, so its projection is no oracle here.
                // The rows that match must survive in the partitions it wrote.
                let kernel = match comparison {
                    CompareType::Equal => arrow_ord::cmp::eq,
                    CompareType::NotEqual => arrow_ord::cmp::neq,
                    CompareType::Less => arrow_ord::cmp::lt,
                    CompareType::LessEqual => arrow_ord::cmp::lt_eq,
                    CompareType::Greater => arrow_ord::cmp::gt,
                    CompareType::GreaterEqual => arrow_ord::cmp::gt_eq,
                };
                for (row, value) in values.iter().enumerate() {
                    let matches = kernel(&scalar(value), &scalar(constant)).unwrap().value(0);
                    assert!(
                        !matches || keep.value(row),
                        "{column_type:?} {value:?} {comparison:?} {constant:?}"
                    );
                }
            } else {
                assert_eq!(
                    keep, expected,
                    "{column_type:?} {transform:?} {comparison:?} {constant:?}"
                );
            }
        }
    }
}

#[test]
fn numeric_partition_pruning_matches_iceberg() {
    let domain = [
        -32769, -32768, -129, -128, -11, -10, -1, 0, 1, 9, 10, 11, 127, 128, 255, 256,
    ];

    for column_type in [
        PrimitiveType::Int,
        PrimitiveType::Long,
        PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        },
        PrimitiveType::Decimal {
            precision: 24,
            scale: 3,
        },
    ] {
        let values: Vec<_> = domain
            .iter()
            .map(|&value| match column_type {
                PrimitiveType::Int => Datum::int(value),
                PrimitiveType::Long => Datum::long(i64::from(value)),
                _ => Datum::try_from_bytes(&i128::from(value).to_be_bytes(), column_type.clone())
                    .unwrap(),
            })
            .collect();
        for transform in [
            IcebergTransform::Identity,
            IcebergTransform::Truncate(10),
            IcebergTransform::Bucket(16),
        ] {
            assert_matches_iceberg(
                column_type.clone(),
                transform,
                values.clone(),
                values.clone(),
            );
        }
    }
}

#[test]
fn partitions_written_before_an_int_column_was_promoted_still_prune() {
    let written_as_int = vec![Datum::int(1), Datum::int(10)];
    let constants = vec![
        Datum::long(-1),
        Datum::long(1),
        Datum::long(10),
        Datum::long(i64::from(i32::MAX) + 1),
    ];

    for transform in [IcebergTransform::Identity, IcebergTransform::Bucket(16)] {
        assert_matches_iceberg(
            PrimitiveType::Long,
            transform,
            written_as_int.clone(),
            constants.clone(),
        );
    }
}

#[test]
fn calendar_partition_pruning_matches_iceberg_including_legacy_dates() {
    let days = [
        -366, -365, -32, -31, -1, 0, 1, 30, 31, 365, 366, 18261, 18262, 18293, 18321,
    ];

    for column_type in [
        PrimitiveType::Date,
        PrimitiveType::Timestamp,
        PrimitiveType::Timestamptz,
    ] {
        let values: Vec<_> = if column_type == PrimitiveType::Date {
            days.iter().map(|&day| Datum::date(day)).collect()
        } else {
            days.iter()
                .flat_map(|&day| {
                    let midnight = i64::from(day) * 86_400_000_000;
                    [midnight - 2, midnight - 1, midnight, midnight + 1]
                        .into_iter()
                        .map(|value| {
                            if column_type == PrimitiveType::Timestamp {
                                Datum::timestamp_micros(value)
                            } else {
                                Datum::timestamptz_micros(value)
                            }
                        })
                })
                .collect()
        };
        for transform in [
            IcebergTransform::Year,
            IcebergTransform::Month,
            IcebergTransform::Day,
            IcebergTransform::Hour,
        ] {
            if column_type == PrimitiveType::Date && transform == IcebergTransform::Hour {
                continue;
            }
            assert_matches_iceberg(
                column_type.clone(),
                transform,
                values.clone(),
                values.clone(),
            );
        }
    }
}

#[test]
fn string_and_identity_partition_pruning_matches_iceberg() {
    let strings: Vec<_> = ["", "a", "ab", "abc", "abcd", "é", "é雪", "é雪a", "雪"]
        .into_iter()
        .map(Datum::string)
        .collect();
    let booleans = vec![Datum::bool(false), Datum::bool(true)];
    let floats = [-1.0_f32, -0.0, 0.0, 1.0].map(Datum::float).to_vec();
    let doubles = [-1.0, -0.0, 0.0, 1.0].map(Datum::double).to_vec();

    for transform in [
        IcebergTransform::Identity,
        IcebergTransform::Truncate(2),
        IcebergTransform::Bucket(16),
    ] {
        assert_matches_iceberg(
            PrimitiveType::String,
            transform,
            strings.clone(),
            strings.clone(),
        );
    }
    for (column_type, values) in [
        (PrimitiveType::Boolean, booleans),
        (PrimitiveType::Float, floats),
        (PrimitiveType::Double, doubles),
    ] {
        assert_matches_iceberg(
            column_type,
            IcebergTransform::Identity,
            values.clone(),
            values,
        );
    }
}

#[test]
fn day_partitions_keep_interior_and_historical_boundaries() {
    // Partitions immediately before and after the epoch, plus a later day.
    let days = Arc::new(Date32Array::from(vec![-1, 0, 1])) as ArrayRef;
    let statistics = Statistics::new(
        3,
        vec![Statistic {
            column: COLUMN.into(),
            transform: Transform::Day,
            bounds: Bounds {
                lower: days.clone(),
                upper: days,
                all_null: BooleanArray::from(vec![false; 3]),
            },
        }],
    );
    let half_day = 43_200_000_000;

    // Iceberg deliberately widens pre-epoch projections for compatibility
    // with older writers. The positive midnight boundary is exact.
    for (comparison, micros, expected) in [
        (CompareType::Less, 0, vec![true, true, false]),
        (CompareType::LessEqual, 0, vec![true, true, false]),
        (CompareType::Less, half_day * 2, vec![true, true, false]),
        (CompareType::LessEqual, half_day * 2, vec![true, true, true]),
        (CompareType::Less, half_day, vec![true, true, false]),
        (CompareType::Greater, -half_day, vec![true, true, true]),
        (CompareType::GreaterEqual, 0, vec![false, true, true]),
    ] {
        let instant =
            Scalar::new(Arc::new(TimestampMicrosecondArray::from(vec![micros])) as ArrayRef);
        let keep = statistics.prune(&[compare(COLUMN, comparison, instant)]);

        assert_eq!(keep, BooleanArray::from(expected));
    }
}
