use super::*;
use ::pruning::{ColumnPredicate, Comparison, PartitionStatistics, StatisticsBatch};
use arrow_array::{ArrayRef, Date32Array, Scalar, TimestampMicrosecondArray};
use iceberg::spec::{NestedField, Schema, Transform, Type};
use std::collections::BTreeMap;
use std::sync::Arc;

use ::pruning::PruningPredicate;
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
};
use iceberg::transform::create_transform_function;

fn scalar(datum: &Datum) -> Scalar<ArrayRef> {
    let physical =
        physical_arrow_type(&to_pivot_type(&Type::Primitive(datum.data_type().clone())).unwrap());
    Scalar::new(build_array(&physical, [Some(datum.literal())].into_iter()))
}

/// Independent reference for the projection formerly used by this adapter.
/// Production only translates metadata; iceberg-rs projection is an oracle in
/// these tests, so the shared crate has no dependency on a storage format.
fn reference(predicate: Predicate, data_type: &DataType) -> PruningPredicate {
    match predicate {
        Predicate::AlwaysTrue => PruningPredicate::Always(true),
        Predicate::AlwaysFalse => PruningPredicate::Always(false),
        Predicate::Binary(expression) => {
            let compare = match expression.op() {
                PredicateOperator::Eq => Comparison::Equal,
                PredicateOperator::NotEq => Comparison::NotEqual,
                PredicateOperator::LessThan => Comparison::Less,
                PredicateOperator::LessThanOrEq => Comparison::LessEqual,
                PredicateOperator::GreaterThan => Comparison::Greater,
                PredicateOperator::GreaterThanOrEq => Comparison::GreaterEqual,
                _ => return PruningPredicate::Always(true),
            };
            PruningPredicate::Compare {
                compare,
                value: Scalar::new(build_array(
                    data_type,
                    [Some(expression.literal().literal())].into_iter(),
                )),
            }
        }
        Predicate::Set(expression) if expression.op() == PredicateOperator::In => {
            PruningPredicate::Or(
                expression
                    .literals()
                    .iter()
                    .map(|literal| PruningPredicate::Compare {
                        compare: Comparison::Equal,
                        value: Scalar::new(build_array(
                            data_type,
                            [Some(literal.literal())].into_iter(),
                        )),
                    })
                    .collect(),
            )
        }
        _ => PruningPredicate::Always(true),
    }
}

fn assert_matches_iceberg(
    source_type: PrimitiveType,
    transform: Transform,
    values: Vec<Datum>,
    constants: Vec<Datum>,
) {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                7,
                "value",
                Type::Primitive(source_type.clone()),
            ))])
            .build()
            .unwrap(),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("value", "partition", transform)
        .unwrap()
        .build()
        .unwrap();
    let field = &spec.fields()[0];
    let column = columns(&schema, &schema, &spec).unwrap().pop().unwrap();
    let function = create_transform_function(&field.transform).unwrap();
    let partitions: Vec<_> = values
        .iter()
        .map(|value| function.transform_literal(value).unwrap())
        .collect();
    let array = build_array(
        &column.data_type,
        partitions
            .iter()
            .map(|value| value.as_ref().map(Datum::literal)),
    );
    let bounds = StatisticsBatch::new(
        values.len(),
        BTreeMap::new(),
        vec![PartitionStatistics {
            expression: column.expression,
            bounds: ColumnBounds {
                lower: Some(array.clone()),
                upper: Some(array),
                nan_free: Some(arrow_array::BooleanArray::from(vec![true; values.len()])),
                ..Default::default()
            },
        }],
    )
    .unwrap();
    for (compare, op) in [
        (Comparison::Equal, PredicateOperator::Eq),
        (Comparison::NotEqual, PredicateOperator::NotEq),
        (Comparison::Less, PredicateOperator::LessThan),
        (Comparison::LessEqual, PredicateOperator::LessThanOrEq),
        (Comparison::Greater, PredicateOperator::GreaterThan),
        (Comparison::GreaterEqual, PredicateOperator::GreaterThanOrEq),
    ] {
        for constant in &constants {
            let bound = Predicate::Binary(BinaryExpression::new(
                op,
                Reference::new("value"),
                constant.clone(),
            ))
            .bind(schema.clone(), true)
            .unwrap();
            let expected = if matches!(bound, BoundPredicate::AlwaysFalse) {
                PruningPredicate::Always(false)
            } else {
                field
                    .transform
                    .project("partition", &bound)
                    .unwrap()
                    .map_or(PruningPredicate::Always(true), |predicate| {
                        reference(predicate, &column.data_type)
                    })
            };
            let filter = ColumnPredicate {
                column_idx: 0,
                path: vec![],
                as_type: None,
                compare_type: compare,
                value: scalar(constant),
            };
            let keep = bounds.prune(&[filter]).unwrap();
            if field.transform == Transform::Day
                && matches!(
                    source_type,
                    PrimitiveType::Timestamp | PrimitiveType::Timestamptz
                )
                && matches!(constant.literal(), PrimitiveLiteral::Long(value) if *value <= 0)
            {
                // iceberg-rs 0.10's subsecond day conversion is not always a
                // floor before the epoch. Its projection is not an oracle for
                // these cases: verify actual matching rows in SDK-written
                // partitions survive. The shared crate separately exercises
                // the same boundaries against spec-compliant floor partitions.
                let kernel = match compare {
                    Comparison::Equal => arrow_ord::cmp::eq,
                    Comparison::NotEqual => arrow_ord::cmp::neq,
                    Comparison::Less => arrow_ord::cmp::lt,
                    Comparison::LessEqual => arrow_ord::cmp::lt_eq,
                    Comparison::Greater => arrow_ord::cmp::gt,
                    Comparison::GreaterEqual => arrow_ord::cmp::gt_eq,
                };
                for (row, value) in values.iter().enumerate() {
                    let matches = kernel(&scalar(value), &scalar(constant)).unwrap().value(0);
                    assert!(
                        !matches || keep.value(row),
                        "{source_type:?} {value:?} {compare:?} {constant:?}"
                    );
                }
            } else {
                assert_eq!(
                    keep,
                    expected
                        .may_match(&bounds.partition_stats()[0].bounds, values.len())
                        .unwrap(),
                    "{source_type:?} {:?} {compare:?} {constant:?}",
                    field.transform
                );
            }
        }
    }
}

#[test]
fn shared_numeric_partition_projection_matches_iceberg() {
    let domain = [
        -32769, -32768, -129, -128, -11, -10, -1, 0, 1, 9, 10, 11, 127, 128, 255, 256,
    ];
    for source in [
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
            .map(|&value| match source {
                PrimitiveType::Int => Datum::int(value),
                PrimitiveType::Long => Datum::long(i64::from(value)),
                _ => {
                    Datum::try_from_bytes(&i128::from(value).to_be_bytes(), source.clone()).unwrap()
                }
            })
            .collect();
        for transform in [
            Transform::Identity,
            Transform::Truncate(10),
            Transform::Bucket(16),
        ] {
            assert_matches_iceberg(source.clone(), transform, values.clone(), values.clone());
        }
    }
    // Old partitions retain their writer's type after an int -> long promotion.
    assert_matches_iceberg(
        PrimitiveType::Int,
        Transform::Bucket(16),
        vec![Datum::int(1), Datum::int(10)],
        vec![
            Datum::long(-1),
            Datum::long(1),
            Datum::long(10),
            Datum::long(i64::from(i32::MAX) + 1),
        ],
    );
}

#[test]
fn shared_calendar_projection_matches_iceberg_including_legacy_dates() {
    let days = [
        -366, -365, -32, -31, -1, 0, 1, 30, 31, 365, 366, 18261, 18262, 18293, 18321,
    ];
    for source in [
        PrimitiveType::Date,
        PrimitiveType::Timestamp,
        PrimitiveType::Timestamptz,
    ] {
        let values: Vec<_> = if source == PrimitiveType::Date {
            days.iter().map(|&day| Datum::date(day)).collect()
        } else {
            days.iter()
                .flat_map(|&day| {
                    let midnight = i64::from(day) * 86_400_000_000;
                    [midnight - 2, midnight - 1, midnight, midnight + 1]
                        .into_iter()
                        .map(|value| {
                            if source == PrimitiveType::Timestamp {
                                Datum::timestamp_micros(value)
                            } else {
                                Datum::timestamptz_micros(value)
                            }
                        })
                })
                .collect()
        };
        for transform in [
            Transform::Year,
            Transform::Month,
            Transform::Day,
            Transform::Hour,
        ] {
            if source == PrimitiveType::Date && transform == Transform::Hour {
                continue;
            }
            assert_matches_iceberg(source.clone(), transform, values.clone(), values.clone());
        }
    }
}

#[test]
fn shared_string_and_identity_projection_matches_iceberg() {
    let strings: Vec<_> = ["", "a", "ab", "abc", "abcd", "é", "é雪", "é雪a", "雪"]
        .into_iter()
        .map(Datum::string)
        .collect();
    for transform in [
        Transform::Identity,
        Transform::Truncate(2),
        Transform::Bucket(16),
    ] {
        assert_matches_iceberg(
            PrimitiveType::String,
            transform,
            strings.clone(),
            strings.clone(),
        );
    }
    let booleans = vec![Datum::bool(false), Datum::bool(true)];
    assert_matches_iceberg(
        PrimitiveType::Boolean,
        Transform::Identity,
        booleans.clone(),
        booleans,
    );
    let floats = [-1.0_f32, -0.0, 0.0, 1.0].map(Datum::float).to_vec();
    assert_matches_iceberg(
        PrimitiveType::Float,
        Transform::Identity,
        floats.clone(),
        floats,
    );
    let doubles = [-1.0, -0.0, 0.0, 1.0].map(Datum::double).to_vec();
    assert_matches_iceberg(
        PrimitiveType::Double,
        Transform::Identity,
        doubles.clone(),
        doubles,
    );
}

fn timestamp_predicate(compare_type: Comparison, micros: i64) -> Vec<ColumnPredicate> {
    vec![ColumnPredicate {
        column_idx: 0,
        path: vec![],
        as_type: None,
        compare_type,
        value: Scalar::new(Arc::new(TimestampMicrosecondArray::from(vec![micros])) as ArrayRef),
    }]
}

#[test]
fn reused_day_bounds_keep_interior_and_historical_boundaries() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                1,
                "at",
                Type::Primitive(PrimitiveType::Timestamp),
            ))])
            .build()
            .unwrap(),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("at", "at_day", Transform::Day)
        .unwrap()
        .build()
        .unwrap();
    let column = columns(&schema, &schema, &spec).unwrap().pop().unwrap();
    // Partitions immediately before and after the epoch, plus a later day.
    let values = Arc::new(Date32Array::from(vec![-1, 0, 1])) as ArrayRef;
    let statistics = StatisticsBatch::new(
        3,
        BTreeMap::new(),
        vec![PartitionStatistics {
            expression: column.expression,
            bounds: ColumnBounds {
                lower: Some(values.clone()),
                upper: Some(values),
                ..Default::default()
            },
        }],
    )
    .unwrap();
    let half_day = 43_200_000_000;
    // Iceberg deliberately widens pre-epoch projections for compatibility
    // with older writers. The positive midnight boundary is exact.
    for (compare, constant, expected) in [
        (Comparison::Less, 0, vec![true, true, false]),
        (Comparison::LessEqual, 0, vec![true, true, false]),
        (Comparison::Less, half_day * 2, vec![true, true, false]),
        (Comparison::LessEqual, half_day * 2, vec![true, true, true]),
        (Comparison::Less, half_day, vec![true, true, false]),
        (Comparison::Greater, -half_day, vec![true, true, true]),
        (Comparison::GreaterEqual, 0, vec![false, true, true]),
        (Comparison::Less, 0, vec![true, true, false]),
    ] {
        assert_eq!(
            statistics
                .prune(&timestamp_predicate(compare, constant))
                .unwrap(),
            arrow_array::BooleanArray::from(expected)
        );
    }
}
