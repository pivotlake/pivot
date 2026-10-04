use std::sync::Arc;

use super::*;
use crate::pruning::{Bounds, Statistic, Statistics};
use arrow_array::{BooleanArray, Date32Array, StringArray};
use arrow_ord::cmp;
use planner::expression::{Compare, Expression, Ref};
use planner::types::Type;

const COMPARISONS: [CompareType; 6] = [
    CompareType::Equal,
    CompareType::NotEqual,
    CompareType::Less,
    CompareType::LessEqual,
    CompareType::Greater,
    CompareType::GreaterEqual,
];

/// `#4 <compare_type> constant`. A transform reads only the constant, so the
/// column's declared type is a stand-in.
fn predicate(compare_type: CompareType, constant: Scalar<ArrayRef>) -> Expression {
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 4,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: Type::Boolean,
    })
}

/// One object per partition value, each bounded by exactly that value.
fn partitions(transform: Transform, partition_values: ArrayRef) -> Statistics {
    Statistics::new(
        partition_values.len(),
        vec![Statistic {
            column: 4.into(),
            transform,
            bounds: Bounds {
                all_null: BooleanArray::from(vec![false; partition_values.len()]),
                lower: partition_values.clone(),
                upper: partition_values,
            },
        }],
    )
}

/// One object per column value, partitioned the way a writer would.
fn partitions_of(transform: Transform, values: &[Scalar<ArrayRef>]) -> Statistics {
    let values: Vec<_> = values
        .iter()
        .map(|value| transform.apply(value).unwrap())
        .collect();
    let arrays: Vec<_> = values.iter().map(|value| value.get().0).collect();
    partitions(transform, arrow_select::concat::concat(&arrays).unwrap())
}

/// Check pruning against the comparison of the column values themselves. A
/// coarse partition may keep an object without a match, never drop one with.
fn assert_inclusive(
    transform: Transform,
    values: &[Scalar<ArrayRef>],
    constants: &[Scalar<ArrayRef>],
) {
    let statistics = partitions_of(transform, values);
    for comparison in COMPARISONS {
        let kernel = match comparison {
            CompareType::Equal => cmp::eq,
            CompareType::NotEqual => cmp::neq,
            CompareType::Less => cmp::lt,
            CompareType::LessEqual => cmp::lt_eq,
            CompareType::Greater => cmp::gt,
            CompareType::GreaterEqual => cmp::gt_eq,
        };
        for constant in constants {
            let keep = statistics.prune(&[predicate(comparison, constant.clone())]);
            for (row, value) in values.iter().enumerate() {
                let matches = kernel(value, constant).unwrap().value(0);
                assert!(
                    !matches || keep.value(row),
                    "{transform:?}: {value:?} {comparison:?} {constant:?}"
                );
            }
        }
    }
}

fn integers(values: &[i128], data_type: &DataType) -> Vec<Scalar<ArrayRef>> {
    values
        .iter()
        .map(|value| scalar::from_integer(*value, data_type).unwrap())
        .collect()
}

#[test]
fn numeric_transforms_are_inclusive_across_negative_values_and_boundaries() {
    let domain = [
        -101, -100, -99, -11, -10, -9, -1, 0, 1, 9, 10, 11, 99, 100, 101,
    ];

    for data_type in [
        DataType::Int32,
        DataType::Int64,
        DataType::Decimal64(9, 2),
        DataType::Decimal128(24, 3),
    ] {
        let values = integers(&domain, &data_type);
        for transform in [
            Transform::Identity,
            Transform::Truncate(10),
            Transform::Bucket(16),
        ] {
            assert_inclusive(transform, &values, &values);
        }
    }
}

#[test]
fn constants_at_the_edges_of_their_type_do_not_overflow() {
    let rows = integers(&[-5, 0, 5], &DataType::Int64);
    let edges = integers(&[i64::MIN.into(), i64::MAX.into()], &DataType::Int64);

    for transform in [Transform::Truncate(10), Transform::Bucket(16)] {
        assert_inclusive(transform, &rows, &edges);
    }
}

#[test]
fn calendar_transforms_are_inclusive_at_epoch_leap_days_and_period_boundaries() {
    let dates = [
        "1968-02-29",
        "1969-12-31",
        "1970-01-01",
        "1970-01-02",
        "2000-02-29",
        "2020-01-01",
        "2020-01-02",
        "2020-02-29",
        "2020-03-01",
        "2021-01-01",
    ];

    for data_type in [
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
    ] {
        let mut domain = Vec::new();
        for date in dates {
            let midnight = NaiveDate::parse_from_str(date, "%Y-%m-%d")
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            let midnight = match &data_type {
                DataType::Date32 => i128::from(midnight / 86_400),
                DataType::Timestamp(unit, _) => i128::from(midnight) * ticks_per_second(unit),
                _ => unreachable!(),
            };
            domain.extend([midnight - 2, midnight - 1, midnight, midnight + 1]);
        }
        let values = integers(&domain, &data_type);
        for transform in [
            Transform::Year,
            Transform::Month,
            Transform::Day,
            Transform::Hour,
        ] {
            if data_type == DataType::Date32 && transform == Transform::Hour {
                continue;
            }
            assert_inclusive(transform, &values, &values);
        }
    }
}

#[test]
fn strict_midnight_uses_the_previous_day_and_an_interior_time_keeps_its_day() {
    let data_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let midnight = NaiveDate::from_ymd_opt(2020, 1, 2)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_micros();
    let rows = [midnight - 1, midnight, midnight + 86_400_000_000].map(i128::from);
    let statistics = partitions_of(Transform::Day, &integers(&rows, &data_type));

    for (comparison, constant, expected) in [
        (CompareType::Less, midnight, vec![true, false, false]),
        (CompareType::Less, midnight + 1, vec![true, true, false]),
        (CompareType::LessEqual, midnight, vec![true, true, false]),
        (CompareType::Greater, midnight - 1, vec![false, true, true]),
    ] {
        let constant = scalar::from_integer(constant.into(), &data_type).unwrap();
        let keep = statistics.prune(&[predicate(comparison, constant)]);

        assert_eq!(keep, BooleanArray::from(expected));
    }
}

#[test]
fn pre_epoch_timestamp_days_keep_both_floor_and_legacy_partition_values() {
    for timezone in [None, Some("UTC".into())] {
        let data_type = DataType::Timestamp(TimeUnit::Microsecond, timezone);
        let statistics = partitions(Transform::Day, Arc::new(Date32Array::from(vec![-2, -1, 0])));
        // 1969-12-30 23:59:59.999998 is day -2. Some writers stored day -1.
        let instant = scalar::from_integer(-86_400_000_002, &data_type).unwrap();

        let keep = statistics.prune(&[predicate(CompareType::Equal, instant)]);

        assert_eq!(keep, BooleanArray::from(vec![true, true, false]));
    }
}

#[test]
fn string_truncation_uses_unicode_codepoints_and_keeps_prefix_boundaries() {
    let strings = [
        "", "a", "ab", "abc", "abcd", "abé", "é", "é雪", "é雪a", "雪",
    ];

    for data_type in [DataType::Utf8, DataType::Utf8View] {
        let values: Vec<_> = strings
            .iter()
            .map(|s| {
                Scalar::new(arrow_cast::cast(&StringArray::from(vec![*s]), &data_type).unwrap())
            })
            .collect();
        assert_inclusive(Transform::Truncate(2), &values, &values);
    }
    let value = Scalar::new(Arc::new(StringViewArray::from(vec!["é雪a"])) as ArrayRef);
    let truncated = Transform::Truncate(2).apply(&value).unwrap();
    assert_eq!(string(truncated.get().0), Some("é雪"));
}

#[test]
fn bucket_hashes_match_spec_vectors_and_integer_promotion() {
    // Published hash vectors, before masking and taking the bucket count.
    for (data_type, value, hash) in [
        (DataType::Int32, 34, 2_017_239_379_i32),
        (DataType::Int64, 34, 2_017_239_379),
        (DataType::Decimal64(9, 2), 1420, -500_754_589),
        (DataType::Decimal128(24, 2), 1420, -500_754_589),
        (DataType::Date32, 17486, -653_330_422),
        (
            DataType::Timestamp(TimeUnit::Microsecond, None),
            1_510_871_468_000_001,
            -1_207_196_810,
        ),
    ] {
        let input = scalar::from_integer(value, &data_type).unwrap();

        let output = Transform::Bucket(100).apply(&input).unwrap();

        assert_eq!(
            scalar::integer(output.get().0),
            Some(i128::from((hash & i32::MAX) % 100)),
            "{data_type:?}"
        );
    }
    let value = Scalar::new(Arc::new(StringViewArray::from(vec!["iceberg"])) as ArrayRef);
    let output = Transform::Bucket(100).apply(&value).unwrap();
    assert_eq!(scalar::integer(output.get().0), Some(89));
}

#[test]
fn comparisons_a_transform_cannot_keep_project_to_nothing() {
    let one = scalar::from_integer(1, &DataType::Int64).unwrap();

    let projections = [
        Transform::Truncate(10).project(CompareType::NotEqual, &one),
        Transform::Bucket(16).project(CompareType::Less, &one),
        Transform::Bucket(0).project(CompareType::Equal, &one),
        Transform::Truncate(0).project(CompareType::Equal, &one),
        Transform::Day.project(CompareType::Equal, &one),
    ];

    assert!(projections.iter().all(Vec::is_empty));
}
