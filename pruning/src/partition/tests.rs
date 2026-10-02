use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;
use crate::{
    ColumnBounds, ColumnPredicate, PartitionExpression, PartitionStatistics, StatisticsBatch,
};
use arrow_array::{BooleanArray, Int64Array, StringArray};
use arrow_ord::cmp;

const COMPARISONS: [Comparison; 6] = [
    Comparison::Equal,
    Comparison::NotEqual,
    Comparison::Less,
    Comparison::LessEqual,
    Comparison::Greater,
    Comparison::GreaterEqual,
];

fn predicate(compare: Comparison, value: Scalar<ArrayRef>) -> ColumnPredicate {
    ColumnPredicate {
        column_idx: 4,
        path: vec![],
        as_type: None,
        compare_type: compare,
        value,
    }
}

fn statistics(
    source_type: &DataType,
    transform: PartitionTransform,
    values: &[Scalar<ArrayRef>],
) -> StatisticsBatch {
    let values: Vec<_> = values
        .iter()
        .map(|value| transform.apply(value).unwrap())
        .collect();
    let refs: Vec<_> = values.iter().map(|value| value.get().0).collect();
    let array = arrow_select::concat::concat(&refs).unwrap();
    StatisticsBatch::new(
        values.len(),
        BTreeMap::new(),
        vec![PartitionStatistics {
            expression: PartitionExpression {
                source: 4.into(),
                source_type: source_type.clone(),
                transform,
            },
            bounds: ColumnBounds {
                lower: Some(array.clone()),
                upper: Some(array),
                ..Default::default()
            },
        }],
    )
    .unwrap()
}

/// Check pruning against actual source-row comparisons, independently of the
/// projected comparison. A coarse partition may retain extras, never a miss.
fn assert_inclusive(
    data_type: DataType,
    transform: PartitionTransform,
    values: Vec<Scalar<ArrayRef>>,
    constants: Vec<Scalar<ArrayRef>>,
) {
    let bounds = statistics(&data_type, transform, &values);
    for compare in COMPARISONS {
        let kernel = match compare {
            Comparison::Equal => cmp::eq,
            Comparison::NotEqual => cmp::neq,
            Comparison::Less => cmp::lt,
            Comparison::LessEqual => cmp::lt_eq,
            Comparison::Greater => cmp::gt,
            Comparison::GreaterEqual => cmp::gt_eq,
        };
        for constant in &constants {
            let predicate = predicate(compare, constant.clone());
            let keep = bounds.prune(&[predicate]).unwrap();
            for (row, value) in values.iter().enumerate() {
                let actual = kernel(value, constant).unwrap();
                assert!(
                    !actual.value(0) || keep.value(row),
                    "{data_type:?} {transform:?}: {value:?} {compare:?} {constant:?}"
                );
            }
        }
    }
}

#[test]
fn numeric_truncation_is_inclusive_across_negative_values_and_boundaries() {
    let domain = [
        -101, -100, -99, -11, -10, -9, -1, 0, 1, 9, 10, 11, 99, 100, 101,
    ];
    for data_type in [
        DataType::Int32,
        DataType::Int64,
        DataType::Decimal64(9, 2),
        DataType::Decimal128(24, 3),
    ] {
        let values: Vec<_> = domain
            .iter()
            .map(|value| scalar::from_integer(*value, &data_type).unwrap())
            .collect();
        for transform in [
            PartitionTransform::Identity,
            PartitionTransform::Truncate(10),
            PartitionTransform::Bucket(16),
        ] {
            assert_inclusive(data_type.clone(), transform, values.clone(), values.clone());
        }
    }
}

#[test]
fn calendar_projection_is_inclusive_at_epoch_leap_days_and_period_boundaries() {
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
        let values: Vec<_> = domain
            .into_iter()
            .map(|value| scalar::from_integer(value, &data_type).unwrap())
            .collect();
        for transform in [
            PartitionTransform::Year,
            PartitionTransform::Month,
            PartitionTransform::Day,
            PartitionTransform::Hour,
        ] {
            if data_type == DataType::Date32 && transform == PartitionTransform::Hour {
                continue;
            }
            assert_inclusive(data_type.clone(), transform, values.clone(), values.clone());
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
    let values = [midnight - 1, midnight, midnight + 86_400_000_000]
        .map(|v| scalar::from_integer(v.into(), &data_type).unwrap());
    let bounds = statistics(&data_type, PartitionTransform::Day, &values);
    for (compare, constant, expected) in [
        (Comparison::Less, midnight, vec![true, false, false]),
        (Comparison::Less, midnight + 1, vec![true, true, false]),
        (Comparison::LessEqual, midnight, vec![true, true, false]),
        (Comparison::Greater, midnight - 1, vec![false, true, true]),
    ] {
        let filter = predicate(
            compare,
            scalar::from_integer(constant.into(), &data_type).unwrap(),
        );
        assert_eq!(
            bounds.prune(&[filter]).unwrap(),
            BooleanArray::from(expected)
        );
    }
}

#[test]
fn pre_epoch_timestamp_days_keep_both_floor_and_legacy_partition_values() {
    for timezone in [None, Some("UTC".into())] {
        let source_type = DataType::Timestamp(TimeUnit::Microsecond, timezone);
        // 1969-12-30 23:59:59.999998 is day -2. Some writers stored day -1.
        let array = Arc::new(arrow_array::Date32Array::from(vec![-2, -1, 0])) as ArrayRef;
        let bounds = StatisticsBatch::new(
            3,
            BTreeMap::new(),
            vec![PartitionStatistics {
                expression: PartitionExpression {
                    source: 4.into(),
                    source_type: source_type.clone(),
                    transform: PartitionTransform::Day,
                },
                bounds: ColumnBounds {
                    lower: Some(array.clone()),
                    upper: Some(array),
                    ..Default::default()
                },
            }],
        )
        .unwrap();
        let filter = predicate(
            Comparison::Equal,
            scalar::from_integer(-86_400_000_002, &source_type).unwrap(),
        );
        assert_eq!(
            bounds.prune(&[filter]).unwrap(),
            BooleanArray::from(vec![true, true, false])
        );
    }
}

#[test]
fn integer_domain_edges_do_not_overflow_or_change_the_query_constant() {
    let values = [i32::MIN + 10, 0, i32::MAX - 10]
        .map(|value| scalar::from_integer(value.into(), &DataType::Int32).unwrap());
    for transform in [
        PartitionTransform::Identity,
        PartitionTransform::Truncate(10),
        PartitionTransform::Bucket(16),
    ] {
        let bounds = statistics(&DataType::Int32, transform, &values);
        for (compare, value) in [
            (Comparison::Equal, i64::from(i32::MAX) + 1),
            (Comparison::Greater, i64::from(i32::MAX)),
            (Comparison::Less, i64::from(i32::MIN)),
        ] {
            // Hashes cannot rule out range predicates within the source type.
            if matches!(transform, PartitionTransform::Bucket(_)) && compare != Comparison::Equal {
                continue;
            }
            let filter = predicate(
                compare,
                Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
            );
            assert_eq!(
                bounds.prune(&[filter]).unwrap(),
                BooleanArray::from(vec![false; 3])
            );
        }
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
        assert_inclusive(
            data_type,
            PartitionTransform::Truncate(2),
            values.clone(),
            values,
        );
    }
    let value = Scalar::new(Arc::new(StringViewArray::from(vec!["é雪a"])) as ArrayRef);
    let truncated = PartitionTransform::Truncate(2).apply(&value).unwrap();
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
        let output = PartitionTransform::Bucket(100).apply(&input).unwrap();
        assert_eq!(
            scalar::integer(output.get().0),
            Some(i128::from((hash & i32::MAX) % 100)),
            "{data_type:?}"
        );
    }
    let value = Scalar::new(Arc::new(StringViewArray::from(vec!["iceberg"])) as ArrayRef);
    let output = PartitionTransform::Bucket(100).apply(&value).unwrap();
    assert_eq!(scalar::integer(output.get().0), Some(89));
}

#[test]
fn unsupported_transforms_and_lossy_float_conversions_supply_no_proof() {
    let value =
        Scalar::new(Arc::new(arrow_array::Float64Array::from(vec![1.00000005])) as ArrayRef);
    for transform in [
        PartitionTransform::Identity,
        PartitionTransform::Bucket(16),
        PartitionTransform::Truncate(10),
    ] {
        assert!(matches!(
            transform.project(&DataType::Float32, Comparison::NotEqual, &value),
            PruningPredicate::Always(true)
        ));
    }
    let value = scalar::from_integer(1, &DataType::Int64).unwrap();
    for transform in [
        PartitionTransform::Bucket(0),
        PartitionTransform::Truncate(0),
        PartitionTransform::Day,
    ] {
        assert!(matches!(
            transform.project(&DataType::Int64, Comparison::Equal, &value),
            PruningPredicate::Always(true)
        ));
    }
}
