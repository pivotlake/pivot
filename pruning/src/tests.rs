use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    ColumnStatistics, Comparison, PartitionExpression, PartitionStatistics, PartitionTransform,
    VariantPathStatistics,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, Scalar,
};

use super::{ColumnBounds, ColumnPredicate, ColumnReference, PruningPredicate, StatisticsBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

const COMPARISONS: [Comparison; 6] = [
    Comparison::Equal,
    Comparison::NotEqual,
    Comparison::Less,
    Comparison::LessEqual,
    Comparison::Greater,
    Comparison::GreaterEqual,
];

fn constant(value: i64) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
}

fn compare_values(left: i64, compare: Comparison, right: i64) -> bool {
    match compare {
        Comparison::Equal => left == right,
        Comparison::NotEqual => left != right,
        Comparison::Less => left < right,
        Comparison::LessEqual => left <= right,
        Comparison::Greater => left > right,
        Comparison::GreaterEqual => left >= right,
    }
}

#[test]
fn bounds_never_exclude_an_object_with_a_matching_row() {
    let domain = [None, Some(-2), Some(0), Some(2)];
    let objects: Vec<_> = domain
        .iter()
        .flat_map(|a| domain.iter().map(move |b| [*a, *b]))
        .collect();

    for loosen in [0, 1] {
        for available in 0..4 {
            let statistics = ColumnBounds {
                validity: None,
                lower: (available & 1 != 0).then(|| {
                    Arc::new(Int64Array::from_iter(objects.iter().map(|rows| {
                        rows.iter().flatten().min().map(|value| value - loosen)
                    }))) as ArrayRef
                }),
                upper: (available & 2 != 0).then(|| {
                    Arc::new(Int64Array::from_iter(objects.iter().map(|rows| {
                        rows.iter().flatten().max().map(|value| value + loosen)
                    }))) as ArrayRef
                }),
                all_null: Some(
                    objects
                        .iter()
                        .map(|rows| rows.iter().all(Option::is_none))
                        .collect(),
                ),
                nan_free: None,
            };
            for compare in COMPARISONS {
                for value in -3..=3 {
                    let keep = statistics
                        .may_match(objects.len(), compare, &constant(value))
                        .unwrap();

                    assert_eq!(keep.null_count(), 0);
                    for (index, rows) in objects.iter().enumerate() {
                        let matches = rows
                            .iter()
                            .flatten()
                            .any(|row| compare_values(*row, compare, value));
                        assert!(
                            !matches || keep.value(index),
                            "{rows:?} {compare:?} {value}"
                        );
                        if available == 3 {
                            let possible = rows
                                .iter()
                                .flatten()
                                .min()
                                .zip(rows.iter().flatten().max())
                                .is_some_and(|(lower, upper)| {
                                    ((*lower - loosen)..=(*upper + loosen))
                                        .any(|row| compare_values(row, compare, value))
                                });
                            assert_eq!(keep.value(index), possible, "{rows:?} {compare:?} {value}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn one_sided_bounds_and_unknown_statistics_are_distinct_from_null_values() {
    let bounds = ColumnBounds {
        validity: None,
        lower: Some(Arc::new(Int64Array::from(vec![Some(10), None, None, None]))),
        upper: Some(Arc::new(Int64Array::from(vec![None, Some(0), None, None]))),
        all_null: Some(BooleanArray::from(vec![
            Some(false),
            Some(false),
            None,
            Some(true),
        ])),
        nan_free: None,
    };

    let keep = bounds
        .may_match(4, Comparison::Equal, &constant(5))
        .unwrap();

    assert_eq!(keep, BooleanArray::from(vec![false, false, true, false]));
}

#[test]
fn equality_keeps_values_inside_a_range_even_when_neither_endpoint_matches() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Int64Array::from(vec![0]))),
        upper: Some(Arc::new(Int64Array::from(vec![10]))),
        ..Default::default()
    };

    let keep = bounds
        .may_match(1, Comparison::Equal, &constant(5))
        .unwrap();

    assert!(keep.value(0));
}

#[test]
fn boolean_combinations_preserve_unknown_alternatives() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Int64Array::from(vec![Some(1), None]))),
        upper: Some(Arc::new(Int64Array::from(vec![Some(1), None]))),
        ..Default::default()
    };
    let equal = || PruningPredicate::Compare {
        compare: Comparison::Equal,
        value: constant(2),
    };
    let disjunction = PruningPredicate::Or(vec![equal(), PruningPredicate::Always(false)])
        .may_match(&bounds, 2)
        .unwrap();
    let conjunction = PruningPredicate::And(vec![equal(), PruningPredicate::Always(false)])
        .may_match(&bounds, 2)
        .unwrap();

    assert_eq!(disjunction, BooleanArray::from(vec![false, true]));
    assert_eq!(conjunction, BooleanArray::from(vec![false, false]));
}

#[test]
fn floating_bounds_require_proof_that_nan_is_absent() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Float64Array::from(vec![1.0; 3]))),
        upper: Some(Arc::new(Float64Array::from(vec![1.0; 3]))),
        nan_free: Some(BooleanArray::from(vec![Some(false), Some(true), None])),
        ..Default::default()
    };
    let two = Scalar::new(Arc::new(Float64Array::from(vec![2.0])) as ArrayRef);

    let keep = bounds.may_match(3, Comparison::Greater, &two).unwrap();

    assert_eq!(keep, BooleanArray::from(vec![true, false, true]));
}

#[test]
fn floating_equality_prunes_without_a_nan_count() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Float64Array::from(vec![1.0]))),
        upper: Some(Arc::new(Float64Array::from(vec![1.0]))),
        ..Default::default()
    };
    let two = Scalar::new(Arc::new(Float64Array::from(vec![2.0])) as ArrayRef);

    let keep = bounds.may_match(1, Comparison::Equal, &two).unwrap();

    assert!(!keep.value(0));
}

#[test]
fn floating_bounds_preserve_matches_for_both_nan_signs_and_signed_zeros() {
    let domain = [
        None,
        Some(-f64::NAN),
        Some(f64::NEG_INFINITY),
        Some(-1.0),
        Some(-0.0),
        Some(0.0),
        Some(1.0),
        Some(f64::INFINITY),
        Some(f64::NAN),
    ];
    let objects: Vec<_> = domain
        .iter()
        .flat_map(|a| domain.iter().map(move |b| [*a, *b]))
        .collect();
    for float32 in [false, true] {
        let array = |values: Vec<Option<f64>>| -> ArrayRef {
            if float32 {
                Arc::new(Float32Array::from_iter(
                    values
                        .into_iter()
                        .map(|value| value.map(|value| value as f32)),
                ))
            } else {
                Arc::new(Float64Array::from(values))
            }
        };
        let actual_rows = array(objects.iter().flatten().copied().collect());
        for counts_known in [false, true] {
            for available in 0..4 {
                let bounds = ColumnBounds {
                    validity: None,
                    lower: (available & 1 != 0).then(|| {
                        array(
                            objects
                                .iter()
                                .map(|rows| {
                                    rows.iter()
                                        .flatten()
                                        .copied()
                                        .filter(|v| !v.is_nan())
                                        .min_by(f64::total_cmp)
                                })
                                .collect(),
                        )
                    }),
                    upper: (available & 2 != 0).then(|| {
                        array(
                            objects
                                .iter()
                                .map(|rows| {
                                    rows.iter()
                                        .flatten()
                                        .copied()
                                        .filter(|v| !v.is_nan())
                                        .max_by(f64::total_cmp)
                                })
                                .collect(),
                        )
                    }),
                    all_null: Some(
                        objects
                            .iter()
                            .map(|rows| rows.iter().all(Option::is_none))
                            .collect(),
                    ),
                    nan_free: counts_known.then(|| {
                        objects
                            .iter()
                            .map(|rows| rows.iter().flatten().all(|v| !v.is_nan()))
                            .collect()
                    }),
                };
                for compare in COMPARISONS {
                    let kernel = match compare {
                        Comparison::Equal => arrow_ord::cmp::eq,
                        Comparison::NotEqual => arrow_ord::cmp::neq,
                        Comparison::Less => arrow_ord::cmp::lt,
                        Comparison::LessEqual => arrow_ord::cmp::lt_eq,
                        Comparison::Greater => arrow_ord::cmp::gt,
                        Comparison::GreaterEqual => arrow_ord::cmp::gt_eq,
                    };
                    for value in domain {
                        let constant = Scalar::new(array(vec![value]));
                        let actual = kernel(&actual_rows, &constant).unwrap();
                        let keep = bounds.may_match(objects.len(), compare, &constant).unwrap();

                        for (index, rows) in objects.iter().enumerate() {
                            let matches = (index * 2..index * 2 + 2)
                                .any(|row| actual.is_valid(row) && actual.value(row));
                            assert!(
                                !matches || keep.value(index),
                                "{rows:?} {compare:?} {value:?}, counts={counts_known}, bounds={available}"
                            );
                            if available == 3
                                && counts_known
                                && rows.iter().flatten().all(|v| !v.is_nan())
                            {
                                // For two values, every comparison except equality has an
                                // exact answer from the two endpoints. Equality can also
                                // match values between them that this pair does not contain.
                                if !matches!(compare, Comparison::Equal) {
                                    assert_eq!(
                                        keep.value(index),
                                        matches,
                                        "{rows:?} {compare:?} {value:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nan_bounds_supply_no_ordering_information() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Float64Array::from(vec![f64::NAN]))),
        upper: Some(Arc::new(Float64Array::from(vec![1.0]))),
        nan_free: Some(BooleanArray::from(vec![true])),
        ..Default::default()
    };
    let zero = Scalar::new(Arc::new(Float64Array::from(vec![0.0])) as ArrayRef);

    let keep = bounds.may_match(1, Comparison::Less, &zero).unwrap();

    assert!(keep.value(0));
}

#[test]
fn incompatible_types_prove_nothing_and_null_constants_match_nothing() {
    let bounds = ColumnBounds {
        lower: Some(Arc::new(Int32Array::from(vec![10]))),
        upper: Some(Arc::new(Int32Array::from(vec![10]))),
        ..Default::default()
    };
    let null = Scalar::new(Arc::new(Int64Array::from(vec![None])) as ArrayRef);

    let mismatched = bounds
        .may_match(1, Comparison::Equal, &constant(5))
        .unwrap();
    let null_comparison = bounds.may_match(1, Comparison::NotEqual, &null).unwrap();

    assert!(mismatched.value(0));
    assert!(!null_comparison.value(0));
}

#[test]
fn empty_and_unknown_batches_keep_object_positions() {
    let unknown = StatisticsBatch::new(3, BTreeMap::new(), Vec::new()).unwrap();
    let empty = StatisticsBatch::new(0, BTreeMap::new(), Vec::new()).unwrap();
    assert_eq!(
        unknown.prune(&[]).unwrap(),
        BooleanArray::from(vec![true; 3])
    );
    assert!(empty.prune(&[]).unwrap().is_empty());
    assert_pruning(&unknown, &[equal(2.into(), 1)], vec![true; 3]);
}

#[test]
fn a_statistics_batch_rejects_misaligned_columns() {
    let result = primitive_statistics(
        2,
        vec![(
            0,
            ColumnBounds {
                lower: Some(Arc::new(Int64Array::from(vec![1]))),
                upper: Some(Arc::new(Int64Array::from(vec![2, 3]))),
                ..Default::default()
            },
        )],
    );
    assert!(result.is_err());
}

fn primitive_statistics(
    len: usize,
    columns: Vec<(usize, ColumnBounds)>,
) -> Result<StatisticsBatch, arrow_schema::ArrowError> {
    StatisticsBatch::new(
        len,
        columns
            .into_iter()
            .map(|(column, bounds)| (column, ColumnStatistics::from(bounds)))
            .collect(),
        Vec::new(),
    )
}

fn exact_bounds(values: Vec<i64>) -> ColumnBounds {
    let values = Arc::new(Int64Array::from(values)) as ArrayRef;
    ColumnBounds {
        lower: Some(values.clone()),
        upper: Some(values),
        ..Default::default()
    }
}

fn equal(reference: ColumnReference, value: i64) -> ColumnPredicate {
    ColumnPredicate {
        column_idx: reference.column_idx,
        path: reference.path,
        as_type: reference.as_type,
        compare_type: Comparison::Equal,
        value: constant(value),
    }
}

fn assert_pruning(
    statistics: &StatisticsBatch,
    predicates: &[ColumnPredicate],
    expected: Vec<bool>,
) {
    assert_eq!(
        statistics.prune(predicates).unwrap(),
        BooleanArray::from(expected.clone())
    );
    for (row, expected) in expected.into_iter().enumerate() {
        assert_eq!(statistics.prune_row(row, predicates).unwrap(), expected);
    }
}

#[test]
fn matching_uses_table_indexes_independently_of_statistics_positions() {
    let statistics = primitive_statistics(
        2,
        vec![
            (9, exact_bounds(vec![10, 20])),
            (2, exact_bounds(vec![100, 200])),
        ],
    )
    .unwrap();

    assert_pruning(&statistics, &[equal(9.into(), 20)], vec![false, true]);
    assert_pruning(&statistics, &[equal(2.into(), 100)], vec![true, false]);
    // Missing table columns supply no proof, even when other columns have bounds.
    assert_pruning(&statistics, &[equal(1.into(), 100)], vec![true, true]);
    assert_pruning(&statistics, &[equal(0.into(), 100)], vec![true, true]);
    assert_pruning(
        &statistics,
        &[equal(9.into(), 20), equal(2.into(), 100)],
        vec![false, false],
    );
    assert!(statistics.prune_row(2, &[]).is_err());
}

#[test]
fn variant_matching_requires_the_path_and_exact_cast_type() {
    let reference = |path: &[&str], as_type| ColumnReference {
        column_idx: 7,
        path: path.iter().map(|part| (*part).into()).collect(),
        as_type,
    };
    let price = reference(&["item", "price"], Some(DataType::Int64));
    let discount = reference(&["discount"], Some(DataType::Int64));
    let root = reference(&[], Some(DataType::Int64));
    let paths = [
        (price.path.clone(), exact_bounds(vec![10, 20])),
        (discount.path.clone(), exact_bounds(vec![20, 10])),
        (root.path.clone(), exact_bounds(vec![30, 40])),
    ]
    .into_iter()
    .map(|(path, bounds)| {
        (
            path,
            VariantPathStatistics {
                data_type: DataType::Int64,
                bounds,
            },
        )
    })
    .collect();
    let statistics = StatisticsBatch::new(
        2,
        BTreeMap::from([
            (7, ColumnStatistics::Variant(paths)),
            (
                9,
                ColumnStatistics::Variant(BTreeMap::from([(
                    price.path.clone(),
                    VariantPathStatistics {
                        data_type: DataType::Int64,
                        bounds: exact_bounds(vec![20, 10]),
                    },
                )])),
            ),
        ]),
        Vec::new(),
    )
    .unwrap();

    assert_pruning(
        &statistics,
        &[equal(
            ColumnReference {
                column_idx: 9,
                ..price.clone()
            },
            20,
        )],
        vec![true, false],
    );
    assert_pruning(&statistics, &[equal(price, 20)], vec![false, true]);
    assert_pruning(&statistics, &[equal(discount, 20)], vec![true, false]);
    assert_pruning(&statistics, &[equal(root, 40)], vec![false, true]);
    for unknown in [
        reference(&["item", "price"], Some(DataType::Float64)),
        reference(&["item", "missing"], Some(DataType::Int64)),
        reference(&[], None),
    ] {
        assert_pruning(&statistics, &[equal(unknown, 0)], vec![true, true]);
    }
}

#[test]
fn all_statistics_for_a_logical_source_can_prove_exclusions() {
    // Different metadata expressions may describe the same logical source,
    // such as column bounds alongside identity-partition bounds.
    let statistics = StatisticsBatch::new(
        2,
        BTreeMap::from([(
            5,
            ColumnStatistics::from(ColumnBounds {
                lower: Some(Arc::new(Int64Array::from(vec![0, 0]))),
                ..Default::default()
            }),
        )]),
        vec![PartitionStatistics {
            expression: PartitionExpression {
                source: 5.into(),
                source_type: DataType::Int64,
                transform: PartitionTransform::Identity,
            },
            bounds: ColumnBounds {
                upper: Some(Arc::new(Int64Array::from(vec![10, 30]))),
                ..Default::default()
            },
        }],
    )
    .unwrap();

    assert_pruning(&statistics, &[equal(5.into(), 20)], vec![false, true]);
    assert_pruning(&statistics, &[equal(5.into(), -1)], vec![false, false]);
}

#[test]
fn invalid_logical_bounds_preserve_raw_statistics_without_proving_exclusions() {
    let raw = Arc::new(Int64Array::from(vec![Some(10), None, Some(10), None])) as ArrayRef;
    let statistics = primitive_statistics(
        4,
        vec![(
            3,
            ColumnBounds {
                lower: Some(raw.clone()),
                upper: Some(raw.clone()),
                all_null: Some(BooleanArray::from(vec![false, true, false, true])),
                validity: Some(NullBuffer::from(vec![false, false, true, true])),
                ..Default::default()
            },
        )],
    )
    .unwrap();

    assert_pruning(
        &statistics,
        &[equal(3.into(), 20)],
        vec![true, true, false, false],
    );
    let bounds = statistics.column_stats()[&3].primitive().unwrap();
    assert!(Arc::ptr_eq(bounds.lower.as_ref().unwrap(), &raw));
    assert!(Arc::ptr_eq(bounds.upper.as_ref().unwrap(), &raw));
}

#[test]
fn selected_and_sliced_statistics_preserve_expression_matching() {
    let statistics = primitive_statistics(3, vec![(4, exact_bounds(vec![10, 20, 30]))]).unwrap();
    assert!(!statistics.column_stats().contains_key(&1));
    assert!(statistics.filter(&BooleanArray::from(vec![true])).is_err());
    let selected = statistics
        .filter(&BooleanArray::from(vec![false, true, true]))
        .unwrap();
    assert_pruning(&selected, &[equal(4.into(), 20)], vec![true, false]);
    let sliced = selected.slice(1, 1);
    assert_pruning(&sliced, &[equal(4.into(), 30)], vec![true]);
    let empty = statistics
        .filter(&BooleanArray::from(vec![false; 3]))
        .unwrap();
    assert!(empty.is_empty());
}

#[test]
fn impossible_partition_projections_apply_only_to_objects_with_that_expression() {
    let statistics = StatisticsBatch::new(
        3,
        BTreeMap::new(),
        vec![PartitionStatistics {
            expression: PartitionExpression {
                source: 4.into(),
                source_type: DataType::Int32,
                transform: PartitionTransform::Identity,
            },
            bounds: ColumnBounds {
                validity: Some(NullBuffer::from(vec![true, false, true])),
                ..Default::default()
            },
        }],
    )
    .unwrap();
    // Only the active Int32 expression proves this impossible. The middle
    // object could use another spec with an Int64 source and contain the value.
    let predicate = equal(4.into(), i64::from(i32::MAX) + 1);
    assert_pruning(
        &statistics,
        std::slice::from_ref(&predicate),
        vec![false, true, false],
    );
    assert_pruning(&statistics.slice(1, 2), &[predicate], vec![true, false]);
}

#[test]
fn filtering_grouped_statistics_preserves_per_path_and_partition_coverage() {
    let price = ColumnReference {
        column_idx: 2,
        path: vec!["price".into()],
        as_type: Some(DataType::Int64),
    };
    let other = ColumnReference {
        path: vec!["other".into()],
        ..price.clone()
    };
    let mut price_bounds = exact_bounds(vec![10, 20, 30, 40]);
    price_bounds.validity = Some(NullBuffer::from(vec![true, true, false, true]));
    // Even an all-null physical typed leaf proves nothing when fallback
    // coverage is unknown for that path. Its sibling is independently usable.
    price_bounds.all_null = Some(BooleanArray::from(vec![false, false, true, false]));
    let paths = BTreeMap::from([
        (
            price.path.clone(),
            VariantPathStatistics {
                data_type: DataType::Int64,
                bounds: price_bounds,
            },
        ),
        (
            other.path.clone(),
            VariantPathStatistics {
                data_type: DataType::Int64,
                bounds: exact_bounds(vec![1, 2, 3, 4]),
            },
        ),
    ]);
    let statistics = StatisticsBatch::new(
        4,
        BTreeMap::from([
            (2, ColumnStatistics::Variant(paths)),
            (7, ColumnStatistics::from(exact_bounds(vec![3, 4, 5, 6]))),
        ]),
        vec![PartitionStatistics {
            expression: PartitionExpression {
                source: 4.into(),
                source_type: DataType::Int32,
                transform: PartitionTransform::Identity,
            },
            bounds: ColumnBounds {
                validity: Some(NullBuffer::from(vec![true, false, true, false])),
                ..Default::default()
            },
        }],
    )
    .unwrap();
    let selected = statistics
        .filter(&BooleanArray::from(vec![
            None,
            Some(false),
            Some(true),
            Some(true),
        ]))
        .unwrap();
    let out_of_range = equal(4.into(), i64::from(i32::MAX) + 1);
    assert_eq!(selected.len(), 2);
    assert_pruning(&selected, &[equal(price.clone(), 50)], vec![true, false]);
    assert_pruning(&selected, &[equal(other.clone(), 4)], vec![false, true]);
    assert_pruning(&selected, &[equal(7.into(), 6)], vec![false, true]);
    assert_pruning(
        &selected,
        std::slice::from_ref(&out_of_range),
        vec![false, true],
    );
    let sliced = selected.slice(1, 1);
    assert_pruning(&sliced, &[equal(price, 50)], vec![false]);
    assert_pruning(&sliced, &[equal(other, 4), out_of_range], vec![true]);
    let empty = selected
        .filter(&BooleanArray::from(vec![false, false]))
        .unwrap();
    assert!(empty.is_empty());
    assert_pruning(&empty, &[equal(7.into(), 6)], vec![]);
}

#[test]
fn variant_statistics_validate_each_paths_shape_and_type() {
    for (data_type, bounds) in [
        (DataType::Int64, exact_bounds(vec![1])),
        (DataType::Float64, exact_bounds(vec![1, 2])),
        (
            DataType::Int64,
            ColumnBounds {
                validity: Some(NullBuffer::from(vec![true])),
                ..Default::default()
            },
        ),
    ] {
        let columns = BTreeMap::from([(
            2,
            ColumnStatistics::Variant(BTreeMap::from([(
                vec!["price".into()],
                VariantPathStatistics { data_type, bounds },
            )])),
        )]);
        assert!(StatisticsBatch::new(2, columns, Vec::new()).is_err());
    }
}
