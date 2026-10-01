use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, Scalar,
};
use planner::expression::CompareType;

use super::{ColumnBounds, PruningPredicate, StatisticsBatch};

const COMPARISONS: [CompareType; 6] = [
    CompareType::Equal,
    CompareType::NotEqual,
    CompareType::Less,
    CompareType::LessEqual,
    CompareType::Greater,
    CompareType::GreaterEqual,
];

fn constant(value: i64) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
}

fn compare_values(left: i64, compare: CompareType, right: i64) -> bool {
    match compare {
        CompareType::Equal => left == right,
        CompareType::NotEqual => left != right,
        CompareType::Less => left < right,
        CompareType::LessEqual => left <= right,
        CompareType::Greater => left > right,
        CompareType::GreaterEqual => left >= right,
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
        .may_match(4, CompareType::Equal, &constant(5))
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
        .may_match(1, CompareType::Equal, &constant(5))
        .unwrap();

    assert!(keep.value(0));
}

#[test]
fn boolean_combinations_preserve_unknown_alternatives() {
    let statistics = StatisticsBatch {
        len: 2,
        columns: vec![ColumnBounds {
            lower: Some(Arc::new(Int64Array::from(vec![Some(1), None]))),
            upper: Some(Arc::new(Int64Array::from(vec![Some(1), None]))),
            ..Default::default()
        }],
    };
    let equal = || PruningPredicate::Compare {
        column: 0,
        compare: CompareType::Equal,
        value: constant(2),
    };

    let disjunction = statistics
        .may_match(&PruningPredicate::Or(vec![
            equal(),
            PruningPredicate::Always(false),
        ]))
        .unwrap();
    let conjunction = statistics
        .may_match(&PruningPredicate::And(vec![
            equal(),
            PruningPredicate::Always(false),
        ]))
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

    let keep = bounds.may_match(3, CompareType::Greater, &two).unwrap();

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

    let keep = bounds.may_match(1, CompareType::Equal, &two).unwrap();

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
                        CompareType::Equal => arrow_ord::cmp::eq,
                        CompareType::NotEqual => arrow_ord::cmp::neq,
                        CompareType::Less => arrow_ord::cmp::lt,
                        CompareType::LessEqual => arrow_ord::cmp::lt_eq,
                        CompareType::Greater => arrow_ord::cmp::gt,
                        CompareType::GreaterEqual => arrow_ord::cmp::gt_eq,
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
                                if !matches!(compare, CompareType::Equal) {
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

    let keep = bounds.may_match(1, CompareType::Less, &zero).unwrap();

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
        .may_match(1, CompareType::Equal, &constant(5))
        .unwrap();
    let null_comparison = bounds.may_match(1, CompareType::NotEqual, &null).unwrap();

    assert!(mismatched.value(0));
    assert!(!null_comparison.value(0));
}
