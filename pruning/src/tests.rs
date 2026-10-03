use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, Scalar};
use arrow_ord::cmp;

use crate::{Bounds, ColumnPath, Comparison, Predicate, Statistic, Statistics, Transform};

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

fn predicate(column: impl Into<ColumnPath>, comparison: Comparison, value: i64) -> Predicate {
    Predicate {
        column: column.into(),
        comparison,
        value: constant(value),
    }
}

fn equal(column: impl Into<ColumnPath>, value: i64) -> Predicate {
    predicate(column, Comparison::Equal, value)
}

fn bounds(lower: Vec<Option<i64>>, upper: Vec<Option<i64>>) -> Bounds {
    Bounds {
        all_null: BooleanArray::from(vec![false; lower.len()]),
        lower: Arc::new(Int64Array::from(lower)),
        upper: Arc::new(Int64Array::from(upper)),
    }
}

fn exact(values: Vec<i64>) -> Bounds {
    let values: Vec<_> = values.into_iter().map(Some).collect();
    bounds(values.clone(), values)
}

fn statistic(column: impl Into<ColumnPath>, bounds: Bounds) -> Statistic {
    Statistic {
        column: column.into(),
        transform: Transform::Identity,
        bounds,
    }
}

fn field(column_idx: usize, path: &[&str]) -> ColumnPath {
    ColumnPath {
        column_idx,
        path: path.iter().map(|name| name.to_string()).collect(),
    }
}

fn kernel(
    comparison: Comparison,
) -> fn(
    &dyn arrow_array::Datum,
    &dyn arrow_array::Datum,
) -> Result<BooleanArray, arrow_schema::ArrowError> {
    match comparison {
        Comparison::Equal => cmp::eq,
        Comparison::NotEqual => cmp::neq,
        Comparison::Less => cmp::lt,
        Comparison::LessEqual => cmp::lt_eq,
        Comparison::Greater => cmp::gt,
        Comparison::GreaterEqual => cmp::gt_eq,
    }
}

/// Whether any of an object's rows satisfies `<comparison> value`.
fn any_row_matches(rows: &[Option<f64>], comparison: Comparison, value: f64) -> bool {
    let rows = Float64Array::from(rows.to_vec());
    let value = Float64Array::new_scalar(value);
    let matches = kernel(comparison)(&rows, &value).unwrap();
    matches.iter().any(|matched| matched == Some(true))
}

#[test]
fn bounds_never_exclude_an_object_with_a_matching_row() {
    let domain = [
        None,
        Some(-f64::NAN),
        Some(-2.0),
        Some(-0.0),
        Some(0.0),
        Some(2.0),
        Some(f64::NAN),
    ];
    let objects: Vec<[Option<f64>; 2]> = domain
        .iter()
        .flat_map(|a| domain.iter().map(move |b| [*a, *b]))
        .collect();
    let bound = |pick: fn(f64, f64) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter(
            objects
                .iter()
                .map(|rows| rows.iter().flatten().copied().reduce(pick)),
        ))
    };
    let statistics = Statistics::new(
        objects.len(),
        vec![statistic(
            0,
            Bounds {
                lower: bound(|a, b| if a.total_cmp(&b).is_le() { a } else { b }),
                upper: bound(|a, b| if a.total_cmp(&b).is_ge() { a } else { b }),
                all_null: objects
                    .iter()
                    .map(|rows| Some(rows.iter().all(Option::is_none)))
                    .collect(),
            },
        )],
    );

    for comparison in COMPARISONS {
        for value in domain.into_iter().flatten() {
            let keep = statistics.prune(&[Predicate {
                column: 0.into(),
                comparison,
                value: Scalar::new(Arc::new(Float64Array::from(vec![value])) as ArrayRef),
            }]);

            for (rows, keep) in objects.iter().zip(keep.values()) {
                let matches = any_row_matches(rows, comparison, value);
                assert!(!matches || keep, "{rows:?} {comparison:?} {value}");
                // Two rows are their own bounds, so only equality with a
                // value strictly between them survives without a match.
                if comparison != Comparison::Equal {
                    assert_eq!(keep, matches, "{rows:?} {comparison:?} {value}");
                }
            }
        }
    }
}

#[test]
fn a_null_bound_is_unknown_and_an_all_null_object_matches_nothing() {
    let statistics = Statistics::new(
        4,
        vec![statistic(
            0,
            Bounds {
                lower: Arc::new(Int64Array::from(vec![Some(10), None, None, None])),
                upper: Arc::new(Int64Array::from(vec![None, Some(0), None, None])),
                all_null: BooleanArray::from(vec![Some(false), Some(false), None, Some(true)]),
            },
        )],
    );

    let keep = statistics.prune(&[equal(0, 5)]);

    assert_eq!(keep, BooleanArray::from(vec![false, false, true, false]));
}

#[test]
fn equality_keeps_a_range_that_contains_the_constant() {
    let statistics = Statistics::new(
        2,
        vec![statistic(
            0,
            bounds(vec![Some(0); 2], vec![Some(10), Some(4)]),
        )],
    );

    let keep = statistics.prune(&[equal(0, 5)]);

    assert_eq!(keep, BooleanArray::from(vec![true, false]));
}

#[test]
fn inequality_excludes_only_an_object_holding_just_the_constant() {
    let statistics = Statistics::new(
        2,
        vec![statistic(
            0,
            bounds(vec![Some(5); 2], vec![Some(5), Some(6)]),
        )],
    );

    let keep = statistics.prune(&[predicate(0, Comparison::NotEqual, 5)]);

    assert_eq!(keep, BooleanArray::from(vec![false, true]));
}

#[test]
fn bounds_of_another_type_than_the_constant_prove_nothing() {
    let statistics = Statistics::new(
        2,
        vec![statistic(
            0,
            Bounds {
                lower: Arc::new(Int32Array::from(vec![10, 10])),
                upper: Arc::new(Int32Array::from(vec![10, 10])),
                all_null: BooleanArray::from(vec![false, true]),
            },
        )],
    );

    let keep = statistics.prune(&[equal(0, 5)]);

    assert_eq!(keep, BooleanArray::from(vec![true, true]));
}

#[test]
fn a_null_constant_matches_nothing_even_without_statistics() {
    let statistics = Statistics::new(2, Vec::new());
    let null = Predicate {
        column: 0.into(),
        comparison: Comparison::NotEqual,
        value: Scalar::new(Arc::new(Int64Array::from(vec![None])) as ArrayRef),
    };

    let keep = statistics.prune(&[null]);

    assert_eq!(keep, BooleanArray::from(vec![false, false]));
}

#[test]
fn objects_without_statistics_for_a_column_are_kept() {
    let statistics = Statistics::new(2, vec![statistic(9, exact(vec![10, 20]))]);
    let empty = Statistics::new(0, Vec::new());

    let other_column = statistics.prune(&[equal(1, 100)]);
    let no_predicates = statistics.prune(&[]);
    let no_objects = empty.prune(&[equal(1, 100)]);

    assert_eq!(other_column, BooleanArray::from(vec![true, true]));
    assert_eq!(no_predicates, BooleanArray::from(vec![true, true]));
    assert!(no_objects.is_empty());
}

#[test]
fn every_predicate_must_be_able_to_match() {
    let statistics = Statistics::new(
        3,
        vec![
            statistic(9, exact(vec![10, 20, 20])),
            statistic(2, exact(vec![100, 100, 200])),
        ],
    );

    let keep = statistics.prune(&[equal(9, 20), equal(2, 100)]);

    assert_eq!(keep, BooleanArray::from(vec![false, true, false]));
}

#[test]
fn a_field_is_matched_by_its_column_and_whole_path() {
    let statistics = Statistics::new(
        2,
        vec![
            statistic(field(7, &["item", "price"]), exact(vec![10, 20])),
            statistic(field(7, &["discount"]), exact(vec![20, 10])),
            statistic(field(7, &[]), exact(vec![30, 40])),
            statistic(field(9, &["item", "price"]), exact(vec![20, 10])),
        ],
    );

    let price = statistics.prune(&[equal(field(7, &["item", "price"]), 20)]);
    let discount = statistics.prune(&[equal(field(7, &["discount"]), 20)]);
    let root = statistics.prune(&[equal(7, 40)]);
    let other_column = statistics.prune(&[equal(field(9, &["item", "price"]), 20)]);
    let missing = statistics.prune(&[equal(field(7, &["item"]), 0)]);

    assert_eq!(price, BooleanArray::from(vec![false, true]));
    assert_eq!(discount, BooleanArray::from(vec![true, false]));
    assert_eq!(root, BooleanArray::from(vec![false, true]));
    assert_eq!(other_column, BooleanArray::from(vec![true, false]));
    assert_eq!(missing, BooleanArray::from(vec![true, true]));
}

#[test]
fn every_statistic_of_a_column_can_exclude_objects() {
    // Column bounds next to that column's partition values.
    let statistics = Statistics::new(
        2,
        vec![
            statistic(5, bounds(vec![Some(0), Some(0)], vec![None, None])),
            statistic(5, bounds(vec![None, None], vec![Some(10), Some(30)])),
        ],
    );

    let above_one = statistics.prune(&[equal(5, 20)]);
    let below_both = statistics.prune(&[equal(5, -1)]);

    assert_eq!(above_one, BooleanArray::from(vec![false, true]));
    assert_eq!(below_both, BooleanArray::from(vec![false, false]));
}

#[test]
fn a_partition_statistic_is_compared_through_its_transform() {
    // Days since the epoch of three files partitioned by day.
    let days = Arc::new(arrow_array::Date32Array::from(vec![0, 1, 2])) as ArrayRef;
    let statistics = Statistics::new(
        3,
        vec![Statistic {
            column: 0.into(),
            transform: Transform::Day,
            bounds: Bounds {
                lower: days.clone(),
                upper: days,
                all_null: BooleanArray::from(vec![false; 3]),
            },
        }],
    );
    let noon_of_day_one = Predicate {
        column: 0.into(),
        comparison: Comparison::GreaterEqual,
        value: Scalar::new(Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
            36 * 3_600_000_000,
        ])) as ArrayRef),
    };

    let keep = statistics.prune(&[noon_of_day_one]);

    assert_eq!(keep, BooleanArray::from(vec![false, true, true]));
}

#[test]
fn selecting_returns_the_objects_the_mask_keeps_in_order() {
    let statistics = Statistics::new(3, vec![statistic(4, exact(vec![10, 20, 30]))]);
    let predicates = [predicate(4, Comparison::Greater, 15)];
    let files = ["a", "b", "c"];

    let selected: Vec<_> = statistics.select(&files, &predicates).collect();

    assert_eq!(selected, [&"b", &"c"]);
}

#[test]
fn the_columns_of_predicates_are_distinct() {
    let predicates = [
        predicate(3, Comparison::Greater, 1),
        equal(field(3, &["a"]), 1),
        predicate(3, Comparison::Less, 9),
    ];

    let columns = Predicate::columns(&predicates);

    assert_eq!(columns, [&field(3, &[]), &field(3, &["a"])]);
}

#[test]
#[should_panic(expected = "one slot per object")]
fn statistics_reject_bounds_of_another_length() {
    Statistics::new(2, vec![statistic(0, exact(vec![1]))]);
}
