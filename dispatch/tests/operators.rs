mod common;

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{Contains, IntKeyExtractor, OrderBy, Projection, StringKeyExtractor, table_input};

#[test]
fn count() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["a", "b", "c", "d", "e"],
        &[1, 2, 3, 4, 5],
    )]);

    let results = table_input(&table, Projection::all(2), false)
        .count()
        .collect();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn filter_string_contains() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["alice", "bob", "alice", "dave", "alice"],
        &[1, 2, 3, 4, 5],
    )]);

    let results = table_input(&table, Projection::columns([0]), false)
        .filter(|| {
            let mut c = Contains::new("alice");
            move |batch: &RecordBatch| {
                c.run(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap(),
                )
            }
        })
        .count()
        .collect();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn filter_no_matches_returns_zero() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(&["alice", "bob", "carol"], &[1, 2, 3])]);

    let results = table_input(&table, Projection::columns([0]), false)
        .filter(|| {
            let mut c = Contains::new("zzz_no_match");
            move |batch: &RecordBatch| {
                c.run(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap(),
                )
            }
        })
        .count()
        .collect();

    assert_eq!(extract_count(&results), 0);
}

#[test]
fn filter_integer_column() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["a", "b", "c", "d", "e"],
        &[5, 15, 25, 35, 45],
    )]);

    let results = table_input(&table, Projection::all(2), false)
        .filter(|| {
            move |batch: &RecordBatch| {
                let col = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |i| {
                    col.value(i) > 20
                }))
            }
        })
        .count()
        .collect();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn project_selects_single_column() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(&["a", "b"], &[10, 20])]);

    let results = table_input(&table, Projection::all(2), false)
        .project(|| {
            let idx = vec![1];
            move |batch: &RecordBatch| batch.project(&idx).unwrap()
        })
        .collect();

    assert_eq!(results[0].num_columns(), 1);
    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![10, 20]);
}

#[test]
fn order_by_ascending_with_limit() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["e", "a", "d", "b", "c"],
        &[50, 10, 40, 20, 30],
    )]);

    let results = table_input(&table, Projection::all(2), false)
        .order_by_limit(vec![OrderBy::new(1, false, false)], 3)
        .collect();

    assert_eq!(collect_i64s(&results, 1), vec![10, 20, 30]);
    assert_eq!(collect_strings(&results, 0), vec!["a", "b", "c"]);
}

#[test]
fn order_by_descending_with_limit() {
    init();
    let (_dir, table) =
        parquet_table(&[strings_and_ints(&["a", "b", "c", "d"], &[10, 40, 20, 30])]);

    let results = table_input(&table, Projection::all(2), false)
        .order_by_limit(vec![OrderBy::new(1, true, false)], 2)
        .collect();

    assert_eq!(collect_i64s(&results, 1), vec![40, 30]);
}

#[test]
fn order_by_limit_exceeds_row_count() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(&["c", "a", "b"], &[30, 10, 20])]);

    let results = table_input(&table, Projection::all(2), false)
        .order_by_limit(vec![OrderBy::new(1, false, false)], 100)
        .collect();

    assert_eq!(collect_i64s(&results, 1), vec![10, 20, 30]);
}

#[test]
fn group_by_count_string_keys() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["alice", "bob", "alice", "carol", "bob", "alice"],
        &[1, 2, 3, 4, 5, 6],
    )]);

    let results = table_input(&table, Projection::columns([0]), false)
        .group_by_count::<StringKeyExtractor>(0)
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect();

    assert_eq!(collect_strings(&results, 0), vec!["alice", "bob", "carol"]);
    assert_eq!(collect_u64s(&results, 1), vec![3, 2, 1]);
}

#[test]
fn group_by_count_int_keys() {
    init();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("cat", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 1, 3, 2, 1, 3, 2]))],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&[batch]);

    let results = table_input(&table, Projection::columns([0]), false)
        .group_by_count::<IntKeyExtractor<Int64Type>>(0)
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect();

    assert_eq!(collect_u64s(&results, 1), vec![3, 3, 2]);
}

#[test]
fn concat_two_tables() {
    init();
    let (_dir1, table1) = parquet_table(&[strings_and_ints(&["a", "b", "c"], &[1, 2, 3])]);
    let (_dir2, table2) = parquet_table(&[strings_and_ints(&["d", "e"], &[4, 5])]);

    let left = table_input(&table1, Projection::all(2), false);
    let right = table_input(&table2, Projection::all(2), false);

    let results = left.concat(right).count().collect();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn concat_same_table() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(&["a", "b", "c"], &[1, 2, 3])]);

    let left = table_input(&table, Projection::all(2), false);
    let right = table_input(&table, Projection::all(2), false);

    let results = left.concat(right).count().collect();

    assert_eq!(extract_count(&results), 6);
}

#[test]
fn concat_then_filter() {
    init();
    let (_dir1, table1) = parquet_table(&[strings_and_ints(&["alice", "bob"], &[1, 2])]);
    let (_dir2, table2) =
        parquet_table(&[strings_and_ints(&["alice", "carol", "alice"], &[3, 4, 5])]);

    let left = table_input(&table1, Projection::columns([0]), false);
    let right = table_input(&table2, Projection::columns([0]), false);

    let results = left
        .concat(right)
        .filter(|| {
            let mut c = Contains::new("alice");
            move |batch: &RecordBatch| {
                c.run(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap(),
                )
            }
        })
        .count()
        .collect();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn concat_with_empty_side() {
    init();
    let (_dir1, table1) = parquet_table(&[strings_and_ints(&["a", "b"], &[1, 2])]);
    let (_dir2, table2) = parquet_table(&[strings_and_ints(&[], &[])]);

    let left = table_input(&table1, Projection::all(2), false);
    let right = table_input(&table2, Projection::all(2), false);

    let results = left.concat(right).count().collect();

    assert_eq!(extract_count(&results), 2);
}
