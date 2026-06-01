mod common;

use arrow_array::{Array, BooleanArray, RecordBatch, StringViewArray};
use arrow_buffer::BooleanBuffer;

use common::*;
use dispatch::{Contains, OrderBy, Projection, StringKeyExtractor, table_input};

#[test]
fn filter_then_project() {
    let dispatcher = dispatch(1);
    let (_dir, table) = parquet_table(
        &dispatcher,
        &[strings_and_ints(
            &["alice", "bob", "alice"],
            &[100, 200, 300],
        )],
        true,
    );

    let results = table_input(&dispatcher, &table, Projection::all(2), false)
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
        .project(|| {
            let idx = vec![1];
            move |batch: RecordBatch| batch.project(&idx).unwrap()
        })
        .collect()
        .unwrap();

    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![100, 300]);
}

#[test]
fn filter_then_count() {
    let dispatcher = dispatch(1);
    let (_dir, table) = parquet_table(
        &dispatcher,
        &[strings_and_ints(
            &["alice", "bob", "alice", "dave", "alice"],
            &[1, 2, 3, 4, 5],
        )],
        true,
    );

    let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
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
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn filter_then_group_by_then_order_by() {
    let dispatcher = dispatch(1);
    let (_dir, table) = parquet_table(
        &dispatcher,
        &[strings_and_ints(
            &[
                "google.com",
                "apple.com",
                "google.com",
                "google.com",
                "meta.com",
                "apple.com",
                "google.com",
                "meta.com",
                "other.com",
            ],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )],
        true,
    );

    let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
        .filter(|| {
            move |batch: &RecordBatch| {
                let col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |i| {
                    col.value(i) != "other.com"
                }))
            }
        })
        .group_by_count::<StringKeyExtractor>(0)
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect()
        .unwrap();

    let keys = collect_strings(&results, 0);
    let counts = collect_u64s(&results, 1);
    assert_eq!(keys[0], "google.com");
    assert_eq!(counts, vec![4, 2, 2]);
}
