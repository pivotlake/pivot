//! Hash-join integration tests: build a hash table from one in-memory input and
//! probe it with another, asserting which probe rows survive.

mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{AggregationKind, AggregationSlot, JoinOutputColumns, values_input};

fn int64_batch(name: &str, values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn two_int64_batch(names: (&str, &str), left: &[i64], right: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(names.0, DataType::Int64, false),
            Field::new(names.1, DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(left.to_vec())),
            Arc::new(Int64Array::from(right.to_vec())),
        ],
    )
    .unwrap()
}

#[test]
fn join_matching_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_on_int32_keys() {
    let d = dispatch(1);
    let int32_batch = |values: &[i32]| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(arrow_array::Int32Array::from(values.to_vec()))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![int32_batch(&[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int32_batch(&[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int32,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    let mut keys: Vec<i32> = results
        .iter()
        .flat_map(|b| {
            use arrow_array::cast::AsArray;
            b.column(1)
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .to_vec()
        })
        .collect();
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_no_matches() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[1, 2, 3])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[4, 5, 6])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn join_duplicate_build_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 20])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![10, 10]);
}

#[test]
fn join_all_keys_match() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[1, 2, 3, 4, 5])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[5, 4, 3, 2, 1])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![1, 2, 3, 4, 5]);
}

#[test]
fn join_large_tables() {
    let d = dispatch(1);
    let build_keys: Vec<i64> = (0..5000).collect();
    let probe_keys: Vec<i64> = (2500..3500).collect();

    let build = values_input(&d, vec![int64_batch("id", &build_keys)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_keys)]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, probe_keys);
}

#[test]
fn join_then_count() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30, 40])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns::keep_all(1, 1),
        )
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 2);
}

#[test]
fn join_keeps_only_listed_columns() {
    let d = dispatch(1);
    let build = values_input(
        &d,
        vec![two_int64_batch(("id", "b_payload"), &[10, 20], &[100, 200])],
    )
    .record_batches();
    let probe = values_input(
        &d,
        vec![two_int64_batch(
            ("id", "p_payload"),
            &[20, 20, 99],
            &[7, 8, 9],
        )],
    )
    .record_batches();

    let results = probe
        .join(
            build,
            0,
            0,
            &DataType::Int64,
            JoinOutputColumns {
                probe: vec![1],
                build: vec![1],
            },
        )
        .collect()
        .unwrap();

    let schema = results[0].schema();
    assert_eq!(schema.field(0).name(), "p_payload");
    assert_eq!(schema.field(1).name(), "b_payload");
    let mut probe_payloads = collect_i64s(&results, 0);
    probe_payloads.sort();
    assert_eq!(probe_payloads, vec![7, 8]);
    assert_eq!(collect_i64s(&results, 1), vec![200, 200]);
}
