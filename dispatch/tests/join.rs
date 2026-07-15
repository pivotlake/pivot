//! Hash-join integration tests: build a hash table from one in-memory input and
//! probe it with another, asserting which probe rows survive.

mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{AggregationKind, AggregationSlot, JoinMode, values_input};

fn int64_batch(name: &str, values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

#[test]
fn join_matching_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_no_matches() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[1, 2, 3])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[4, 5, 6])]).record_batches();

    let results = probe
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
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
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
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
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
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
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
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
        .join(build, vec![0], vec![0], vec![false], JoinMode::Inner)
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
fn semi_join_emits_each_matching_probe_row_once() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20, 30, 10])]).record_batches();

    let results = probe
        .join(build, vec![0], vec![0], vec![false], JoinMode::Semi)
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 0);
    keys.sort();
    assert_eq!(keys, vec![10, 10, 30]);
}

#[test]
fn anti_join_emits_probe_rows_without_matches() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20, 30, 40])]).record_batches();

    let results = probe
        .join(build, vec![0], vec![0], vec![false], JoinMode::Anti)
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 0);
    keys.sort();
    assert_eq!(keys, vec![20, 40]);
}

#[test]
fn anti_join_with_empty_build_passes_everything() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[1, 2])]).record_batches();

    let results = probe
        .join(build, vec![0], vec![0], vec![false], JoinMode::Anti)
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 0);
    keys.sort();
    assert_eq!(keys, vec![1, 2]);
}
