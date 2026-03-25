mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{Projection, table_input};

fn int64_table(name: &str, values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

#[test]
fn join_matching_keys() {
    init();
    let (_bd, build_table) = parquet_table(&[int64_table("id", &[10, 20, 30])]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &[20, 30, 99])]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).collect();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_no_matches() {
    init();
    let (_bd, build_table) = parquet_table(&[int64_table("id", &[1, 2, 3])]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &[4, 5, 6])]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).collect();

    assert!(results.is_empty());
}

#[test]
fn join_duplicate_build_keys() {
    init();
    let (_bd, build_table) = parquet_table(&[int64_table("id", &[10, 10, 20])]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &[10])]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).collect();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![10, 10]);
}

#[test]
fn join_all_keys_match() {
    init();
    let (_bd, build_table) = parquet_table(&[int64_table("id", &[1, 2, 3, 4, 5])]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &[5, 4, 3, 2, 1])]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).collect();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![1, 2, 3, 4, 5]);
}

#[test]
fn join_large_tables() {
    init();
    let build_keys: Vec<i64> = (0..5000).collect();
    let probe_keys: Vec<i64> = (2500..3500).collect();

    let (_bd, build_table) = parquet_table(&[int64_table("id", &build_keys)]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &probe_keys)]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).collect();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, probe_keys);
}

#[test]
fn join_then_count() {
    init();
    let (_bd, build_table) = parquet_table(&[int64_table("id", &[10, 20, 30, 40])]);
    let (_pd, probe_table) = parquet_table(&[int64_table("id", &[20, 30, 99])]);

    let build = table_input(&build_table, Projection::all(1), false);
    let probe = table_input(&probe_table, Projection::all(1), false);
    let results = probe.join(build, 0, 0).count().collect();

    assert_eq!(extract_count(&results), 2);
}
