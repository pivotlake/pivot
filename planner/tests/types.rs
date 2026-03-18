#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Int8Array, Int16Array, Int32Array, Int64Array, StringViewArray,
};

use common::*;
use planner::types::Type;

#[test]
#[ignore = "currently not supported in dispatch"]
fn boolean_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "flag",
            Type::Boolean,
            Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
        )],
    );

    // Intended end-to-end behavior once dispatch supports BOOLEAN parquet decoding.
    let results = planner
        .plan("SELECT flag FROM test WHERE flag <> false")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["flag"], true);
    assert_eq!(rows[1]["flag"], true);
}

#[test]
#[ignore = "currently not supported in dispatch"]
fn int8_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "value",
            Type::Int8,
            Arc::new(Int8Array::from(vec![1_i8, 2, 3])) as ArrayRef,
        )],
    );

    // Intended end-to-end behavior once dispatch supports Int8 parquet decoding.
    let results = planner
        .plan("SELECT value FROM test WHERE value <> 2")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 1);
    assert_eq!(rows[1]["value"], 3);
}

#[test]
fn int16_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "value",
            Type::Int16,
            Arc::new(Int16Array::from(vec![10_i16, 20, 30])) as ArrayRef,
        )],
    );

    let results = planner
        .plan("SELECT value FROM test WHERE value <> 20")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 10);
    assert_eq!(rows[1]["value"], 30);
}

#[test]
fn int32_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "value",
            Type::Int32,
            Arc::new(Int32Array::from(vec![100_i32, 200, 300])) as ArrayRef,
        )],
    );

    let results = planner
        .plan("SELECT value FROM test WHERE value <> 200")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 100);
    assert_eq!(rows[1]["value"], 300);
}

#[test]
fn int64_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "value",
            Type::Int64,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
        )],
    );

    let results = planner
        .plan("SELECT value FROM test WHERE value <> 2")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 1);
    assert_eq!(rows[1]["value"], 3);
}

#[test]
fn utf8_query_runs_end_to_end() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "name",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["alice", "bob", "charlie"])) as ArrayRef,
        )],
    );

    let results = planner
        .plan("SELECT name FROM test WHERE name <> 'bob'")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["name"].as_str().unwrap().to_string());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "charlie");
}
