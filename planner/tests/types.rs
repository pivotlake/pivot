#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Int8Array, Int16Array, Int64Array};
use common::*;
use planner::types::Type;
use rstest::rstest;

#[rstest]
#[ignore = "currently not supported in dispatch"]
fn boolean_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "bools",
        &[(
            "flag",
            Type::Boolean,
            Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT flag FROM bools WHERE flag <> false")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["flag"], true);
    assert_eq!(rows[1]["flag"], true);
}

#[rstest]
#[ignore = "currently not supported in dispatch"]
fn int8_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "int8s",
        &[(
            "value",
            Type::Int8,
            Arc::new(Int8Array::from(vec![1_i8, 2, 3])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT value FROM int8s WHERE value <> 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 1);
    assert_eq!(rows[1]["value"], 3);
}

#[rstest]
fn int16_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "int16s",
        &[(
            "value",
            Type::Int16,
            Arc::new(Int16Array::from(vec![10_i16, 20, 30])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT value FROM int16s WHERE value <> 20")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 10);
    assert_eq!(rows[1]["value"], 30);
}

#[rstest]
fn int32_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    // example_table.c is the Int32 column; rows are 100, 200, 300, 400, 500.
    let results = testing_planner
        .planner
        .plan("SELECT c FROM example_table WHERE c <> 200")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["c"].as_i64().unwrap());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0]["c"], 100);
    assert_eq!(rows[1]["c"], 300);
    assert_eq!(rows[2]["c"], 400);
    assert_eq!(rows[3]["c"], 500);
}

#[rstest]
fn int64_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "int64s",
        &[(
            "value",
            Type::Int64,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT value FROM int64s WHERE value <> 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["value"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["value"], 1);
    assert_eq!(rows[1]["value"], 3);
}

#[rstest]
fn utf8_query_runs_end_to_end(mut testing_planner: TestingPlanner) {
    // example_table.name is the Utf8 column; rows are alice/bob/charlie/dave/alice.
    let results = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE name <> 'bob'")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|row| row["name"].as_str().unwrap().to_string());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "alice");
    assert_eq!(rows[2]["name"], "charlie");
    assert_eq!(rows[3]["name"], "dave");
}
