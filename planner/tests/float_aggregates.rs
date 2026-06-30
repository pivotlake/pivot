//! `SUM`/`MIN`/`MAX` over `Float64` columns, folded in `f64` and emitted as a
//! `Float64` column, on both the global (no GROUP BY) and grouped paths.

#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, Int64Array};
use arrow_schema::DataType;
use common::*;
use rstest::rstest;
use serde_json::Value;

fn add_floats_table(planner: &TestingPlanner) {
    planner.add_table(
        "floats",
        &[
            (
                "g",
                planner::types::Type::Int32,
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2])),
            ),
            (
                "f",
                planner::types::Type::Float64,
                Arc::new(Float64Array::from(vec![1.5, 2.5, 10.0, 20.0, 30.0])),
            ),
            (
                "big",
                planner::types::Type::Int64,
                Arc::new(Int64Array::from(vec![1, 1, 1, 1, 1])),
            ),
        ],
    );
}

fn rows(planner: &mut TestingPlanner, sql: &str) -> Vec<Value> {
    let results = planner
        .planner
        .plan(sql)
        .unwrap()
        .compile(planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    batches_to_json(&results)
}

#[rstest]
fn global_float_aggregates(mut testing_planner: TestingPlanner) {
    add_floats_table(&testing_planner);

    let rows = rows(
        &mut testing_planner,
        "SELECT SUM(f) AS s, MIN(f) AS mn, MAX(f) AS mx FROM floats",
    );

    assert_eq!(rows[0]["s"].as_f64().unwrap(), 64.0);
    assert_eq!(rows[0]["mn"].as_f64().unwrap(), 1.5);
    assert_eq!(rows[0]["mx"].as_f64().unwrap(), 30.0);
}

#[rstest]
fn global_float_sum_output_is_float64(mut testing_planner: TestingPlanner) {
    add_floats_table(&testing_planner);

    let results = testing_planner
        .planner
        .plan("SELECT SUM(f) AS s FROM floats")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(results[0].schema().field(0).data_type(), &DataType::Float64);
}

#[rstest]
fn grouped_float_aggregates(mut testing_planner: TestingPlanner) {
    add_floats_table(&testing_planner);

    let mut rows = rows(
        &mut testing_planner,
        "SELECT g, SUM(f) AS s, MIN(f) AS mn, MAX(f) AS mx FROM floats GROUP BY g ORDER BY g",
    );
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());

    assert_eq!(rows[0]["s"].as_f64().unwrap(), 4.0);
    assert_eq!(rows[0]["mn"].as_f64().unwrap(), 1.5);
    assert_eq!(rows[0]["mx"].as_f64().unwrap(), 2.5);
    assert_eq!(rows[1]["s"].as_f64().unwrap(), 60.0);
    assert_eq!(rows[1]["mn"].as_f64().unwrap(), 10.0);
    assert_eq!(rows[1]["mx"].as_f64().unwrap(), 30.0);
}

#[rstest]
fn grouped_float_sum_beside_wide_int_sum(mut testing_planner: TestingPlanner) {
    // A 64-bit integer `SUM` forces the wide (`i128`) cell array; the float `SUM`
    // then rides an `i128` cell, exercising the wide-cell float render path.
    add_floats_table(&testing_planner);

    let mut rows = rows(
        &mut testing_planner,
        "SELECT g, SUM(big) AS b, SUM(f) AS s FROM floats GROUP BY g ORDER BY g",
    );
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());

    assert_eq!(rows[0]["s"].as_f64().unwrap(), 4.0);
    assert_eq!(rows[1]["s"].as_f64().unwrap(), 60.0);
}

#[rstest]
fn global_avg_over_float(mut testing_planner: TestingPlanner) {
    // DuckDB lowers `AVG(f)` to `SUM(f) / COUNT(f)`, so the float `SUM` slot drives
    // the average too.
    add_floats_table(&testing_planner);

    let rows = rows(&mut testing_planner, "SELECT AVG(f) AS a FROM floats");

    assert_eq!(rows[0]["a"].as_f64().unwrap(), 64.0 / 5.0);
}
