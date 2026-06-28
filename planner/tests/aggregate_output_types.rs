//! An aggregate's output column carries its declared type, independent of the
//! accumulator's storage width: a `COUNT` co-located with a string extreme sits
//! in a wide (`i128`) cell yet stays `Int64`, and a narrow integer `SUM` widens
//! to `Decimal128`. DuckDB-inserted casts above these columns (a `HAVING`, an
//! arithmetic) then line up by type instead of crashing.

#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use arrow_schema::DataType;
use common::*;
use rstest::rstest;

#[rstest]
fn grouped_count_beside_string_min_is_int64(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT COUNT(*) AS n, MIN(name) AS mn FROM example_table GROUP BY a")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let count = results[0]
        .schema()
        .fields()
        .iter()
        .find(|f| f.name() == "n")
        .unwrap()
        .data_type()
        .clone();
    assert_eq!(count, DataType::Int64);
}

#[rstest]
fn grouped_count_having_on_count_does_not_crash(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT MIN(name) AS mn, COUNT(*) AS n FROM example_table GROUP BY a HAVING COUNT(*) > 0")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

#[rstest]
fn global_sum_of_narrow_int_is_decimal128(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT SUM(a) AS s FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let sum = results[0].schema().field(0).data_type().clone();
    assert_eq!(sum, DataType::Decimal128(38, 0));
}
