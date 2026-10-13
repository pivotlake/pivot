//! End-to-end set operation tests: UNION ALL through DuckDB's binding and
//! pushdown, the planner's union translation, and dispatch execution.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, StringViewArray};

use crate::common::*;
use planner::types::Type;
use rstest::rstest;

fn int_col(values: Vec<Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn int64_col(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

fn add_two_shops(planner: &TestingPlanner) {
    planner.add_table(
        "north_sales",
        &[
            (
                "item",
                Type::Int32,
                int_col(vec![Some(1), Some(2), Some(3)]),
            ),
            (
                "qty",
                Type::Int32,
                int_col(vec![Some(10), Some(20), Some(30)]),
            ),
        ],
    );
    planner.add_table(
        "south_sales",
        &[
            ("item", Type::Int32, int_col(vec![Some(3), Some(4)])),
            ("qty", Type::Int32, int_col(vec![Some(30), Some(40)])),
        ],
    );
}

#[rstest]
fn union_all_keeps_every_row_of_every_input(mut testing_planner: TestingPlanner) {
    add_two_shops(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT item FROM north_sales UNION ALL SELECT item FROM south_sales",
    );

    rows.sort_by_key(|row| row["item"].as_i64().unwrap());
    assert_eq!(
        rows,
        serde_json::json!([{"item": 1}, {"item": 2}, {"item": 3}, {"item": 3}, {"item": 4}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn filter_on_a_cte_union_all_applies_to_both_inputs(mut testing_planner: TestingPlanner) {
    add_two_shops(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "WITH all_sales AS (SELECT item, qty FROM north_sales \
                            UNION ALL SELECT item, qty FROM south_sales) \
         SELECT item, qty FROM all_sales WHERE qty >= 30",
    );

    rows.sort_by_key(|row| row["item"].as_i64().unwrap());
    assert_eq!(
        rows,
        serde_json::json!([{"item": 3, "qty": 30}, {"item": 3, "qty": 30}, {"item": 4, "qty": 40}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn union_all_feeds_a_group_by(mut testing_planner: TestingPlanner) {
    add_two_shops(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT item, sum(qty) AS total FROM \
           (SELECT item, qty FROM north_sales UNION ALL SELECT item, qty FROM south_sales) \
         GROUP BY item",
    );

    rows.sort_by_key(|row| row["item"].as_i64().unwrap());
    assert_eq!(
        rows,
        serde_json::json!([
            {"item": 1, "total": 10},
            {"item": 2, "total": 20},
            {"item": 3, "total": 60},
            {"item": 4, "total": 40},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

/// One input has NULLs and the other does not, and the sort above the union
/// merges batches from both: the union's output has to declare the column
/// nullable for every batch, whichever input it came from.
#[rstest]
fn union_all_sorts_nullable_and_null_free_inputs_together(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "with_nulls",
        &[("v", Type::Int32, int_col(vec![Some(5), None, Some(1)]))],
    );
    testing_planner.add_table(
        "without_nulls",
        &[("v", Type::Int32, int_col(vec![Some(3), Some(2)]))],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT v FROM with_nulls UNION ALL SELECT v FROM without_nulls ORDER BY v NULLS LAST",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"v": 1}, {"v": 2}, {"v": 3}, {"v": 5}, {}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn union_all_casts_its_inputs_to_a_common_type(mut testing_planner: TestingPlanner) {
    testing_planner.add_table("narrow_ids", &[("id", Type::Int32, int_col(vec![Some(1)]))]);
    testing_planner.add_table(
        "wide_ids",
        &[("id", Type::Int64, int64_col(vec![5_000_000_000]))],
    );

    let mut rows = run(
        &mut testing_planner,
        "SELECT id FROM narrow_ids UNION ALL SELECT id FROM wide_ids",
    );

    rows.sort_by_key(|row| row["id"].as_i64().unwrap());
    assert_eq!(
        rows,
        serde_json::json!([{"id": 1}, {"id": 5_000_000_000i64}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn union_all_of_string_columns(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "first_names",
        &[("name", Type::Utf8, str_col(vec!["ann", "bob"]))],
    );
    testing_planner.add_table("last_names", &[("name", Type::Utf8, str_col(vec!["cole"]))]);

    let rows = run(
        &mut testing_planner,
        "SELECT name FROM first_names UNION ALL SELECT name FROM last_names ORDER BY name",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"name": "ann"}, {"name": "bob"}, {"name": "cole"}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn plain_union_drops_duplicate_rows(mut testing_planner: TestingPlanner) {
    add_two_shops(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT item FROM north_sales UNION SELECT item FROM south_sales",
    );

    rows.sort_by_key(|row| row["item"].as_i64().unwrap());
    assert_eq!(
        rows,
        serde_json::json!([{"item": 1}, {"item": 2}, {"item": 3}, {"item": 4}])
            .as_array()
            .unwrap()
            .clone()
    );
}
