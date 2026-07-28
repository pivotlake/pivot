//! End-to-end inner hash equi-join tests: SQL through DuckDB planning, the
//! pivot plan walk, and dispatch execution.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, StringViewArray};

use crate::common::*;
use planner::types::Type;
use rstest::rstest;

fn int64_col(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

fn add_orders_and_items(planner: &TestingPlanner) {
    planner.add_table(
        "orders",
        &[
            ("o_key", Type::Int64, int64_col(vec![1, 2, 3])),
            (
                "o_status",
                Type::Utf8,
                str_col(vec!["open", "closed", "open"]),
            ),
        ],
    );
    planner.add_table(
        "items",
        &[
            ("i_order", Type::Int64, int64_col(vec![1, 1, 2, 9])),
            ("i_qty", Type::Int64, int64_col(vec![10, 20, 30, 40])),
        ],
    );
}

#[rstest]
fn inner_join_emits_matching_pairs(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT i_qty, o_status FROM items JOIN orders ON i_order = o_key",
    );
    rows.sort_by_key(|r| r["i_qty"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"i_qty": 10, "o_status": "open"},
            {"i_qty": 20, "o_status": "open"},
            {"i_qty": 30, "o_status": "closed"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn join_key_is_selectable_from_both_sides(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT o_key, i_order FROM items JOIN orders ON i_order = o_key",
    );
    rows.sort_by_key(|r| r["o_key"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"o_key": 1, "i_order": 1},
            {"o_key": 1, "i_order": 1},
            {"o_key": 2, "i_order": 2},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn join_with_filter_on_one_side(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT i_qty FROM items JOIN orders ON i_order = o_key WHERE o_status = 'open'",
    );
    rows.sort_by_key(|r| r["i_qty"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([{"i_qty": 10}, {"i_qty": 20}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn join_then_aggregate(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT COUNT(*) AS matches, SUM(i_qty) AS total \
         FROM items JOIN orders ON i_order = o_key",
    );

    assert_eq!(rows, vec![serde_json::json!({"matches": 3, "total": 60})]);
}

#[rstest]
fn join_output_without_key_columns(mut testing_planner: TestingPlanner) {
    // Neither side's key appears in the SELECT list, so DuckDB's
    // column-lifetime pass trims it via the join's projection maps.
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT o_status FROM items JOIN orders ON i_order = o_key WHERE i_qty > 15",
    );
    rows.sort_by_key(|r| r["o_status"].as_str().unwrap().to_string());

    assert_eq!(
        rows,
        serde_json::json!([{"o_status": "closed"}, {"o_status": "open"}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn join_sides_flip_with_table_sizes(mut testing_planner: TestingPlanner) {
    // The bigger side written second: whichever side DuckDB picks as build,
    // the join's result is the same.
    testing_planner.add_table("small", &[("s_key", Type::Int64, int64_col(vec![5, 6]))]);
    testing_planner.add_table(
        "big",
        &[(
            "b_key",
            Type::Int64,
            int64_col((0..1000).collect::<Vec<i64>>()),
        )],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT COUNT(*) AS n FROM big JOIN small ON b_key = s_key",
    );

    assert_eq!(rows, vec![serde_json::json!({"n": 2})]);
}

#[rstest]
fn non_equality_join_reports_unsupported(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let error = testing_planner
        .plan("SELECT i_qty FROM items JOIN orders ON i_order < o_key")
        .unwrap_err();

    assert!(error.to_string().contains("join"), "got: {error}");
}

#[rstest]
fn join_with_case_in_list_and_date_arithmetic(mut testing_planner: TestingPlanner) {
    // The shape of TPC-H q12: IN list, column-to-column date compares, a
    // constant-folded date + interval bound, and SUM(CASE ...) per group.
    testing_planner.add_table(
        "orders2",
        &[
            ("o_orderkey", Type::Int64, int64_col(vec![1, 2, 3])),
            (
                "o_orderpriority",
                Type::Utf8,
                str_col(vec!["1-URGENT", "3-MEDIUM", "2-HIGH"]),
            ),
        ],
    );
    testing_planner.add_table(
        "lineitem2",
        &[
            ("l_orderkey", Type::Int64, int64_col(vec![1, 2, 3, 3])),
            (
                "l_shipmode",
                Type::Utf8,
                str_col(vec!["MAIL", "SHIP", "MAIL", "AIR"]),
            ),
            (
                "l_shipdate",
                Type::Date,
                Arc::new(arrow_array::Date32Array::from(vec![8700; 4])),
            ),
            (
                "l_commitdate",
                Type::Date,
                Arc::new(arrow_array::Date32Array::from(vec![8760; 4])),
            ),
            (
                "l_receiptdate",
                Type::Date,
                Arc::new(arrow_array::Date32Array::from(vec![8790; 4])),
            ),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "select l_shipmode, \
            sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' \
                then 1 else 0 end) as high_line_count, \
            sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' \
                then 1 else 0 end) as low_line_count \
         from orders2, lineitem2 \
         where o_orderkey = l_orderkey \
           and l_shipmode in ('MAIL', 'SHIP') \
           and l_commitdate < l_receiptdate \
           and l_shipdate < l_commitdate \
           and l_receiptdate >= date '1993-11-01' \
           and l_receiptdate < date '1993-11-01' + interval '1' year \
         group by l_shipmode \
         order by l_shipmode",
    );

    assert_eq!(
        rows,
        serde_json::json!([
            {"l_shipmode": "MAIL", "high_line_count": 2, "low_line_count": 0},
            {"l_shipmode": "SHIP", "high_line_count": 0, "low_line_count": 1},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn inner_join_count_star(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT count(*) AS n FROM items JOIN orders ON i_order = o_key",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"n": 3}]).as_array().unwrap().clone()
    );
}

#[rstest]
fn left_join_keeps_a_row_that_matched_nothing(mut testing_planner: TestingPlanner) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT o_key, i_qty FROM orders LEFT JOIN items ON o_key = i_order",
    );
    rows.sort_by_key(|r| (r["o_key"].as_i64().unwrap(), r["i_qty"].as_i64()));

    // Order 3 has no item, and comes out with a NULL quantity rather than not
    // at all.
    assert_eq!(
        rows,
        serde_json::json!([
            {"o_key": 1, "i_qty": 10},
            {"o_key": 1, "i_qty": 20},
            {"o_key": 2, "i_qty": 30},
            {"o_key": 3},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn counting_a_column_over_a_left_join_skips_the_rows_it_filled_in(
    mut testing_planner: TestingPlanner,
) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT o_key, count(i_qty) AS items FROM orders LEFT JOIN items ON o_key = i_order \
         GROUP BY o_key",
    );
    rows.sort_by_key(|r| r["o_key"].as_i64().unwrap());

    // Counting a column counts its values, and the row the join filled in for
    // order 3 holds none, so that order counts zero items rather than one.
    assert_eq!(
        rows,
        serde_json::json!([
            {"o_key": 1, "items": 2},
            {"o_key": 2, "items": 1},
            {"o_key": 3, "items": 0},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn aggregating_a_column_over_a_left_join_skips_the_rows_it_filled_in(
    mut testing_planner: TestingPlanner,
) {
    add_orders_and_items(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT o_key, min(i_qty) AS smallest, sum(i_qty) AS total FROM orders \
         LEFT JOIN items ON o_key = i_order GROUP BY o_key",
    );
    rows.sort_by_key(|r| r["o_key"].as_i64().unwrap());

    // An order with no items has nothing to take an extreme or a total of, so
    // both come back NULL rather than reading the filled-in row's value.
    assert_eq!(
        rows,
        serde_json::json!([
            {"o_key": 1, "smallest": 10, "total": 30},
            {"o_key": 2, "smallest": 30, "total": 30},
            {"o_key": 3},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}
