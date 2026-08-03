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

/// The q18 shape: a probe side wide and long enough that DuckDB builds the hash
/// table from the subquery's keys, which is what makes the semi join a
/// probe-side one. Were the sizes the other way round it would flip the
/// children and ask for a build-side semi join instead.
fn add_wide_orders_and_repeated_keys(planner: &TestingPlanner) {
    planner.add_table(
        "wide_orders",
        &[
            ("w_key", Type::Int64, int64_col((0..1000).collect())),
            (
                "w_total",
                Type::Int64,
                int64_col((0..1000).map(|key| key * 10).collect()),
            ),
        ],
    );
    planner.add_table(
        "big_items",
        &[("g_order", Type::Int64, int64_col(vec![5, 5, 5, 6]))],
    );
}

#[rstest]
fn semi_join_keeps_a_probe_row_that_matches_at_all(mut testing_planner: TestingPlanner) {
    add_wide_orders_and_repeated_keys(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT w_key FROM wide_orders WHERE w_key IN (SELECT g_order FROM big_items)",
    );
    rows.sort_by_key(|r| r["w_key"].as_i64().unwrap());

    // Order 5 is named by three items and still comes out once; every order no
    // item names does not come out at all.
    assert_eq!(
        rows,
        serde_json::json!([{"w_key": 5}, {"w_key": 6}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn a_semi_join_emits_the_probe_columns_asked_for(mut testing_planner: TestingPlanner) {
    add_wide_orders_and_repeated_keys(&testing_planner);

    let mut rows = run(
        &mut testing_planner,
        "SELECT w_total FROM wide_orders WHERE w_key IN (SELECT g_order FROM big_items)",
    );
    rows.sort_by_key(|r| r["w_total"].as_i64().unwrap());

    // The key the join filters on is not in the output at all, only the column
    // asked for.
    assert_eq!(
        rows,
        serde_json::json!([{"w_total": 50}, {"w_total": 60}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn aggregating_over_a_semi_join_counts_each_probe_row_once(mut testing_planner: TestingPlanner) {
    add_wide_orders_and_repeated_keys(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT COUNT(*) AS orders, SUM(w_total) AS total FROM wide_orders \
         WHERE w_key IN (SELECT g_order FROM big_items)",
    );

    // Two orders match, though one of them is named by three items.
    assert_eq!(rows, vec![serde_json::json!({"orders": 2, "total": 110})]);
}

#[rstest]
fn a_join_on_two_equality_conditions_matches_pairwise(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "assignments",
        &[
            ("a_person", Type::Int64, int64_col(vec![1, 1, 2])),
            ("a_day", Type::Int64, int64_col(vec![5, 6, 5])),
            ("a_task", Type::Utf8, str_col(vec!["a", "b", "c"])),
        ],
    );
    testing_planner.add_table(
        "shifts",
        &[
            ("s_person", Type::Int64, int64_col(vec![1, 1, 2])),
            ("s_day", Type::Int64, int64_col(vec![5, 6, 6])),
            ("s_room", Type::Utf8, str_col(vec!["r1", "r2", "r3"])),
        ],
    );

    let mut rows = run(
        &mut testing_planner,
        "SELECT a_task, s_room FROM assignments \
         JOIN shifts ON a_person = s_person AND a_day = s_day",
    );
    rows.sort_by_key(|r| r["a_task"].as_str().unwrap().to_string());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a_task": "a", "s_room": "r1"},
            {"a_task": "b", "s_room": "r2"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn a_join_on_a_string_key_matches_exact_text(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "people",
        &[
            ("p_city", Type::Utf8, str_col(vec!["lyon", "oslo", "kyiv"])),
            ("p_name", Type::Utf8, str_col(vec!["ana", "bo", "cy"])),
        ],
    );
    testing_planner.add_table(
        "cities",
        &[
            ("c_name", Type::Utf8, str_col(vec!["oslo", "lyon"])),
            ("c_pop", Type::Int64, int64_col(vec![700, 500])),
        ],
    );

    let mut rows = run(
        &mut testing_planner,
        "SELECT p_name, c_pop FROM people JOIN cities ON p_city = c_name",
    );
    rows.sort_by_key(|r| r["p_name"].as_str().unwrap().to_string());

    assert_eq!(
        rows,
        serde_json::json!([
            {"p_name": "ana", "c_pop": 500},
            {"p_name": "bo", "c_pop": 700},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn a_join_on_an_int_and_a_string_condition_matches_pairwise(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "readings",
        &[
            ("r_sensor", Type::Int64, int64_col(vec![1, 1, 2])),
            ("r_unit", Type::Utf8, str_col(vec!["c", "f", "c"])),
            ("r_value", Type::Int64, int64_col(vec![20, 68, 21])),
        ],
    );
    testing_planner.add_table(
        "limits",
        &[
            ("l_sensor", Type::Int64, int64_col(vec![1, 2])),
            ("l_unit", Type::Utf8, str_col(vec!["f", "c"])),
            ("l_max", Type::Int64, int64_col(vec![100, 30])),
        ],
    );

    let mut rows = run(
        &mut testing_planner,
        "SELECT r_value, l_max FROM readings \
         JOIN limits ON r_sensor = l_sensor AND r_unit = l_unit",
    );
    rows.sort_by_key(|r| r["r_value"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"r_value": 21, "l_max": 30},
            {"r_value": 68, "l_max": 100},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}
