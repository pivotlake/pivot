//! End-to-end delim join tests: correlated subqueries through DuckDB's
//! decorrelation, the planner's delim translation, and dispatch execution.

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

fn add_parts_and_sales(planner: &TestingPlanner) {
    planner.add_table(
        "parts",
        &[
            ("p_key", Type::Int64, int64_col(vec![1, 2, 3])),
            ("p_region", Type::Int64, int64_col(vec![10, 10, 20])),
            ("p_flag", Type::Utf8, str_col(vec!["a", "a", "b"])),
        ],
    );
    planner.add_table(
        "sales",
        &[
            ("s_part", Type::Int64, int64_col(vec![1, 1, 1, 2, 2, 3])),
            (
                "s_region",
                Type::Int64,
                int64_col(vec![10, 10, 10, 10, 20, 20]),
            ),
            ("s_qty", Type::Int64, int64_col(vec![10, 20, 60, 10, 30, 5])),
        ],
    );
}

/// The TPC-H q17 shape: a correlated scalar aggregate subquery under a filtered
/// outer join, which DuckDB decorrelates into a LEFT delim join. Part 1's
/// average quantity is 30 so only its 10-quantity sale passes; part 2's is 20
/// so none of its sales do; part 3 is filtered out.
#[rstest]
fn correlated_scalar_subquery_filters_through_a_delim_join(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT sum(s_qty) AS total FROM sales, parts \
         WHERE p_key = s_part AND p_flag = 'a' \
           AND s_qty < (SELECT 0.5 * avg(s2.s_qty) FROM sales s2 WHERE s2.s_part = p_key)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"total": 10}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// A subquery correlated on two columns de-duplicates a two-column key. The
/// average for (part 1, region 10) is 30 so its 10- and 20-quantity sales
/// pass; (part 2, region 10) averages 10, rejecting both of part 2's sales
/// (its 30-quantity sale sits in region 20, outside the correlated pair).
#[rstest]
fn two_column_correlation_deduplicates_both_keys(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT sum(s_qty) AS total FROM sales, parts \
         WHERE p_key = s_part AND p_flag = 'a' \
           AND s_qty < (SELECT avg(s2.s_qty) FROM sales s2 \
                        WHERE s2.s_part = p_key AND s2.s_region = p_region)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"total": 30}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// The TPC-H q04 shape: a correlated EXISTS under a filtered outer side, which
/// DuckDB decorrelates into a SEMI delim join. Only part 1 has a sale above 35.
#[rstest]
fn filtered_correlated_exists_runs_as_a_semi_delim_join(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT p_key FROM parts WHERE p_flag = 'a' \
           AND EXISTS (SELECT 1 FROM sales WHERE s_part = p_key AND s_qty > 35)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"p_key": 1}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// Two correlated EXISTS in one WHERE decorrelate into two delim joins, the
/// second taking the first's output as its outer side. Only part 1 has both a
/// sale above 35 and one below 15.
#[rstest]
fn chained_correlated_exists_run_as_two_delim_joins(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT p_key FROM parts WHERE p_flag = 'a' \
           AND EXISTS (SELECT 1 FROM sales WHERE s_part = p_key AND s_qty > 35) \
           AND EXISTS (SELECT 1 FROM sales WHERE s_part = p_key AND s_qty < 15)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"p_key": 1}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// A correlated EXISTS inside a correlated scalar subquery decorrelates into a
/// delim join nested in another's subquery side, so the inner subtree holds
/// delim scans of both joins and each must resolve to its own dedup. The
/// average only counts sales whose region has some sale above 55 (region 10),
/// leaving part 1 at avg 30 (sales 10 and 20 pass) and part 2 at avg 10
/// (nothing passes).
#[rstest]
fn nested_correlated_subqueries_resolve_their_own_delim_scans(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT sum(s_qty) AS total FROM sales, parts \
         WHERE p_key = s_part AND p_flag = 'a' \
           AND s_qty < (SELECT avg(s2.s_qty) FROM sales s2 WHERE s2.s_part = p_key \
                        AND EXISTS (SELECT 1 FROM sales s3 \
                                    WHERE s3.s_region = s2.s_region AND s3.s_qty > 55))",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"total": 30}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// A text correlation column de-duplicates and joins as a string key. Flag 'a'
/// sales average 30, passing the 10- and 20-quantity sales of the two
/// region-10 parts.
#[rstest]
fn string_correlation_key_deduplicates_text(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);
    testing_planner.add_table(
        "labeled_sales",
        &[
            ("l_part", Type::Int64, int64_col(vec![1, 1, 2, 2, 3])),
            ("l_flag", Type::Utf8, str_col(vec!["a", "a", "b", "a", "b"])),
            ("l_qty", Type::Int64, int64_col(vec![10, 30, 20, 50, 6])),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT sum(l_qty) AS total FROM labeled_sales, parts \
         WHERE p_key = l_part AND p_region = 10 \
           AND l_qty < (SELECT avg(l2.l_qty) FROM labeled_sales l2 WHERE l2.l_flag = p_flag)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"total": 30}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// An outer row whose correlation column is NULL cannot match any subquery
/// answer; the LEFT delim join must still emit it, with a NULL answer, rather
/// than lose it.
#[rstest]
fn null_correlation_key_pads_instead_of_matching(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);
    testing_planner.add_table(
        "nparts",
        &[
            ("np_key", Type::Int64, int64_col(vec![1, 2, 3, 4])),
            (
                "np_region",
                Type::Int64,
                Arc::new(Int64Array::from(vec![Some(10), Some(20), None, Some(10)])),
            ),
            ("np_flag", Type::Utf8, str_col(vec!["a", "a", "a", "b"])),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT np_key, (SELECT avg(s_qty) FROM sales WHERE s_region = np_region) AS region_avg \
         FROM nparts WHERE np_flag = 'a' ORDER BY np_key",
    );

    assert_eq!(
        rows,
        serde_json::json!([
            {"np_key": 1, "region_avg": 25.0},
            {"np_key": 2, "region_avg": 17.5},
            {"np_key": 3},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

/// The NOT EXISTS mirror of the q04 shape: DuckDB decorrelates it into an
/// ANTI delim join, which keeps a probing row only when no subquery answer
/// exists. Part 1 has a sale above 35, part 2 does not, part 3 is filtered.
#[rstest]
fn filtered_not_exists_runs_as_an_anti_delim_join(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT p_key FROM parts WHERE p_flag = 'a' \
           AND NOT EXISTS (SELECT 1 FROM sales WHERE s_part = p_key AND s_qty > 35)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"p_key": 2}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// An outer row whose correlation column is NULL finds no subquery answer
/// (its correlated equality compares against NULL), and NOT EXISTS over an
/// empty subquery holds, so the ANTI delim join must keep the row.
#[rstest]
fn null_correlation_key_survives_a_not_exists(mut testing_planner: TestingPlanner) {
    add_parts_and_sales(&testing_planner);
    testing_planner.add_table(
        "nparts",
        &[
            ("np_key", Type::Int64, int64_col(vec![1, 2, 3, 4])),
            (
                "np_region",
                Type::Int64,
                Arc::new(Int64Array::from(vec![Some(10), Some(20), None, Some(10)])),
            ),
            ("np_flag", Type::Utf8, str_col(vec!["a", "a", "a", "b"])),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT np_key FROM nparts WHERE np_flag = 'a' \
           AND NOT EXISTS (SELECT 1 FROM sales WHERE s_region = np_region AND s_qty > 25) \
         ORDER BY np_key",
    );

    // Regions 10 and 20 both hold a sale above 25, so only the NULL-region
    // part survives.
    assert_eq!(
        rows,
        serde_json::json!([{"np_key": 3}])
            .as_array()
            .unwrap()
            .clone()
    );
}
