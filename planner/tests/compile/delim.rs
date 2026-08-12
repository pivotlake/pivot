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

/// The q20 shape: an IN whose outer table is far smaller than its subquery
/// side, which DuckDB therefore flips to a RIGHT_SEMI (build-side semi)
/// join, over a subquery that nests a second IN and a correlated aggregate
/// (a delim join). The tables keep realistic relative sizes: shrinking them
/// evenly makes DuckDB flip the delim join too, a shape the planner rejects.
#[rstest]
fn a_small_outer_in_over_a_correlated_aggregate_runs_build_side_semi(
    mut testing_planner: TestingPlanner,
) {
    testing_planner.add_table(
        "nation",
        &[
            ("n_nationkey", Type::Int64, int64_col((0..25).collect())),
            (
                "n_name",
                Type::Utf8,
                str_col(
                    (0..25)
                        .map(|i| if i == 5 { "CANADA" } else { "OTHER" })
                        .collect(),
                ),
            ),
        ],
    );
    testing_planner.add_table(
        "supplier",
        &[
            ("s_suppkey", Type::Int64, int64_col((0..100).collect())),
            (
                "s_nationkey",
                Type::Int64,
                int64_col((0..100).map(|i| i % 25).collect()),
            ),
        ],
    );
    testing_planner.add_table(
        "part",
        &[
            ("p_partkey", Type::Int64, int64_col((0..2000).collect())),
            (
                "p_name",
                Type::Utf8,
                str_col(
                    (0..2000)
                        .map(|i| {
                            if i % 100 == 0 {
                                "forest green"
                            } else {
                                "misty rose"
                            }
                        })
                        .collect(),
                ),
            ),
        ],
    );
    testing_planner.add_table(
        "partsupp",
        &[
            (
                "ps_partkey",
                Type::Int64,
                int64_col((0..8000).map(|i| i % 2000).collect()),
            ),
            (
                "ps_suppkey",
                Type::Int64,
                int64_col((0..8000).map(|i| i % 97).collect()),
            ),
            (
                "ps_availqty",
                Type::Int64,
                int64_col((0..8000).map(|i| i % 9999).collect()),
            ),
        ],
    );
    testing_planner.add_table(
        "lineitem",
        &[
            (
                "l_partkey",
                Type::Int64,
                int64_col((0..60000).map(|i| i % 2000).collect()),
            ),
            (
                "l_suppkey",
                Type::Int64,
                int64_col((0..60000).map(|i| i % 97).collect()),
            ),
            (
                "l_quantity",
                Type::Int64,
                int64_col((0..60000).map(|i| i % 50 + 1).collect()),
            ),
            (
                "l_shipdate",
                Type::Int64,
                int64_col((0..60000).map(|i| i % 2557).collect()),
            ),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT s_suppkey FROM supplier, nation \
         WHERE s_suppkey IN ( \
           SELECT ps_suppkey FROM partsupp \
           WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%') \
             AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem \
                                WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey \
                                  AND l_shipdate >= 730 AND l_shipdate < 1095)) \
           AND s_nationkey = n_nationkey AND n_name = 'CANADA' \
         ORDER BY s_suppkey",
    );

    // Verified against DuckDB on the same data: suppliers 5, 30, and 80 hold
    // forest stock above half of the window's delivered quantity and sit in
    // the filtered nation; supplier 55 is in the nation but holds none.
    assert_eq!(
        rows,
        serde_json::json!([{"s_suppkey": 5}, {"s_suppkey": 30}, {"s_suppkey": 80}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// A tiny outer table against a big subquery table: DuckDB flips the delim
/// join (the outer rows become the RHS, the dedup source, and the build),
/// and the join type arrives mirrored. Request 5's part is NULL.
fn add_requests_and_history(planner: &TestingPlanner) {
    planner.add_table(
        "requests",
        &[
            ("r_id", Type::Int64, int64_col(vec![1, 2, 3, 4, 5])),
            (
                "r_part",
                Type::Int64,
                Arc::new(Int64Array::from(vec![
                    Some(10),
                    Some(10),
                    Some(20),
                    Some(30),
                    None,
                ])),
            ),
            ("r_grp", Type::Int64, int64_col(vec![1, 2, 1, 2, 1])),
            ("r_qty", Type::Int64, int64_col(vec![5, 60, 7, 1, 9])),
        ],
    );
    planner.add_table(
        "history",
        &[
            (
                "h_part",
                Type::Int64,
                int64_col((0..1000).map(|i| i % 25 * 10).collect()),
            ),
            (
                "h_grp",
                Type::Int64,
                int64_col((0..1000).map(|i| i % 3).collect()),
            ),
            (
                "h_qty",
                Type::Int64,
                int64_col((0..1000).map(|i| i % 100).collect()),
            ),
        ],
    );
}

/// A correlated EXISTS whose outer side is the small one arrives as a flipped
/// RIGHT_SEMI delim join, running as a build-side semi over the outer rows.
/// Every non-NULL part holds history above the threshold; the NULL-part
/// request matches nothing and drops.
#[rstest]
fn a_small_outer_exists_runs_as_a_flipped_semi_delim_join(mut testing_planner: TestingPlanner) {
    add_requests_and_history(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT r_id FROM requests \
         WHERE EXISTS (SELECT 1 FROM history WHERE h_part = r_part AND h_qty > 70) \
         ORDER BY r_id",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"r_id": 1}, {"r_id": 2}, {"r_id": 3}, {"r_id": 4}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// The NOT EXISTS mirror arrives as a flipped RIGHT_ANTI delim join, running
/// as a build-side anti over the outer rows. Part 10 peaks at quantity 76 and
/// so has nothing above the threshold, and the NULL-part request finds no
/// history at all (its correlated equality compares against NULL); the anti
/// join keeps those, including the tuple-less NULL-keyed build row.
#[rstest]
fn a_small_outer_not_exists_runs_as_a_flipped_anti_delim_join(mut testing_planner: TestingPlanner) {
    add_requests_and_history(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT r_id FROM requests \
         WHERE NOT EXISTS (SELECT 1 FROM history WHERE h_part = r_part AND h_qty > 76) \
         ORDER BY r_id",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"r_id": 1}, {"r_id": 2}, {"r_id": 5}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// An EXISTS and a NOT EXISTS over the same big table, correlated on two
/// columns, decorrelate into two nested flipped delim joins (RIGHT_ANTI over
/// RIGHT_SEMI) each de-duplicating both. Per (part, group) the history peaks
/// at quantity 76, 76, 77, and 78 for the four keyed requests, so only the
/// part-10 requests pass both thresholds.
#[rstest]
fn nested_flipped_semi_and_anti_delim_joins_run_the_double_exists(
    mut testing_planner: TestingPlanner,
) {
    add_requests_and_history(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT r_id FROM requests \
         WHERE EXISTS (SELECT 1 FROM history \
                       WHERE h_part = r_part AND h_grp = r_grp AND h_qty > 70) \
           AND NOT EXISTS (SELECT 1 FROM history \
                           WHERE h_part = r_part AND h_grp = r_grp AND h_qty > 76) \
         ORDER BY r_id",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"r_id": 1}, {"r_id": 2}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// The q21 shape: the EXISTS carries a non-equality correlated condition
/// beside its key (`h_qty <> r_qty`), which reaches the subquery side's inner
/// join as a comparison-form condition and must ride it as a residual. The
/// quantity column joins the correlation key in the two-column dedup.
#[rstest]
fn a_correlated_not_equal_rides_the_subquery_join_as_a_residual(
    mut testing_planner: TestingPlanner,
) {
    add_requests_and_history(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT r_id FROM requests \
         WHERE EXISTS (SELECT 1 FROM history \
                       WHERE h_part = r_part AND h_qty <> r_qty AND h_qty > 70) \
           AND NOT EXISTS (SELECT 1 FROM history \
                           WHERE h_part = r_part AND h_qty <> r_qty AND h_qty > 76) \
         ORDER BY r_id",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"r_id": 1}, {"r_id": 2}])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// The q20 shape at small scale, where DuckDB flips the delim join too: a
/// two-key correlated aggregate arrives as a flipped RIGHT delim join (a
/// build-side outer over the stock rows) under a build-side semi join over
/// the suppliers. One stock row's (part, supplier) pair has no deliveries at
/// all, so its aggregate pads to NULL and the comparison drops it, which is
/// what keeps supplier delta out.
#[rstest]
fn a_flipped_two_key_aggregate_delim_join_pads_its_build_side(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "suppliers",
        &[
            ("sup_key", Type::Int64, int64_col(vec![10, 20, 30, 40])),
            (
                "sup_name",
                Type::Utf8,
                str_col(vec!["alpha", "beta", "gamma", "delta"]),
            ),
        ],
    );
    testing_planner.add_table(
        "catalog_parts",
        &[
            ("p_key", Type::Int64, int64_col((1..=20).collect())),
            (
                "p_flag",
                Type::Utf8,
                str_col(
                    (1..=20)
                        .map(|i| if i % 2 == 0 { "a" } else { "b" })
                        .collect(),
                ),
            ),
        ],
    );
    let mut stock_parts: Vec<i64> = (0..60).map(|i| i % 20 + 1).collect();
    let mut stock_suppliers: Vec<i64> = (0..60).map(|i| (i % 3 + 1) * 10).collect();
    let mut stock_quantities: Vec<i64> = (0..60).map(|i| i * 7 % 60).collect();
    stock_parts.push(2);
    stock_suppliers.push(40);
    stock_quantities.push(99);
    testing_planner.add_table(
        "stock",
        &[
            ("ps_part", Type::Int64, int64_col(stock_parts)),
            ("ps_sup", Type::Int64, int64_col(stock_suppliers)),
            ("ps_avail", Type::Int64, int64_col(stock_quantities)),
        ],
    );
    testing_planner.add_table(
        "deliveries",
        &[
            (
                "d_part",
                Type::Int64,
                int64_col((0..120).map(|i| i % 20 + 1).collect()),
            ),
            (
                "d_sup",
                Type::Int64,
                int64_col((0..120).map(|i| (i % 3 + 1) * 10).collect()),
            ),
            (
                "d_qty",
                Type::Int64,
                int64_col((0..120).map(|i| i * 13 % 50).collect()),
            ),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT sup_name FROM suppliers WHERE sup_key IN ( \
           SELECT ps_sup FROM stock \
           WHERE ps_part IN (SELECT p_key FROM catalog_parts WHERE p_flag = 'a') \
             AND ps_avail > (SELECT 0.5 * sum(d_qty) FROM deliveries \
                             WHERE d_part = ps_part AND d_sup = ps_sup)) \
         ORDER BY sup_name",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"sup_name": "alpha"}, {"sup_name": "beta"}, {"sup_name": "gamma"}])
            .as_array()
            .unwrap()
            .clone()
    );
}
