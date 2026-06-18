use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use dispatch::Dispatch;

use crate::common::*;
use planner::Error as PlannerError;
use planner::Planner;
use planner::catalog::{Catalog, CreateTableRequest, Table};
use planner::types::Type;
use rstest::rstest;

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

#[rstest]
fn select_column_subset(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_all_columns(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b, c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100},
            {"a": 2, "b": 20, "c": 200},
            {"a": 3, "b": 30, "c": 300},
            {"a": 4, "b": 40, "c": 400},
            {"a": 5, "b": 50, "c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_single_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["c"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"c": 100},
            {"c": 200},
            {"c": 300},
            {"c": 400},
            {"c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_columns(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_with_dup",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 10])),
            ("b", Type::Int32, int_col(vec![10, 20, 30, 10])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM pairs_with_dup WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    // (10, 10) is excluded.
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_no_matches(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3])),
            ("b", Type::Int32, int_col(vec![1, 2, 3])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT a FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_not_equal_all_pass(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1},
            {"a": 3},
            {"a": 4},
            {"a": 5},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = 3")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([{"a": 3}]).as_array().unwrap().clone()
    );
}

#[rstest]
fn filter_equal_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = 999")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty(), "expected no rows, got: {rows:?}");
}

// ---------------------------------------------------------------------------
// OrderBy tests
// ---------------------------------------------------------------------------

#[rstest]
fn order_by_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table ORDER BY a ASC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn order_by_descending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table ORDER BY a DESC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn top_n_limit_1(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 1")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["a"], 5);
}

#[rstest]
fn top_n_limit_exceeds_row_count(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a LIMIT 100")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

#[rstest]
fn top_n_limit_2_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT b FROM example_table ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"b": 10},
            {"b": 20},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

// `SELECT *` over a filtered Top-N is exactly the late-materialization shape:
// DuckDB scans only `a`/`name` for the predicate+sort, then materializes the
// full row for the survivors. Exercises multi-column materialize + reordering
// back to schema order, plus that metadata survives the narrow projection.
#[rstest]
fn select_star_filtered_top_n_late_materializes(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT * FROM example_table WHERE name <> 'bob' ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100, "name": "alice"},
            {"a": 3, "b": 30, "c": 300, "name": "charlie"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn group_by_int_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Each value of a (1..=5) appears exactly once.
    assert_eq!(rows.len(), 5);
    for row in &rows {
        assert_eq!(row["v0"], 1);
    }
}

#[rstest]
fn group_by_string_column_with_duplicates(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name, COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice appears twice, bob/charlie/dave each once.
    assert_eq!(rows.len(), 4);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["v0"], 2);
    let bob = rows.iter().find(|r| r["key"] == "bob").unwrap();
    assert_eq!(bob["v0"], 1);
}

#[rstest]
fn group_by_count_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,1,2,2,3   x: 10,10,20,30,30,40
    // distinct x per g: g=1 -> {10,20}=2, g=2 -> {30}=1, g=3 -> {40}=1
    testing_planner.add_table(
        "gx",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 1, 2, 2, 3])),
            ("x", Type::Int32, int_col(vec![10, 10, 20, 30, 30, 40])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, COUNT(DISTINCT x) FROM gx GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["key"], 1);
    assert_eq!(rows[0]["v0"], 2);
    assert_eq!(rows[1]["key"], 2);
    assert_eq!(rows[1]["v0"], 1);
    assert_eq!(rows[2]["key"], 3);
    assert_eq!(rows[2]["v0"], 1);
}

#[rstest]
fn global_count_distinct_string(mut testing_planner: TestingPlanner) {
    // example_table.name = alice, bob, charlie, dave, alice -> 4 distinct.
    let results = testing_planner
        .planner
        .plan("SELECT COUNT(DISTINCT name) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn global_count_distinct_int(mut testing_planner: TestingPlanner) {
    // distinct {7, 8, 9, 0} = 4 (includes 0, exercising the HashOnly extractor's
    // mix64(0)==0 / empty-sentinel edge through the keys-only count path).
    testing_planner.add_table(
        "ints",
        &[("v", Type::Int32, int_col(vec![7, 7, 7, 8, 9, 9, 0, 0]))],
    );
    let results = testing_planner
        .planner
        .plan("SELECT COUNT(DISTINCT v) FROM ints")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn group_by_mixed_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,2  x: 10,10,20  v: 5,5,7
    // per g: SUM(v), COUNT(*), COUNT(DISTINCT x)
    //   g=1 -> sum=10, count=2, distinct=1 ; g=2 -> sum=7, count=1, distinct=1
    testing_planner.add_table(
        "mixed",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            ("x", Type::Int32, int_col(vec![10, 10, 20])),
            ("v", Type::Int32, int_col(vec![5, 5, 7])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, SUM(v), COUNT(*), COUNT(DISTINCT x) FROM mixed GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    // Value columns are v0/v1/v2 in DuckDB's expression order; assert the
    // multiset so the test is robust to that order (all three values differ).
    let vals = |r: &serde_json::Value| -> Vec<i64> {
        ["v0", "v1", "v2"]
            .iter()
            .map(|c| r[*c].as_i64().unwrap())
            .collect()
    };
    assert_eq!(rows[0]["key"], 1);
    let v1 = vals(&rows[0]);
    assert!(
        v1.contains(&10) && v1.contains(&2) && v1.contains(&1),
        "g=1 {v1:?}"
    );
    assert_eq!(rows[1]["key"], 2);
    let v2 = vals(&rows[1]);
    assert!(v2.contains(&7) && v2.contains(&1), "g=2 {v2:?}");
}

// ---------------------------------------------------------------------------
// Combined operator tests
// ---------------------------------------------------------------------------

#[rstest]
fn filter_then_order_by(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table WHERE a <> b ORDER BY a DESC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_then_count(mut testing_planner: TestingPlanner) {
    // Every row has a == b, so WHERE a <> b yields 0 rows.
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
            ("b", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT COUNT(*) FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["count"], 0);
}

#[rstest]
fn global_avg_is_lowered_to_sum_and_count(mut testing_planner: TestingPlanner) {
    // AVG never reaches the global aggregate operator as a dedicated kind:
    // DuckDB lowers it to sum/count plus a divide projection. After the
    // `AggKind::Avg` removal this must still compile (no
    // `UnsupportedAggregateExpression`) and produce the right average through
    // the sum + count slots. avg(a) over [1,2,3,4,5] = 3.0.
    let results = testing_planner
        .planner
        .plan("SELECT AVG(a) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let avg = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(avg.as_f64().unwrap(), 3.0);
}

#[rstest]
fn global_sum_count_avg_together(mut testing_planner: TestingPlanner) {
    // The multi-aggregate global path with a SUM, a COUNT(*) and an AVG mixed
    // (q02's shape). Must compile and run end-to-end; count = 5 and avg(b) over
    // [10,20,30,40,50] = 30.0 (sum(a) is a Decimal128, skipped by the f64 scan).
    let results = testing_planner
        .planner
        .plan("SELECT SUM(a), COUNT(*), AVG(b) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let vals: Vec<f64> = rows[0]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|v| v.as_f64())
        .collect();
    assert!(vals.contains(&5.0), "expected count 5 in {vals:?}");
    assert!(vals.contains(&30.0), "expected avg(b) 30.0 in {vals:?}");
}

#[rstest]
fn global_min_max(mut testing_planner: TestingPlanner) {
    // MIN/MAX fold the extreme over the whole column (no GROUP BY): over
    // a=[1,2,3,4,5] and b=[10,20,30,40,50], min(a)=1 and max(b)=50.
    let results = testing_planner
        .planner
        .plan("SELECT MIN(a), MAX(b) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["min"], 1);
    assert_eq!(rows[0]["max"], 50);
}

#[rstest]
fn global_string_min_max(mut testing_planner: TestingPlanner) {
    // Global string MIN/MAX (no GROUP BY) fold the byte-lexicographic extreme over
    // the whole column, emitting one Utf8View row each.
    testing_planner.add_table(
        "gs_global",
        &[(
            "s",
            Type::Utf8,
            str_col(vec!["banana", "apple", "date", "cherry"]),
        )],
    );
    let results = testing_planner
        .planner
        .plan("SELECT MIN(s), MAX(s) FROM gs_global")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["min"], "apple");
    assert_eq!(rows[0]["max"], "date");
}

#[rstest]
fn grouped_min_max(mut testing_planner: TestingPlanner) {
    // Per-group extremes through the dynamic value extractor's MIN/MAX slots.
    // g=1 -> v {10,40} (min 10, max 40); g=2 -> v {20,5} (min 5, max 20). The
    // consume fold and the partition merge are both kind-aware.
    testing_planner.add_table(
        "gv",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            ("v", Type::Int32, int_col(vec![10, 40, 20, 5])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, MIN(v), MAX(v) FROM gv GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    // A single integer key goes through the row-encoded key extractor, whose key
    // column is named `k0` (multi-key groups would add `k1`, …).
    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["k0"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["k0"], 1);
    assert_eq!(rows[0]["v0"], 10); // min
    assert_eq!(rows[0]["v1"], 40); // max
    assert_eq!(rows[1]["k0"], 2);
    assert_eq!(rows[1]["v0"], 5); // min
    assert_eq!(rows[1]["v1"], 20); // max
}

#[rstest]
fn grouped_string_min_and_max(mut testing_planner: TestingPlanner) {
    // Per-group string extremes via StringExtreme (lazy arena persist + zero-copy
    // StringView output). Homogeneous direction only, so MIN and MAX are separate
    // queries. g=1 -> {"banana","apple"}; g=2 -> {"cherry","date"}.
    testing_planner.add_table(
        "gs",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            (
                "s",
                Type::Utf8,
                str_col(vec!["banana", "apple", "cherry", "date"]),
            ),
        ],
    );

    let run = |p: &mut TestingPlanner, sql: &str| {
        let results = p
            .planner
            .plan(sql)
            .unwrap()
            .compile(p.dispatcher())
            .unwrap()
            .collect()
            .unwrap();
        let mut rows = batches_to_json(&results);
        rows.sort_by_key(|r| r["k0"].as_i64().unwrap());
        rows
    };

    let mins = run(&mut testing_planner, "SELECT g, MIN(s) FROM gs GROUP BY g");
    assert_eq!(mins[0]["v0"], "apple");
    assert_eq!(mins[1]["v0"], "cherry");

    let maxes = run(&mut testing_planner, "SELECT g, MAX(s) FROM gs GROUP BY g");
    assert_eq!(maxes[0]["v0"], "banana");
    assert_eq!(maxes[1]["v0"], "date");
}

/// A string extreme mixed with the opposite direction (or with integer
/// aggregates) routes to the runtime `Dynamic` value (widened to `i128` so the
/// `ArenaKey` cell fits) rather than the homogeneous `Compiled` tuple. Both the
/// string MIN+MAX mix and a string-extreme-beside-an-integer-extreme mix must
/// compile and return the right per-group values.
#[rstest]
fn mixed_string_extreme_via_dynamic(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "gs2",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            (
                "s",
                Type::Utf8,
                str_col(vec!["banana", "apple", "cherry", "date"]),
            ),
            ("v", Type::Int32, int_col(vec![10, 30, 5, 20])),
        ],
    );

    // MIN(s) + MAX(s): a string mix of opposite directions in one value.
    let results = testing_planner
        .planner
        .plan("SELECT g, MIN(s), MAX(s) FROM gs2 GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["k0"].as_i64().unwrap());
    assert_eq!(rows[0]["v0"], "apple"); // g=1 min
    assert_eq!(rows[0]["v1"], "banana"); // g=1 max
    assert_eq!(rows[1]["v0"], "cherry"); // g=2 min
    assert_eq!(rows[1]["v1"], "date"); // g=2 max

    // MIN(s) (string) + MAX(v) (integer): a string extreme beside a numeric one.
    let mixed = testing_planner
        .planner
        .plan("SELECT g, MIN(s), MAX(v) FROM gs2 GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    let mut mixed_rows = batches_to_json(&mixed);
    mixed_rows.sort_by_key(|r| r["k0"].as_i64().unwrap());
    assert_eq!(mixed_rows[0]["v0"], "apple"); // g=1 min(s)
    assert_eq!(mixed_rows[0]["v1"], 30); // g=1 max(v)
    assert_eq!(mixed_rows[1]["v0"], "cherry"); // g=2 min(s)
    assert_eq!(mixed_rows[1]["v1"], 20); // g=2 max(v)
}

#[rstest]
fn filter_then_top_n(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["a"], 5);
    assert_eq!(rows[1]["a"], 4);
}

/// Build a single-column `events` table whose group sizes are all distinct, so
/// `COUNT(*)` orderings are unambiguous: g=1 -> 4, g=2 -> 3, g=3 -> 2, g=4 -> 1.
fn add_events_table(testing_planner: &TestingPlanner) {
    testing_planner.add_table(
        "events",
        &[(
            "g",
            Type::Int32,
            int_col(vec![1, 1, 1, 1, 2, 2, 2, 3, 3, 4]),
        )],
    );
}

// `GROUP BY … ORDER BY COUNT(*) DESC LIMIT k OFFSET m`: the `OFFSET` window must
// survive the group → Top-N pipeline. DESC by count is g=1,2,3,4; skipping the
// top 2 and taking 2 leaves g=3 then g=4.
#[rstest]
fn group_order_by_count_desc_with_offset(mut testing_planner: TestingPlanner) {
    add_events_table(&testing_planner);
    let results = testing_planner
        .planner
        .plan(
            "SELECT g, COUNT(*) FROM events \
             GROUP BY g ORDER BY COUNT(*) DESC LIMIT 2 OFFSET 2",
        )
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let keys: Vec<i64> = rows.iter().map(|r| r["key"].as_i64().unwrap()).collect();
    assert_eq!(keys, vec![3, 4]);
}

// `GROUP BY … ORDER BY <group key> ASC LIMIT k OFFSET m`: ordering by the group
// key (not an aggregate) with an `OFFSET`. Keys ascending are 1,2,3,4; skipping
// 1 and taking 2 leaves g=2 then g=3.
#[rstest]
fn group_order_by_key_asc_with_offset(mut testing_planner: TestingPlanner) {
    add_events_table(&testing_planner);
    let results = testing_planner
        .planner
        .plan(
            "SELECT g, COUNT(*) FROM events \
             GROUP BY g ORDER BY g ASC LIMIT 2 OFFSET 1",
        )
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let keys: Vec<i64> = rows.iter().map(|r| r["key"].as_i64().unwrap()).collect();
    assert_eq!(keys, vec![2, 3]);
}

#[derive(Debug, Default)]
struct RecordingCatalog {
    created_tables: Mutex<Vec<CreateTableRequest>>,
}

impl Catalog for RecordingCatalog {
    fn table(&self, _name: &str) -> Option<Box<dyn Table>> {
        None
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &dispatch::DataFlowDispatcher,
    ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
        self.created_tables.lock().unwrap().push(request);
        // CREATE TABLE yields no rows; a real catalog would commit the table here.
        Ok(dispatch::RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoRowsNullary::default()),
        ))
    }
}

/// A nullary that emits nothing and finishes immediately — backs the empty
/// result of this test's `CREATE TABLE`.
#[derive(Default)]
struct NoRowsNullary {
    ran: bool,
}

impl dispatch::NullaryFactory<RecordBatch> for NoRowsNullary {
    type Nullary = NoRowsNullary;

    fn build_nullary(self) -> NoRowsNullary {
        self
    }
}

impl dispatch::Nullary<RecordBatch> for NoRowsNullary {
    fn run<S: dispatch::Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::NullaryResult<dispatch::WorkStatus> {
        if self.ran {
            return Ok(dispatch::WorkStatus::Pending);
        }
        self.ran = true;
        Ok(dispatch::WorkStatus::Ran)
    }

    fn finish<S: dispatch::Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::NullaryResult<bool> {
        Ok(self.ran)
    }
}

// CREATE TABLE tests use a custom recording catalog (which the shared
// `TestCatalog` can't impersonate without adding state we don't otherwise
// need), so they construct their own `Planner` rather than going through the
// `testing_planner` fixture.
#[test]
fn create_table_calls_catalog_once() {
    let dispatch = Dispatch::spin_up(1, 32);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 2);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(created[0].columns[1].name, "name");
    assert_eq!(created[0].columns[1].col_type, Type::Utf8);
    assert!(created[0].options.is_empty());
    assert!(!created[0].if_not_exists);
}

#[test]
fn create_table_passes_with_options_to_catalog() {
    let dispatch = Dispatch::spin_up(1, 32);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (existing_path='/asdf')")
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 1);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(
        created[0].options.get("existing_path").map(String::as_str),
        Some("/asdf")
    );
}

#[rstest]
fn unsupported_aggregate_returns_error(mut testing_planner: TestingPlanner) {
    // `stddev` has no pivot lowering (unlike SUM/COUNT/MIN/MAX/AVG), so it must
    // surface as a plan-conversion error rather than silently mis-aggregating.
    let result = testing_planner
        .planner
        .plan("SELECT STDDEV(b) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
