use std::sync::Arc;

use arrow_array::{ArrayRef, StringViewArray};

use crate::common::*;
use planner::Error as PlannerError;
use planner::types::Type;
use rstest::rstest;

#[rstest]
fn filter_contains_substring(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name, b FROM example_table WHERE contains(name, 'ali')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["b"].as_i64().unwrap());

    // "alice" appears at rows with b=10 and b=50.
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "alice");
}

#[rstest]
fn filter_contains_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE contains(name, 'zzz')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_contains_matches_all(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "substrings",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["aaa", "baab", "caaac"])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT s FROM substrings WHERE contains(s, 'aa')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 3);
}

#[rstest]
fn contains_then_group_by(mut testing_planner: TestingPlanner) {
    // Filter to names containing "a" (alice, charlie, dave), then group by name.
    let results = testing_planner
        .planner
        .plan("SELECT name, COUNT(*) FROM example_table WHERE contains(name, 'a') GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice (2 rows), charlie (1), dave (1) all contain "a"; bob does not.
    assert_eq!(rows.len(), 3);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["v0"], 2);
    assert!(rows.iter().all(|r| r["key"] != "bob"));
}

#[rstest]
fn case_expression_in_projection(mut testing_planner: TestingPlanner) {
    // a = [1,2,3,4,5]; a < 3 -> 'low' (a=1,2), else 'high' (a=3,4,5).
    let results = testing_planner
        .planner
        .plan("SELECT CASE WHEN a < 3 THEN 'low' ELSE 'high' END AS c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut labels = batches_to_json(&results)
        .iter()
        .map(|r| r["col0"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    labels.sort();
    assert_eq!(labels, vec!["high", "high", "high", "low", "low"]);
}

#[rstest]
fn case_expression_multi_arm_group_key(mut testing_planner: TestingPlanner) {
    // Three-arm CASE bucketing a into lo/mid/hi, grouped and counted:
    //   a=1     -> "lo"  (1 row)
    //   a=2,3   -> "mid" (2 rows)
    //   a=4,5   -> "hi"  (2 rows)
    let results = testing_planner
        .planner
        .plan(
            "SELECT CASE WHEN a < 2 THEN 'lo' WHEN a < 4 THEN 'mid' ELSE 'hi' END AS c, COUNT(*) \
             FROM example_table GROUP BY 1",
        )
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());
    let counts: Vec<(String, i64)> = rows
        .iter()
        .map(|r| (r["key"].as_str().unwrap().to_string(), r["v0"].as_i64().unwrap()))
        .collect();
    assert_eq!(
        counts,
        vec![
            ("hi".to_string(), 2),
            ("lo".to_string(), 1),
            ("mid".to_string(), 2),
        ]
    );
}

#[rstest]
fn unsupported_scalar_function_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT lower(name) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
