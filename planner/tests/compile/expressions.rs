use std::sync::Arc;

use arrow_array::{ArrayRef, StringViewArray};

use crate::common::*;
use planner::Error as PlannerError;
use planner::types::Type;

#[test]
fn filter_contains_substring() {
    init();
    let mut planner = string_table();

    let results = planner
        .plan("SELECT name, value FROM test WHERE contains(name, 'ali')")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["value"].as_i64().unwrap());

    // "alice" contains "ali" — appears at rows with value 10 and 50
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "alice");
}

#[test]
fn filter_contains_no_match() {
    init();
    let mut planner = string_table();

    let results = planner
        .plan("SELECT name FROM test WHERE contains(name, 'zzz')")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[test]
fn filter_contains_matches_all() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["aaa", "baab", "caaac"])) as ArrayRef,
        )],
    );

    let results = planner
        .plan("SELECT s FROM test WHERE contains(s, 'aa')")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 3);
}

#[test]
fn contains_then_group_by() {
    init();
    let mut planner = string_table();

    // Filter to names containing "a" (alice, charlie, dave), then group by name
    let results = planner
        .plan("SELECT name, COUNT(*) FROM test WHERE contains(name, 'a') GROUP BY name")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice (2 rows), charlie (1), dave (1) all contain "a"; bob does not
    assert_eq!(rows.len(), 3);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["value"], 2);
    assert!(rows.iter().all(|r| r["key"] != "bob"));
}

#[test]
fn unsupported_scalar_function_returns_error() {
    init();
    let mut planner = string_table();

    let result = planner.plan("SELECT lower(name) FROM test");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
