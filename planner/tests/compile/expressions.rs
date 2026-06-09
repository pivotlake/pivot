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
fn group_by_minute_of_timestamp(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    // EventTime as epoch seconds. Minute-of-hour = (t mod 3600) / 60:
    //   0 -> 0, 90 -> 1, 150 -> 2, 3690 -> 1 (3690 mod 3600 = 90).
    testing_planner.add_table(
        "events",
        &[(
            "EventTime",
            Type::Timestamp,
            Arc::new(Int64Array::from(vec![0i64, 90, 150, 3690])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM events GROUP BY m")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Minutes 0 and 2 occur once; minute 1 occurs twice.
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.iter().map(|r| r["key"].as_i64().unwrap()).collect::<Vec<_>>(), vec![0, 1, 2]);
    let minute_one = rows.iter().find(|r| r["key"] == 1).unwrap();
    assert_eq!(minute_one["v0"], 2);
}

#[rstest]
fn unsupported_scalar_function_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT lower(name) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
