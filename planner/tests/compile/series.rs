//! End-to-end tests of `range` and `generate_series` over timestamps.

use crate::common::*;
use rstest::rstest;
use serde_json::{Value, json};

/// The values of the result's one column, in the order the rows came out.
fn single_column_values(rows: &[Value]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            let columns = row.as_object().unwrap();
            assert_eq!(columns.len(), 1);
            columns.values().next().unwrap().clone()
        })
        .collect()
}

#[rstest]
fn generate_series_steps_between_timestamps_by_an_interval(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT * FROM generate_series(\
            TIMESTAMP '2026-11-12 10:15:00', TIMESTAMP '2026-11-12 10:15:20', \
            INTERVAL '10 seconds') ORDER BY 1",
    );

    assert_eq!(
        rows,
        vec![
            json!({"generate_series": "2026-11-12T10:15:00"}),
            json!({"generate_series": "2026-11-12T10:15:10"}),
            json!({"generate_series": "2026-11-12T10:15:20"}),
        ]
    );
}

#[rstest]
fn range_stops_before_the_last_timestamp(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT * FROM range(\
            TIMESTAMP '2026-11-12 10:15:00', TIMESTAMP '2026-11-12 10:15:20', \
            INTERVAL '10 seconds') ORDER BY 1",
    );

    assert_eq!(
        rows,
        vec![
            json!({"range": "2026-11-12T10:15:00"}),
            json!({"range": "2026-11-12T10:15:10"}),
        ]
    );
}

#[rstest]
fn timestamptz_bounds_give_a_timestamptz_series(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT * FROM generate_series(\
            TIMESTAMPTZ '2026-11-12 10:15:00+00', TIMESTAMPTZ '2026-11-12 12:15:00+00', \
            INTERVAL '1 hour') ORDER BY 1",
    );

    assert_eq!(
        single_column_values(&rows),
        vec![
            json!("2026-11-12T10:15:00Z"),
            json!("2026-11-12T11:15:00Z"),
            json!("2026-11-12T12:15:00Z"),
        ]
    );
}

/// A month step lands on the same day of the next month, or on that month's
/// last day when it is shorter, and each value steps from the one before it.
#[rstest]
#[case::forwards(
    "TIMESTAMP '2024-01-31', TIMESTAMP '2024-03-31', INTERVAL '1 month'",
    &["2024-01-31T00:00:00", "2024-02-29T00:00:00", "2024-03-29T00:00:00"]
)]
#[case::backwards(
    "TIMESTAMP '2024-03-31', TIMESTAMP '2024-01-01', INTERVAL '-1 month'",
    &["2024-01-29T00:00:00", "2024-02-29T00:00:00", "2024-03-31T00:00:00"]
)]
#[case::with_days_and_time(
    "TIMESTAMP '2024-01-31', TIMESTAMP '2024-04-05', INTERVAL '1 month 1 day 12 hours'",
    &["2024-01-31T00:00:00", "2024-03-01T12:00:00", "2024-04-03T00:00:00"]
)]
fn month_steps_follow_the_calendar(
    mut testing_planner: TestingPlanner,
    #[case] arguments: &str,
    #[case] expected_ascending: &[&str],
) {
    let rows = run(
        &mut testing_planner,
        &format!("SELECT * FROM generate_series({arguments}) ORDER BY 1"),
    );

    let expected_ascending: Vec<Value> = expected_ascending
        .iter()
        .map(|value| json!(value))
        .collect();
    assert_eq!(single_column_values(&rows), expected_ascending);
}

#[rstest]
#[case::integer_bound("generate_series(1, NULL)")]
#[case::timestamp_bound("generate_series(TIMESTAMP '2026-01-01', NULL, INTERVAL '1 day')")]
#[case::interval_step("range(TIMESTAMP '2026-01-01', TIMESTAMP '2026-01-03', NULL)")]
fn a_null_argument_gives_no_rows(mut testing_planner: TestingPlanner, #[case] call: &str) {
    let rows = run(
        &mut testing_planner,
        &format!("SELECT count(*) AS row_count FROM {call}"),
    );

    assert_eq!(rows, vec![json!({"row_count": 0})]);
}

#[rstest]
#[case::zero_step(
    "TIMESTAMP '2026-01-01', TIMESTAMP '2026-01-02', INTERVAL '0 seconds'",
    "step must not be zero"
)]
#[case::step_in_two_directions(
    "TIMESTAMP '2026-01-01', TIMESTAMP '2027-01-01', INTERVAL '1 month -1 day'",
    "point in different directions"
)]
#[case::infinite_bound(
    "TIMESTAMP '2026-01-01', TIMESTAMP 'infinity', INTERVAL '1 day'",
    "the bounds cannot be infinite"
)]
fn an_unusable_timestamp_series_fails_during_binding(
    mut testing_planner: TestingPlanner,
    #[case] arguments: &str,
    #[case] expected_message: &str,
) {
    let result = testing_planner.plan(&format!("SELECT * FROM generate_series({arguments})"));

    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected_message), "{error}");
}
