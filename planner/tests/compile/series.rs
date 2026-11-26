//! End-to-end tests of `range` and `generate_series` over timestamps, and of
//! lateral calls, whose arguments are columns of an earlier `FROM` item.

use std::sync::Arc;

use crate::common::*;
use arrow_array::{ArrayRef, Int64Array, TimestampMicrosecondArray};
use planner::types::Type;
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

/// Spans of integers: span 2 has no last value and span 3 no first.
///
/// ```text
/// id | first | last
/// ---+-------+-----
///  1 |     1 |    2
///  2 |     5 | NULL
///  3 |  NULL |    3
///  4 |     7 |    7
/// ```
fn add_spans(planner: &TestingPlanner) {
    let int64_col = |values: Vec<Option<i64>>| Arc::new(Int64Array::from(values)) as ArrayRef;
    planner.add_table(
        "spans",
        &[
            (
                "id",
                Type::Int64,
                int64_col(vec![Some(1), Some(2), Some(3), Some(4)]),
            ),
            (
                "first",
                Type::Int64,
                int64_col(vec![Some(1), Some(5), None, Some(7)]),
            ),
            (
                "last",
                Type::Int64,
                int64_col(vec![Some(2), None, Some(3), Some(7)]),
            ),
        ],
    );
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

#[rstest]
fn series_bounds_come_from_an_earlier_from_item(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT buckets.step \
         FROM (SELECT 2 AS first_step, 5 AS last_step) AS observed, \
              generate_series(observed.first_step, observed.last_step) AS buckets(step) \
         ORDER BY step",
    );

    assert_eq!(
        single_column_values(&rows),
        vec![json!(2), json!(3), json!(4), json!(5)]
    );
}

/// Each row gets its own series, and a row with a NULL bound gets none.
#[rstest]
fn a_lateral_series_is_generated_for_each_row(mut testing_planner: TestingPlanner) {
    add_spans(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT id, step FROM spans, generate_series(spans.first, spans.last) AS steps(step) \
         ORDER BY id, step",
    );

    assert_eq!(
        rows,
        vec![
            json!({"id": 1, "step": 1}),
            json!({"id": 1, "step": 2}),
            json!({"id": 4, "step": 7}),
        ]
    );
}

#[rstest]
fn a_lateral_range_excludes_each_rows_stop(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT a, step FROM example_table, range(a) AS steps(step) WHERE a <= 2 \
         ORDER BY a, step",
    );

    assert_eq!(
        rows,
        vec![
            json!({"a": 1, "step": 0}),
            json!({"a": 2, "step": 0}),
            json!({"a": 2, "step": 1}),
        ]
    );
}

#[rstest]
fn a_left_lateral_join_keeps_rows_whose_series_is_empty(mut testing_planner: TestingPlanner) {
    add_spans(&testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT id, step FROM spans \
         LEFT JOIN LATERAL generate_series(spans.first, spans.last) AS steps(step) ON true \
         ORDER BY id, step",
    );

    assert_eq!(
        rows,
        vec![
            json!({"id": 1, "step": 1}),
            json!({"id": 1, "step": 2}),
            json!({"id": 2}),
            json!({"id": 3}),
            json!({"id": 4, "step": 7}),
        ]
    );
}

/// A series can read the rows of the series before it.
#[rstest]
fn a_lateral_series_takes_its_bounds_from_another_series(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT outer_steps.step AS outer_step, inner_steps.step AS inner_step \
         FROM example_table, \
              generate_series(1, a) AS outer_steps(step), \
              generate_series(1, outer_steps.step) AS inner_steps(step) \
         WHERE a = 2 ORDER BY outer_step, inner_step",
    );

    assert_eq!(
        rows,
        vec![
            json!({"outer_step": 1, "inner_step": 1}),
            json!({"outer_step": 2, "inner_step": 1}),
            json!({"outer_step": 2, "inner_step": 2}),
        ]
    );
}

/// A bound computed by a subquery is an argument that is not a constant, with
/// no earlier `FROM` item involved.
#[rstest]
fn a_series_bound_can_be_a_subquery(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT * FROM generate_series(4, (SELECT max(a) FROM example_table)) ORDER BY 1",
    );

    assert_eq!(single_column_values(&rows), vec![json!(4), json!(5)]);
}

/// The five rows' series hold 60,000 values between them, so the series of one
/// row spans several output batches.
#[rstest]
fn a_lateral_series_longer_than_a_batch_is_generated_in_full(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT count(*) AS step_count, sum(step) AS step_total \
         FROM example_table, generate_series(1, a * 4000) AS steps(step)",
    );

    assert_eq!(
        rows,
        vec![json!({"step_count": 60000, "step_total": 440030000})]
    );
}

/// Every bucket between the first and the last one that has rows, with the
/// count of the rows in it.
#[rstest]
fn a_lateral_timestamp_series_fills_empty_time_buckets(mut testing_planner: TestingPlanner) {
    // 2026-11-12 10:15:00, in microseconds since the epoch.
    let first_bucket_micros: i64 = 1_794_478_500_000_000;
    let seconds_into_first_bucket = [3, 8, 31, 52];
    testing_planner.add_table(
        "events",
        &[(
            "timestamp",
            Type::Timestamp,
            Arc::new(TimestampMicrosecondArray::from(
                seconds_into_first_bucket
                    .map(|seconds| first_bucket_micros + seconds * 1_000_000)
                    .to_vec(),
            )) as ArrayRef,
        )],
    );

    let rows = run(
        &mut testing_planner,
        "WITH counts AS ( \
            SELECT time_bucket(INTERVAL '10 seconds', \"timestamp\") AS bucket, count(*) AS count \
            FROM events GROUP BY bucket) \
         SELECT buckets.bucket AS bucket, counts.count AS count \
         FROM (SELECT min(bucket) AS first_bucket, max(bucket) AS last_bucket FROM counts) \
                AS observed, \
              generate_series(observed.first_bucket, observed.last_bucket, INTERVAL '10 seconds') \
                AS buckets(bucket) \
         LEFT JOIN counts ON counts.bucket = buckets.bucket \
         ORDER BY bucket ASC",
    );

    assert_eq!(
        rows,
        vec![
            json!({"bucket": "2026-11-12T10:15:00", "count": 2}),
            json!({"bucket": "2026-11-12T10:15:10"}),
            json!({"bucket": "2026-11-12T10:15:20"}),
            json!({"bucket": "2026-11-12T10:15:30", "count": 1}),
            json!({"bucket": "2026-11-12T10:15:40"}),
            json!({"bucket": "2026-11-12T10:15:50", "count": 1}),
        ]
    );
}

#[rstest]
fn a_zero_step_read_from_a_row_fails_the_query(mut testing_planner: TestingPlanner) {
    let error = run_expecting_error(
        &mut testing_planner,
        "SELECT step FROM example_table, generate_series(1, 3, a - 1) AS steps(step)",
    );

    assert!(error.contains("step must not be zero"), "{error}");
}

/// The rows a call returns are joined back to their row on the columns the
/// arguments read, and a NULL never joins. Span 2's series is not empty
/// although its `last` is NULL, so the query fails instead of losing it.
#[rstest]
fn rows_returned_for_a_null_argument_column_fail_the_query(mut testing_planner: TestingPlanner) {
    add_spans(&testing_planner);

    let error = run_expecting_error(
        &mut testing_planner,
        "SELECT id, step FROM spans, \
         generate_series(1, CASE WHEN last IS NULL THEN 2 ELSE last END) AS steps(step)",
    );

    assert!(
        error.contains("returned rows for a row where a column its arguments read is NULL"),
        "{error}"
    );
}
