//! Window operator integration tests: feed in-memory `(team, score)` rows
//! through a window and assert the function columns appended to them.

mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{Dispatch, OrderBy, WindowFunction, WindowRowLimit, WindowSpec, values_input};

const TEAM: usize = 0;
const SCORE: usize = 1;

/// A batch of `(team, score)` rows.
fn team_score_batch(rows: &[(Option<i64>, i64)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("team", DataType::Int64, true),
        Field::new("score", DataType::Int64, false),
    ]));
    let teams: Int64Array = rows.iter().map(|&(team, _)| team).collect();
    let scores: Int64Array = rows.iter().map(|&(_, score)| Some(score)).collect();
    RecordBatch::try_new(schema, vec![Arc::new(teams), Arc::new(scores)]).unwrap()
}

/// `functions OVER (PARTITION BY team ORDER BY score)`.
fn per_team_by_score(functions: Vec<WindowFunction>) -> WindowSpec {
    WindowSpec {
        partition_by: vec![TEAM],
        order_by: vec![OrderBy::new(SCORE, false, false)],
        functions,
        row_limit: None,
    }
}

/// Runs the window over `batches` on `workers` workers and returns every
/// output row as its column values, sorted, so a test does not depend on the
/// order rows are emitted in.
fn run_window(
    spec: WindowSpec,
    workers: usize,
    batches: Vec<RecordBatch>,
) -> Vec<Vec<Option<i64>>> {
    run_window_on(&dispatch(workers), spec, batches)
}

/// [`run_window`] on the workers and memory of `dispatch`.
fn run_window_on(
    dispatch: &DispatchGuard,
    spec: WindowSpec,
    batches: Vec<RecordBatch>,
) -> Vec<Vec<Option<i64>>> {
    let output = values_input(dispatch, batches)
        .record_batches()
        .window(spec)
        .collect()
        .unwrap();
    let mut rows: Vec<Vec<Option<i64>>> = output
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(|row| {
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        let values = column.as_primitive::<Int64Type>();
                        values.is_valid(row).then(|| values.value(row))
                    })
                    .collect()
            })
        })
        .collect();
    rows.sort();
    rows
}

/// Shorthand for an expected output row with no NULLs.
fn row(values: &[i64]) -> Vec<Option<i64>> {
    values.iter().copied().map(Some).collect()
}

#[test]
fn row_number_restarts_in_each_partition() {
    let batches = vec![
        team_score_batch(&[(Some(1), 30), (Some(2), 5), (Some(1), 10)]),
        team_score_batch(&[(Some(1), 20), (Some(2), 7)]),
    ];

    let rows = run_window(
        per_team_by_score(vec![WindowFunction::RowNumber]),
        2,
        batches,
    );

    assert_eq!(
        rows,
        vec![
            row(&[1, 10, 1]),
            row(&[1, 20, 2]),
            row(&[1, 30, 3]),
            row(&[2, 5, 1]),
            row(&[2, 7, 2]),
        ]
    );
}

#[test]
fn rank_skips_after_ties_and_dense_rank_does_not() {
    let batches = vec![
        team_score_batch(&[(Some(1), 10), (Some(1), 20)]),
        team_score_batch(&[(Some(1), 10), (Some(1), 30)]),
    ];

    let rows = run_window(
        per_team_by_score(vec![WindowFunction::Rank, WindowFunction::DenseRank]),
        2,
        batches,
    );

    assert_eq!(
        rows,
        vec![
            row(&[1, 10, 1, 1]),
            row(&[1, 10, 1, 1]),
            row(&[1, 20, 3, 2]),
            row(&[1, 30, 4, 3]),
        ]
    );
}

#[test]
fn a_row_limit_keeps_the_first_rows_of_each_partition() {
    let batches = vec![
        team_score_batch(&[(Some(1), 30), (Some(2), 5), (Some(1), 10)]),
        team_score_batch(&[(Some(1), 20), (Some(1), 40), (Some(2), 7)]),
    ];
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.row_limit = Some(WindowRowLimit {
        function_index: 0,
        limit: 2,
    });

    let rows = run_window(spec, 2, batches);

    assert_eq!(
        rows,
        vec![
            row(&[1, 10, 1]),
            row(&[1, 20, 2]),
            row(&[2, 5, 1]),
            row(&[2, 7, 2]),
        ]
    );
}

#[test]
fn a_rank_limit_keeps_every_row_tied_at_the_limit() {
    let batches = vec![team_score_batch(&[
        (Some(1), 10),
        (Some(1), 20),
        (Some(1), 20),
        (Some(1), 30),
    ])];
    let mut spec = per_team_by_score(vec![WindowFunction::Rank]);
    spec.row_limit = Some(WindowRowLimit {
        function_index: 0,
        limit: 2,
    });

    let rows = run_window(spec, 2, batches);

    assert_eq!(
        rows,
        vec![row(&[1, 10, 1]), row(&[1, 20, 2]), row(&[1, 20, 2])]
    );
}

#[test]
fn null_partition_keys_form_one_partition() {
    let batches = vec![
        team_score_batch(&[(None, 2), (Some(1), 9)]),
        team_score_batch(&[(None, 1)]),
    ];

    let rows = run_window(
        per_team_by_score(vec![WindowFunction::RowNumber]),
        2,
        batches,
    );

    assert_eq!(
        rows,
        vec![
            vec![None, Some(1), Some(1)],
            vec![None, Some(2), Some(2)],
            row(&[1, 9, 1]),
        ]
    );
}

#[test]
fn a_descending_order_key_numbers_from_the_highest_value() {
    let batches = vec![team_score_batch(&[
        (Some(1), 10),
        (Some(1), 30),
        (Some(1), 20),
    ])];
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.order_by = vec![OrderBy::new(SCORE, true, false)];

    let rows = run_window(spec, 2, batches);

    assert_eq!(
        rows,
        vec![row(&[1, 10, 3]), row(&[1, 20, 2]), row(&[1, 30, 1])]
    );
}

#[test]
fn without_partition_keys_the_whole_input_is_one_partition() {
    let batches = vec![
        team_score_batch(&[(Some(1), 30), (Some(2), 10)]),
        team_score_batch(&[(Some(3), 20)]),
    ];
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.partition_by = Vec::new();

    let rows = run_window(spec, 2, batches);

    assert_eq!(
        rows,
        vec![row(&[1, 30, 3]), row(&[2, 10, 1]), row(&[3, 20, 2])]
    );
}

#[test]
fn without_any_keys_every_row_gets_its_own_number() {
    let batches = vec![
        team_score_batch(&[(Some(1), 30), (Some(2), 10)]),
        team_score_batch(&[(Some(3), 20)]),
    ];
    let spec = WindowSpec {
        partition_by: Vec::new(),
        order_by: Vec::new(),
        functions: vec![WindowFunction::RowNumber],
        row_limit: None,
    };

    let rows = run_window(spec, 2, batches);

    let mut row_numbers: Vec<_> = rows.iter().map(|row| row[2]).collect();
    row_numbers.sort();
    assert_eq!(row_numbers, vec![Some(1), Some(2), Some(3)]);
}

#[test]
fn empty_input_emits_nothing() {
    let rows = run_window(
        per_team_by_score(vec![WindowFunction::RowNumber]),
        2,
        Vec::new(),
    );

    assert!(rows.is_empty());
}

/// `row_count` rows over `team_count` teams, with a distinct score per row so
/// the expected numbering has no ties.
fn scattered_rows(row_count: usize, team_count: usize) -> Vec<(Option<i64>, i64)> {
    (0..row_count)
        .map(|index| {
            let team = (index % team_count) as i64;
            let score = ((index * 7919) % 1_000_003) as i64;
            (Some(team), score)
        })
        .collect()
}

fn split_into_batches(rows: &[(Option<i64>, i64)]) -> Vec<RecordBatch> {
    rows.chunks(4_096).map(team_score_batch).collect()
}

/// The rows of `rows` with their row number per team by ascending score,
/// keeping those numbered at most `limit`, sorted like [`run_window`] output.
fn expected_row_numbers(rows: &[(Option<i64>, i64)], limit: i64) -> Vec<Vec<Option<i64>>> {
    let mut sorted_rows = rows.to_vec();
    sorted_rows.sort();
    let mut expected = Vec::new();
    let mut row_number = 0;
    for (index, &(team, score)) in sorted_rows.iter().enumerate() {
        let starts_team = index == 0 || sorted_rows[index - 1].0 != team;
        row_number = if starts_team { 1 } else { row_number + 1 };
        if row_number <= limit {
            expected.push(vec![team, Some(score), Some(row_number)]);
        }
    }
    expected
}

/// Enough rows for every worker to sort its pending rows into runs more than
/// once, so buckets are merged from several runs.
#[test]
fn many_rows_over_many_runs_are_numbered_per_partition() {
    let input_rows = scattered_rows(600_000, 37);

    let rows = run_window(
        per_team_by_score(vec![WindowFunction::RowNumber]),
        3,
        split_into_batches(&input_rows),
    );

    assert_eq!(rows, expected_row_numbers(&input_rows, i64::MAX));
}

#[test]
fn a_row_limit_over_many_runs_keeps_the_first_rows_of_each_partition() {
    let input_rows = scattered_rows(600_000, 37);
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.row_limit = Some(WindowRowLimit {
        function_index: 0,
        limit: 5,
    });

    let rows = run_window(spec, 3, split_into_batches(&input_rows));

    assert_eq!(rows, expected_row_numbers(&input_rows, 5));
}

/// Few rows per partition, and enough rows for the worker to stop applying
/// the limit while it consumes and leave it to the output. That many rows
/// need more memory than the other tests run with.
#[test]
fn a_row_limit_over_many_small_partitions_keeps_the_first_rows_of_each_partition() {
    let dispatch = DispatchGuard::new(Dispatch::spin_up(1, 128, None));
    let input_rows = scattered_rows(1_300_000, 400_000);
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.row_limit = Some(WindowRowLimit {
        function_index: 0,
        limit: 2,
    });

    let rows = run_window_on(&dispatch, spec, split_into_batches(&input_rows));

    assert_eq!(rows, expected_row_numbers(&input_rows, 2));
}

/// One bucket with more rows than a merge slice holds, so its merge is split
/// into several slices.
#[test]
fn an_unpartitioned_window_larger_than_one_merge_slice_is_numbered_in_order() {
    let input_rows = scattered_rows(600_000, 1);
    let mut spec = per_team_by_score(vec![WindowFunction::RowNumber]);
    spec.partition_by = Vec::new();

    let rows = run_window(spec, 3, split_into_batches(&input_rows));

    assert_eq!(rows, expected_row_numbers(&input_rows, i64::MAX));
}
