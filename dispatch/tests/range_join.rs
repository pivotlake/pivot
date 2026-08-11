//! Range join integration tests: SQL-free dataflows joining two inputs on a
//! single `<`/`<=`/`>`/`>=` comparison, through the real ORDER BY build phase.

mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{DataFlowDispatcher, RECORD_BATCH_SIZE, RangeCompare, RangeJoinSpec, values_input};

fn keyed_batch(keys: &[Option<i64>], values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, true),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(keys.to_vec())),
            Arc::new(Int64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

/// Join `probe` and `build` on `probe.key OP build.key`, emitting both key
/// columns and both value columns.
fn range_join(
    dispatcher: &DataFlowDispatcher,
    compare: RangeCompare,
    probe: Vec<RecordBatch>,
    build: Vec<RecordBatch>,
) -> Vec<RecordBatch> {
    let field = |name: &str| Field::new(name, DataType::Int64, true);
    let spec = RangeJoinSpec {
        probe_key_index: 0,
        build_key_index: 0,
        compare,
        probe_output_indices: vec![0, 1],
        build_output_indices: vec![0, 1],
        probe_fields: vec![field("probe_key"), field("probe_value")],
        build_fields: vec![field("build_key"), field("build_value")],
    };
    let probe = values_input(dispatcher, probe).record_batches();
    let build = values_input(dispatcher, build).record_batches();
    probe
        .range_join(build, &DataType::Int64, spec)
        .collect()
        .unwrap()
}

/// Every output row's `(probe_key, build_key)` pair, sorted.
fn key_pairs(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
    let probe = collect_i64s(batches, 0);
    let build = collect_i64s(batches, 2);
    let mut pairs: Vec<_> = probe.into_iter().zip(build).collect();
    pairs.sort();
    pairs
}

#[test]
fn less_than_matches_strictly_greater_build_keys() {
    let d = dispatch(4);
    let build = keyed_batch(&[Some(1), Some(5), Some(9)], &[10, 50, 90]);
    let probe = keyed_batch(&[Some(5)], &[1]);

    let out = range_join(&d, RangeCompare::Less, vec![probe], vec![build]);

    assert_eq!(key_pairs(&out), vec![(5, 9)]);
}

#[test]
fn less_or_equal_keeps_equal_keys_and_duplicates() {
    let d = dispatch(4);
    let build = keyed_batch(&[Some(3), Some(5), Some(5), Some(7)], &[1, 2, 3, 4]);
    let probe = keyed_batch(&[Some(5)], &[1]);

    let out = range_join(&d, RangeCompare::LessEq, vec![probe], vec![build]);

    assert_eq!(key_pairs(&out), vec![(5, 5), (5, 5), (5, 7)]);
}

#[test]
fn greater_or_equal_keeps_equal_keys_and_duplicates() {
    let d = dispatch(4);
    let build = keyed_batch(&[Some(3), Some(5), Some(5), Some(7)], &[1, 2, 3, 4]);
    let probe = keyed_batch(&[Some(5)], &[1]);

    let out = range_join(&d, RangeCompare::GreaterEq, vec![probe], vec![build]);

    assert_eq!(key_pairs(&out), vec![(5, 3), (5, 5), (5, 5)]);
}

#[test]
fn greater_than_joins_unsorted_multi_batch_build() {
    let d = dispatch(4);
    let build = vec![
        keyed_batch(&[Some(4), Some(1)], &[1, 2]),
        keyed_batch(&[Some(2), Some(8)], &[3, 4]),
    ];
    let probe = keyed_batch(&[Some(5), Some(0)], &[1, 2]);

    let out = range_join(&d, RangeCompare::Greater, vec![probe], build);

    assert_eq!(key_pairs(&out), vec![(5, 1), (5, 2), (5, 4)]);
}

#[test]
fn null_keys_match_nothing_on_either_side() {
    let d = dispatch(4);
    let build = keyed_batch(&[Some(1), None], &[1, 2]);
    let probe = keyed_batch(&[None, Some(9)], &[1, 2]);

    let out = range_join(&d, RangeCompare::Greater, vec![probe], vec![build]);

    assert_eq!(key_pairs(&out), vec![(9, 1)]);
}

#[test]
fn empty_build_side_emits_nothing() {
    let d = dispatch(4);
    let probe = keyed_batch(&[Some(1)], &[1]);

    let out = range_join(&d, RangeCompare::Less, vec![probe], vec![]);

    assert_eq!(key_pairs(&out), vec![]);
}

#[test]
fn wide_fan_out_spans_chunks_and_emits_bounded_batches() {
    // A build side several sorted chunks long, matched whole by one probe
    // row: the output must stream as bounded batches and cover every row.
    let d = dispatch(4);
    let n = 3 * RECORD_BATCH_SIZE as i64 + 17;
    let keys: Vec<Option<i64>> = (0..n).map(Some).collect();
    let values: Vec<i64> = (0..n).collect();
    let build = keyed_batch(&keys, &values);
    let probe = keyed_batch(&[Some(n)], &[1]);

    let out = range_join(&d, RangeCompare::Greater, vec![probe], vec![build]);

    let total: usize = out.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, n as usize);
    assert!(out.iter().all(|b| b.num_rows() <= RECORD_BATCH_SIZE));
    let mut build_keys = collect_i64s(&out, 2);
    build_keys.sort();
    assert_eq!(build_keys, (0..n).collect::<Vec<_>>());
}

#[test]
fn boundary_between_chunks_splits_exactly() {
    // Duplicate keys straddling a chunk boundary: a probe key equal to them
    // must take all of them or none, per the comparison's strictness.
    let d = dispatch(4);
    let n = RECORD_BATCH_SIZE as i64;
    let keys: Vec<Option<i64>> = (0..n).map(|_| Some(7)).chain([Some(9)].repeat(5)).collect();
    let values: Vec<i64> = (0..keys.len() as i64).collect();
    let build = keyed_batch(&keys, &values);
    let probe = keyed_batch(&[Some(7)], &[1]);

    let strict = range_join(
        &d,
        RangeCompare::Less,
        vec![probe.clone()],
        vec![build.clone()],
    );
    let inclusive = range_join(&d, RangeCompare::LessEq, vec![probe], vec![build]);

    assert_eq!(strict.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert_eq!(
        inclusive.iter().map(|b| b.num_rows()).sum::<usize>(),
        n as usize + 5
    );
}
