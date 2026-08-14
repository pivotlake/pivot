//! Cartesian-product integration tests over the public record-batch API.

mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{DataFlowDispatcher, values_input};

fn nullable_i64_batch(name: &str, values: Vec<Option<i64>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap()
}

fn execute_cross_join(
    dispatcher: &DataFlowDispatcher,
    left: Vec<RecordBatch>,
    right: Vec<RecordBatch>,
) -> Vec<RecordBatch> {
    let left = values_input(dispatcher, left).record_batches();
    let right = values_input(dispatcher, right).record_batches();
    left.cross_join(
        right,
        vec![Field::new("left", DataType::Int64, true)],
        vec![Field::new("right", DataType::Int64, true)],
    )
    .collect()
    .unwrap()
}

fn pairs(batches: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
    let mut pairs: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            let left = batch
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>();
            let right = batch
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>();
            (0..batch.num_rows()).map(move |row| {
                (
                    left.is_valid(row).then(|| left.value(row)),
                    right.is_valid(row).then(|| right.value(row)),
                )
            })
        })
        .collect();
    pairs.sort();
    pairs
}

#[test]
fn cross_join_matches_rows_across_batches_and_preserves_nulls() {
    // Setup
    let dispatcher = dispatch(4);
    let left = vec![
        nullable_i64_batch("left", vec![Some(1), None]),
        nullable_i64_batch("left", vec![Some(2)]),
    ];
    let right = vec![
        nullable_i64_batch("right", vec![Some(10)]),
        nullable_i64_batch("right", vec![Some(20), Some(30)]),
    ];

    // Execute
    let results = execute_cross_join(&dispatcher, left, right);

    // Assert
    assert_eq!(
        pairs(&results),
        vec![
            (None, Some(10)),
            (None, Some(20)),
            (None, Some(30)),
            (Some(1), Some(10)),
            (Some(1), Some(20)),
            (Some(1), Some(30)),
            (Some(2), Some(10)),
            (Some(2), Some(20)),
            (Some(2), Some(30)),
        ]
    );
}

#[test]
fn cross_join_with_either_input_empty_emits_nothing() {
    // Setup
    let dispatcher = dispatch(4);
    let rows = || nullable_i64_batch("rows", vec![Some(1), Some(2)]);
    let empty = || nullable_i64_batch("empty", Vec::new());

    // Execute
    let empty_right = execute_cross_join(&dispatcher, vec![rows()], vec![empty()]);
    let empty_left = execute_cross_join(&dispatcher, vec![empty()], vec![rows()]);

    // Assert
    assert!(empty_right.is_empty());
    assert!(empty_left.is_empty());
}

#[test]
fn cross_join_chunks_large_outputs() {
    // Setup
    let dispatcher = dispatch(4);
    let values = (0..100).map(Some).collect();

    // Execute
    let results = execute_cross_join(
        &dispatcher,
        vec![nullable_i64_batch("left", values)],
        vec![nullable_i64_batch("right", (0..100).map(Some).collect())],
    );

    // Assert
    assert_eq!(
        results.iter().map(RecordBatch::num_rows).sum::<usize>(),
        10_000
    );
    assert!(
        results
            .iter()
            .all(|batch| batch.num_rows() <= dispatch::RECORD_BATCH_SIZE)
    );
}
