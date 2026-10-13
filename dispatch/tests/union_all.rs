//! UNION ALL integration tests: several chains' rows arrive as one stream.

mod common;

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::values_input;

fn int64_batch(name: &str, nullable: bool, values: Vec<Option<i64>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            name,
            DataType::Int64,
            nullable,
        )])),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap()
}

/// The NULL count and the sorted non-NULL values of every batch's first column.
fn nulls_and_values(batches: &[RecordBatch]) -> (usize, Vec<i64>) {
    let mut nulls = 0;
    let mut values = Vec::new();
    for batch in batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        nulls += column.null_count();
        values.extend(column.iter().flatten());
    }
    values.sort();
    (nulls, values)
}

#[test]
fn every_input_contributes_all_of_its_rows() {
    let d = dispatch(4);
    let left: Vec<_> = (0..8)
        .map(|b| int64_batch("id", false, vec![Some(b * 2), Some(b * 2 + 1)]))
        .collect();
    let middle: Vec<_> = (8..12)
        .map(|b| int64_batch("id", false, vec![Some(b * 2), Some(b * 2 + 1)]))
        .collect();
    let right: Vec<_> = (12..16)
        .map(|b| int64_batch("id", false, vec![Some(b * 2), Some(b * 2 + 1)]))
        .collect();

    let results: Vec<RecordBatch> = values_input(&d, left)
        .record_batches()
        .union_all(
            vec![
                values_input(&d, middle).record_batches(),
                values_input(&d, right).record_batches(),
            ],
            vec![false],
        )
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, (0..32).collect::<Vec<_>>());
}

/// Inputs name and declare their columns differently; every output batch
/// carries the one uniform schema, so a stage downstream that takes the first
/// batch's schema for all of them sees no mismatch.
#[test]
fn output_batches_share_one_schema() {
    let d = dispatch(2);
    let left = vec![int64_batch("left_id", false, vec![Some(1), Some(2)])];
    let right = vec![int64_batch("right_id", true, vec![Some(3), None])];

    let results: Vec<RecordBatch> = values_input(&d, left)
        .record_batches()
        .union_all(vec![values_input(&d, right).record_batches()], vec![true])
        .collect()
        .unwrap();

    let schemas: Vec<_> = results.iter().map(|batch| batch.schema()).collect();
    assert!(schemas.iter().all(|schema| schema == &schemas[0]));
    assert!(schemas[0].field(0).is_nullable());
    assert_eq!(nulls_and_values(&results), (1, vec![1, 2, 3]));
}

/// An input that produces nothing must not hold up the union: the stage
/// finishes once every input chain has, rows or not.
#[test]
fn an_empty_input_does_not_stall_the_union() {
    let d = dispatch(4);
    let left = vec![int64_batch("id", false, vec![Some(1), Some(2)])];

    let results: Vec<RecordBatch> = values_input(&d, left)
        .record_batches()
        .union_all(
            vec![values_input(&d, Vec::<RecordBatch>::new()).record_batches()],
            vec![false],
        )
        .collect()
        .unwrap();

    assert_eq!(nulls_and_values(&results), (0, vec![1, 2]));
}
