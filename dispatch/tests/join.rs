//! Hash-join integration tests: build a hash table from one in-memory input and
//! probe it with another, asserting which probe rows survive.

mod common;

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{
    AggregationKind, AggregationSlot, JoinKind, JoinOutputColumns, JoinSpec, values_input,
};

/// An inner join keyed on column 0 of both sides.
fn inner_join(output_columns: JoinOutputColumns) -> JoinSpec {
    JoinSpec {
        build_key_columns: vec![0],
        probe_key_columns: vec![0],
        output_columns,
        kind: JoinKind::Inner,
    }
}

fn int64_batch(name: &str, values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn two_int64_batch(names: (&str, &str), left: &[i64], right: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(names.0, DataType::Int64, false),
            Field::new(names.1, DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(left.to_vec())),
            Arc::new(Int64Array::from(right.to_vec())),
        ],
    )
    .unwrap()
}

#[test]
fn join_matching_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_on_int32_keys() {
    let d = dispatch(1);
    let int32_batch = |values: &[i32]| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(arrow_array::Int32Array::from(values.to_vec()))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![int32_batch(&[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int32_batch(&[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int32],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut keys: Vec<i32> = results
        .iter()
        .flat_map(|b| {
            use arrow_array::cast::AsArray;
            b.column(1)
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .to_vec()
        })
        .collect();
    keys.sort();
    assert_eq!(keys, vec![20, 30]);
}

#[test]
fn join_no_matches() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[1, 2, 3])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[4, 5, 6])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn join_duplicate_build_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 20])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![10, 10]);
}

#[test]
fn join_all_keys_match() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[1, 2, 3, 4, 5])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[5, 4, 3, 2, 1])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, vec![1, 2, 3, 4, 5]);
}

#[test]
fn join_large_tables() {
    let d = dispatch(1);
    let build_keys: Vec<i64> = (0..5000).collect();
    let probe_keys: Vec<i64> = (2500..3500).collect();

    let build = values_input(&d, vec![int64_batch("id", &build_keys)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_keys)]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut keys = collect_i64s(&results, 1);
    keys.sort();
    assert_eq!(keys, probe_keys);
}

#[test]
fn join_then_count() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30, 40])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 2);
}

#[test]
fn join_keeps_only_listed_columns() {
    let d = dispatch(1);
    let build = values_input(
        &d,
        vec![two_int64_batch(("id", "b_payload"), &[10, 20], &[100, 200])],
    )
    .record_batches();
    let probe = values_input(
        &d,
        vec![two_int64_batch(
            ("id", "p_payload"),
            &[20, 20, 99],
            &[7, 8, 9],
        )],
    )
    .record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            inner_join(JoinOutputColumns {
                probe: vec![1],
                build: vec![1],
            }),
        )
        .collect()
        .unwrap();

    let schema = results[0].schema();
    assert_eq!(schema.field(0).name(), "p_payload");
    assert_eq!(schema.field(1).name(), "b_payload");
    let mut probe_payloads = collect_i64s(&results, 0);
    probe_payloads.sort();
    assert_eq!(probe_payloads, vec![7, 8]);
    assert_eq!(collect_i64s(&results, 1), vec![200, 200]);
}

/// A build-side outer join keyed on column 0 of both sides, whose probe side
/// contributes one nullable `Int64` column to the output.
fn build_outer_join(output_columns: JoinOutputColumns) -> JoinSpec {
    JoinSpec {
        build_key_columns: vec![0],
        probe_key_columns: vec![0],
        output_columns,
        kind: JoinKind::BuildOuter {
            probe_fields: vec![Field::new("id", DataType::Int64, true)],
        },
    }
}

#[test]
fn build_outer_join_emits_unmatched_build_rows() {
    let d = dispatch(4);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 99])]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            build_outer_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut rows: Vec<(bool, i64)> = results
        .iter()
        .flat_map(|batch| {
            let probe_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let build_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|row| (probe_ids.is_valid(row), build_ids.value(row)))
                .collect::<Vec<_>>()
        })
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(false, 10), (false, 30), (true, 20)]);
}

#[test]
fn every_build_row_reaches_a_build_outer_join_output() {
    let d = dispatch(4);
    let build_ids: Vec<i64> = (0..30_000).collect();
    let probe_ids: Vec<i64> = (0..30_000).step_by(3).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64],
            build_outer_join(JoinOutputColumns::keep_all(1, 1)),
        )
        .collect()
        .unwrap();

    let mut build_ids_out = collect_i64s(&results, 1);
    build_ids_out.sort();
    assert_eq!(build_ids_out, (0..30_000).collect::<Vec<i64>>());
}

/// A probe-side semi join keyed on column 0 of both sides, emitting the listed
/// probe columns and, as every semi join must, no build column.
fn probe_semi_join(probe_columns: Vec<usize>) -> JoinSpec {
    JoinSpec {
        build_key_columns: vec![0],
        probe_key_columns: vec![0],
        output_columns: JoinOutputColumns {
            probe: probe_columns,
            build: Vec::new(),
        },
        kind: JoinKind::ProbeSemi,
    }
}

#[test]
fn probe_semi_join_emits_a_matched_probe_row_once() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 20])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20, 99])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![0]))
        .collect()
        .unwrap();

    // Key 10 sits in the build side twice, and still brings its probe row out
    // once; 99 is in neither.
    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, vec![10, 20]);
    assert!(results.iter().all(|batch| batch.num_columns() == 1));
}

#[test]
fn probe_semi_join_keeps_every_copy_of_a_repeated_probe_row() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 10, 10])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![0]))
        .collect()
        .unwrap();

    // The de-duplication is per probe row, not per key: each of the three rows
    // has a match of its own.
    assert_eq!(collect_i64s(&results, 0), vec![10, 10, 10]);
}

#[test]
fn probe_semi_join_carries_the_probe_columns_it_lists() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[20])]).record_batches();
    let probe = values_input(
        &d,
        vec![two_int64_batch(("id", "payload"), &[10, 20], &[100, 200])],
    )
    .record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![1]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), vec![200]);
}

#[test]
fn probe_semi_join_over_an_empty_build_side_emits_nothing() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), Vec::<i64>::new());
}

#[test]
fn probe_semi_join_never_matches_a_null_key() {
    let d = dispatch(1);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![nullable(vec![Some(10), None])]).record_batches();
    let probe = values_input(&d, vec![nullable(vec![None, Some(10), None])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), vec![10]);
}

#[test]
fn a_semi_join_spanning_batches_emits_each_matched_probe_row_once() {
    let d = dispatch(4);
    // Every key three times over on the build side, so a row that came out per
    // match would come out three times too.
    let build_ids: Vec<i64> = (0..30_000).flat_map(|id| [id, id, id]).collect();
    let probe_ids: Vec<i64> = (0..30_000).step_by(3).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_semi_join(vec![0]))
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, probe_ids);
}
