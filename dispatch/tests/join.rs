//! Hash-join integration tests: build a hash table from one in-memory input and
//! probe it with another, asserting which probe rows survive.

mod common;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{
    AggregationKind, AggregationSlot, BoundarySlot, JoinBuildFilter, JoinKind, JoinResidualFn,
    JoinResidualSpec, JoinSpec, KeyBitsetSlot, values_input,
};

/// Non-nullable `Int64` output fields, which is what every column of these
/// tests' batches is.
fn int64_fields(count: usize) -> Vec<Field> {
    (0..count)
        .map(|i| Field::new(format!("f{i}"), DataType::Int64, false))
        .collect()
}

/// An inner join keyed on column 0 of both sides.
fn inner_join(probe_output_indices: Vec<usize>, build_output_indices: Vec<usize>) -> JoinSpec {
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_fields: int64_fields(probe_output_indices.len()),
        build_fields: int64_fields(build_output_indices.len()),
        probe_output_indices,
        build_output_indices,
        kind: JoinKind::Inner,
        residual_filters: None,
        build_filters: Vec::new(),
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

fn int64_columns(columns: &[(&str, &[i64])]) -> RecordBatch {
    RecordBatch::try_from_iter(columns.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
        )
    }))
    .unwrap()
}

fn residual_input(output: &[i64], predicate: Option<&[i64]>) -> RecordBatch {
    match predicate {
        Some(predicate) => int64_columns(&[
            ("key", &[1, 2]),
            ("output", output),
            ("predicate", predicate),
        ]),
        None => int64_columns(&[("key", &[1, 2]), ("output", output)]),
    }
}

fn residual(
    probe_column_indices: Vec<usize>,
    build_column_indices: Vec<usize>,
    evaluate: fn(&RecordBatch) -> BooleanArray,
) -> JoinResidualSpec {
    JoinResidualSpec::new(
        Arc::new(move || Box::new(evaluate) as JoinResidualFn),
        probe_column_indices,
        build_column_indices,
    )
}

fn int64_column(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
}

fn first_less_than_second(batch: &RecordBatch) -> BooleanArray {
    let first = int64_column(batch, 0);
    let second = int64_column(batch, 1);
    BooleanArray::from_iter(
        (0..batch.num_rows()).map(|row| Some(first.value(row) < second.value(row))),
    )
}

fn first_above_ten(batch: &RecordBatch) -> BooleanArray {
    let first = int64_column(batch, 0);
    BooleanArray::from_iter((0..batch.num_rows()).map(|row| Some(first.value(row) > 10)))
}

fn true_without_columns(batch: &RecordBatch) -> BooleanArray {
    assert_eq!(batch.num_columns(), 0);
    BooleanArray::from(vec![true; batch.num_rows()])
}

fn run_residual_join(
    probe_batch: RecordBatch,
    build_batch: RecordBatch,
    residual: JoinResidualSpec,
) -> Vec<RecordBatch> {
    let d = dispatch(1);
    let probe = values_input(&d, vec![probe_batch]).record_batches();
    let build = values_input(&d, vec![build_batch]).record_batches();
    let mut spec = inner_join(vec![1], vec![1]);
    spec.residual_filters = Some(residual);
    probe
        .join(build, &[DataType::Int64], spec)
        .collect()
        .unwrap()
}

#[test]
fn a_residual_gathers_only_its_selected_columns() {
    let probe = residual_input(&[7, 8], Some(&[5, 25]));
    let build = residual_input(&[100, 200], Some(&[10, 20]));
    let residual = residual(vec![2], vec![2], first_less_than_second);

    let results = run_residual_join(probe, build, residual);

    assert_eq!(collect_i64s(&results, 0), vec![7]);
    assert_eq!(collect_i64s(&results, 1), vec![100]);
}

#[test]
fn a_probe_only_residual_needs_no_build_columns() {
    let probe = residual_input(&[7, 8], Some(&[5, 15]));
    let build = residual_input(&[100, 200], None);
    let residual = residual(vec![2], vec![], first_above_ten);

    let results = run_residual_join(probe, build, residual);

    assert_eq!(collect_i64s(&results, 0), vec![8]);
    assert_eq!(collect_i64s(&results, 1), vec![200]);
}

#[test]
fn a_build_only_residual_needs_no_probe_columns() {
    let probe = residual_input(&[7, 8], None);
    let build = residual_input(&[100, 200], Some(&[5, 15]));
    let residual = residual(vec![], vec![2], first_above_ten);

    let results = run_residual_join(probe, build, residual);

    assert_eq!(collect_i64s(&results, 0), vec![8]);
    assert_eq!(collect_i64s(&results, 1), vec![200]);
}

#[test]
fn a_column_free_residual_preserves_the_candidate_count() {
    let probe = residual_input(&[7, 8], None);
    let build = residual_input(&[100, 200], None);
    let residual = residual(vec![], vec![], true_without_columns);

    let results = run_residual_join(probe, build, residual);

    assert_eq!(collect_i64s(&results, 0), vec![7, 8]);
    assert_eq!(collect_i64s(&results, 1), vec![100, 200]);
}

#[test]
fn join_matching_keys() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 30, 99])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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

    let int32_field = |name: &str| vec![Field::new(name, DataType::Int32, false)];
    let results = probe
        .join(
            build,
            &[DataType::Int32],
            JoinSpec {
                probe_key_indices: vec![0],
                build_key_indices: vec![0],
                probe_output_indices: vec![0],
                build_output_indices: vec![0],
                probe_fields: int32_field("probe_id"),
                build_fields: int32_field("build_id"),
                kind: JoinKind::Inner,
                residual_filters: None,
                build_filters: Vec::new(),
            },
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
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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
        .join(build, &[DataType::Int64], inner_join(vec![0], vec![0]))
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

    // The spec names the output fields; the input batches' own names are
    // never consulted.
    let results = probe
        .join(
            build,
            &[DataType::Int64],
            JoinSpec {
                probe_key_indices: vec![0],
                build_key_indices: vec![0],
                probe_output_indices: vec![1],
                build_output_indices: vec![1],
                probe_fields: vec![Field::new("p_payload", DataType::Int64, false)],
                build_fields: vec![Field::new("b_payload", DataType::Int64, false)],
                kind: JoinKind::Inner,
                residual_filters: None,
                build_filters: Vec::new(),
            },
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
fn build_outer_join(
    probe_output_indices: Vec<usize>,
    build_output_indices: Vec<usize>,
) -> JoinSpec {
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices,
        build_fields: int64_fields(build_output_indices.len()),
        build_output_indices,
        probe_fields: vec![Field::new("id", DataType::Int64, true)],
        kind: JoinKind::BuildOuter,
        residual_filters: None,
        build_filters: Vec::new(),
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
            build_outer_join(vec![0], vec![0]),
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
            build_outer_join(vec![0], vec![0]),
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
    let probe_fields = int64_fields(probe_columns.len());
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: probe_columns,
        build_output_indices: Vec::new(),
        probe_fields,
        build_fields: Vec::new(),
        kind: JoinKind::ProbeSemi,
        residual_filters: None,
        build_filters: Vec::new(),
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

/// A probe-side anti join keyed on column 0 of both sides, emitting the listed
/// probe columns for every probe row nothing matched, and no build column.
fn probe_anti_join(probe_columns: Vec<usize>) -> JoinSpec {
    let probe_fields = int64_fields(probe_columns.len());
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: probe_columns,
        build_output_indices: Vec::new(),
        probe_fields,
        build_fields: Vec::new(),
        kind: JoinKind::ProbeAnti,
        residual_filters: None,
        build_filters: Vec::new(),
    }
}

#[test]
fn probe_anti_join_emits_only_the_probe_rows_without_a_match() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 20])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20, 99])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_anti_join(vec![0]))
        .collect()
        .unwrap();

    // Key 10 sits in the build side twice and still keeps its probe row out;
    // only 99 is in neither side.
    assert_eq!(collect_i64s(&results, 0), vec![99]);
    assert!(results.iter().all(|batch| batch.num_columns() == 1));
}

#[test]
fn probe_anti_join_keeps_every_copy_of_an_unmatched_probe_row() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[99, 99, 10])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_anti_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), vec![99, 99]);
}

#[test]
fn probe_anti_join_carries_the_probe_columns_it_lists() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[20])]).record_batches();
    let probe = values_input(
        &d,
        vec![two_int64_batch(("id", "payload"), &[10, 20], &[100, 200])],
    )
    .record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_anti_join(vec![1]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), vec![100]);
}

#[test]
fn probe_anti_join_over_an_empty_build_side_emits_every_probe_row() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 20])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_anti_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 0), vec![10, 20]);
}

#[test]
fn probe_anti_join_emits_a_null_keyed_probe_row() {
    let d = dispatch(1);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![nullable(vec![Some(10), None])]).record_batches();
    let probe = values_input(&d, vec![nullable(vec![None, Some(10), Some(30)])]).record_batches();

    // The null-keyed probe row reaches the output, so the output field must
    // admit nulls.
    let mut spec = probe_anti_join(vec![0]);
    spec.probe_fields = vec![Field::new("f0", DataType::Int64, true)];
    let results = probe
        .join(build, &[DataType::Int64], spec)
        .collect()
        .unwrap();

    // A null key matches nothing, not even the build side's null: the null
    // probe row and 30 come out, 10 stays in.
    let mut ids: Vec<Option<i64>> = results
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort();
    assert_eq!(ids, vec![None, Some(30)]);
}

/// A build-side anti join keyed on column 0 of both sides, emitting the
/// listed build columns for every build row no probe row matched, and no
/// probe column.
fn build_anti_join(build_columns: Vec<usize>) -> JoinSpec {
    let build_fields = int64_fields(build_columns.len());
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: Vec::new(),
        build_output_indices: build_columns,
        probe_fields: Vec::new(),
        build_fields,
        kind: JoinKind::BuildAnti,
        residual_filters: None,
        build_filters: Vec::new(),
    }
}

#[test]
fn build_anti_join_emits_only_the_unmatched_build_rows() {
    let d = dispatch(4);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 20, 99])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], build_anti_join(vec![0]))
        .collect()
        .unwrap();

    // 20 is matched (twice, which changes nothing), 99 matches no build row.
    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, vec![10, 30]);
    assert!(results.iter().all(|batch| batch.num_columns() == 1));
}

#[test]
fn every_unmatched_build_row_reaches_a_build_anti_join_output() {
    let d = dispatch(4);
    let build_ids: Vec<i64> = (0..30_000).collect();
    let probe_ids: Vec<i64> = (0..30_000).step_by(3).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], build_anti_join(vec![0]))
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    let expected: Vec<i64> = (0..30_000).filter(|id| id % 3 != 0).collect();
    assert_eq!(ids, expected);
}

/// A build-side semi join keyed on column 0 of both sides, emitting the
/// listed build columns once for every build row some probe row matched, and
/// no probe column.
fn build_semi_join(build_columns: Vec<usize>) -> JoinSpec {
    let build_fields = int64_fields(build_columns.len());
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: Vec::new(),
        build_output_indices: build_columns,
        probe_fields: Vec::new(),
        build_fields,
        kind: JoinKind::BuildSemi,
        residual_filters: None,
        build_filters: Vec::new(),
    }
}

#[test]
fn build_semi_join_emits_only_the_matched_build_rows() {
    let d = dispatch(4);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20, 30])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[20, 20, 99])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], build_semi_join(vec![0]))
        .collect()
        .unwrap();

    // 20 is matched twice and still comes out once; 99 names no build row.
    assert_eq!(collect_i64s(&results, 0), vec![20]);
    assert!(results.iter().all(|batch| batch.num_columns() == 1));
}

#[test]
fn every_matched_build_row_reaches_a_build_semi_join_output() {
    let d = dispatch(4);
    let build_ids: Vec<i64> = (0..30_000).collect();
    let probe_ids: Vec<i64> = (0..30_000).step_by(3).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], build_semi_join(vec![0]))
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    let expected: Vec<i64> = (0..30_000).step_by(3).collect();
    assert_eq!(ids, expected);
}

#[test]
fn a_build_semi_join_carries_the_build_columns_it_lists() {
    let d = dispatch(4);
    let build = values_input(
        &d,
        vec![two_int64_batch(("id", "score"), &[10, 20, 30], &[1, 2, 3])],
    )
    .record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[30, 10])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], build_semi_join(vec![1]))
        .collect()
        .unwrap();

    // Only the listed score column comes out, not the key it was matched on.
    let mut scores = collect_i64s(&results, 0);
    scores.sort();
    assert_eq!(scores, vec![1, 3]);
    assert!(results.iter().all(|batch| batch.num_columns() == 1));
}

/// A mark join keyed on column 0 of both sides, emitting the listed probe
/// columns plus the nullable boolean marker column after them.
fn probe_mark_join(probe_columns: Vec<usize>) -> JoinSpec {
    let probe_fields = int64_fields(probe_columns.len());
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: probe_columns,
        build_output_indices: Vec::new(),
        probe_fields,
        build_fields: Vec::new(),
        kind: JoinKind::ProbeMark,
        residual_filters: None,
        build_filters: Vec::new(),
    }
}

/// Every output row's first column and marker, sorted.
fn collect_marked_keys(results: &[RecordBatch]) -> Vec<(Option<i64>, Option<bool>)> {
    let mut rows: Vec<(Option<i64>, Option<bool>)> = results
        .iter()
        .flat_map(|batch| {
            let keys = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let marks = batch
                .column(batch.num_columns() - 1)
                .as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .unwrap();
            (0..batch.num_rows())
                .map(|i| {
                    (
                        keys.is_valid(i).then(|| keys.value(i)),
                        marks.is_valid(i).then(|| marks.value(i)),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn mark_join_flags_each_probe_row_with_its_match() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 20])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 30])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(
        collect_marked_keys(&results),
        vec![(Some(10), Some(true)), (Some(30), Some(false)),]
    );
}

#[test]
fn a_probe_row_matching_several_build_rows_is_marked_once() {
    let d = dispatch(1);
    let build = values_input(&d, vec![int64_batch("id", &[10, 10, 10])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_marked_keys(&results), vec![(Some(10), Some(true))]);
}

#[test]
fn a_null_keyed_probe_row_marks_null() {
    let d = dispatch(1);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![int64_batch("id", &[10])]).record_batches();
    let probe = values_input(&d, vec![nullable(vec![None, Some(10), Some(30)])]).record_batches();

    let mut spec = probe_mark_join(vec![0]);
    spec.probe_fields = vec![Field::new("f0", DataType::Int64, true)];
    let results = probe
        .join(build, &[DataType::Int64], spec)
        .collect()
        .unwrap();

    // The null key compares as unknown against a non-empty build side; the
    // plain miss stays a definitive FALSE because no build key is null.
    assert_eq!(
        collect_marked_keys(&results),
        vec![
            (None, None),
            (Some(10), Some(true)),
            (Some(30), Some(false)),
        ]
    );
}

#[test]
fn a_null_keyed_build_row_turns_misses_into_null_marks() {
    let d = dispatch(1);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![nullable(vec![None, Some(10)])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[10, 30])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    // 30 matches nothing, but the null build key could have been anything,
    // so its miss is unknown rather than FALSE.
    assert_eq!(
        collect_marked_keys(&results),
        vec![(Some(10), Some(true)), (Some(30), None),]
    );
}

#[test]
fn a_null_key_seen_by_any_build_worker_turns_misses_into_null_marks() {
    let d = dispatch(4);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(
        &d,
        vec![
            nullable(vec![Some(10)]),
            nullable(vec![Some(20)]),
            nullable(vec![None]),
            nullable(vec![Some(40)]),
        ],
    )
    .record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &[30])]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    assert_eq!(collect_marked_keys(&results), vec![(Some(30), None)]);
}

#[test]
fn an_empty_build_side_marks_every_probe_row_false() {
    let d = dispatch(1);
    let nullable = |values: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap()
    };
    let build = values_input(&d, vec![int64_batch("id", &[])]).record_batches();
    let probe = values_input(&d, vec![nullable(vec![None, Some(10)])]).record_batches();

    let mut spec = probe_mark_join(vec![0]);
    spec.probe_fields = vec![Field::new("f0", DataType::Int64, true)];
    let results = probe
        .join(build, &[DataType::Int64], spec)
        .collect()
        .unwrap();

    // IN over an empty set is FALSE for every row, the null key included.
    assert_eq!(
        collect_marked_keys(&results),
        vec![(None, Some(false)), (Some(10), Some(false)),]
    );
}

#[test]
fn marked_batches_stay_within_the_output_batch_size() {
    let d = dispatch(1);
    // One oversized probe batch: the join must re-chunk its marked output,
    // since downstream accumulators size to RECORD_BATCH_SIZE.
    let probe_ids: Vec<i64> = (0..dispatch::RECORD_BATCH_SIZE as i64 * 3 + 7).collect();
    let build = values_input(&d, vec![int64_batch("id", &[5])]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    assert!(
        results
            .iter()
            .all(|batch| batch.num_rows() <= dispatch::RECORD_BATCH_SIZE)
    );
    assert_eq!(
        results.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        probe_ids.len()
    );
}

#[test]
fn a_mark_join_spanning_batches_marks_every_probe_row() {
    let d = dispatch(4);
    let build_ids: Vec<i64> = (0..30_000).step_by(3).collect();
    let probe_ids: Vec<i64> = (0..30_000).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_mark_join(vec![0]))
        .collect()
        .unwrap();

    let marked = collect_marked_keys(&results);
    assert_eq!(marked.len(), 30_000);
    let trues = marked
        .iter()
        .filter(|(_, mark)| *mark == Some(true))
        .count();
    let falses = marked
        .iter()
        .filter(|(_, mark)| *mark == Some(false))
        .count();
    assert_eq!(trues, 10_000);
    assert_eq!(falses, 20_000);
}

#[test]
fn an_anti_join_spanning_batches_emits_each_unmatched_probe_row_once() {
    let d = dispatch(4);
    // Every third key three times over on the build side, so the other two
    // thirds of the probe rows are the anti output.
    let build_ids: Vec<i64> = (0..30_000).step_by(3).flat_map(|id| [id, id, id]).collect();
    let probe_ids: Vec<i64> = (0..30_000).collect();
    let build = values_input(&d, vec![int64_batch("id", &build_ids)]).record_batches();
    let probe = values_input(&d, vec![int64_batch("id", &probe_ids)]).record_batches();

    let results = probe
        .join(build, &[DataType::Int64], probe_anti_join(vec![0]))
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    let expected: Vec<i64> = (0..30_000).filter(|id| id % 3 != 0).collect();
    assert_eq!(ids, expected);
}

#[test]
fn join_on_two_key_columns_needs_both_to_match() {
    let d = dispatch(1);
    let build = values_input(
        &d,
        vec![two_int64_batch(("k0", "k1"), &[1, 1, 2], &[10, 20, 30])],
    )
    .record_batches();
    let probe = values_input(
        &d,
        vec![two_int64_batch(("k0", "k1"), &[1, 2, 7], &[20, 30, 30])],
    )
    .record_batches();

    let results = probe
        .join(
            build,
            &[DataType::Int64, DataType::Int64],
            JoinSpec {
                probe_key_indices: vec![0, 1],
                build_key_indices: vec![0, 1],
                probe_output_indices: vec![0, 1],
                build_output_indices: vec![0, 1],
                probe_fields: int64_fields(2),
                build_fields: int64_fields(2),
                kind: JoinKind::Inner,
                residual_filters: None,
                build_filters: Vec::new(),
            },
        )
        .collect()
        .unwrap();

    // Only (1, 20) and (2, 30) exist on both sides.
    let mut pairs: Vec<(i64, i64)> = collect_i64s(&results, 0)
        .into_iter()
        .zip(collect_i64s(&results, 1))
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(1, 20), (2, 30)]);
}

#[test]
fn a_build_filter_publishes_the_key_bounds() {
    let probe = int64_batch("key", &[1, 2, 3]);
    let build = int64_batch("key", &[42, 7, 19]);
    let min_slot = Arc::new(BoundarySlot::new());
    let max_slot = Arc::new(BoundarySlot::new());
    let key_bitset_slot = Arc::new(KeyBitsetSlot::new());
    let mut spec = inner_join(vec![0], vec![0]);
    spec.build_filters = vec![JoinBuildFilter {
        build_column: 0,
        min_slot: min_slot.clone(),
        max_slot: max_slot.clone(),
        key_bitset_slot: key_bitset_slot.clone(),
    }];

    let d = dispatch(2);
    let probe_input = values_input(&d, vec![probe]).record_batches();
    let build_input = values_input(&d, vec![build]).record_batches();
    probe_input
        .join(build_input, &[DataType::Int64], spec)
        .collect()
        .unwrap();

    assert_eq!(boundary_i64(&min_slot), 7);
    assert_eq!(boundary_i64(&max_slot), 42);
    let bitset = key_bitset_slot.get().expect("sparse keys seal a bitset");
    let mut survivors = Vec::new();
    let probe_keys: ArrayRef = Arc::new(Int64Array::from(vec![6, 7, 19, 42, 43]));
    bitset.select(&probe_keys, &mut survivors);
    assert_eq!(survivors, vec![1, 2, 3]);
}

fn boundary_i64(slot: &BoundarySlot) -> i64 {
    use arrow_array::Datum;
    let boundary = slot.boundary().expect("the sealed build published a bound");
    let (array, _) = boundary.get();
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("an Int64 key publishes an Int64 bound")
        .value(0)
}

#[test]
fn sparse_build_keys_join_exactly() {
    // Sparse keys over a wide domain seal the probe's key bitset; only rows
    // whose key the build side holds may come out.
    let probe = int64_batch("key", &[3, 10_000, 42, 777, 5]);
    let build = two_int64_batch(("key", "payload"), &[5, 10_000, 9_999], &[50, 100, 99]);

    let d = dispatch(2);
    let results = values_input(&d, vec![probe])
        .record_batches()
        .join(
            values_input(&d, vec![build]).record_batches(),
            &[DataType::Int64],
            inner_join(vec![0], vec![1]),
        )
        .collect()
        .unwrap();

    let mut pairs: Vec<(i64, i64)> = collect_i64s(&results, 0)
        .into_iter()
        .zip(collect_i64s(&results, 1))
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(5, 50), (10_000, 100)]);
}

#[test]
fn a_probe_anti_join_keeps_rows_a_key_filter_would_drop() {
    // An anti join owes every non-matching probe row an output, so no key
    // bitset is sealed for it even over sparse build keys.
    let probe = int64_batch("key", &[1, 5_000, 9_999]);
    let build = int64_batch("key", &[5_000]);
    let mut spec = inner_join(vec![0], vec![]);
    spec.kind = JoinKind::ProbeAnti;

    let d = dispatch(2);
    let results = values_input(&d, vec![probe])
        .record_batches()
        .join(
            values_input(&d, vec![build]).record_batches(),
            &[DataType::Int64],
            spec,
        )
        .collect()
        .unwrap();

    let mut kept = collect_i64s(&results, 0);
    kept.sort();
    assert_eq!(kept, vec![1, 9_999]);
}
