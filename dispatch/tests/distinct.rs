//! Distinct integration tests: the keys-only group emitting the keys, alone and
//! as the duplicate-elimination stage of a delim join shape.

mod common;

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{
    Distinct, IntKeyExtractor, JoinKind, JoinSpec, RecordBatchOperatorSpec, values_input,
};

fn int64_batch(values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn inner_join_on_id() -> JoinSpec {
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: vec![0],
        build_output_indices: vec![0],
        probe_fields: vec![Field::new("id", DataType::Int64, false)],
        build_fields: vec![Field::new("id", DataType::Int64, false)],
        kind: JoinKind::Inner,
        residual_filters: None,
    }
}

#[test]
fn distinct_emits_each_key_once() {
    let d = dispatch(4);
    let batches = vec![int64_batch(&[1, 2, 2, 3]), int64_batch(&[3, 3, 4])];

    let results = values_input(&d, batches)
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Distinct>(vec![0], Vec::new(), None, ())
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4]);
}

/// The delim join dataflow shape: one producer feeds both a join side and a
/// distinct, and the distinct's rows feed the join's other side. Every original
/// row matches its de-duplicated key exactly once, so the join returns the
/// producer's rows and any site seeing a share of them shows up as missing rows.
#[test]
fn distinct_of_a_shared_producer_feeds_the_join_back() {
    const OUTER: usize = 0;
    const DELIM: usize = 1;
    let d = dispatch(4);
    let batches = vec![int64_batch(&[1, 2, 2, 3]), int64_batch(&[3, 3, 4])];
    let definition = values_input(&d, batches).record_batches();
    let deduplicated = RecordBatchOperatorSpec::cte_scan(&d, OUTER)
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Distinct>(vec![0], Vec::new(), None, ());

    let results = RecordBatchOperatorSpec::cte_scan(&d, DELIM)
        .join(
            RecordBatchOperatorSpec::cte_scan(&d, OUTER),
            &[DataType::Int64],
            inner_join_on_id(),
        )
        .with_cte(deduplicated, DELIM, 1)
        .with_cte(definition, OUTER, 2)
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2, 2, 3, 3, 3, 4]);
}
