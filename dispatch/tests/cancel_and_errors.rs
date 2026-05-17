mod common;

use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_buffer::BooleanBuffer;

use common::*;
use dispatch::{Projection, table_input};

/// A panicking filter on every batch should surface as an error from
/// `.collect()`, not a silent empty result.
#[test]
fn panic_in_filter_returns_error() {
    let dispatcher = dispatch(1);
    let names: Vec<&str> = (0..1000).map(|_| "x").collect();
    let values: Vec<i64> = (0..1000).collect();
    let (_dir, table) = parquet_table(&dispatcher, &[strings_and_ints(&names, &values)], true);
    let result = table_input(&dispatcher, &table, Projection::columns([1]), false)
        .filter(|| move |_batch: &RecordBatch| panic!("intentional panic in filter"))
        .count()
        .collect();

    // Assert: collect() returned an error mentioning the panic.
    let err = result.expect_err("expected an error from a panicking filter");
    let msg = err.to_string();
    assert!(
        msg.contains("intentional panic in filter"),
        "error message should surface the panic; got: {msg}"
    );
}

/// Cancelling a query before draining its output should:
///   1. not hang `.collect()`,
///   2. not return an error (cancel is not a failure mode),
///   3. return at most the full result, often less.
#[test]
fn cancelled_query_returns_without_hanging() {
    // Setup: a table large enough that finishing in one tick is unlikely.
    let dispatcher = dispatch(1);
    let n = 200_000;
    let names: Vec<&str> = (0..n).map(|_| "x").collect();
    let values: Vec<i64> = (0..n as i64).collect();
    let (_dir, table) = parquet_table(&dispatcher, &[strings_and_ints(&names, &values)], true);

    // Execute: kick off the query, immediately cancel, then drain the handle.
    let handle = table_input(&dispatcher, &table, Projection::columns([1]), false)
        .filter(|| {
            move |batch: &RecordBatch| {
                let col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |_| true))
            }
        })
        .execute();
    handle.cancel();
    let result = handle.collect();

    // Assert: collect succeeded and returned no more rows than the table holds.
    let batches = result.expect("cancelled query should not return an error");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total <= n,
        "cancelled query returned more rows ({total}) than the table ({n})"
    );
}
