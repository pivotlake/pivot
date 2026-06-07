#![allow(dead_code)]
//! Shared helpers for dispatch's format-agnostic integration tests. The
//! Parquet-specific helpers live in `catalog`'s copy of this file, alongside the
//! Parquet reader tests that moved there.

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::ops::Deref;
use std::sync::{Arc, Once};

use dispatch::{DataFlowDispatcher, Dispatch};

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init()
    });
}

/// RAII wrapper around the owned `Dispatch`: derefs to its `DataFlowDispatcher`
/// and calls [`Dispatch::exit`] on drop to clean up worker threads.
pub struct DispatchGuard(Option<Dispatch>);

impl Deref for DispatchGuard {
    type Target = DataFlowDispatcher;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap().dispatcher()
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.0.take().unwrap().exit();
    }
}

/// Spin a `Dispatch` up on the calling thread and return a guard for it.
pub fn dispatch(workers: usize) -> DispatchGuard {
    init_tracing();
    DispatchGuard(Some(Dispatch::spin_up(workers, 10)))
}

pub fn strings_and_ints(names: &[&str], values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringViewArray::from(names.to_vec())),
            Arc::new(Int64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

pub fn extract_count(batches: &[RecordBatch]) -> u64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

pub fn collect_strings(batches: &[RecordBatch], col: usize) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i).to_string())
        })
        .collect()
}

pub fn collect_i64s(batches: &[RecordBatch], col: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b.column(col).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}

pub fn collect_u64s(batches: &[RecordBatch], col: usize) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}
