#![allow(dead_code)]

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::ops::Deref;
use std::sync::{Arc, Once};
use tempfile::TempDir;

use goose::parquet::ParquetTable;
use dispatch::{DataFlowDispatcher, Dispatch};

// The process-wide dispatcher, created on the first `init*` call.
// static DISPATCHER: OnceLock<Mutex<DataFlowDispatcher>> = OnceLock::new();

// pub fn init() {
//     init_with_workers(core_affinity::get_core_ids().unwrap().len());
// }

// pub fn init_with_workers(num_workers: usize) {
//     // Each test thread also drops result `RecordBatch`es here, so it needs its
//     // own `MemoryContext`. Create a 1-worker context and install it.
//     let factory = MemoryContextFactory::create_many(1)
//         .into_iter()
//         .next()
//         .unwrap();
//     init_memory_context(factory.create_memory_ctx());
//
//     DISPATCHER.get_or_init(|| {
//         let (exit, handles, dispatcher) = Dispatch::new(num_workers).into_parts();
//         Mutex::new(dispatcher)
//     });
// }

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

/// RAII wrapper around the owned `Dispatch`: derefs to its
/// `DataFlowDispatcher` for `table_input(&dispatch, …)` usage and calls
/// [`Dispatch::exit`] on drop to clean up worker threads.
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

pub fn parquet_table(
    dispatch: &DispatchGuard,
    batches: &[RecordBatch],
    dictionary: bool,
) -> (TempDir, Arc<ParquetTable>) {
    // Write the parquet file on the test thread (`ArrowWriter` is plain
    // `std::fs` IO — no `MemoryContext` needed).
    let dir = TempDir::new().unwrap();
    let schema = batches[0].schema();
    let path = dir.path().join("data.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(dictionary)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();

    let table = parquet_table_from_dir(dispatch, dir.path());
    (dir, table)
}

/// Load a `ParquetTable` from an already-populated directory. Runs
/// `ParquetTable::from_directory` on a worker because it touches
/// `memory_ctx()` via the file cache.
pub fn parquet_table_from_dir(
    dispatch: &DispatchGuard,
    dir: &std::path::Path,
) -> Arc<ParquetTable> {
    let dir_path = dir.to_owned();
    dispatch
        .run_on_worker(move || Arc::new(ParquetTable::from_directory(&dir_path).unwrap()))
        .expect("ParquetTable::from_directory failed")
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
