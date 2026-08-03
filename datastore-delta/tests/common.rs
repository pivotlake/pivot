#![allow(dead_code)]

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::ops::Deref;
use std::sync::{Arc, Once};
use tempfile::TempDir;

use datastore::DatastoreTransaction;
use datastore_delta::parquet::ParquetTable;
use dispatch::{DataFlowDispatcher, Dispatch};
use planner::catalog::Result as CatalogResult;

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
///
/// The buffer count is a process total that `spin_up` divides across NUMA node
/// regions, and workers only allocate from their own region. A multi-node
/// runner therefore halves what each worker can reach, and tests with a few
/// megabytes of pinned results (cached footers, zero-copy string views) run a
/// 10-slot ring dry. 32 leaves headroom either way.
pub fn dispatch(workers: usize) -> DispatchGuard {
    dispatch_with_buffers(workers, 32)
}

/// Like [`dispatch`], but with an explicit file-cache buffer count — needed when
/// a test keeps many row groups (or footers) in flight at once, since each pins
/// a cache slot.
pub fn dispatch_with_buffers(workers: usize, buffers: usize) -> DispatchGuard {
    init_tracing();
    DispatchGuard(Some(Dispatch::spin_up(workers, buffers, None)))
}

/// Commit one datastore transaction to completion. Writing commits hop to
/// Tokio's blocking pool, so direct datastore tests use this small runtime just
/// as the server's async query handler does.
pub fn commit_datastore_transaction(
    transaction: Arc<dyn DatastoreTransaction>,
) -> CatalogResult<()> {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(transaction.commit())
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

/// Reload `name` to its latest committed manifest and return its current row
/// groups for inspection. The background refresh does the same sweep; here we
/// drive it on a cloned-out table handle.
pub fn current_parquet(
    datastore: &datastore_delta::DeltaDatastore,
    name: &str,
) -> Arc<ParquetTable> {
    let mut table = datastore.table_handle(name).expect("table exists");
    table.refresh().expect("manifest reload");
    table.build_scan_view(&[]).expect("build scan view")
}

/// Load a `ParquetTable` from an already-populated directory. Drives the
/// metadata-fetch dataflow, so it runs on the coordinator (the test thread), not
/// inside `run_on_worker`.
pub fn parquet_table_from_dir(
    dispatch: &DispatchGuard,
    dir: &std::path::Path,
) -> Arc<ParquetTable> {
    Arc::new(
        ParquetTable::from_directory(dispatch, dir, &[])
            .expect("ParquetTable::from_directory failed"),
    )
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

pub fn extract_count(batches: &[RecordBatch]) -> i64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
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

/// Write `batches` into `dir` as Parquet, through the engine's own write path:
/// a table over that directory, an INSERT of the rows, and a commit. This is how
/// a test gets files our writer produced, and it exercises what production runs
/// rather than a separate entry point kept alive for tests.
///
/// The columns are taken from the batches' schema, so a caller only has to pass
/// the rows it wants written.
pub fn write_parquet_files(
    dispatch: &DispatchGuard,
    dir: &std::path::Path,
    batches: Vec<RecordBatch>,
) {
    use planner::catalog::{Column, CreateTableRequest};

    let schema = batches[0].schema();
    let columns: Vec<Column> = schema
        .fields()
        .iter()
        .map(|field| Column {
            name: field.name().clone(),
            col_type: planner::types::type_from_physical(field.data_type())
                .expect("a test writes a column type the engine has"),
        })
        .collect();

    let database = TempDir::new().unwrap();
    let datastore = datastore_delta::DeltaDatastore::open_local(database.path(), dispatch).unwrap();
    let mut options = std::collections::HashMap::new();
    options.insert("path".to_string(), dir.to_string_lossy().into_owned());
    let creation = datastore.clone().begin_transaction();
    creation
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            name: "written".to_string(),
            columns,
            options,
            if_not_exists: false,
        })
        .unwrap()
        .compile(dispatch)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(creation).unwrap();

    let insert = datastore.begin_transaction();
    let table = insert
        .bind_table(planner::DEFAULT_DATASTORE_NAME, "written")
        .expect("the table was created");
    let rows = dispatch::values_input(dispatch, batches).record_batches();
    table
        .compile_insert(rows, dispatch)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(insert).unwrap();
}
