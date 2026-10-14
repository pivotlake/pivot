//! A harness for the pivot-managed mode: a datastore over a temporary local
//! directory, wrapped in a Pivot catalog and planner so tests speak SQL to it
//! exactly as a server would.

#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use catalog::datastore::DatastoreTransaction as _;
use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use catalog::{Datastore, PivotCatalog};
use datastore_pivotlake::{
    CompactionConfig, MaintenanceConfig, PivotlakeDatastore, TableFormat, VacuumConfig,
};
use dispatch::{DataFlowDispatcher, Dispatch};
use object_storage::{LocalStore, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;
use planner::{DEFAULT_DATASTORE_NAME, Planner};
use tempfile::TempDir;

/// A metastore serving no datastores and only the built-in trusted user:
/// the tests wrap an already-open datastore in a catalog themselves.
#[derive(Debug)]
struct TrustMetastore;

impl Metastore for TrustMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DEFAULT_DATASTORE_NAME
    }

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
    }
}

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init()
    });
}

/// One dispatch pool shared by every test in a binary.
pub fn shared_dispatcher() -> DataFlowDispatcher {
    static DISPATCH: std::sync::OnceLock<Dispatch> = std::sync::OnceLock::new();
    init_tracing();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(2, 64, None))
        .dispatcher()
        .clone()
}

/// A managed datastore over a temporary directory, with the catalog and
/// planner that speak SQL to it. Dropping it stops the datastore's background
/// tasks before the directory goes away.
pub struct Database {
    /// Taken by [`close`](Self::close), which hands the directory to a reopen.
    /// `None` for a database over a remote store.
    dir: Option<TempDir>,
    /// The root the datastore's store was opened on: the directory, or the
    /// bucket URI.
    root: String,
    pub datastore: Arc<PivotlakeDatastore>,
    catalog: PivotCatalog,
    planner: Planner,
    runtime: tokio::runtime::Runtime,
}

impl Deref for Database {
    type Target = PivotCatalog;

    fn deref(&self) -> &Self::Target {
        &self.catalog
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        self.catalog.abort();
    }
}

impl Database {
    /// A fresh database with no maintenance running.
    pub fn new() -> Self {
        let dir = TempDir::new().unwrap();
        Self::open_at(dir, None)
    }

    /// Stop this database and hand back its directory, for a reopen.
    pub fn close(mut self) -> TempDir {
        self.catalog.abort();
        self.dir.take().expect("the directory is held until close")
    }

    /// Reopen the database in `dir`, the way a restarted server would.
    pub fn reopen(dir: TempDir) -> Self {
        Self::open_at(dir, None)
    }

    pub fn path(&self) -> &Path {
        self.dir
            .as_ref()
            .expect("the directory is held until close")
            .path()
    }

    /// A fresh database whose refresh and vacuum loops run at `interval`.
    pub fn with_maintenance(interval: Duration) -> Self {
        let dir = TempDir::new().unwrap();
        Self::open_at(
            dir,
            Some(MaintenanceConfig {
                refresh_interval: interval,
                compaction: None,
                vacuum: Some(VacuumConfig {
                    poll_interval: interval,
                }),
            }),
        )
    }

    /// A fresh database whose `COMPACT` statements run with `compaction`'s
    /// sizes, and no timers.
    pub fn with_compaction(compaction: CompactionConfig) -> Self {
        let dir = TempDir::new().unwrap();
        Self::open_at(
            dir,
            Some(MaintenanceConfig {
                refresh_interval: Duration::from_secs(3600),
                compaction: Some(compaction),
                vacuum: None,
            }),
        )
    }

    fn open_at(dir: TempDir, maintenance: Option<MaintenanceConfig>) -> Self {
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let root = dir.path().to_string_lossy().into_owned();
        Self::open_over(Some(dir), root, store, maintenance)
    }

    /// A database over `store`, opened on `root` (a bucket URI the
    /// object-store harness provides), with no maintenance running.
    pub fn over_store(root: String, store: Arc<dyn ObjectStore>) -> Self {
        Self::open_over(None, root, store, None)
    }

    fn open_over(
        dir: Option<TempDir>,
        root: String,
        store: Arc<dyn ObjectStore>,
        maintenance: Option<MaintenanceConfig>,
    ) -> Self {
        let dispatcher = shared_dispatcher();
        let datastore = PivotlakeDatastore::from_store(
            store,
            &dispatcher,
            maintenance,
            Some(TableFormat::Iceberg),
        )
        .expect("open the managed datastore");
        let catalog = PivotCatalog::new(
            HashMap::from([(
                DEFAULT_DATASTORE_NAME.to_string(),
                Arc::clone(&datastore) as Arc<dyn Datastore>,
            )]),
            DEFAULT_DATASTORE_NAME.to_string(),
            Arc::new(TrustMetastore),
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        {
            let _runtime = runtime.enter();
            catalog.start();
        }
        let planner = Planner::from_datastore_names(
            vec![DEFAULT_DATASTORE_NAME.to_string()],
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .unwrap();
        Self {
            dir,
            root,
            datastore,
            catalog,
            planner,
            runtime,
        }
    }

    /// The root the datastore was opened on, as a URI the test-support
    /// helpers open their own store from.
    pub fn root_uri(&self) -> String {
        self.root.clone()
    }

    /// INSERT `batches` into the table `name` through a transaction of the
    /// datastore's own and commit it, the way one client's statement runs.
    pub fn insert_rows(&self, name: &str, batches: Vec<RecordBatch>) {
        insert_rows(&self.datastore, name, batches);
    }

    /// Run one SQL statement to completion and commit it, returning its
    /// result batches.
    pub fn execute(&mut self, sql: &str) -> Vec<RecordBatch> {
        self.try_execute(sql)
            .unwrap_or_else(|error| panic!("`{sql}` failed: {error}"))
    }

    pub fn try_execute(&mut self, sql: &str) -> Result<Vec<RecordBatch>, String> {
        let transaction = self.catalog.begin_transaction();
        let batches = self
            .planner
            .plan(sql, transaction.clone())
            .map_err(|error| error.to_string())?
            .compile(&shared_dispatcher(), transaction.as_ref())
            .map_err(|error| error.to_string())?
            .collect()
            .map_err(|error| error.to_string())?;
        self.runtime
            .block_on(transaction.commit())
            .map_err(|error| error.to_string())?;
        Ok(batches)
    }

    /// Compact the table `name` (in the default schema) to completion, the
    /// way the server runs a `COMPACT` statement: through the transaction's
    /// catalog view, `final_sweep` dropping the routine guards. Returns the
    /// number of passes the round took.
    pub fn compact(&mut self, name: &str, final_sweep: bool) -> u64 {
        let transaction = self.catalog.begin_transaction();
        let table = SchemaQualifiedTableName::in_default_schema(name);
        self.runtime
            .block_on(transaction.compact(DEFAULT_DATASTORE_NAME, &table, final_sweep))
            .unwrap_or_else(|error| panic!("compacting `{name}` failed: {error}"))
    }

    /// Run `sql` until its result satisfies `is_seen`, or panic after
    /// `timeout`: how a test waits for a background sweep.
    pub fn execute_until(
        &mut self,
        sql: &str,
        timeout: Duration,
        is_seen: impl Fn(&[RecordBatch]) -> bool,
    ) -> Vec<RecordBatch> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let result = self.try_execute(sql);
            if let Ok(batches) = &result
                && is_seen(batches)
            {
                return result.unwrap();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "`{sql}` did not see the change within {timeout:?}: {result:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// The directory of the table `name` (in the default schema), relative
    /// to the database root: what the datastore named its storage.
    pub fn table_directory(&self, name: &str) -> String {
        self.datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("the table exists")
            .id()
            .to_string()
    }

    pub fn table_path(&self, name: &str) -> PathBuf {
        self.path().join(self.table_directory(name))
    }
}

/// INSERT `batches` into the table `name` of `datastore` through a
/// transaction of its own and commit it, the way one client's statement
/// runs: what a test uses to run several inserts at once from several
/// threads, which one planner cannot.
pub fn insert_rows(datastore: &Arc<PivotlakeDatastore>, name: &str, batches: Vec<RecordBatch>) {
    let transaction = Arc::clone(datastore).begin_transaction();
    let table = transaction
        .bind_table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema(name),
        )
        .expect("table lookup succeeds")
        .expect("the table was created");
    let dispatcher = shared_dispatcher();
    let rows = dispatch::values_input(&dispatcher, batches).record_batches();
    table
        .compile_insert(rows, &dispatcher)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(transaction.commit())
        .unwrap();
}

/// Every value of the `Int64` column `column` across `batches`, in order.
pub fn collect_i64s(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Every value of the string column `column` across `batches`, in order.
pub fn collect_strings(batches: &[RecordBatch], column: usize) -> Vec<String> {
    batches
        .iter()
        .flat_map(|batch| {
            let array = batch.column(column);
            (0..array.len())
                .map(|row| {
                    if let Some(strings) = array.as_string_view_opt() {
                        strings.value(row).to_string()
                    } else {
                        array.as_string::<i32>().value(row).to_string()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn single_i64(batches: &[RecordBatch]) -> i64 {
    let values = collect_i64s(batches, 0);
    assert_eq!(values.len(), 1, "one row expected");
    values[0]
}

/// A one-column (`value BIGINT`) batch of `values`.
pub fn int_batch(values: &[i64]) -> RecordBatch {
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow_array::Int64Array::from(values.to_vec())) as _],
    )
    .unwrap()
}

/// `values` as a one-column (`value BIGINT`) Parquet file, through the
/// reference writer, so the file carries no Parquet field ids.
pub fn int_parquet_bytes(values: &[i64]) -> Vec<u8> {
    use parquet::arrow::ArrowWriter;
    let batch = int_batch(values);
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    bytes
}

/// Write `values` as a one-column (`value BIGINT`) Parquet file at `path`.
pub fn write_int_parquet(path: &Path, values: &[i64]) {
    std::fs::write(path, int_parquet_bytes(values)).unwrap();
}

/// Past the default four-hour retention window.
pub const FIVE_HOURS_MS: u64 = 5 * 60 * 60 * 1000;

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Backdate a file's storage mtime past the retention window.
pub fn backdate_past_retention(path: &Path) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(std::time::SystemTime::now() - Duration::from_millis(FIVE_HOURS_MS))
        .unwrap();
}

/// The names of the Parquet files directly under `dir`, sorted.
pub fn parquet_files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".parquet"))
        .collect();
    names.sort();
    names
}

/// The names of the objects directly under `dir`, sorted.
pub fn names_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}
