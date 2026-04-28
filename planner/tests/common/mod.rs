use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_json::ArrayWriter;
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::fixture;
use serde_json::Value;
use tempfile::TempDir;

use dispatch::{ParquetTable, Projection, RecordBatchOperatorSpec, table_input};
use planner::Planner;
use planner::catalog::{Catalog, Column, Table};
use planner::types::Type;

static INIT: Once = Once::new();

/// Initialize the `dispatch` worker pool. Idempotent; safe to call from every
/// test. The `testing_planner` fixture calls this for you, so most tests don't
/// need to invoke it directly — only tests that build their own `Planner`
/// (e.g. with a custom `Catalog`) need to.
pub fn init() {
    INIT.call_once(|| dispatch::init(core_affinity::get_core_ids().unwrap().len()));
}

#[derive(Debug)]
struct TestTable {
    _dir: TempDir,
    parquet_table: Arc<ParquetTable>,
    columns: Vec<Column>,
}

impl TestTable {
    fn new(columns: &[(&str, Type, ArrayRef)]) -> Self {
        let dir = TempDir::new().unwrap();

        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, _, array)| Field::new(*name, array.data_type().clone(), false))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = columns.iter().map(|(_, _, array)| array.clone()).collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

        let file_path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = std::fs::File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let parquet_table = Arc::new(ParquetTable::from_directory(dir.path()).unwrap());
        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();

        TestTable {
            _dir: dir,
            parquet_table,
            columns: cols,
        }
    }
}

impl Table for TestTable {
    fn compile(&self, projection: Projection) -> RecordBatchOperatorSpec {
        table_input(&self.parquet_table, projection, false)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }
}

/// In-memory catalog used by tests. Pre-populated with `example_table` (see
/// the [`catalog`] fixture) and exposes [`TestCatalog::add_table`] so tests
/// that need extra schemas can register them at the start of the test.
#[derive(Debug)]
pub struct TestCatalog {
    tables: Mutex<HashMap<String, Arc<dyn Table>>>,
}

impl TestCatalog {
    fn new() -> Self {
        Self {
            tables: Mutex::new(HashMap::new()),
        }
    }

    /// Register an ad-hoc table backed by the given column data. Mirrors the
    /// shape of the old `make_planner_with_table` helper, but adds the table
    /// to the shared catalog so the existing `testing_planner` can plan
    /// queries against it.
    pub fn add_table(&self, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        let table: Arc<dyn Table> = Arc::new(TestTable::new(columns));
        self.tables
            .lock()
            .unwrap()
            .insert(name.to_string(), table);
    }
}

impl Catalog for TestCatalog {
    fn table(&self, name: &str) -> Option<Arc<dyn Table>> {
        self.tables.lock().unwrap().get(name).cloned()
    }

    fn create_table(
        &self,
        _request: ::planner::catalog::CreateTableRequest,
    ) -> ::planner::catalog::Result<()> {
        unreachable!("test helper catalog does not support CREATE TABLE")
    }
}

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

/// A `Planner` paired with a handle to its backing catalog.
///
/// The two are bundled in one fixture (rather than two separate fixtures)
/// because rstest does not share fixture instances between a sub-fixture
/// dependency and a direct test parameter — they would resolve to different
/// `TestCatalog` instances and any tables registered through the test's
/// `catalog` would never reach the planner.
pub struct TestingPlanner {
    pub planner: Planner,
    // Read by the `compile` and `types` test binaries to register extra
    // tables; the `plan` binary only uses `.planner`, which would otherwise
    // trip dead-code there since each binary lints this module independently.
    #[allow(dead_code)]
    pub catalog: Arc<TestCatalog>,
}

/// Shared planner backed by a catalog seeded with `example_table`:
/// `(a Int32, b Int32, c Int32, name Utf8)`, 5 rows:
///
/// ```text
/// a | b  | c   | name
/// --+----+-----+--------
/// 1 | 10 | 100 | alice
/// 2 | 20 | 200 | bob
/// 3 | 30 | 300 | charlie
/// 4 | 40 | 400 | dave
/// 5 | 50 | 500 | alice
/// ```
///
/// Tests that need a different schema reach into the returned `catalog` and
/// call [`TestCatalog::add_table`] before planning.
#[fixture]
pub fn testing_planner() -> TestingPlanner {
    init();
    let catalog = Arc::new(TestCatalog::new());
    catalog.add_table(
        "example_table",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
            ("b", Type::Int32, int_col(vec![10, 20, 30, 40, 50])),
            ("c", Type::Int32, int_col(vec![100, 200, 300, 400, 500])),
            (
                "name",
                Type::Utf8,
                str_col(vec!["alice", "bob", "charlie", "dave", "alice"]),
            ),
        ],
    );
    let planner = Planner::new(catalog.clone() as Arc<dyn Catalog>);
    TestingPlanner { planner, catalog }
}

// Used only by compile tests, but this module is shared across test binaries.
#[allow(dead_code)]
pub fn batches_to_json(batches: &[RecordBatch]) -> Vec<Value> {
    let mut writer = ArrayWriter::new(Vec::new());
    writer
        .write_batches(&batches.iter().collect::<Vec<_>>())
        .unwrap();
    writer.finish().unwrap();
    serde_json::from_slice(&writer.into_inner()).unwrap()
}