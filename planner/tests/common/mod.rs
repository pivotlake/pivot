use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_json::ArrayWriter;
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::fixture;
use serde_json::Value;
use tempfile::TempDir;

use dispatch::{DataFlowDispatcher, Dispatch, Projection, RecordBatchOperatorSpec};
use goose::parquet::{ParquetTable, row_group_filter_from, table_input_with_filter};
use planner::Planner;
use planner::catalog::{Catalog, Column, DynamicScanPredicate, Table};
use planner::types::Type;

#[derive(Clone, Debug)]
struct TestTable {
    _dir: Arc<TempDir>,
    parquet_table: Arc<ParquetTable>,
    columns: Vec<Column>,
}

impl TestTable {
    fn new(dispatch: &Dispatch, columns: &[(&str, Type, ArrayRef)]) -> Self {
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

        // Read the footers once, over the dispatch worker pool. This drives a
        // dataflow, so it runs on the coordinator (here), not via run_on_worker.
        let parquet_table = Arc::new(
            ParquetTable::from_directory(dispatch.dispatcher(), dir.path())
                .expect("ParquetTable::from_directory failed"),
        );
        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();

        TestTable {
            _dir: Arc::new(dir),
            parquet_table,
            columns: cols,
        }
    }
}

impl Table for TestTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> planner::catalog::Result<RecordBatchOperatorSpec> {
        Ok(table_input_with_filter(
            dispatcher,
            &self.parquet_table,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> RecordBatchOperatorSpec {
        goose::parquet::materialize(input, self.parquet_table.clone(), projection)
    }
}

/// In-memory catalog used by tests. Pre-populated with `example_table` (see
/// the [`catalog`] fixture) and exposes [`TestCatalog::add_table`] so tests
/// that need extra schemas can register them at the start of the test.
#[derive(Debug)]
pub struct TestCatalog {
    tables: Mutex<HashMap<String, TestTable>>,
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
    pub fn add_table(&self, dispatch: &Dispatch, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.tables
            .lock()
            .unwrap()
            .insert(name.to_string(), TestTable::new(dispatch, columns));
    }
}

impl Catalog for TestCatalog {
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        self.tables
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .map(|t| Box::new(t) as _)
    }

    fn create_table(
        &self,
        _request: ::planner::catalog::CreateTableRequest,
        _dispatcher: &::dispatch::DataFlowDispatcher,
    ) -> ::planner::catalog::Result<::dispatch::RecordBatchOperatorSpec> {
        unreachable!("test helper catalog does not support CREATE TABLE")
    }
}

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

pub fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

/// A `Planner` paired with a handle to its backing catalog and a private
/// `Dispatch` for the workers running compiled plans.
///
/// The first two are bundled in one fixture (rather than two separate
/// fixtures) because rstest does not share fixture instances between a
/// sub-fixture dependency and a direct test parameter — they would resolve
/// to different `TestCatalog` instances and any tables registered through
/// the test's `catalog` would never reach the planner.
///
/// `dispatch` is owned by the fixture so each test gets its own worker pool
/// and matching `MemoryContext`, and tear-down is automatic when the fixture
/// drops at end of test.
pub struct TestingPlanner {
    pub planner: Planner,
    #[allow(dead_code)]
    pub catalog: Arc<TestCatalog>,
    // Held only for its `Drop` side-effect; access the dispatcher via
    // [`TestingPlanner::dispatcher`].
    #[allow(dead_code)]
    dispatch: Dispatch,
}

#[allow(dead_code)] // not all test binaries call every helper
impl TestingPlanner {
    /// Borrow the dispatcher to hand to `Plan::compile`.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        self.dispatch.dispatcher()
    }

    /// Register an ad-hoc table in the catalog. Uses this fixture's own
    /// `Dispatch` so the parquet write/open happens on a worker.
    pub fn add_table(&self, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.catalog.add_table(&self.dispatch, name, columns);
    }
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
    // `TestTable::new` calls `ParquetTable::from_directory`, which touches
    // `memory_ctx()` (file cache) and so must run on a worker — see
    // `TestTable::new` for the `run_on_worker` hop.
    let dispatch = Dispatch::spin_up(1, 32);
    let catalog = Arc::new(TestCatalog::new());
    catalog.add_table(
        &dispatch,
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
    TestingPlanner {
        planner,
        catalog,
        dispatch,
    }
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
