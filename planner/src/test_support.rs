//! In-crate blackbox test harness, available to the `#[cfg(test)]` unit tests
//! co-located with each expression module.
//!
//! A [`TestingPlanner`] fixture wraps a [`Planner`] over an in-memory
//! [`TestCatalog`] seeded with `example_table`, plus its own [`Dispatch`]
//! worker pool so compiled plans actually run. [`run`] is the one-liner the
//! inline tests drive: plan a SQL string, compile it, collect the batches, and
//! hand back JSON rows.
//!
//! Unlike the integration harness in `tests/common`, the table source here is
//! a plain in-memory [`dispatch`] nullary that replays a stored `RecordBatch`
//! — deliberately *not* goose's parquet reader. `goose` depends on `planner`,
//! so linking it into `planner`'s own unit tests would pull a second copy of
//! `planner` into the graph and the catalog types wouldn't unify.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_json::ArrayWriter;
use arrow_schema::{Field, Schema};
use rstest::fixture;
use serde_json::Value;

use crate::Planner;
use crate::catalog::{Catalog, Column, DynamicScanPredicate, Table};
use crate::types::Type;
use dispatch::{
    DataFlowDispatcher, Dispatch, Nullary, NullaryFactory, NullaryResult, Projection,
    RecordBatchOperatorSpec, Sender, WorkStatus,
};

/// A goose-free table source: replays a single in-memory `RecordBatch` once.
struct ReplayFactory {
    batch: RecordBatch,
}

impl NullaryFactory<RecordBatch> for ReplayFactory {
    type Nullary = Replay;
    fn build_nullary(self) -> Self::Nullary {
        Replay {
            batch: Some(self.batch),
        }
    }
}

struct Replay {
    batch: Option<RecordBatch>,
}

impl Nullary<RecordBatch> for Replay {
    fn run<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> NullaryResult<WorkStatus> {
        match self.batch.take() {
            Some(batch) => {
                sender.send(batch)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.batch.is_none())
    }
}

#[derive(Clone, Debug)]
struct TestTable {
    batch: RecordBatch,
    columns: Vec<Column>,
}

impl TestTable {
    fn new(columns: &[(&str, Type, ArrayRef)]) -> Self {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, _, array)| Field::new(*name, array.data_type().clone(), false))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = columns.iter().map(|(_, _, array)| array.clone()).collect();
        let batch = RecordBatch::try_new(schema, arrays).unwrap();
        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();
        TestTable {
            batch,
            columns: cols,
        }
    }
}

impl Table for TestTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
        _ctx: &dyn crate::catalog::QueryContext,
    ) -> crate::catalog::Result<RecordBatchOperatorSpec> {
        let projected = self
            .batch
            .project(projection.indices())
            .expect("projection indices are valid for the test schema");
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            [ReplayFactory { batch: projected }],
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
        _input: RecordBatchOperatorSpec,
        _projection: Projection,
        _ctx: &dyn crate::catalog::QueryContext,
    ) -> crate::catalog::Result<RecordBatchOperatorSpec> {
        unreachable!("the in-memory test table is never late-materialized")
    }
}

/// In-memory catalog seeded with `example_table`; [`TestCatalog::add_table`]
/// registers extra schemas for tests that need them.
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

    /// Register an ad-hoc table backed by the given column data.
    pub fn add_table(&self, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.tables
            .lock()
            .unwrap()
            .insert(name.to_string(), TestTable::new(columns));
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
        _request: crate::catalog::CreateTableRequest,
        _dispatcher: &DataFlowDispatcher,
    ) -> crate::catalog::Result<RecordBatchOperatorSpec> {
        unreachable!("test helper catalog does not support CREATE TABLE")
    }
}

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

/// A `Planner` paired with a handle to its backing catalog and a private
/// `Dispatch` for the workers running compiled plans. The catalog handle lets
/// a test register extra tables before planning; `dispatch` is owned so each
/// test gets its own worker pool, torn down when the fixture drops.
pub struct TestingPlanner {
    pub planner: Planner,
    pub catalog: Arc<TestCatalog>,
    dispatch: Dispatch,
}

impl TestingPlanner {
    /// Register an ad-hoc table in the catalog.
    pub fn add_table(&self, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.catalog.add_table(name, columns);
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
#[fixture]
pub fn testing_planner() -> TestingPlanner {
    let dispatch = Dispatch::spin_up(1, 32);
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
    TestingPlanner {
        planner,
        catalog,
        dispatch,
    }
}

/// Plan, compile, and run `sql` on the fixture, returning the result rows as
/// JSON. The one call the inline blackbox tests build their setup/execute/assert
/// around.
pub fn run(planner: &mut TestingPlanner, sql: &str) -> Vec<Value> {
    let results = planner
        .planner
        .plan(sql)
        .unwrap()
        .compile(planner.dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    batches_to_json(&results)
}

/// Serialize record batches to JSON rows.
pub fn batches_to_json(batches: &[RecordBatch]) -> Vec<Value> {
    let mut writer = ArrayWriter::new(Vec::new());
    writer
        .write_batches(&batches.iter().collect::<Vec<_>>())
        .unwrap();
    writer.finish().unwrap();
    serde_json::from_slice(&writer.into_inner()).unwrap()
}
