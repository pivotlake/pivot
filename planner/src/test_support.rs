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

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{
    ArrayRef, Date32Array, Int32Array, Int64Array, RecordBatch, Scalar, StringViewArray,
    TimestampMicrosecondArray,
};
use arrow_json::ArrayWriter;
use arrow_schema::{DataType, Field, Schema};
use rstest::fixture;
use serde_json::Value;

use crate::catalog::{
    BoundTable, CatalogTransaction, Column, DynamicScanPredicate, TableReference, TableRevision,
};
use crate::types::{Type, physical_arrow_type};
use crate::{DEFAULT_DATASTORE_NAME, Planner};
use dispatch::{
    DataFlowDispatcher, Dispatch, Nullary, NullaryFactory, NullaryResult, OperatorIO, Projection,
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
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        match self.batch.take() {
            Some(batch) => {
                sender.send(batch)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.batch.is_none())
    }
}

#[derive(Clone, Debug)]
struct TestTable {
    reference: TableReference,
    batch: RecordBatch,
    columns: Vec<Column>,
}

impl TestTable {
    fn new(name: &str, columns: &[(&str, Type, ArrayRef)]) -> Self {
        // Surface each column as the real arrow type its pivot `Type` decodes
        // into, mirroring the reader: a temporal column becomes Date32/Timestamp,
        // others keep the array they were given.
        let arrays: Vec<ArrayRef> = columns
            .iter()
            .map(|(_, col_type, array)| match col_type {
                Type::Date | Type::Timestamp => {
                    arrow::compute::cast(array, &physical_arrow_type(col_type)).unwrap()
                }
                _ => array.clone(),
            })
            .collect();
        let fields: Vec<Field> = columns
            .iter()
            .zip(&arrays)
            .map(|((name, _, _), array)| {
                Field::new(*name, array.data_type().clone(), array.null_count() > 0)
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays).unwrap();
        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();
        TestTable {
            reference: TableReference {
                datastore: DEFAULT_DATASTORE_NAME.to_string(),
                schema: crate::DEFAULT_SCHEMA_NAME.to_string(),
                table: name.to_string(),
            },
            batch,
            columns: cols,
        }
    }
}

impl BoundTable for TestTable {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{}:{}", self.reference.datastore, self.reference.table),
            version: "0".to_string(),
        }
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
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

    fn nullability(&self) -> Vec<bool> {
        // Report the batch's actual nullability, as a real binding derives it
        // from footers, so NULL-free test data keeps the fast paths.
        self.batch
            .columns()
            .iter()
            .map(|array| array.null_count() > 0)
            .collect()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        _input: RecordBatchOperatorSpec,
        _projection: Projection,
    ) -> crate::catalog::Result<RecordBatchOperatorSpec> {
        unreachable!("the in-memory test table is never late-materialized")
    }

    fn estimate_row_count(&self) -> crate::catalog::Result<Option<u64>> {
        Ok(Some(self.batch.num_rows() as u64))
    }

    /// Exact min/max over the stored column as a scalar of its physical int type,
    /// so the no-scan global MIN/MAX peephole ([`Aggregate::try_compile_from_stats`])
    /// can be exercised. Only the int columns the peephole supports are answered.
    fn column_min_max(
        &self,
        column: usize,
    ) -> crate::catalog::Result<Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)>> {
        Ok(self.column_extremes(column))
    }
}

impl TestTable {
    /// The least and greatest value of `column`, for a column of a type the
    /// peephole answers that holds no NULL.
    fn column_extremes(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        let arr = self.batch.column(column);
        // `values()` below reads the zero-filled null slots too, which would
        // fabricate a 0 extreme; a nullable column just skips the peephole.
        if arr.null_count() > 0 {
            return None;
        }
        match arr.data_type() {
            DataType::Int32 => {
                let values = arr.as_primitive::<Int32Type>().values();
                let lo = *values.iter().min()?;
                let hi = *values.iter().max()?;
                Some((
                    Scalar::new(Arc::new(Int32Array::from(vec![lo])) as ArrayRef),
                    Scalar::new(Arc::new(Int32Array::from(vec![hi])) as ArrayRef),
                ))
            }
            DataType::Int64 => {
                let values = arr.as_primitive::<Int64Type>().values();
                let lo = *values.iter().min()?;
                let hi = *values.iter().max()?;
                Some((
                    Scalar::new(Arc::new(Int64Array::from(vec![lo])) as ArrayRef),
                    Scalar::new(Arc::new(Int64Array::from(vec![hi])) as ArrayRef),
                ))
            }
            DataType::Date32 => {
                let values = arr.as_primitive::<Date32Type>().values();
                let lo = *values.iter().min()?;
                let hi = *values.iter().max()?;
                Some((
                    Scalar::new(Arc::new(Date32Array::from(vec![lo])) as ArrayRef),
                    Scalar::new(Arc::new(Date32Array::from(vec![hi])) as ArrayRef),
                ))
            }
            DataType::Timestamp(_, _) => {
                let values = arr.as_primitive::<TimestampMicrosecondType>().values();
                let lo = *values.iter().min()?;
                let hi = *values.iter().max()?;
                Some((
                    Scalar::new(Arc::new(TimestampMicrosecondArray::from(vec![lo])) as ArrayRef),
                    Scalar::new(Arc::new(TimestampMicrosecondArray::from(vec![hi])) as ArrayRef),
                ))
            }
            _ => None,
        }
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
            .insert(name.to_string(), TestTable::new(name, columns));
    }
}

/// One test transaction: a frozen clone of the catalog's table map, so binding
/// mirrors the production shape (snapshot per query) without any storage.
#[derive(Debug)]
struct TestTransaction {
    tables: HashMap<String, TestTable>,
}

impl CatalogTransaction for TestTransaction {
    // Every test table lives in the default schema, so that is the only schema
    // this catalog defines.
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> crate::catalog::Result<bool> {
        Ok(schema == crate::DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        reference: &TableReference,
    ) -> crate::catalog::Result<Option<Box<dyn BoundTable>>> {
        Ok(self
            .tables
            .get(&reference.table)
            .cloned()
            .map(|t| Box::new(t) as _))
    }

    fn table_revision(
        &self,
        reference: &TableReference,
    ) -> crate::catalog::Result<Option<TableRevision>> {
        Ok(self
            .tables
            .contains_key(&reference.table)
            .then(|| TableRevision {
                identity: format!("{}:{}", reference.datastore, reference.table),
                version: "0".to_string(),
            }))
    }
}

impl TestCatalog {
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        Arc::new(TestTransaction {
            tables: self.tables.lock().unwrap().clone(),
        })
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

    /// Plan `sql` inside a fresh transaction on this fixture's catalog.
    pub fn plan(&mut self, sql: &str) -> Result<crate::Plan, crate::Error> {
        let transaction = self.catalog.begin_transaction();
        self.planner.plan(sql, transaction)
    }

    /// A fresh transaction on this fixture's catalog, for compiling plans.
    pub fn transaction(&self) -> Arc<dyn CatalogTransaction> {
        self.catalog.begin_transaction()
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
/// and `nullable_table`: `(a Int32, b Int32, s Utf8)`, 6 rows where `b` and
/// `s` carry NULLs:
///
/// ```text
/// a | b    | s
/// --+------+------
/// 1 | 10   | x
/// 2 | NULL | NULL
/// 3 | 30   | y
/// 4 | NULL | NULL
/// 5 | 50   | x
/// 6 | NULL | z
/// ```
#[fixture]
pub fn testing_planner() -> TestingPlanner {
    let dispatch = Dispatch::spin_up(1, 32, None);
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
    catalog.add_table(
        "nullable_table",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5, 6])),
            (
                "b",
                Type::Int32,
                Arc::new(Int32Array::from(vec![
                    Some(10),
                    None,
                    Some(30),
                    None,
                    Some(50),
                    None,
                ])),
            ),
            (
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    Some("x"),
                    None,
                    Some("y"),
                    None,
                    Some("x"),
                    Some("z"),
                ])),
            ),
        ],
    );
    let planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
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
    batches_to_json(&run_batches(planner, sql))
}

/// Plan, compile, and run `sql`, returning the raw result batches. Lets a test
/// inspect the output arrow schema (e.g. that a DATE column surfaces as
/// `Date32`) rather than only the JSON-rendered values.
pub fn run_batches(planner: &mut TestingPlanner, sql: &str) -> Vec<RecordBatch> {
    collect_batches(planner, sql).unwrap()
}

/// Plan, compile, and run `sql`, returning the error the failing dataflow
/// reports. For tests asserting that bad data fails the query.
pub fn run_expecting_error(planner: &mut TestingPlanner, sql: &str) -> String {
    collect_batches(planner, sql).unwrap_err().to_string()
}

/// The error compiling `sql`'s plan reports, for tests pinning a shape pivot
/// declines to execute. Planning must succeed and compiling must not.
pub fn compile_expecting_error(planner: &mut TestingPlanner, sql: &str) -> String {
    let plan = planner.plan(sql).unwrap();
    plan.compile(
        planner.dispatch.dispatcher(),
        planner.transaction().as_ref(),
    )
    .err()
    .expect("compiling was expected to fail")
    .to_string()
}

/// The shared plan-compile-collect pipeline behind [`run_batches`] and
/// [`run_expecting_error`]: planning and compilation must succeed, running the
/// dataflow may fail.
fn collect_batches(
    planner: &mut TestingPlanner,
    sql: &str,
) -> Result<Vec<RecordBatch>, dispatch::DataFlowError> {
    planner
        .plan(sql)
        .unwrap()
        .compile(
            planner.dispatch.dispatcher(),
            planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
}

/// The value of a row's single column, by position rather than name. For tests
/// that check a computed select item's value, not its (DuckDB-derived) name.
pub fn only_column(row: &Value) -> &Value {
    let cols = row.as_object().expect("row is a JSON object");
    assert_eq!(cols.len(), 1, "only_column expects exactly one column");
    cols.values().next().unwrap()
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
