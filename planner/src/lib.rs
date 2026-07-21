//! `planner` turns SQL into an executable dataflow.
//!
//! The crate sits between [`duckdb_planner`] (which parses SQL and produces a
//! generic logical plan via an embedded DuckDB) and [`dispatch`] (which runs
//! physical dataflows on a thread-per-core worker pool). Its job is twofold:
//!
//! 1. **Translate** — Convert DuckDB's logical plan into a Pivot-native
//!    [`Plan`] tree, wiring each table reference to a concrete
//!    [`Table`](catalog::Table) from the user's [`Catalog`]
//!    and replacing foreign types (expressions, operators, scalars) with the
//!    ones in [`operator`] and [`expression`].
//! 2. **Compile** — Walk the Pivot plan and build a
//!    [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec) that
//!    `dispatch` can execute in parallel across all workers.
//!
//! The two steps live in distinct modules ([`plan`]/[`operator`]/[`expression`]
//! for the translation, [`compile`] for the lowering) so the Pivot plan is a
//! stable, inspectable IR independent of any particular executor, and so the
//! conversion-vs-compile failure modes stay separate.
//!
//! # Example
//!
//! A query over a minimal read-only catalog backed by an in-memory `HashMap`,
//! built around a parquet directory on disk:
//!
//! ```ignore
//! use std::collections::HashMap;
//! use std::path::Path;
//! use std::sync::Arc;
//!
//! use catalog::parquet::{ParquetTable, table_input};
//! use dispatch::{DataFlowDispatcher, Dispatch, Projection, RecordBatchOperatorSpec};
//! use planner::Planner;
//! use planner::catalog::{
//!     Catalog, CatalogTransaction, Column, CreateTableRequest, DynamicScanPredicate, Table,
//! };
//! use planner::types::Type;
//!
//! #[derive(Debug)]
//! struct MyTable {
//!     parquet: Arc<ParquetTable>,
//!     columns: Vec<Column>,
//! }
//!
//! impl Table for MyTable {
//!     fn compile(&self, dispatcher: &DataFlowDispatcher, projection: Projection, _filters: Vec<DynamicScanPredicate>, _emit_row_group_metadata: bool, _transaction: &dyn CatalogTransaction) -> planner::catalog::Result<RecordBatchOperatorSpec> {
//!         Ok(table_input(dispatcher, &self.parquet, projection, false))
//!     }
//!     fn columns(&self) -> Vec<Column> { self.columns.clone() }
//!     fn clone_box(&self) -> Box<dyn Table> { Box::new(MyTable { parquet: self.parquet.clone(), columns: self.columns.clone() }) }
//! }
//!
//! #[derive(Clone, Debug)]
//! struct MyTableTemplate {
//!     parquet: Arc<ParquetTable>,
//!     columns: Vec<Column>,
//! }
//!
//! #[derive(Debug)]
//! struct MyCatalog {
//!     tables: HashMap<String, MyTableTemplate>,
//! }
//!
//! // Tables resolve through a per-query transaction: a frozen snapshot of the
//! // catalog.
//! #[derive(Debug)]
//! struct MyTransaction {
//!     tables: HashMap<String, MyTableTemplate>,
//! }
//!
//! impl CatalogTransaction for MyTransaction {
//!     fn table(&self, name: &str) -> Option<Box<dyn Table>> {
//!         self.tables
//!             .get(name)
//!             .cloned()
//!             .map(|t| Box::new(MyTable { parquet: t.parquet, columns: t.columns }) as Box<dyn Table>)
//!     }
//!     fn as_any(&self) -> &dyn std::any::Any { self }
//! }
//!
//! impl Catalog for MyCatalog {
//!     fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
//!         Arc::new(MyTransaction { tables: self.tables.clone() })
//!     }
//!     fn create_table(&self, _req: CreateTableRequest, _dispatcher: &DataFlowDispatcher) -> planner::catalog::Result<RecordBatchOperatorSpec> {
//!         unimplemented!("this catalog is read-only")
//!     }
//! }
//!
//! // Wire one parquet directory into the catalog under the name "hits". The
//! // footers are read once here, over the dispatch worker pool.
//! let dispatch = Dispatch::spin_up(1, 10, None);
//! let columns = vec![Column { name: "URL".into(), col_type: Type::Utf8 }];
//! let parquet = Arc::new(ParquetTable::from_directory(dispatch.dispatcher(), Path::new("/tmp/hits"), &columns).unwrap());
//! let template = MyTableTemplate { parquet, columns };
//!
//! let mut tables = HashMap::new();
//! tables.insert("hits".to_string(), template);
//! let catalog: Arc<dyn Catalog> = Arc::new(MyCatalog { tables });
//!
//! let mut planner = Planner::new(catalog.clone());
//!
//! // One transaction per query: SQL -> Pivot Plan -> dispatch spec -> execution.
//! let transaction = catalog.begin_transaction();
//! let plan = planner.plan("SELECT COUNT(*) FROM hits WHERE URL <> 'foo'", transaction.clone()).unwrap();
//! let spec = plan.compile(dispatch.dispatcher(), transaction.as_ref()).unwrap();
//! let batches = spec.collect();
//! ```
//!
//! # Module map
//!
//! - [`catalog`] — [`Catalog`] / [`Table`](catalog::Table)
//!   traits the caller implements, plus the DuckDB adapters required to plug
//!   them into `duckdb-planner`.
//! - [`types`] — The small set of column types Pivot supports, plus
//!   conversions to and from DuckDB's [`LogicalTypeId`].
//! - [`expression`] — Pivot-native expression tree produced during translation.
//! - [`operator`] — Pivot-native logical operators (`Input`, `Projection`,
//!   `Filter`, …).
//! - [`plan`] — The [`Plan`] tree that wraps [`operator::Operator`]s.
//! - [`compile`] — Lowering from a [`Plan`] into a
//!   [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec).

mod build;
pub mod catalog;
pub mod compile;
pub mod dynamic_filter;
pub mod expression;
pub mod operator;
pub mod plan;
#[cfg(test)]
mod test_support;
pub mod types;
use std::sync::Arc;

use crate::catalog::Catalog;
pub use operator::{Operator, SetVariable, TableFunction, TableFunctionSignature};
pub use plan::{Plan, PlanNode};
use thiserror::Error;

use crate::catalog::{CatalogTransaction, DuckDBCatalogAdapter, DuckDBTransactionAdapter};
pub use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, ScalarValue,
};

/// The datastore a single-catalog [`Planner`] attaches its catalog under (and
/// makes DuckDB's current database). Kept in sync with the catalog crate's
/// `DEFAULT_DATASTORE_NAME`; `planner` sits below `catalog` and can't import it.
const DEFAULT_DATASTORE: &str = "default";

/// Errors surfaced by [`Planner::plan`].
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Planning(#[from] duckdb_planner::Error),
    #[error("Error converting plan: {0}")]
    PlanConversion(#[from] plan::Error),
}

/// Entry point for using crate: plans SQL statements against a [`Catalog`] into a Pivot [`Plan`].
///
/// A `Planner` owns a [`duckdb_planner::PlannerContext`] wrapped around a
/// [`DuckDBCatalogAdapter`], so DuckDB can resolve table names against the
/// Pivot catalog during binding. A single `Planner` instance can be reused
/// for many queries.
pub struct Planner {
    catalog: Arc<dyn Catalog>,
    planner_context: duckdb_planner::PlannerContext,
}

impl Planner {
    /// Create a `Planner` backed by a single `catalog`, attached to DuckDB as the
    /// [`DEFAULT_DATASTORE`] database (its current database, so unqualified names
    /// resolve against it). The degenerate one-datastore case of
    /// [`with_datastores`](Self::with_datastores).
    pub fn new(catalog: Arc<dyn Catalog>) -> Self {
        Self::with_datastores(
            vec![DEFAULT_DATASTORE.to_string()],
            DEFAULT_DATASTORE.to_string(),
            catalog,
        )
    }

    /// Create a `Planner` over several named datastores, each attached to DuckDB
    /// as its own database so a query can name it (`db.schema.t`). `default_name`
    /// is the current database. `composite` is the compile-time catalog — it must
    /// span every datastore for `begin_transaction`/`create_table` (typically a
    /// `catalog::PivotCatalog` over the same set); its transaction routes each
    /// table to the right datastore's snapshot by the attach name.
    pub fn with_datastores(
        database_names: Vec<String>,
        default_name: String,
        composite: Arc<dyn Catalog>,
    ) -> Self {
        Self {
            // The static provider only answers generic scalar functions, so a
            // single one over the composite serves every attached datastore.
            planner_context: duckdb_planner::PlannerContext::new(
                Arc::new(DuckDBCatalogAdapter {
                    catalog: composite.clone(),
                }),
                database_names,
                default_name,
            ),
            catalog: composite,
        }
    }

    /// Plan a SQL statement into a Pivot [`Plan`], inside `transaction`.
    ///
    /// The statement is first planned by DuckDB, which resolves every table
    /// reference through the transaction's catalog snapshot (carried across the
    /// bridge on the DuckDB transaction the planner starts internally), then the
    /// resulting plan handles are walked into a Pivot [`PlanNode`] tree (see the
    /// `build` module).
    pub fn plan(
        &mut self,
        query: &str,
        transaction: Arc<dyn CatalogTransaction>,
    ) -> Result<Plan, Error> {
        let adapter = Arc::new(DuckDBTransactionAdapter { transaction });
        let planned = self.planner_context.plan(query, adapter)?;
        let mut root = build::build_plan(planned.root())?;
        // Push a top-k limit into a grouped aggregate that feeds ORDER BY DESC.
        root.annotate_group_topn();
        // Push a plain LIMIT (no ORDER BY) into a grouped aggregate beneath it.
        root.annotate_group_limit();
        Ok(Plan {
            catalog: self.catalog.clone(),
            root,
            output_names: planned.into_output_names(),
        })
    }
}
