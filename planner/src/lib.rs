//! `planner` turns SQL into an executable dataflow.
//!
//! The crate sits between [`duckdb_planner`] (which parses SQL and produces a
//! generic logical plan via an embedded DuckDB) and [`dispatch`] (which runs
//! physical dataflows on a thread-per-core worker pool). Its job is twofold:
//!
//! 1. **Translate** — Convert DuckDB's logical plan into a Pivot-native
//!    [`Plan`] tree, wiring each table reference to a concrete
//!    [`BoundTable`](catalog::BoundTable) from the user's catalog
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
//! use datastore_delta::parquet::{ParquetTable, table_input};
//! use dispatch::{DataFlowDispatcher, Dispatch, Projection, RecordBatchOperatorSpec};
//! use planner::Planner;
//! use planner::catalog::{
//!     CatalogTransaction, Column, DynamicScanPredicate, BoundTable,
//! };
//! use planner::types::Type;
//!
//! #[derive(Debug)]
//! struct MyTable {
//!     parquet: Arc<ParquetTable>,
//!     columns: Vec<Column>,
//! }
//!
//! impl BoundTable for MyTable {
//!     fn compile(&self, dispatcher: &DataFlowDispatcher, projection: Projection, _filters: Vec<DynamicScanPredicate>, _emit_row_group_metadata: bool) -> planner::catalog::Result<RecordBatchOperatorSpec> {
//!         Ok(table_input(dispatcher, &self.parquet, projection, false))
//!     }
//!     fn columns(&self) -> Vec<Column> { self.columns.clone() }
//!     fn clone_box(&self) -> Box<dyn BoundTable> { Box::new(MyTable { parquet: self.parquet.clone(), columns: self.columns.clone() }) }
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
//! // A single read-only database presented to the planner as a one-entry
//! // catalog: it ignores the datastore qualifier and resolves by table name.
//! impl CatalogTransaction for MyTransaction {
//!     fn bind_table(&self, _datastore: &str, name: &str) -> Option<Box<dyn BoundTable>> {
//!         self.tables
//!             .get(name)
//!             .cloned()
//!             .map(|t| Box::new(MyTable { parquet: t.parquet, columns: t.columns }) as Box<dyn BoundTable>)
//!     }
//! }
//!
//! impl MyCatalog {
//!     fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
//!         Arc::new(MyTransaction { tables: self.tables.clone() })
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
//! let catalog = MyCatalog { tables };
//!
//! let mut planner = Planner::from_datastore_names(vec!["default".to_string()], "default".to_string());
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
//! - [`catalog`] — [`CatalogTransaction`](catalog::CatalogTransaction) / [`BoundTable`](catalog::BoundTable)
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

pub use operator::{Operator, SetVariable, TableFunction, TableFunctionSignature};
pub use plan::{Plan, PlanNode};
use thiserror::Error;

use crate::catalog::{CatalogTransaction, DuckDBScalarFunctionBinder, DuckDBTransactionAdapter};
pub use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, ScalarValue,
};

/// Errors surfaced by [`Planner::plan`].
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Planning(#[from] duckdb_planner::Error),
    #[error("Error converting plan: {0}")]
    PlanConversion(#[from] plan::Error),
}

/// Entry point for using crate: plans SQL statements into a Pivot [`Plan`].
///
/// A `Planner` owns a [`duckdb_planner::PlannerContext`] configured with the
/// datastore names to attach as DuckDB databases; it holds no catalog. Every
/// query hands [`plan`](Self::plan) a [`CatalogTransaction`], and all table and
/// DDL resolution flows through that. A single `Planner` instance can be reused
/// for many queries.
pub struct Planner {
    planner_context: duckdb_planner::PlannerContext,
}

impl Planner {
    /// Create a `Planner` over several named datastores, each attached to DuckDB
    /// as its own database so a query can name it (`db.schema.t`). `default_name`
    /// is the current database. The planner holds no catalog: the per-query
    /// [`CatalogTransaction`] passed to [`plan`](Self::plan) must span these
    /// datastores and routes each table to the right one's snapshot by the attach
    /// name.
    pub fn from_datastore_names(database_names: Vec<String>, default_name: String) -> Self {
        Self {
            // The static provider only answers generic scalar functions, shared
            // across every attached datastore.
            planner_context: duckdb_planner::PlannerContext::new(
                Arc::new(DuckDBScalarFunctionBinder),
                database_names,
                default_name,
            ),
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
            root,
            output_names: planned.into_output_names(),
        })
    }
}
