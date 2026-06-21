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
//! use planner::catalog::{Catalog, Column, CreateTableRequest, DynamicScanPredicate, Table};
//! use planner::types::Type;
//!
//! #[derive(Debug)]
//! struct MyTable {
//!     parquet: Arc<ParquetTable>,
//!     columns: Vec<Column>,
//! }
//!
//! impl Table for MyTable {
//!     fn compile(&self, dispatcher: &DataFlowDispatcher, projection: Projection, _filters: Vec<DynamicScanPredicate>, _emit_row_group_metadata: bool) -> planner::catalog::Result<RecordBatchOperatorSpec> {
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
//! impl Catalog for MyCatalog {
//!     fn table(&self, name: &str) -> Option<Box<dyn Table>> {
//!         self.tables
//!             .get(name)
//!             .cloned()
//!             .map(|t| Box::new(MyTable { parquet: t.parquet, columns: t.columns }) as Box<dyn Table>)
//!     }
//!     fn create_table(&self, _req: CreateTableRequest, _dispatcher: &DataFlowDispatcher) -> planner::catalog::Result<RecordBatchOperatorSpec> {
//!         unimplemented!("this catalog is read-only")
//!     }
//! }
//!
//! // Wire one parquet directory into the catalog under the name "hits". The
//! // footers are read once here, over the dispatch worker pool.
//! let dispatch = Dispatch::spin_up(1, 10);
//! let parquet = Arc::new(ParquetTable::from_directory(dispatch.dispatcher(), Path::new("/tmp/hits")).unwrap());
//! let template = MyTableTemplate {
//!     parquet,
//!     columns: vec![Column { name: "URL".into(), col_type: Type::Utf8 }],
//! };
//!
//! let mut tables = HashMap::new();
//! tables.insert("hits".to_string(), template);
//! let catalog: Arc<dyn Catalog> = Arc::new(MyCatalog { tables });
//!
//! let mut planner = Planner::new(catalog);
//!
//! // SQL -> Pivot Plan -> dispatch operator spec -> execution.
//! let plan = planner.plan("SELECT COUNT(*) FROM hits WHERE URL <> 'foo'").unwrap();
//! let spec = plan.compile(dispatch.dispatcher()).unwrap();
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
pub use operator::{Operator, SetVariable};
pub use plan::{Plan, PlanNode};
use thiserror::Error;

use crate::catalog::DuckDBCatalogAdapter;
pub use duckdb_planner::{DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId};

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
    /// Create a new `Planner` backed by `catalog`.
    ///
    /// The catalog is cloned into the internal DuckDB adapter so both the
    /// translation layer and DuckDB's binder see the same tables.
    pub fn new(catalog: Arc<dyn Catalog>) -> Self {
        Self {
            catalog: catalog.clone(),
            planner_context: duckdb_planner::PlannerContext::new(Arc::new(DuckDBCatalogAdapter {
                catalog,
            })),
        }
    }

    /// Plan a SQL statement into a Pivot [`Plan`].
    ///
    /// The statement is first planned by DuckDB (which resolves references
    /// through the catalog), then each [`duckdb_planner::PlanNode`] is
    /// translated into a [`PlanNode`].
    pub fn plan(&mut self, query: &str) -> Result<Plan, Error> {
        let duckdb_plan = self.planner_context.plan(query)?;
        let mut root = PlanNode::try_from(duckdb_plan)?;
        // Push a top-k limit into a grouped aggregate that feeds ORDER BY DESC.
        root.annotate_group_topn();
        Ok(Plan {
            catalog: self.catalog.clone(),
            root,
        })
    }
}
