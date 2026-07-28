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
//! let columns = vec![Column::new("URL".into(), Type::Utf8)];
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

/// Errors surfaced by [`Planner::plan`].
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Planning(#[from] duckdb_planner::Error),
    #[error("Error converting plan: {0}")]
    PlanConversion(#[from] plan::Error),
    #[error("{0}")]
    GenerationExpression(String),
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
        let adapter: Arc<dyn DuckDBTransaction> =
            Arc::new(DuckDBTransactionAdapter { transaction });
        let planned = self.planner_context.plan(query, adapter.clone())?;
        let mut root = build::build_plan(planned.root())?;
        self.widen_rows_for_generated_columns(&mut root, &adapter)?;
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

    /// Widen the rows flowing into every INSERT whose target has generated
    /// columns, so what reaches the write is the table's full row.
    ///
    /// A generated column's expression is stored as text (it travels with the
    /// schema), so it is bound here by planning `SELECT <every column> FROM
    /// <target>` with the target's generated columns declared as generated,
    /// which makes DuckDB resolve each of those into its expression over the
    /// table's other columns. That is the same binder, with the same functions,
    /// casts and type rules the original `CREATE TABLE` used, and it hands back
    /// the stored columns as plain references in the same breath, so the result
    /// is precisely the projection the insert needs.
    ///
    /// This cannot happen while the statement's own plan is being walked, since
    /// that walk is what discovers the target, so it runs as this second pass.
    fn widen_rows_for_generated_columns(
        &mut self,
        node: &mut PlanNode,
        transaction: &Arc<dyn DuckDBTransaction>,
    ) -> Result<(), Error> {
        for input in &mut node.inputs {
            self.widen_rows_for_generated_columns(input, transaction)?;
        }
        let Operator::Insert(insert) = &node.operator else {
            return Ok(());
        };
        let schema = insert.table.columns();
        if schema.iter().all(|column| column.generated.is_none()) {
            return Ok(());
        }

        let projection = self.bind_row_of(&insert.table_name, &schema, transaction)?;
        let input = node.inputs.remove(0);
        node.inputs.push(PlanNode {
            name: "generated columns".to_string(),
            operator: Operator::Projection(operator::Projection {
                projections: projection,
            }),
            inputs: vec![input],
        });
        Ok(())
    }

    /// Bind the expression for each of `table`'s columns as written into a row:
    /// a reference to the row's own value for a stored column, and the
    /// generation expression for a generated one.
    ///
    /// The stored columns are selected first and in order, which is the order
    /// the row supplies them in, so the references the generation expressions
    /// resolve to address that row directly.
    fn bind_row_of(
        &mut self,
        table: &str,
        schema: &[catalog::Column],
        transaction: &Arc<dyn DuckDBTransaction>,
    ) -> Result<Vec<expression::Expression>, Error> {
        let (stored, generated): (Vec<usize>, Vec<usize>) =
            (0..schema.len()).partition(|&column| schema[column].generated.is_none());
        let selected = stored
            .iter()
            .chain(&generated)
            .map(|&column| quote_identifier(&schema[column].name))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!("SELECT {selected} FROM {}", quote_identifier(table));

        let bound = self.planner_context.plan_binding_generated_columns_of(
            &query,
            transaction.clone(),
            table,
        )?;
        let bound = build::build_selected_expressions(bound.root())
            .map_err(|e| Error::GenerationExpression(e.to_string()))?;

        // The stored columns were selected first and in order, so the scan reads
        // exactly them, in the order the row holds them, and the references the
        // expressions carry are already positions in that row. Anything else
        // means they address a row this insert does not supply.
        if bound.scanned_columns.iter().ne(stored.iter()) {
            return Err(Error::GenerationExpression(format!(
                "the columns of {table} are bound against table columns {:?} rather than \
                 the {} a row supplies",
                bound.scanned_columns,
                stored.len()
            )));
        }
        if bound.expressions.len() != schema.len() {
            return Err(Error::GenerationExpression(format!(
                "binding the columns of {table} produced {} expressions for {} columns",
                bound.expressions.len(),
                schema.len()
            )));
        }

        // Selected stored-then-generated; put each back where the table
        // declares it.
        let mut expressions = bound.expressions;
        let mut generated = expressions.split_off(stored.len()).into_iter();
        let mut stored = expressions.into_iter();
        schema
            .iter()
            .map(|column| match column.generated {
                Some(_) => generated.next(),
                None => stored.next(),
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::GenerationExpression(format!("{table} bound too few column expressions"))
            })
    }
}

/// Quote an identifier for use in the column-binding query, so a name that needs
/// quoting (or contains a quote) still resolves to itself.
fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}
