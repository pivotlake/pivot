//! Catalog: how the planner discovers tables.
//!
//! The planner does not know anything about persistence — it defers to a
//! caller-supplied [`Catalog`] implementation to resolve table names into
//! [`Table`]s, each of which exposes a schema (as a list of Pivot [`Column`]s)
//! and knows how to compile itself into a dispatch scan spec.
//!
//! Because DuckDB owns SQL binding, the catalog and tables also need to be
//! visible to it: the adapters [`DuckDBCatalogAdapter`] and
//! [`DuckDBTableAdapter`] implement DuckDB's [`DuckDBBind`] /
//! [`DuckDBTable`] traits over our Pivot types. They exist as
//! standalone wrapper structs (rather than blanket impls) because the orphan
//! rule prevents implementing a foreign trait for `Box<dyn Table>` directly.

use std::collections::HashMap;

use crate::expression::{CompareType, TableFilter};
use crate::types::{Type, logical_from_type};
use dispatch::{DataFlowDispatcher, DynamicFilterSlot, Projection, RecordBatchOperatorSpec};
use duckdb_planner::DuckDBColumn;
use duckdb_planner::catalog_provider::{DuckDBBind, DuckDBTable};
use duckdb_planner::expression::TableFilter as DuckDBTableFilter;
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single-column predicate whose constant is supplied at runtime from a shared
/// [`DynamicFilterSlot`] (filled by a Top-N as it tightens its boundary).
///
/// It is a purely logical predicate — "column `column_idx` `compare_type` the
/// current slot value". A storage backend may use it to skip data that cannot
/// match the live boundary (e.g. Parquet row-group elimination), or ignore it
/// entirely; ignoring is always correct, just without the optimization.
pub struct DynamicScanPredicate {
    pub column_idx: usize,
    pub compare_type: CompareType,
    pub slot: Arc<DynamicFilterSlot>,
}

/// A single column in a [`Table`]'s schema: name plus Pivot [`Type`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Column {
    pub name: String,
    pub col_type: Type,
}

/// Description of a table to be created — produced by translating a
/// `CREATE TABLE` statement, consumed by [`Catalog::create_table`].
///
/// `options` carries the `WITH (...)` clause verbatim so the catalog
/// implementation can decide what to do with backend-specific keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTableRequest {
    pub name: String,
    pub columns: Vec<Column>,
    pub options: HashMap<String, String>,
    pub if_not_exists: bool,
}

/// A table that the planner can read from.
///
/// Implementations expose two pieces of information: the column list (used
/// during planning, both for our own translation and to feed DuckDB through
/// [`DuckDBTableAdapter`]) and a way to compile a scan into a dispatch
/// [`RecordBatchOperatorSpec`].
pub trait Table: Debug + Send + Sync {
    /// Build a dispatch scan spec that reads this table.
    ///
    /// `dynamic_filters` are logical single-column predicates whose constants are
    /// filled in at runtime (by a Top-N above the scan tightening its boundary).
    /// A backend may use them to skip data that can't match — e.g. Parquet
    /// row-group elimination — or ignore them; ignoring is always correct, just
    /// without the optimization.
    ///
    /// `emit_row_group_metadata` asks the scan to tag each emitted row with the
    /// metadata a downstream [`materialize`](Table::materialize) needs (e.g. its
    /// row-group ID and per-row index). Backends that don't materialize can
    /// ignore it; the bridge only sets it on a late-materialized query's narrow
    /// scan.
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> Result<RecordBatchOperatorSpec>;

    /// Return the table's schema.
    fn columns(&self) -> Vec<Column>;

    /// Clone this table into a fresh boxed trait object.
    ///
    /// A late-materialized query references one table from both its narrow scan
    /// and its [`Materialize`](crate::operator::Materialize); `Box<dyn Table>`
    /// isn't `Clone`, so backends expose cloning through this method.
    fn clone_box(&self) -> Box<dyn Table>;

    /// Fetch `projection` for the rows that survived `input` (whose scan was
    /// tagged via `emit_row_group_metadata`), emitting them in `projection`
    /// order.
    ///
    /// Only reached for a [`Materialize`](crate::operator::Materialize) node,
    /// which the bridge only emits for tables that support it; the default
    /// panics.
    fn materialize(
        &self,
        _input: RecordBatchOperatorSpec,
        _projection: Projection,
    ) -> RecordBatchOperatorSpec {
        unreachable!("materialize called on a table that does not support late materialization")
    }

    /// Try to push a filter into the table. Returns `Ok(true)` if it was
    /// *FULLY* consumed (no upstream `Filter` operator required), `Ok(false)`
    /// if it was kept above. Errors propagate to the FFI boundary as C++
    /// exceptions.
    fn pushdown_filter(&mut self, _filter: TableFilter) -> Result<bool> {
        Ok(false)
    }

    /// Downcast hook, so a re-resolved table can read another instance's
    /// concrete file set in [`rebind_onto`](Table::rebind_onto).
    fn as_any(&self) -> &dyn std::any::Any;

    /// This table's catalog name, if it has one.
    ///
    /// Used at compile time to re-resolve the table's *current* file set from
    /// the live catalog, so a reused (cached) plan still sees data committed
    /// since it was planned. `None` for backends that are not catalog-resolved
    /// (e.g. test stubs), which then skip the refresh.
    fn name(&self) -> Option<&str> {
        None
    }

    /// Return a copy of `self` that scans `fresh`'s file set instead of its
    /// own, keeping `self`'s pushed-down predicates. `fresh` is the latest
    /// predicate-less resolution of the *same* table from the catalog (see
    /// [`Catalog::table`]). Default: ignore `fresh` and clone unchanged, for
    /// backends with no separable file set.
    fn rebind_onto(&self, _fresh: &dyn Table) -> Box<dyn Table> {
        self.clone_box()
    }
}

/// Adapts a Pivot [`Table`] to DuckDB's [`DuckDBTable`] trait,
/// converting our column types into DuckDB logical types. Required because
/// Rust's orphan rule prevents implementing a foreign trait for a foreign type.
#[derive(Debug)]
pub struct DuckDBTableAdapter {
    pub table: Box<dyn Table>,
}

impl DuckDBTable for DuckDBTableAdapter {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(DuckDBTableAdapter {
            table: self.table.clone_box(),
        })
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        self.table
            .columns()
            .iter()
            .map(|col| DuckDBColumn {
                name: col.name.clone(),
                duckdb_logical_type_id: logical_from_type(&col.col_type) as u8,
            })
            .collect()
    }

    fn pushdown_filter(
        &mut self,
        filter: DuckDBTableFilter,
    ) -> duckdb_planner::catalog_provider::Result<bool> {
        let filter: TableFilter = filter.try_into()?;
        Ok(self.table.pushdown_filter(filter)?)
    }
}

/// The set of tables the planner can resolve names against.
///
/// This is the only thing a caller has to provide to use the planner — DuckDB
/// will call into it (via [`DuckDBCatalogAdapter`]) during binding, and the
/// translation layer will call it when wiring [`Operator::Input`](crate::operator::Operator::Input)
/// nodes to concrete [`Table`]s.
///
/// `create_table` compiles a `CREATE TABLE` statement: it does the up-front work
/// (e.g. reading every data file's footer, in parallel, into a materialized
/// table) on the coordinator and returns the dataflow plan that *writes* the
/// result into the catalog when executed.
pub trait Catalog: Debug + Send + Sync {
    /// Resolve a table name to a fresh, independently-mutable [`Table`], or
    /// `None` if no such table exists. Each call returns a unique `Box`, so
    /// per-query filter pushdown can mutate the table without affecting
    /// concurrent queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>>;

    /// Compile a `CREATE TABLE` statement into the dataflow that writes the new
    /// table into the catalog.
    ///
    /// Called on the **coordinator** at plan-compile time, so the backend may
    /// run a dataflow here to build the table (e.g. fetch every Parquet footer
    /// in parallel over the dispatch worker pool) before returning the plan that
    /// commits it. The returned spec, when executed, performs the catalog write
    /// and yields no rows. Errors surface as [`catalog::Error`](enum@Error).
    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec>;
}

/// Adapts a Pivot [`Catalog`] to DuckDB's [`DuckDBBind`] trait so DuckDB can
/// resolve table names during SQL binding. Looks up the table on the wrapped
/// catalog and wraps it in a [`DuckDBTableAdapter`].
pub struct DuckDBCatalogAdapter {
    pub catalog: Arc<dyn Catalog>,
}

impl DuckDBBind for DuckDBCatalogAdapter {
    fn try_bind(&self, name: &str) -> Option<Box<dyn DuckDBTable>> {
        let table = self.catalog.table(name)?;
        Some(Box::new(DuckDBTableAdapter { table }))
    }
}
