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
//! [`GetDuckDBTypedColumns`] traits over our Pivot types. They exist as
//! standalone wrapper structs (rather than blanket impls) because the orphan
//! rule prevents implementing a foreign trait for `Arc<dyn Table>` directly.

use std::collections::HashMap;

use crate::types::{Type, logical_from_type};
use dispatch::{Projection, RecordBatchOperatorSpec};
use duckdb_planner::DuckDBColumn;
use duckdb_planner::catalog_provider::{DuckDBBind, GetDuckDBTypedColumns};
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single column in a [`Table`]'s schema: name plus Pivot [`Type`].
#[derive(Debug, Clone, PartialEq, Eq)]
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
    fn compile(&self, projection: Projection) -> RecordBatchOperatorSpec;

    /// Return the table's schema.
    fn columns(&self) -> Vec<Column>;
}

/// Adapts a Pivot [`Table`] to DuckDB's [`GetDuckDBTypedColumns`] trait,
/// converting our column types into DuckDB logical types. Required because
/// Rust's orphan rule prevents implementing a foreign trait for a foreign type.
#[derive(Debug)]
pub struct DuckDBTableAdapter {
    pub table: Arc<dyn Table>,
}

impl GetDuckDBTypedColumns for DuckDBTableAdapter {
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
}

/// The set of tables the planner can resolve names against.
///
/// This is the only thing a caller has to provide to use the planner — DuckDB
/// will call into it (via [`DuckDBCatalogAdapter`]) during binding, and the
/// translation layer will call it when wiring [`Operator::Input`](crate::operator::Operator::Input)
/// nodes to concrete [`Table`]s.
///
/// `create_table` is invoked at execution time (not at plan time) by the
/// nullary operator that compiles a `CREATE TABLE` statement.
pub trait Catalog: Debug + Send + Sync {
    /// Resolve a table name to a [`Table`], or `None` if no such table exists.
    fn table(&self, name: &str) -> Option<Arc<dyn Table>>;

    /// Create a new table from a [`CreateTableRequest`]. Invoked at execution
    /// time by the nullary operator compiled from a `CREATE TABLE` statement,
    /// not during planning. Errors are surfaced to the caller as
    /// [`catalog::Error`](enum@Error).
    fn create_table(&self, request: CreateTableRequest) -> Result<()>;
}

/// Adapts a Pivot [`Catalog`] to DuckDB's [`DuckDBBind`] trait so DuckDB can
/// resolve table names during SQL binding. Looks up the table on the wrapped
/// catalog and wraps it in a [`DuckDBTableAdapter`].
pub struct DuckDBCatalogAdapter {
    pub catalog: Arc<dyn Catalog>,
}

impl DuckDBBind for DuckDBCatalogAdapter {
    fn try_bind(&self, name: &str) -> Option<Arc<dyn GetDuckDBTypedColumns>> {
        let table = self.catalog.table(name)?;
        Some(Arc::new(DuckDBTableAdapter { table }))
    }
}
