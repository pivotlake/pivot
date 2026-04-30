use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi::CatalogGetTableResult;
use crate::duckdb_bridge::ffi::DuckDBColumn;
use crate::expression::TableFilter;

/// Describes a table that the planner can reference during query planning.
///
/// Implement this trait for tables in your schema and return instances
/// from [`DuckDBBind::try_bind`]. The planner uses [`duckdb_typed_columns`](DuckDBTable::duckdb_typed_columns)
/// to resolve column names and types, and attaches the `Box<dyn DuckDBTable>` to the
/// resulting [`Input`](crate::operator::Input) operator so downstream consumers
/// can identify which table is being scanned.
/// Type-erased error that crosses the FFI boundary as a C++ exception (CXX
/// converts via `Display`).
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Result alias for fallible [`DuckDBTable`] / bridge operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub trait DuckDBTable: Any {
    /// The ordered list of columns in this table, used by the planner to
    /// resolve column references and determine output types.
    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn>;

    /// Called from DuckDB's `pushdown_complex_filter` hook with the current
    /// scan-local filter expressions deserialized into a [`TableFilter`].
    ///
    /// Returns `Ok(true)` when the filter was fully consumed by the table
    /// (DuckDB drops it from the scan), `Ok(false)` to keep it. Errors are
    /// surfaced to C++ as exceptions.
    fn pushdown_filter(&mut self, _filter: TableFilter) -> Result<bool> {
        Ok(false)
    }
}

/// Newtype around `Option<Box<dyn DuckDBTable>>` needed because CXX cannot declare
/// generic opaque types in `extern "Rust"` blocks. This gives us a concrete
/// name that CXX can reference while still carrying an optional
/// `DuckDBTable` trait object across the FFI boundary.
#[derive(Default)]
pub struct OptionalTableWrapper {
    pub table: Option<Box<dyn DuckDBTable>>,
}

pub trait DuckDBBind {
    /// Given a table name, return a table/object that implements [`DuckDBTable`] with column definitions.
    /// Returns `None` if the table doesn't exist.
    fn try_bind(&self, name: &str) -> Option<Box<dyn DuckDBTable>>;
}

/// Wraps an `Arc<dyn DuckDBBind>` for the C++ bridge.
///
/// CXX requires owned, concrete types to cross the FFI boundary — it cannot
/// pass an `Arc<dyn Trait>` directly. `Box<CatalogContext>` satisfies that
/// constraint while keeping the inner provider cheaply cloneable via `Arc`.
///
/// Users don't interact with this type; [`PlannerContext::new`](crate::PlannerContext::new)
/// accepts `Arc<dyn DuckDBBind>` and wraps it internally.
pub struct CatalogContext {
    provider: Arc<dyn DuckDBBind>,
}

impl CatalogContext {
    pub(crate) fn new(provider: Arc<dyn DuckDBBind>) -> Self {
        CatalogContext { provider }
    }
}

pub(crate) fn catalog_get_table(ctx: &CatalogContext, name: &str) -> CatalogGetTableResult {
    match ctx.provider.try_bind(name) {
        Some(table) => {
            let columns = table.duckdb_typed_columns();
            CatalogGetTableResult {
                found: true,
                columns,
                table: Box::new(OptionalTableWrapper { table: Some(table) }),
            }
        }
        None => CatalogGetTableResult {
            found: false,
            columns: Vec::new(),
            table: Box::new(OptionalTableWrapper { table: None }),
        },
    }
}

pub(crate) fn pushdown_filter(table: &mut OptionalTableWrapper, filter_json: &str) -> Result<bool> {
    let table = table
        .table
        .as_mut()
        .ok_or_else(|| -> Error { "pushdown_filter called on unbound table".into() })?;
    let filter = serde_json::from_str::<TableFilter>(filter_json)?;
    table.pushdown_filter(filter)
}
