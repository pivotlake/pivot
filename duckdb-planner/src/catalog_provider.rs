use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi::{CatalogGetTableResult, DuckDBColumn};

/// Describes a table that the planner can reference during query planning.
///
/// Implement this trait for tables in your schema and return instances
/// from [`DuckDBBind::try_bind`]. The planner uses [`duckdb_typed_columns`](GetDuckDBTypedColumns::duckdb_typed_columns)
/// to resolve column names and types, and attaches the `Arc<dyn GetDuckDBTypedColumns>` to the
/// resulting [`Input`](crate::operator::Input) operator so downstream consumers
/// can identify which table is being scanned.
pub trait GetDuckDBTypedColumns: Any {
    /// The ordered list of columns in this table, used by the planner to
    /// resolve column references and determine output types.
    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn>;
}

/// Newtype around `Option<Arc<dyn GetDuckDBTypedColumns>>` needed because CXX cannot declare
/// generic opaque types in `extern "Rust"` blocks. This gives us a concrete
/// name that CXX can reference while still carrying an optional
/// `GetDuckDBTypedColumns` trait object across the FFI boundary.
#[derive(Default)]
pub struct OptionalTableWrapper {
    pub table: Option<Arc<dyn GetDuckDBTypedColumns>>,
}

pub trait DuckDBBind {
    /// Given a table name, return a table/object that implements [`GetDuckDBTypedColumns`] with column definitions.
    /// Returns `None` if the table doesn't exist.
    fn try_bind(&self, name: &str) -> Option<Arc<dyn GetDuckDBTypedColumns>>;
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
