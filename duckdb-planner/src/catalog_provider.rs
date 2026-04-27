use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi::CatalogGetTableResult;
use crate::duckdb_bridge::ffi::DuckDBColumn;
use crate::expression::TableFilter;

/// Describes a table that the planner can reference during query planning.
///
/// Implement this trait for tables in your schema and return instances
/// from [`DuckDBBind::try_bind`]. The planner uses [`duckdb_typed_columns`](DuckDBTable::duckdb_typed_columns)
/// to resolve column names and types, and attaches the `Arc<dyn DuckDBTable>` to the
/// resulting [`Input`](crate::operator::Input) operator so downstream consumers
/// can identify which table is being scanned.
pub trait DuckDBTable: Any {
    /// The ordered list of columns in this table, used by the planner to
    /// resolve column references and determine output types.
    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn>;

    /// Called from DuckDB's `pushdown_complex_filter` hook with the current
    /// scan-local filter expressions serialized as JSON.
    ///
    /// The JSON payload is a list of DuckDB expression objects in the same
    /// shape used by the planner bridge internally. Implementors that care
    /// about pre-statistics filter state can deserialize or stash the values
    /// here. The default implementation ignores the callback.
    fn pushdown_filter(&self, _filter: TableFilter) -> bool {
        false
    }
}

/// Newtype around `Option<Arc<dyn DuckDBTable>>` needed because CXX cannot declare
/// generic opaque types in `extern "Rust"` blocks. This gives us a concrete
/// name that CXX can reference while still carrying an optional
/// `DuckDBTable` trait object across the FFI boundary.
#[derive(Default)]
pub struct OptionalTableWrapper {
    pub table: Option<Arc<dyn DuckDBTable>>,
}

pub trait DuckDBBind {
    /// Given a table name, return a table/object that implements [`DuckDBTable`] with column definitions.
    /// Returns `None` if the table doesn't exist.
    fn try_bind(&self, name: &str) -> Option<Arc<dyn DuckDBTable>>;
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

pub(crate) fn pushdown_filter(table: &OptionalTableWrapper, filter_json: &str) -> bool {
    let Some(table) = &table.table else {
        return false;
    };

    let filter = match serde_json::from_str::<TableFilter>(filter_json) {
        Ok(filter) => filter,
        Err(err) => panic!("invalid pushdown filter payload from C++ bridge: {err}"),
    };
    table.pushdown_filter(filter)
}
