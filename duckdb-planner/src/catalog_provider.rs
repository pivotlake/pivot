use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi::CatalogGetScalarFunctionResult;
use crate::duckdb_bridge::ffi::CatalogGetTableFunctionResult;
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

    /// Clone this table into a fresh boxed trait object.
    ///
    /// A single `table_id` can be referenced by more than one plan node — a
    /// late-materialized query's narrow scan and its `Materialize` share one —
    /// so `resolve_inputs` hands each reference its own clone rather than moving
    /// the one resolved table out.
    fn clone_box(&self) -> Box<dyn DuckDBTable>;

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

/// A table-valued function the catalog provides, described for DuckDB's binder:
/// its argument types and full output columns, both as DuckDB logical type id
/// discriminants. This is all the bridge needs to register and type-check the
/// function, so the schema is defined once (in the provider) rather than also in
/// the C++ bridge.
pub struct TableFunctionDef {
    pub arg_type_ids: Vec<u8>,
    pub columns: Vec<DuckDBColumn>,
}

/// A scalar function the provider defines, described for DuckDB's binder: its
/// argument types and return type (DuckDB logical type id discriminants), plus
/// whether it must be marked `VOLATILE` so the optimizer can't fold the call
/// away before pivot re-plans it.
pub struct ScalarFunctionDef {
    pub arg_type_ids: Vec<u8>,
    pub return_type_id: u8,
    pub is_volatile: bool,
}

pub trait DuckDBBind {
    /// Given a table name, return a table/object that implements [`DuckDBTable`] with column definitions.
    /// Returns `None` if the table doesn't exist.
    fn try_bind(&self, name: &str) -> Option<Box<dyn DuckDBTable>>;

    /// Given a function name, return its binding signature, or `None` if the
    /// provider has no such table function. The bridge registers it on demand
    /// during binding; functions the provider doesn't know (e.g. DuckDB built-ins
    /// like `generate_series`) return `None` and resolve elsewhere. Default: none.
    fn table_function(&self, _name: &str) -> Option<TableFunctionDef> {
        None
    }

    /// Like [`table_function`](Self::table_function), but for scalar functions
    /// (e.g. `drop_cache`). Names the provider doesn't define return `None` and
    /// resolve against DuckDB's own built-ins. Default: none.
    fn scalar_function(&self, _name: &str) -> Option<ScalarFunctionDef> {
        None
    }
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

pub(crate) fn catalog_get_table_function(
    ctx: &CatalogContext,
    name: &str,
) -> CatalogGetTableFunctionResult {
    match ctx.provider.table_function(name) {
        Some(def) => CatalogGetTableFunctionResult {
            found: true,
            arg_type_ids: def.arg_type_ids,
            columns: def.columns,
        },
        None => CatalogGetTableFunctionResult {
            found: false,
            arg_type_ids: Vec::new(),
            columns: Vec::new(),
        },
    }
}

pub(crate) fn catalog_get_scalar_function(
    ctx: &CatalogContext,
    name: &str,
) -> CatalogGetScalarFunctionResult {
    match ctx.provider.scalar_function(name) {
        Some(def) => CatalogGetScalarFunctionResult {
            found: true,
            arg_type_ids: def.arg_type_ids,
            return_type_id: def.return_type_id,
            is_volatile: def.is_volatile,
        },
        None => CatalogGetScalarFunctionResult {
            found: false,
            arg_type_ids: Vec::new(),
            return_type_id: 0,
            is_volatile: false,
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
