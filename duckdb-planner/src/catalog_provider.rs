use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi;
use crate::duckdb_bridge::ffi::CatalogGetScalarFunctionResult;
use crate::duckdb_bridge::ffi::CatalogGetTableFunctionResult;
use crate::duckdb_bridge::ffi::CatalogGetTableResult;
use crate::duckdb_bridge::ffi::DuckDBColumn;
use crate::handle::Expr;

/// Describes a table that the planner can reference during query planning.
///
/// Implement this trait for tables in your schema and return instances
/// from [`DuckDBTransaction::table`]. The planner uses [`duckdb_typed_columns`](DuckDBTable::duckdb_typed_columns)
/// to resolve column names and types, and attaches the `Box<dyn DuckDBTable>` to the
/// resulting scan operator so downstream consumers
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
    /// A single `table_id` can be referenced by more than one plan node, a
    /// late-materialized query's narrow scan and its `Materialize` share one;
    /// so `resolve_inputs` hands each reference its own clone rather than moving
    /// the one resolved table out.
    fn clone_box(&self) -> Box<dyn DuckDBTable>;

    /// Called from DuckDB's `pushdown_complex_filter` hook with a borrowed
    /// handle to the current scan-local filter expression. The implementor reads
    /// the handle (translating it however it likes) to decide eligibility.
    ///
    /// Returns `Ok(true)` when the filter was fully consumed by the table
    /// (DuckDB drops it from the scan), `Ok(false)` to keep it. Errors are
    /// surfaced to C++ as exceptions.
    fn pushdown_filter(&mut self, _filter: Expr<'_>) -> Result<bool> {
        Ok(false)
    }

    /// The table's total row count, if the backend can answer from metadata
    /// alone. Feeds DuckDB's cardinality hook during optimization, so its cost
    /// model (join ordering, hash-join build/probe side choice) sees real
    /// table sizes. `None` means unknown; DuckDB then falls back to its own
    /// defaults.
    fn estimate_row_count(&self) -> Option<u64> {
        None
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

pub use crate::duckdb_bridge::ffi::ScalarFunctionDef;

pub trait DuckDBBind {
    /// Given a scalar function name (e.g. `drop_cache`), return its binding
    /// signature. Names the provider doesn't define return `None` and resolve
    /// against DuckDB's own built-ins. Default: none.
    fn scalar_function(&self, _name: &str) -> Option<ScalarFunctionDef> {
        None
    }
}

/// One planning transaction's binder: resolves table and table-function names
/// against the snapshot the transaction was opened on, so everything a single
/// plan binds comes from one consistent view of the catalog. Only names that
/// are static registry (scalar functions) still resolve through [`DuckDBBind`].
pub trait DuckDBTransaction: Send + Sync {
    /// Given a table name in datastore `datastore` (the DuckDB database qualifier),
    /// return a table/object that implements [`DuckDBTable`] with column
    /// definitions, resolved against that datastore's snapshot. Returns `None`
    /// if the table doesn't exist. Single-datastore transactions ignore
    /// `datastore`; a composite over several datastores routes by it.
    fn bind_table(&self, datastore: &str, name: &str) -> Option<Box<dyn DuckDBTable>>;

    /// Given a function name in datastore `datastore`, return its binding
    /// signature, or `None` if that datastore has no such table function. The
    /// bridge registers it on demand during binding; functions it doesn't know
    /// (e.g. DuckDB built-ins like `generate_series`) return `None` and resolve
    /// elsewhere. Default: none.
    fn bind_table_function(&self, _datastore: &str, _name: &str) -> Option<TableFunctionDef> {
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
/// accepts the pieces and wraps them internally.
///
/// The static provider is a single one shared by every attached datastore:
/// scalar functions are generic (not per-datastore). `database_names` are the
/// datastores to `ATTACH` (one DuckDB database each) and `default_name` the one
/// to `USE` as the current database.
pub struct CatalogContext {
    provider: Arc<dyn DuckDBBind>,
    database_names: Vec<String>,
    default_name: String,
}

impl CatalogContext {
    pub(crate) fn new(
        provider: Arc<dyn DuckDBBind>,
        database_names: Vec<String>,
        default_name: String,
    ) -> Self {
        CatalogContext {
            provider,
            database_names,
            default_name,
        }
    }
}

/// The datastore names to attach, one DuckDB `ATTACH` per name. Called from the
/// C++ context constructor.
pub(crate) fn catalog_context_names(ctx: &CatalogContext) -> Vec<String> {
    ctx.database_names.clone()
}

/// The datastore DuckDB should make its current database (`USE`). Called from the
/// C++ context constructor.
pub(crate) fn catalog_context_default(ctx: &CatalogContext) -> String {
    ctx.default_name.clone()
}

/// Wraps an `Arc<dyn DuckDBTransaction>` for the C++ bridge, the same way
/// [`CatalogContext`] wraps the provider. One is created per
/// [`PlannerContext::plan`](crate::PlannerContext::plan) call; the C++ side
/// publishes a pointer to it on the pivot storage info for the duration of
/// that plan, and table lookups during binding come back through it.
pub struct TransactionContext {
    transaction: Arc<dyn DuckDBTransaction>,
}

impl TransactionContext {
    pub(crate) fn new(transaction: Arc<dyn DuckDBTransaction>) -> Self {
        TransactionContext { transaction }
    }
}

pub(crate) fn catalog_get_table(
    transaction: &TransactionContext,
    datastore: &str,
    name: &str,
) -> CatalogGetTableResult {
    match transaction.transaction.bind_table(datastore, name) {
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
    transaction: &TransactionContext,
    datastore: &str,
    name: &str,
) -> CatalogGetTableFunctionResult {
    match transaction.transaction.bind_table_function(datastore, name) {
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
        Some(function) => CatalogGetScalarFunctionResult {
            found: true,
            function,
        },
        None => CatalogGetScalarFunctionResult {
            found: false,
            function: ScalarFunctionDef {
                arg_type_ids: Vec::new(),
                return_type_id: 0,
                is_volatile: false,
            },
        },
    }
}

pub(crate) fn pushdown_filter(
    table: &mut OptionalTableWrapper,
    expr: &ffi::Expression,
) -> Result<bool> {
    let table = table
        .table
        .as_mut()
        .ok_or_else(|| -> Error { "pushdown_filter called on unbound table".into() })?;
    table.pushdown_filter(Expr::from_raw(expr))
}

pub(crate) fn table_estimate_row_count(table: &OptionalTableWrapper) -> ffi::CardinalityEstimate {
    let table = table
        .table
        .as_ref()
        .expect("estimate_row_count called on unbound table");
    match table.estimate_row_count() {
        Some(rows) => ffi::CardinalityEstimate { known: true, rows },
        None => ffi::CardinalityEstimate {
            known: false,
            rows: 0,
        },
    }
}
