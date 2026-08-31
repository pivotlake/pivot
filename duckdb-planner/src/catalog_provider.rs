use std::any::Any;
use std::sync::Arc;

use crate::duckdb_bridge::ffi;
use crate::duckdb_bridge::ffi::CatalogGetScalarFunctionResult;
use crate::duckdb_bridge::ffi::CatalogGetTableResult;
use crate::duckdb_bridge::ffi::DuckDBColumn;
use crate::handle::Expr;

/// Describes a table that the planner can reference during query planning.
///
/// Implement this trait for tables in your schema and return instances
/// from [`DuckDBTransaction::bind_table`]. The planner uses [`duckdb_typed_columns`](DuckDBTable::duckdb_typed_columns)
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

    /// Whether DuckDB may rewrite this table's scan into a narrow row-ID scan
    /// followed by late materialization in the execution planner.
    fn supports_late_materialization(&self) -> bool {
        false
    }

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

/// One globally registered table-function overload. Each overload repeats the
/// SQL name so the C++ bridge can group the flat list into DuckDB function
/// sets while constructing the planner context.
pub struct TableFunctionDef {
    pub name: String,
    pub arg_type_ids: Vec<u8>,
    pub supports_late_materialization: bool,
}

pub use crate::duckdb_bridge::ffi::ScalarFunctionDef;

/// One planning transaction's binder: resolves table names against the
/// snapshot the transaction was opened on, so everything a single plan binds
/// comes from one consistent view of the catalog. Table-function signatures
/// are static planner registrations, but each invocation binds through this
/// transaction.
pub trait DuckDBTransaction: Send + Sync {
    /// Given a scalar function name (e.g. `drop_cache`), return its binding
    /// signature. Names the transaction doesn't define return `None` and
    /// resolve against DuckDB's own built-ins. Default: none.
    fn scalar_function(&self, _name: &str) -> Option<ScalarFunctionDef> {
        None
    }

    /// Whether datastore `datastore` defines a schema named `schema`. The bridge
    /// asks this when the binder looks a schema up, before any table inside it is
    /// resolved, so a reference into a schema that does not exist fails as an
    /// unknown schema rather than as an unknown table.
    fn does_schema_exist(&self, datastore: &str, schema: &str) -> Result<bool>;

    /// Given a table name in schema `schema` of datastore `datastore` (the
    /// DuckDB database qualifier), return a table/object that implements
    /// [`DuckDBTable`] with column definitions, resolved against that
    /// datastore's snapshot. Returns `None` if the table doesn't exist.
    /// Single-datastore transactions ignore `datastore`; a composite over
    /// several datastores routes by it.
    fn bind_table(
        &self,
        datastore: &str,
        schema: &str,
        name: &str,
    ) -> Result<Option<Box<dyn DuckDBTable>>>;

    /// Bind a globally registered table-function invocation through the
    /// current query transaction.
    fn bind_table_function(
        &self,
        _name: &str,
        _arguments: Vec<crate::ScalarValue>,
    ) -> Result<Box<dyn DuckDBTable>> {
        Err("table-function binding is not supported by this transaction".into())
    }
}

/// Static configuration owned by the C++ planner context. Table-function
/// overloads must be registered before any query is planned; invocation binding
/// and all other catalog resolution use the per-query [`DuckDBTransaction`].
pub struct CatalogContext {
    table_functions: Vec<TableFunctionDef>,
    database_names: Vec<String>,
    default_name: String,
}

impl CatalogContext {
    pub(crate) fn new(
        table_functions: Vec<TableFunctionDef>,
        database_names: Vec<String>,
        default_name: String,
    ) -> Self {
        CatalogContext {
            table_functions,
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

/// The global table-function overloads to install in DuckDB's system catalog.
pub(crate) fn catalog_table_functions(ctx: &CatalogContext) -> Vec<ffi::TableFunctionDef> {
    ctx.table_functions
        .iter()
        .map(|function| ffi::TableFunctionDef {
            name: function.name.clone(),
            arg_type_ids: function.arg_type_ids.clone(),
            supports_late_materialization: function.supports_late_materialization,
        })
        .collect()
}

/// Wraps an `Arc<dyn DuckDBTransaction>` for the C++ bridge. One is created per
/// [`PlannerContext::plan`](crate::PlannerContext::plan) call; the C++ side
/// publishes a pointer to it on the pivot storage info for the duration of that
/// plan, and catalog lookups during binding come back through it.
pub struct TransactionContext {
    transaction: Arc<dyn DuckDBTransaction>,
}

impl TransactionContext {
    pub(crate) fn new(transaction: Arc<dyn DuckDBTransaction>) -> Self {
        TransactionContext { transaction }
    }
}

/// Whether `datastore` defines `schema`, answered from the transaction's
/// snapshot. Called from `PivotCatalog::LookupSchema` before it materializes a
/// schema entry.
pub(crate) fn catalog_does_schema_exist(
    transaction: &TransactionContext,
    datastore: &str,
    schema: &str,
) -> Result<bool> {
    transaction.transaction.does_schema_exist(datastore, schema)
}

pub(crate) fn catalog_get_table(
    transaction: &TransactionContext,
    datastore: &str,
    schema: &str,
    name: &str,
) -> Result<CatalogGetTableResult> {
    Ok(
        match transaction
            .transaction
            .bind_table(datastore, schema, name)?
        {
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
        },
    )
}

pub(crate) fn catalog_bind_table_function(
    transaction: &TransactionContext,
    name: &str,
    arguments: &cxx::CxxVector<ffi::Value>,
) -> Result<ffi::CatalogBindTableFunctionResult> {
    let arguments = arguments
        .iter()
        .map(crate::handle::scalar_from_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let table = transaction
        .transaction
        .bind_table_function(name, arguments)?;
    let columns = table.duckdb_typed_columns();
    Ok(ffi::CatalogBindTableFunctionResult {
        columns,
        table: Box::new(OptionalTableWrapper { table: Some(table) }),
    })
}

/// Clone a table handle held in DuckDB function bind data. Optimizer copies of
/// a logical get must own independent Rust wrappers, while the underlying bound
/// table decides what state is shared through [`DuckDBTable::clone_box`].
pub(crate) fn clone_table_function_table(
    table: &OptionalTableWrapper,
) -> Box<OptionalTableWrapper> {
    Box::new(OptionalTableWrapper {
        table: table.table.as_ref().map(|table| table.clone_box()),
    })
}

pub(crate) fn catalog_get_scalar_function(
    transaction: &TransactionContext,
    name: &str,
) -> CatalogGetScalarFunctionResult {
    match transaction.transaction.scalar_function(name) {
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

/// Whether the currently bound table implements the materialization half of
/// DuckDB's late-materialization rewrite.
pub(crate) fn table_supports_late_materialization(table: &OptionalTableWrapper) -> bool {
    table
        .table
        .as_ref()
        .expect("supports_late_materialization called on unbound table")
        .supports_late_materialization()
}
