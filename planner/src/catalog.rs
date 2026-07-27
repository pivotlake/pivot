//! Catalog: how the planner discovers tables.
//!
//! The planner does not know anything about persistence; it defers to a
//! caller-supplied [`Catalog`] implementation to resolve table names into
//! [`BoundTable`]s, each of which exposes a schema (as a list of Pivot [`Column`]s)
//! and knows how to compile itself into a dispatch scan spec.
//!
//! Because DuckDB owns SQL binding, the catalog and tables also need to be
//! visible to it: the adapters [`DuckDBScalarFunctionBinder`] and
//! [`DuckDBTableAdapter`] implement DuckDB's [`DuckDBBind`] /
//! [`DuckDBTable`] traits over our Pivot types. They exist as
//! standalone wrapper structs (rather than blanket impls) because the orphan
//! rule prevents implementing a foreign trait for `Box<dyn BoundTable>` directly.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::expression::{CompareType, TableFilter};
use crate::operator::TableFunction;
use crate::types::{Type, logical_from_type};
use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, DynamicFilterSlot, Projection, RecordBatchOperatorSpec};
use duckdb_planner::DuckDBColumn;
use duckdb_planner::Expr;
use duckdb_planner::catalog_provider::{
    DuckDBBind, DuckDBTable, DuckDBTransaction, ScalarFunctionDef, TableFunctionDef,
};
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
/// It is a purely logical predicate: "column `column_idx` `compare_type` the
/// current slot value". A storage backend may use it to skip data that cannot
/// match the live boundary (e.g. Parquet row-group elimination), or ignore it
/// entirely; ignoring is always correct, just without the optimization.
pub struct DynamicScanPredicate {
    pub column_idx: usize,
    pub compare_type: CompareType,
    pub slot: Arc<DynamicFilterSlot>,
}

/// A single column in a [`BoundTable`]'s schema: name plus Pivot [`Type`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Column {
    pub name: String,
    pub col_type: Type,
}

/// Description of a table to be created, produced by translating a
/// `CREATE TABLE` statement, consumed by a transaction's `bind_create_table`
/// ([`CatalogTransaction::bind_create_table`], which routes to the target
/// datastore).
///
/// `options` carries the `WITH (...)` clause verbatim so the target datastore
/// implementation can decide what to do with backend-specific keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTableRequest {
    pub datastore_name: Option<String>,
    pub name: String,
    pub columns: Vec<Column>,
    pub options: HashMap<String, String>,
    pub if_not_exists: bool,
}

/// One query's transaction: a consistent **snapshot** of the catalog, opened by
/// [`Catalog::begin_transaction`] before the query is planned and held until
/// [`Catalog::commit_transaction`] or [`Catalog::rollback_transaction`]. Every
/// table the query binds resolves through this snapshot (never through the live
/// catalog, which a background refresh may be updating concurrently), so a
/// plan's scans, its late materialize, and its metadata peepholes all see one
/// frozen view. A backend may also hold pending writes here until commit.
#[async_trait]
pub trait CatalogTransaction: Debug + Send + Sync {
    /// Resolve `name` in datastore `datastore` to a fresh, independently-mutable
    /// [`BoundTable`], or `None` if that datastore holds no such table.
    fn bind_table(&self, datastore: &str, name: &str) -> Option<Box<dyn BoundTable>>;

    /// A backend-specific table-valued function `name` in datastore `datastore`,
    /// or `None`.
    fn bind_table_function(&self, _datastore: &str, _name: &str) -> Option<Box<dyn TableFunction>> {
        None
    }

    /// A backend table function by bare `name`, resolved against the default
    /// datastore.
    fn bind_default_table_function(&self, _name: &str) -> Option<Box<dyn TableFunction>> {
        None
    }

    /// Resolve a `CREATE TABLE` by routing to the datastore
    /// [`CreateTableRequest::datastore_name`] names (the default when unqualified)
    /// and deferring to that datastore's own `bind_create_table`.
    fn bind_create_table(&self, _request: CreateTableRequest) -> Result<Box<dyn TableCreation>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this catalog does not support CREATE TABLE",
        )
        .into())
    }

    /// Commit this transaction: publish whatever it staged (an INSERT's uploaded
    /// files, a CREATE's table). A read-only transaction is a no-op. Async so a
    /// backend can hop blocking store I/O to the blocking pool; the composite
    /// awaits each sub-transaction it opened.
    async fn commit(&self) -> Result<()> {
        Ok(())
    }

    /// Roll back this transaction: discard whatever it staged. Default: nothing,
    /// the snapshot is released when the last reference drops.
    fn rollback(&self) {}
}

/// A resolved `CREATE TABLE`, bound to its target datastore and ready to be
/// compiled into the dataflow that creates the table. Resolution (routing,
/// validation, locating the table's existing files) happens when
/// [`CatalogTransaction::bind_create_table`] produces it; [`compile`](Self::compile)
/// then builds the dataflow with the worker pool.
pub trait TableCreation: Send + Sync {
    /// Build the dataflow that fetches the new table's file footers over the pool
    /// and, at its terminal, commits the table into the datastore.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec>;
}

/// A table that the planner can read from.
///
/// Implementations expose two pieces of information: the column list (used
/// during planning, both for our own translation and to feed DuckDB through
/// [`DuckDBTableAdapter`]) and a way to compile a scan into a dispatch
/// [`RecordBatchOperatorSpec`].
pub trait BoundTable: Debug + Send + Sync {
    /// Build a dispatch scan spec that reads this table.
    ///
    /// `dynamic_filters` are logical single-column predicates whose constants are
    /// filled in at runtime (by a Top-N above the scan tightening its boundary).
    /// A backend may use them to skip data that can't match (e.g. Parquet
    /// row-group elimination) or ignore them; ignoring is always correct, just
    /// without the optimization.
    ///
    /// `emit_row_group_metadata` asks the scan to tag each emitted row with the
    /// metadata a downstream [`materialize`](BoundTable::materialize) needs (e.g. its
    /// row-group ID and per-row index). Backends that don't materialize can
    /// ignore it; the bridge only sets it on a late-materialized query's narrow
    /// scan.
    ///
    /// The binding is self-contained: it captured its datastore snapshot at bind
    /// time, so it resolves its own file set here without a transaction handle.
    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> Result<RecordBatchOperatorSpec>;

    /// Build a dispatch spec that writes the rows produced by `input` into this
    /// table and emits one `BIGINT` row with the count. Durable publication
    /// belongs to the query transaction's
    /// [`commit_transaction`](Catalog::commit_transaction). The default rejects
    /// INSERT (a read-only or virtual table).
    fn compile_insert(
        &self,
        _input: RecordBatchOperatorSpec,
        _dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec> {
        Err(
            Box::<dyn std::error::Error + Send + Sync>::from("this table does not support INSERT")
                .into(),
        )
    }

    /// Return the table's schema.
    fn columns(&self) -> Vec<Column>;

    /// Clone this table into a fresh boxed trait object.
    ///
    /// A late-materialized query references one table from both its narrow scan
    /// and its [`Materialize`](crate::operator::Materialize); `Box<dyn BoundTable>`
    /// isn't `Clone`, so backends expose cloning through this method.
    fn clone_box(&self) -> Box<dyn BoundTable>;

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
    ) -> Result<RecordBatchOperatorSpec> {
        unreachable!("materialize called on a table that does not support late materialization")
    }

    /// Try to push a filter into the table. Returns `Ok(true)` if it was
    /// *FULLY* consumed (no upstream `Filter` operator required), `Ok(false)`
    /// if it was kept above. Errors propagate to the FFI boundary as C++
    /// exceptions.
    fn pushdown_filter(&mut self, _filter: TableFilter) -> Result<bool> {
        Ok(false)
    }

    /// The `column`'s min and max derived purely from table metadata, if they
    /// can be answered without scanning any rows (e.g. Parquet row-group
    /// statistics covering every row group, with no predicates pushed into this
    /// binding). The scalars carry the column's physical storage type. `None`
    /// means "unknown, scan instead" and is always a safe answer. Answered from
    /// the binding's captured snapshot, as in [`compile_scan`](BoundTable::compile_scan).
    fn column_min_max(&self, _column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        None
    }

    /// The table's total row count derived purely from metadata, if it can be
    /// answered without scanning any rows (e.g. summing Parquet row-group row
    /// counts, with no predicates pushed into this binding). `None` means
    /// "unknown, scan instead" and is always a safe answer. Answered from the
    /// binding's captured snapshot, as in [`compile_scan`](BoundTable::compile_scan).
    fn row_count(&self) -> Option<i64> {
        None
    }
}

/// Convert Pivot columns into the DuckDB-typed columns the binder consumes.
/// Shared by base-table and table-function binding.
fn duckdb_columns(columns: &[Column]) -> Vec<DuckDBColumn> {
    columns
        .iter()
        .map(|column| logical_from_type(&column.col_type).to_duckdb_column(column.name.clone()))
        .collect()
}

/// Adapts a Pivot [`BoundTable`] to DuckDB's [`DuckDBTable`] trait,
/// converting our column types into DuckDB logical types. Required because
/// Rust's orphan rule prevents implementing a foreign trait for a foreign type.
#[derive(Debug)]
pub struct DuckDBTableAdapter {
    pub table: Box<dyn BoundTable>,
}

impl DuckDBTable for DuckDBTableAdapter {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(DuckDBTableAdapter {
            table: self.table.clone_box(),
        })
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        duckdb_columns(&self.table.columns())
    }

    fn pushdown_filter(
        &mut self,
        filter: Expr<'_>,
    ) -> duckdb_planner::catalog_provider::Result<bool> {
        // Translate the borrowed DuckDB filter expression into a Pivot one (the
        // only filter shape the bridge pushes is a bound expression).
        let filter = TableFilter::Expression(Box::new(crate::expression::Expression::from_handle(
            filter,
        )?));
        Ok(self.table.pushdown_filter(filter)?)
    }
}

/// The DuckDB [`DuckDBBind`] provider: resolves the static
/// (transaction-independent) names during SQL binding, today only the planner's
/// built-in scalar functions, which are generic across datastores. It holds no
/// catalog: tables and table functions resolve through a per-query
/// [`DuckDBTransactionAdapter`] instead, since both are answered from the
/// transaction's snapshot.
pub struct DuckDBScalarFunctionBinder;

/// Adapts a Pivot [`CatalogTransaction`] to DuckDB's [`DuckDBTransaction`]
/// trait: table and table-function lookups during one plan's binding resolve
/// against this transaction's snapshot.
pub struct DuckDBTransactionAdapter {
    pub transaction: Arc<dyn CatalogTransaction>,
}

impl DuckDBTransaction for DuckDBTransactionAdapter {
    fn bind_table(&self, datastore: &str, name: &str) -> Option<Box<dyn DuckDBTable>> {
        let table = self.transaction.bind_table(datastore, name)?;
        Some(Box::new(DuckDBTableAdapter { table }))
    }

    fn bind_table_function(&self, datastore: &str, name: &str) -> Option<TableFunctionDef> {
        // The function's own signature is the single source of truth; convert its
        // Pivot types to DuckDB logical type ids for the binder.
        let signature = self
            .transaction
            .bind_table_function(datastore, name)?
            .signature();
        Some(TableFunctionDef {
            arg_type_ids: signature
                .arguments
                .iter()
                .map(|arg_type| logical_from_type(arg_type).id as u8)
                .collect(),
            columns: duckdb_columns(&signature.columns),
        })
    }
}

impl DuckDBBind for DuckDBScalarFunctionBinder {
    fn scalar_function(&self, name: &str) -> Option<ScalarFunctionDef> {
        // Pivot's own scalar functions (e.g. drop_cache) are generic, not
        // catalog-specific, so their signatures live in the planner rather than
        // on the catalog.
        let signature = crate::expression::builtin_scalar_function(name)?;
        Some(ScalarFunctionDef {
            arg_type_ids: signature
                .arguments
                .iter()
                .map(|arg_type| logical_from_type(arg_type).id as u8)
                .collect(),
            return_type_id: logical_from_type(&signature.return_type).id as u8,
            is_volatile: signature.volatile,
        })
    }
}
