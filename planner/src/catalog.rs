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

use std::any::Any;
use std::collections::{BTreeMap, HashMap};

use crate::expression::{CompareType, TableFilter};
use crate::operator::TableFunction;
use crate::types::{Type, logical_from_type};
use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, DynamicFilterSlot, Projection, RecordBatchOperatorSpec};
use duckdb_planner::DuckDBColumn;
use duckdb_planner::Expr;
use duckdb_planner::catalog_provider::{
    DuckDBBind, DuckDBTable, ScalarFunctionDef, TableFunctionDef,
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

/// Whether a secret option's value is sensitive and must render as `redacted`
/// anywhere a human might read it - operator `Display`, logs, or a secrets
/// listing. Matches DuckDB's redaction set for S3 secrets.
pub fn secret_option_is_redacted(key: &str) -> bool {
    ["secret", "session_token"]
        .iter()
        .any(|redacted| key.eq_ignore_ascii_case(redacted))
}

/// Description of a secret to create - produced by translating a
/// `CREATE SECRET` statement, consumed by [`Catalog::create_secret`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSecretRequest {
    pub name: String,
    /// The secret's TYPE (e.g. `s3`), lowercased.
    pub secret_type: String,
    /// How the secret's values were produced (`config` for explicit values).
    pub provider: String,
    /// Path prefixes the secret applies to; the longest matching prefix wins
    /// when a path looks a secret up.
    pub scope: Vec<String>,
    /// The key-value options (`key_id`, `secret`, `region`, ...), keys
    /// lowercased. Ordered so displays and listings are deterministic.
    pub options: BTreeMap<String, String>,
    /// A temporary secret lives in memory only and is gone on restart; a
    /// persistent one is written to the catalog's store.
    pub temporary: bool,
    pub or_replace: bool,
    pub if_not_exists: bool,
}

/// Description of a secret to drop - produced by translating a `DROP SECRET`
/// statement, consumed by [`Catalog::drop_secret`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSecretRequest {
    pub name: String,
    pub if_exists: bool,
    /// `Some(true)` drops only a temporary secret, `Some(false)` only a
    /// persistent one, `None` whichever holds the name (erroring when both do).
    pub temporary: Option<bool>,
}

/// A bound secret statement, ready to apply to a [`Catalog`]. Like `SET`, a
/// secret statement compiles to no dataflow: the server extracts it from the
/// plan ([`Plan::as_secret_command`](crate::Plan::as_secret_command)) and
/// applies it here, acknowledging with [`tag`](Self::tag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretCommand {
    Create(CreateSecretRequest),
    Drop(DropSecretRequest),
}

impl SecretCommand {
    pub fn apply(self, catalog: &dyn Catalog) -> Result<()> {
        match self {
            SecretCommand::Create(request) => catalog.create_secret(request),
            SecretCommand::Drop(request) => catalog.drop_secret(request),
        }
    }

    /// The command tag acknowledging this statement on the wire.
    pub fn tag(&self) -> &'static str {
        match self {
            SecretCommand::Create(_) => "CREATE SECRET",
            SecretCommand::Drop(_) => "DROP SECRET",
        }
    }
}

/// An opaque per-query context, created once per [`Plan::compile`](crate::Plan::compile)
/// (via [`Catalog::query_context`]) and threaded to every [`Table::compile`]. The
/// planner treats it as a black box; a backend's [`Table`] downcasts it to its
/// own concrete context and reads whatever it needs. The catalog, for instance,
/// uses it to reload + pin each table's files once per query — so a reused
/// (cached) plan sees data committed since it was planned, and a scan and its
/// late materialize share one snapshot.
pub trait QueryContext {
    /// Downcast hook. The trait carries no behaviour of its own — a backend
    /// recovers its concrete context from this and reads whatever it needs.
    fn as_any(&self) -> &dyn Any;
}

/// The default [`QueryContext`]: carries nothing, for backends whose tables are
/// always current (e.g. in-memory test stubs) and never downcast it.
pub struct NoQueryContext;

impl QueryContext for NoQueryContext {
    fn as_any(&self) -> &dyn Any {
        self
    }
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
    /// `ctx` is the per-query [`QueryContext`]; a backend reading mutable storage
    /// downcasts it to reload itself to the latest committed version before
    /// scanning, so a reused (cached) plan sees data committed since it was
    /// planned. Backends with no such notion ignore it.
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
        ctx: &dyn QueryContext,
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
        _ctx: &dyn QueryContext,
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
    /// means "unknown, scan instead" and is always a safe answer. `ctx` is the
    /// per-query context a mutable-storage backend downcasts to resolve its
    /// current files, exactly as [`compile`](Table::compile) does.
    fn column_min_max(
        &self,
        _column: usize,
        _ctx: &dyn QueryContext,
    ) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        None
    }

    /// The table's total row count derived purely from metadata, if it can be
    /// answered without scanning any rows (e.g. summing Parquet row-group row
    /// counts, with no predicates pushed into this binding). `None` means
    /// "unknown, scan instead" and is always a safe answer. `ctx` resolves the
    /// binding's current files, as in [`column_min_max`](Table::column_min_max).
    fn row_count(&self, _ctx: &dyn QueryContext) -> Option<i64> {
        None
    }
}

/// Convert Pivot columns into the DuckDB-typed columns the binder consumes
/// (logical type as a `u8` discriminant). Shared by base-table and table-function
/// binding.
fn duckdb_columns(columns: &[Column]) -> Vec<DuckDBColumn> {
    columns
        .iter()
        .map(|column| DuckDBColumn {
            name: column.name.clone(),
            duckdb_logical_type_id: logical_from_type(&column.col_type) as u8,
        })
        .collect()
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

    /// A fresh [`QueryContext`] for one [`Plan::compile`](crate::Plan::compile).
    /// Default: an empty context, for always-current backends.
    fn query_context(&self) -> Box<dyn QueryContext> {
        Box::new(NoQueryContext)
    }

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

    /// Create a secret (and, unless it is temporary, persist it in the
    /// catalog's durable storage). Secrets are control-plane metadata: this
    /// runs synchronously on the coordinator, no dataflow. The default rejects
    /// the statement, for backends that hold no secrets.
    fn create_secret(&self, _request: CreateSecretRequest) -> Result<()> {
        Err(Error::Other("this catalog does not support secrets".into()))
    }

    /// Drop a secret by name (see [`DropSecretRequest`] for the qualifier
    /// semantics). The default rejects the statement, for backends that hold
    /// no secrets.
    fn drop_secret(&self, _request: DropSecretRequest) -> Result<()> {
        Err(Error::Other("this catalog does not support secrets".into()))
    }

    /// A backend-specific table-valued function by `name`, or `None`. This is how
    /// a catalog contributes functions only it can answer (e.g. `metadata`, which
    /// needs the backend's row-group metadata) without the generic planner
    /// knowing about them. The generic functions (`generate_series`, `range`) are
    /// resolved by the planner itself and never reach here. Default: none.
    fn table_function(&self, _name: &str) -> Option<Box<dyn TableFunction>> {
        None
    }
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

    fn table_function(&self, name: &str) -> Option<TableFunctionDef> {
        // The function's own signature is the single source of truth; convert its
        // Pivot types to DuckDB logical type ids for the binder.
        let signature = self.catalog.table_function(name)?.signature();
        Some(TableFunctionDef {
            arg_type_ids: signature
                .arguments
                .iter()
                .map(|arg_type| logical_from_type(arg_type) as u8)
                .collect(),
            columns: duckdb_columns(&signature.columns),
        })
    }

    fn scalar_function(&self, name: &str) -> Option<ScalarFunctionDef> {
        // Pivot's own scalar functions (e.g. drop_cache) are generic, not
        // catalog-specific, so their signatures live in the planner rather than
        // on the catalog.
        let signature = crate::expression::builtin_scalar_function(name)?;
        Some(ScalarFunctionDef {
            arg_type_ids: signature
                .arguments
                .iter()
                .map(|arg_type| logical_from_type(arg_type) as u8)
                .collect(),
            return_type_id: logical_from_type(&signature.return_type) as u8,
            is_volatile: signature.volatile,
        })
    }
}
