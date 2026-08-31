//! Catalog: how the planner discovers tables.
//!
//! The planner does not know anything about persistence; it defers to a
//! caller-supplied [`CatalogTransaction`] to resolve table names into
//! [`BoundTable`]s, each of which exposes a schema (as a list of Pivot [`Column`]s)
//! and knows how to compile itself into a dispatch scan spec.
//!
//! Because DuckDB owns SQL binding, the catalog and tables also need to be
//! visible to it: [`DuckDBTransactionAdapter`] and [`DuckDBTableAdapter`]
//! implement DuckDB's [`DuckDBTransaction`] / [`DuckDBTable`] traits over our
//! Pivot types. They exist as
//! standalone wrapper structs (rather than blanket impls) because the orphan
//! rule prevents implementing a foreign trait for `Box<dyn BoundTable>` directly.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::expression::{CompareType, TableFilter};
use crate::operator::built_in_table_function;
use crate::types::{Type, logical_from_type};
use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, DynamicFilterSlot, Projection, RecordBatchOperatorSpec};
use duckdb_planner::DuckDBColumn;
use duckdb_planner::Expr;
use duckdb_planner::ScalarValue;
use duckdb_planner::catalog_provider::{DuckDBTable, DuckDBTransaction, ScalarFunctionDef};
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

/// A table name qualified by the schema holding it, with no datastore
/// qualifier: how one datastore names its own tables. A [`TableReference`] is
/// this plus the datastore that owns it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchemaQualifiedTableName {
    pub schema: String,
    pub table: String,
}

impl SchemaQualifiedTableName {
    pub fn new(schema: impl Into<String>, table: impl Into<String>) -> Self {
        Self {
            schema: schema.into(),
            table: table.into(),
        }
    }

    /// The name in the default schema: for a caller that names a table with no
    /// schema of its own (an embedded API, a test).
    pub fn in_default_schema(table: impl Into<String>) -> Self {
        Self::new(crate::DEFAULT_SCHEMA_NAME, table)
    }
}

/// The diagnostic form, for error text and UI labels. Not a SQL identifier:
/// nothing is quoted or escaped, so a name containing a dot or a quote renders
/// ambiguously. Build SQL from the two fields separately.
impl std::fmt::Display for SchemaQualifiedTableName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.schema, self.table)
    }
}

/// One table name as DuckDB resolved it: the datastore/database, the schema
/// inside it, and the table name inside that schema. Plan-cache dependencies use
/// the fully-qualified triple so equal table names in different schemas or
/// datastores never collide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableReference {
    pub datastore: String,
    pub schema: String,
    pub table: String,
}

impl TableReference {
    /// This reference with its datastore qualifier dropped: the name the owning
    /// datastore resolves the table by.
    pub fn schema_qualified_name(&self) -> SchemaQualifiedTableName {
        SchemaQualifiedTableName::new(self.schema.clone(), self.table.clone())
    }
}

/// The immutable identity and snapshot version of one table.
///
/// `identity` distinguishes a dropped/recreated table from its predecessor even
/// when both are at version zero. It is deliberately opaque to the planner; a
/// backend chooses a stable representation (Pivot's datastore uses its
/// manifest table ID).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRevision {
    pub identity: String,
    pub version: u64,
}

/// Description of a table to be created, produced by translating a
/// `CREATE TABLE` statement, consumed by a transaction's `bind_create_table`
/// ([`CatalogTransaction::bind_create_table`], which routes to the target
/// datastore).
///
/// `options` carries the `WITH (...)` clause verbatim so the target datastore
/// implementation can decide what to do with backend-specific keys.
///
/// `datastore_name` and `schema_name` are the qualifiers this create targets.
/// DuckDB resolves both while binding the statement, so in practice each is
/// `Some` even for an unqualified `CREATE TABLE t`, which arrives carrying the
/// current database and schema. They stay optional so a caller building a
/// request by hand (an embedded API, a test) can leave the choice to the
/// catalog, which reads a `None` as the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTableRequest {
    pub datastore_name: Option<String>,
    pub schema_name: Option<String>,
    pub name: String,
    pub columns: Vec<Column>,
    pub options: HashMap<String, String>,
    pub if_not_exists: bool,
}

impl CreateTableRequest {
    /// The name the target datastore will register the new table under: the
    /// schema the statement named, or the default when it named none.
    pub fn schema_qualified_name(&self) -> SchemaQualifiedTableName {
        SchemaQualifiedTableName::new(
            self.schema_name
                .as_deref()
                .unwrap_or(crate::DEFAULT_SCHEMA_NAME),
            self.name.clone(),
        )
    }
}

/// Description of a table to be dropped, produced by translating a
/// `DROP TABLE` statement and consumed by
/// [`CatalogTransaction::bind_drop_table`], which routes it to the datastore
/// holding the table.
///
/// `datastore_name` and `schema_name` are the qualifiers DuckDB resolved while
/// binding the statement, so in practice each is `Some` whenever the table was
/// found. They stay optional both for a caller building a request by hand and
/// for an `IF EXISTS` drop of a missing table, where DuckDB has nothing to
/// resolve and leaves the qualifiers as written (possibly absent); the catalog
/// reads a `None` as the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTableRequest {
    pub datastore_name: Option<String>,
    pub schema_name: Option<String>,
    pub name: String,
    pub if_exists: bool,
}

impl DropTableRequest {
    /// The name the target datastore resolves the dropped table by: the schema
    /// the statement named, or the default when it named none.
    pub fn schema_qualified_name(&self) -> SchemaQualifiedTableName {
        SchemaQualifiedTableName::new(
            self.schema_name
                .as_deref()
                .unwrap_or(crate::DEFAULT_SCHEMA_NAME),
            self.name.clone(),
        )
    }
}

/// Description of a schema to be created, produced by translating a
/// `CREATE SCHEMA` statement and consumed by
/// [`CatalogTransaction::bind_create_schema`], which routes it to the target
/// datastore.
///
/// `datastore_name` is the qualifier the statement wrote (`CREATE SCHEMA db.s`),
/// or `None` when unqualified, which routes to the default datastore. Unlike a
/// `CREATE TABLE`, this really is `None` for an unqualified statement: naming a
/// schema to create involves no lookup of an existing one, so DuckDB has
/// nothing to resolve the qualifier against and leaves it as written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSchemaRequest {
    pub datastore_name: Option<String>,
    pub name: String,
    pub if_not_exists: bool,
}

/// Description of a user to be created, produced by translating a
/// `CREATE USER` statement and consumed by
/// [`CatalogTransaction::bind_create_user`]. Users are server-wide (they live
/// in the metastore, not a datastore), so it carries no qualifier.
#[derive(Clone, PartialEq, Eq)]
pub struct CreateUserRequest {
    pub name: String,
    /// The password the stored credential is derived from, or `None` for a
    /// trusted user.
    pub password: Option<String>,
}

/// Redacted: the password must not reach a log through a `{:?}` of some plan
/// or transaction that happens to hold a request.
impl Debug for CreateUserRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateUserRequest")
            .field("name", &self.name)
            .field("password", &self.password.as_ref().map(|_| "redacted"))
            .finish()
    }
}

/// One query's transaction: a consistent **snapshot** of the catalog, opened
/// before the query is planned and held until [`commit`](Self::commit) or
/// [`rollback`](Self::rollback). Every table the query binds resolves through
/// this snapshot (never through the live catalog, which a background refresh
/// may be updating concurrently), so a plan's scans, its late materialize, and
/// its metadata peepholes all see one frozen view. A backend may also hold
/// pending writes here until commit.
#[async_trait]
pub trait CatalogTransaction: Debug + Send + Sync {
    /// Whether `datastore` defines a schema named `schema` in this transaction's
    /// frozen snapshot. Asked before any table in that schema is resolved, so a
    /// reference to a schema that does not exist is reported as such rather than
    /// as a missing table. `false` for a datastore this transaction doesn't know.
    fn does_schema_exist(&self, datastore: &str, schema: &str) -> bool;

    /// Resolve `reference` to a fresh, independently-mutable [`BoundTable`], or
    /// `None` if its datastore holds no such table in that schema.
    fn bind_table(&self, reference: &TableReference) -> Option<Box<dyn BoundTable>>;

    /// Bind the planner-owned `read_parquet(path)` function using this
    /// catalog's external-file policy. The composite catalog implements this
    /// as a global capability rather than routing it to a named datastore.
    fn bind_read_parquet(&self, _location: &str) -> Result<Box<dyn BoundTable>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "read_parquet is not supported by this catalog",
        )
        .into())
    }

    /// The identity and version of `reference` in this transaction's frozen
    /// snapshot of its datastore, or `None` if no such table exists. This must
    /// return `Some` for every table returned by [`bind_table`](Self::bind_table).
    fn table_revision(&self, reference: &TableReference) -> Option<TableRevision>;

    /// Compact a table from this transaction's frozen catalog view.
    async fn compact(
        &self,
        _datastore: &str,
        _table: &SchemaQualifiedTableName,
        _final_sweep: bool,
    ) -> Result<u64> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this catalog does not support COMPACT",
        )
        .into())
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

    /// Resolve a `DROP TABLE` by routing to the datastore
    /// [`DropTableRequest::datastore_name`] names (the default when unqualified)
    /// and deferring to that datastore's own `bind_drop_table`.
    fn bind_drop_table(&self, _request: DropTableRequest) -> Result<Box<dyn TableDrop>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this catalog does not support DROP TABLE",
        )
        .into())
    }

    /// Resolve a `CREATE SCHEMA` by routing to the datastore
    /// [`CreateSchemaRequest::datastore_name`] names (the default when
    /// unqualified) and deferring to that datastore's own `bind_create_schema`.
    fn bind_create_schema(&self, _request: CreateSchemaRequest) -> Result<Box<dyn SchemaCreation>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this catalog does not support CREATE SCHEMA",
        )
        .into())
    }

    /// Resolve a `CREATE USER` by routing to wherever users live (the
    /// metastore); users are server-wide, not a datastore's.
    fn bind_create_user(&self, _request: CreateUserRequest) -> Result<Box<dyn UserCreation>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this catalog does not support CREATE USER",
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
/// then builds the dataflow with the worker pool. The dataflow stages its result;
/// durable creation belongs to [`CatalogTransaction::commit`].
pub trait TableCreation: Send + Sync {
    /// Build the dataflow that fetches the new table's file footers over the pool
    /// and stages the completed creation for the transaction's commit.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec>;
}

/// A resolved `DROP TABLE`, bound to the datastore holding the table and ready
/// to be compiled into the dataflow that stages the drop.
///
/// A drop reads no data, so the dataflow this compiles to does nothing but
/// stage it. It exists so that the catalog changes when the statement *runs*
/// rather than when it is planned: resolving and compiling a statement must
/// leave the catalog untouched, or merely planning one (to report an error, to
/// render `EXPLAIN`) would drop the table. Durable removal belongs to
/// [`CatalogTransaction::commit`], as durable creation does for a table.
pub trait TableDrop: Send + Sync {
    /// Build the dataflow that stages this drop for the transaction's commit.
    /// It emits no rows.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec>;
}

/// A resolved `CREATE SCHEMA`, bound to its target datastore and ready to be
/// compiled into the dataflow that creates it, exactly as [`TableCreation`] is
/// for a table.
///
/// A schema has no data to read, so the dataflow this compiles to does nothing
/// but stage the creation. It exists so that the catalog changes when the
/// statement *runs* rather than when it is planned: resolving and compiling a
/// statement must leave the catalog untouched, or merely planning one (to
/// report an error, to render `EXPLAIN`) would create the schema. Durable
/// creation belongs to [`CatalogTransaction::commit`], as it does for a table.
pub trait SchemaCreation: Send + Sync {
    /// Build the dataflow that stages this creation for the transaction's
    /// commit. It emits no rows.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec>;
}

/// A resolved `CREATE USER`, ready to be compiled into the dataflow that
/// creates it, exactly as [`SchemaCreation`] is for a schema: the dataflow
/// does nothing but stage the creation, so the user appears when the statement
/// *runs* rather than when it is planned, and durable creation belongs to
/// [`CatalogTransaction::commit`].
pub trait UserCreation: Send + Sync {
    /// Build the dataflow that stages this creation for the transaction's
    /// commit. It emits no rows.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec>;
}

/// A table that the planner can read from.
///
/// Implementations expose their catalog identity and frozen revision, the
/// column list used during planning, and a way to compile a scan into a
/// dispatch [`RecordBatchOperatorSpec`].
pub trait BoundTable: Debug + Send + Sync {
    /// The qualified name the catalog resolved for this binding, as it was bound.
    fn table_reference(&self) -> TableReference;

    /// The exact table snapshot captured by this binding.
    fn table_revision(&self) -> TableRevision;

    /// Whether a plan containing this binding may be retained after the query.
    /// Virtual metadata tables return `false` because their rows were captured
    /// from the catalog transactions that planned the query.
    fn is_plan_cacheable(&self) -> bool {
        true
    }

    /// Whether DuckDB may rewrite this table's scan into a narrow row-ID scan
    /// followed by [`materialize`](BoundTable::materialize).
    ///
    /// Implementations opting in must honor `emit_row_group_metadata` in
    /// [`compile_scan`](BoundTable::compile_scan) and implement `materialize`.
    fn supports_late_materialization(&self) -> bool {
        false
    }

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

    /// Whether a scan of this table applies the variant field extracts carried
    /// in its [`Projection`].
    ///
    /// Unlike `dynamic_filters`, a pushed extract is not an optimization a
    /// backend may quietly decline: a scan that ignored one would emit the whole
    /// variant document where the query asked for a single field, which is a
    /// wrong answer rather than a slow one. The default is therefore `false`,
    /// and the planner keeps the extraction as an expression above the scan. A
    /// backend that can resolve a path against its own storage layout, and so
    /// read only the leaves the path needs, overrides this to `true`.
    fn applies_variant_extracts(&self) -> bool {
        false
    }

    /// Build a dispatch spec that writes the rows produced by `input` into this
    /// table and emits one `BIGINT` row with the count. Durable publication
    /// belongs to the query transaction's [`commit`](CatalogTransaction::commit).
    /// The default rejects INSERT (a read-only or virtual table).
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

    /// Whether each column (in [`columns`](Self::columns) order) can actually
    /// hold SQL NULLs. The planner routes a nullable column through the
    /// null-aware execution paths; a `false` keeps the branch-free fast paths,
    /// so a binding should report `false` whenever it can prove the data has
    /// no NULLs (e.g. from parquet statistics), not merely echo a declared
    /// `OPTIONAL`. The default reports every column nullable, which is always
    /// sound.
    fn nullability(&self) -> Vec<bool> {
        vec![true; self.columns().len()]
    }

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

    /// The table's estimated total row count for cost-based planning (join
    /// ordering, hash-join build/probe side choice). Unlike
    /// [`row_count`](BoundTable::row_count) this is an estimate, not an exact
    /// answer: predicates pushed into this binding don't invalidate it, since
    /// the cost model wants the base table's size and applies filter
    /// selectivity itself. `None` means unknown; the planner then uses its own
    /// defaults. Answered from the binding's captured snapshot, as in
    /// [`compile_scan`](BoundTable::compile_scan).
    fn estimate_row_count(&self) -> Option<u64> {
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

    fn supports_late_materialization(&self) -> bool {
        self.table.supports_late_materialization()
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

    fn estimate_row_count(&self) -> Option<u64> {
        self.table.estimate_row_count()
    }
}

/// Adapts a Pivot [`CatalogTransaction`] to DuckDB's [`DuckDBTransaction`]
/// trait: table lookups during one plan's binding resolve against this
/// transaction's snapshot.
pub struct DuckDBTransactionAdapter {
    pub transaction: Arc<dyn CatalogTransaction>,
}

impl DuckDBTransaction for DuckDBTransactionAdapter {
    fn scalar_function(&self, name: &str) -> Option<ScalarFunctionDef> {
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

    fn does_schema_exist(&self, datastore: &str, schema: &str) -> bool {
        self.transaction.does_schema_exist(datastore, schema)
    }

    fn bind_table(
        &self,
        datastore: &str,
        schema: &str,
        name: &str,
    ) -> Option<Box<dyn DuckDBTable>> {
        let reference = TableReference {
            datastore: datastore.to_string(),
            schema: schema.to_string(),
            table: name.to_string(),
        };
        let table = self.transaction.bind_table(&reference)?;
        Some(Box::new(DuckDBTableAdapter { table }))
    }

    fn bind_table_function(
        &self,
        name: &str,
        arguments: Vec<ScalarValue>,
    ) -> duckdb_planner::catalog_provider::Result<Box<dyn DuckDBTable>> {
        let function = built_in_table_function(name)
            .ok_or_else(|| format!("no registered table function named `{name}`"))?;
        let table = function.bind(&arguments, self.transaction.as_ref())?;
        Ok(Box::new(DuckDBTableAdapter { table }))
    }
}
