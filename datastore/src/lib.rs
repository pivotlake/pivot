//! The **datastore** layer: one named data source and the per-query transaction
//! it opens.
//!
//! A [`Datastore`] is a single named source the planner can resolve tables
//! against; it opens a [`DatastoreTransaction`], a consistent snapshot held for
//! one query's lifetime. A composite over several datastores is a
//! [`planner::catalog::CatalogTransaction`] per query, whose
//! [`planner::catalog::CatalogTransaction`] routes each resolution to the right
//! datastore's transaction by name.

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use planner::TableFunction;
use planner::catalog::{
    BoundTable, CreateSchemaRequest, CreateTableRequest, DropTableRequest, Result, SchemaCreation,
    SchemaQualifiedTableName, TableCreation, TableDrop, TableRevision,
};
use planner::types::Type;

/// One table exposed by a datastore transaction's frozen catalog view, with the
/// columns it declares and the data files it holds at that snapshot.
///
/// The cross-datastore catalog uses this backend-neutral description to build
/// virtual metadata relations without reaching into a concrete datastore's
/// manifest or in-memory index. `id` is the datastore catalog's durable table
/// identifier; Delta datastores return the ID recorded in their Pivot manifest,
/// not the independent ID in the Delta log's `metaData` action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatastoreTableMetadata {
    pub name: SchemaQualifiedTableName,
    pub id: String,
    /// The columns the table declares, in schema order.
    pub columns: Vec<DatastoreColumnMetadata>,
    /// The table's sort columns, in order; empty when the table is unsorted.
    pub sort_by: Vec<String>,
    /// The table's partition columns, in order; empty when the table is
    /// unpartitioned.
    pub partition_by: Vec<String>,
    /// How many rows the table's committed files hold.
    pub total_rows: u64,
    /// How many bytes those files occupy in storage.
    pub bytes: u64,
    /// What those bytes hold decoded, before the storage format compressed them.
    pub bytes_uncompressed: u64,
    /// The table's committed files, in the order its datastore holds them.
    /// Files a pending write staged are not committed to the snapshot and so
    /// are absent, exactly as a table the same transaction is still creating is.
    pub files: Vec<DatastoreFileMetadata>,
}

/// One column of a [`DatastoreTableMetadata`]: what the table declares it as,
/// and what the table's committed files spend on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatastoreColumnMetadata {
    pub name: String,
    pub column_type: Type,
    /// The column's position in the table's schema, counted from zero.
    pub position: usize,
    /// How many bytes this column occupies across the table's committed files.
    pub bytes: u64,
    /// What those bytes hold decoded, before the storage format compressed them.
    pub bytes_uncompressed: u64,
    pub is_partition_key: bool,
    pub is_sort_key: bool,
}

/// One data file of a [`DatastoreTableMetadata`], described the same
/// backend-neutral way as the table that owns it.
///
/// The owning table is the one this hangs off, so a file names no table of its
/// own. `path` names the file the way its datastore does, relative to that
/// datastore's storage root, and `bytes` is the size the catalog recorded for
/// it when the file was committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatastoreFileMetadata {
    pub path: String,
    pub bytes: u64,
    /// What the file's bytes hold decoded, before the storage format compressed
    /// them.
    pub bytes_uncompressed: u64,
    /// The partition the file belongs to: `column=value` pairs in the table's
    /// partition order, comma-separated. Empty when the table is unpartitioned,
    /// or when the file predates its partitioning.
    pub partition: String,
}

/// One query's transaction against a **single datastore**: a consistent
/// snapshot of that datastore, opened by [`Datastore::begin_transaction`] before
/// the query is planned and held until its own [`commit`](Self::commit) or
/// [`rollback`](Self::rollback). Every table the query binds resolves
/// through this snapshot (never through the live datastore, which a background
/// refresh may be updating concurrently), so a plan's scans, its late
/// materialize, and its metadata peepholes all see one frozen view. A backend
/// may also hold pending writes here until commit.
///
/// This is the per-datastore half of the pair: it resolves tables by
/// schema-qualified name, with no datastore qualifier. A
/// [`planner::catalog::CatalogTransaction`] composes several of these and routes
/// to them by datastore name.
#[async_trait]
pub trait DatastoreTransaction: Debug + Send + Sync {
    /// Whether this datastore defines a schema named `schema` in the
    /// transaction's frozen snapshot. Every datastore defines
    /// [`planner::DEFAULT_SCHEMA_NAME`], so a table named with no schema always
    /// has a schema to resolve in.
    fn does_schema_exist(&self, schema: &str) -> bool;

    /// Resolve the schema-qualified `name`, in the datastore the catalog serves
    /// as `datastore`, to a fresh, independently-mutable [`BoundTable`] bound to
    /// this transaction's snapshot, or `None` if no such table exists in the
    /// snapshot. Each call returns a unique `Box`, so per-query filter pushdown
    /// can mutate the table without affecting concurrent queries. The returned
    /// binding captures the snapshot's copy of the table, so its compile needs
    /// no transaction handle.
    ///
    /// `datastore` is the name the *catalog* holds this datastore under, passed
    /// in rather than known here: a datastore is registered by its owner, so the
    /// binding can only learn the qualifier it was reached through from the
    /// caller that routed to it. The binding records it, so re-resolving the
    /// table later uses the very key it was bound by.
    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> Option<Box<dyn BoundTable>>;

    /// The identity and version of `name` in this transaction's frozen
    /// snapshot, or `None` if no such table exists. This must return `Some` for
    /// every table returned by [`bind_table`](Self::bind_table).
    fn table_revision(&self, name: &SchemaQualifiedTableName) -> Option<TableRevision>;

    /// Every stored table defined by this transaction's frozen snapshot.
    ///
    /// Virtual catalog relations are not datastore objects and therefore are not
    /// included.
    fn tables(&self) -> Vec<DatastoreTableMetadata>;

    /// A backend-specific table-valued function by `name`, or `None`. This is
    /// how a backend contributes functions only it can answer (e.g. `metadata`,
    /// which needs the backend's row-group metadata). It lives on the
    /// transaction rather than the datastore because such a function reads data,
    /// which it captures from this transaction's snapshot when resolved. The
    /// generic functions (`generate_series`, `range`) are resolved by the
    /// planner itself and never reach here. Default: none.
    fn bind_table_function(&self, _name: &str) -> Option<Box<dyn TableFunction>> {
        None
    }

    /// Resolve a `CREATE TABLE` against this datastore into a [`TableCreation`]
    /// the caller compiles into the dataflow that writes the new table. Resolution
    /// runs on the coordinator (validating the request and locating the table's
    /// files); compiling it runs the footer-fetch dataflow over the pool. Split so
    /// DDL resolution and dataflow construction are two steps, like table binding.
    /// The default rejects DDL (a read-only datastore).
    fn bind_create_table(&self, _request: CreateTableRequest) -> Result<Box<dyn TableCreation>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this datastore does not support CREATE TABLE",
        )
        .into())
    }

    /// Resolve a `DROP TABLE` against this datastore into a [`TableDrop`] the
    /// caller compiles into the dataflow that stages it. Split the same way
    /// [`bind_create_table`](Self::bind_create_table) is, so that resolving a
    /// statement never changes the catalog by itself: the drop is staged when
    /// the compiled dataflow runs, and made durable by [`commit`](Self::commit).
    /// The default rejects DDL (a read-only datastore).
    fn bind_drop_table(&self, _request: DropTableRequest) -> Result<Box<dyn TableDrop>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this datastore does not support DROP TABLE",
        )
        .into())
    }

    /// Resolve a `CREATE SCHEMA` against this datastore into a [`SchemaCreation`]
    /// the caller compiles into the dataflow that stages it. Split the same way
    /// [`bind_create_table`](Self::bind_create_table) is, so that resolving a
    /// statement never changes the catalog by itself: the schema is staged when
    /// the compiled dataflow runs, and made durable by
    /// [`commit`](Self::commit). The default rejects DDL (a read-only
    /// datastore).
    fn bind_create_schema(&self, _request: CreateSchemaRequest) -> Result<Box<dyn SchemaCreation>> {
        Err(Box::<dyn std::error::Error + Send + Sync>::from(
            "this datastore does not support CREATE SCHEMA",
        )
        .into())
    }

    /// Commit this transaction: publish whatever it staged against this datastore
    /// (the files an INSERT injected). A read-only transaction is a no-op. Async
    /// so the backend can hop the blocking store I/O of a writing commit to the
    /// blocking pool and finish a read-only one inline.
    async fn commit(&self) -> Result<()> {
        Ok(())
    }

    /// Roll back this transaction: discard whatever it staged. Default: nothing,
    /// the snapshot is released when the last reference drops.
    fn rollback(&self) {}
}

/// One data source the planner can resolve tables against.
#[async_trait]
pub trait Datastore: Debug + Send + Sync {
    /// Open a transaction: snapshot this datastore as it stands right now. All
    /// binding for one query resolves through the returned snapshot, so the
    /// query reads a single consistent view regardless of concurrent refreshes
    /// or commits. The caller holds the transaction for the query's lifetime and
    /// passes it to commit or rollback when the query finishes. The `Arc<Self>`
    /// receiver lets the transaction hold the datastore alive, so a `CREATE TABLE`
    /// it commits can publish the new table straight back into the datastore.
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction>;

    /// The format this datastore stores its data in (`delta`), as
    /// `system.datastores` reports it. A constant of the backend: what a
    /// configuration selects the backend by.
    fn kind(&self) -> &'static str;

    /// Where this datastore's data lives, as a URI (`file:///var/lib/pivot`,
    /// `s3://bucket/prefix`). Every path a table or a file of this datastore
    /// reports is relative to it.
    fn data_path(&self) -> String;

    /// Start this datastore's background maintenance (e.g. periodic refresh and
    /// compaction), spawning its tasks onto the ambient async runtime. Called
    /// once, when the server begins serving. The `Arc<Self>` receiver lets the
    /// spawned tasks hold the datastore alive. Default: no maintenance.
    fn start(self: Arc<Self>) {}

    /// Stop the background maintenance [`start`](Self::start) spawned, aborting
    /// its tasks. Called on shutdown *before* the worker pool is torn down, so a
    /// maintenance sweep cannot race the pool's teardown. Default: nothing to
    /// stop.
    fn abort(&self) {}

    /// Merge `table`'s small files into target-sized ones now, synchronously,
    /// the way this datastore's own background compaction would over time.
    /// Backs the `COMPACT` statement, so it runs on the coordinator's async
    /// context, never on a dispatch worker. One call is one sweep;
    /// `final_sweep` keeps sweeping until a sweep advances no version (a merge
    /// changes the file list, so one pass can leave a tail). Returns the
    /// number of sweeps performed, or an error if `table` does not exist.
    async fn compact(
        self: Arc<Self>,
        table: &SchemaQualifiedTableName,
        final_sweep: bool,
    ) -> Result<u64>;

    /// Downcast hook (owned): recover the concrete backend as an owned `Arc`, for
    /// a server feature specific to one datastore format (the web dashboard's
    /// Parquet-level introspection). Implemented as
    /// `fn into_any_arc(self: Arc<Self>) { self }`.
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}
