//! A concrete [`datastore::Datastore`] implementation backed by Parquet.
//!
//! The datastore stores tables in a `RwLock<HashMap>` keyed by name, so multiple
//! threads can resolve and create tables concurrently: many readers (lookups)
//! coexist with infrequent writers (`CREATE TABLE`, file registrations).
//! Backing the map is the database's [`ObjectStore`], opened on an explicit
//! local directory or remote `s3://` root.
//!
//! Durable state lives in two places, both in the store:
//!
//! - the [`manifest`]: which schemas exist, and which tables exist in them
//!   (schema-qualified name and location);
//! - each table's standard Delta `_delta_log`: its schema, partitioning,
//!   version, and active Parquet files, interpreted by Delta Kernel.
//!
//! Each table's in-memory [`table::CatalogTable`] pairs its definition with the
//! per-file row groups at one log version. The in-memory set is kept current by
//! **push, not pull**: a periodic [`refresh_from_store`](DeltaDatastore::refresh_from_store)
//! sweep (the server runs one on an interval) reloads every table to its latest
//! committed version and fetches any new files' footers, and an in-process
//! writer (INSERT, compaction) [`publish_table`](DeltaDatastore::publish_table)s
//! its committed copy immediately. Queries never touch the store: a query opens
//! a transaction ([`Datastore::begin_transaction`]) whose [`DeltaSnapshot`]
//! freezes the table set as of that moment, and every binding, scan, and late
//! materialize of that query reads the frozen snapshot with zero I/O.
//!
//! Each resolve hands back a fresh [`TableBinding`], so per-query filter
//! pushdown accumulates on that binding alone.

mod binding;
mod insert_sink;
mod table;

pub use binding::TableBinding;

use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use uuid::Uuid;

use crate::manifest::{self, CatalogManifest, DeltaFileEntry};
use crate::parquet::ParquetTableError;
use crate::store::{self, DataFile, FileRef, LocalStore, ObjectPath, ObjectStore, open_store};
use async_trait::async_trait;
use crossbeam_deque::{Injector, Steal};
use datastore::{Datastore, DatastoreTransaction};
use dispatch::{DataFlowDispatcher, DataFlowError, OneShotNullaryFactory, RecordBatchOperatorSpec};
use planner::catalog::{
    BoundTable, CreateSchemaRequest, CreateTableRequest, Error as CatalogError,
    Result as CatalogResult, SchemaCreation, SchemaQualifiedTableName, TableCreation,
    TableRevision,
};
pub use table::CatalogTable;
pub use table::TableFile;
use thiserror::Error as ThisError;

/// `WITH (with_pre_existing_parquets = 'dir')`: a directory of Parquet files the new
/// table adopts as its initial data, where they already sit.
const PRE_EXISTING_PARQUETS_OPTION: &str = "with_pre_existing_parquets";
/// `WITH (partition_by = 'a, b')` — ordered, comma-separated partition columns.
const PARTITION_BY_OPTION: &str = "partition_by";
/// `WITH (sort_by = 'a, b')` — ordered, comma-separated sort columns.
const SORT_BY_OPTION: &str = "sort_by";

/// Every option `CREATE TABLE` understands. An option outside this set is
/// rejected rather than ignored: a statement that asks for something the
/// datastore does not implement has not been carried out, and silently
/// dropping it is how a retired option (or a typo) turns into a table that is
/// quietly not what was asked for.
const TABLE_OPTIONS: [&str; 3] = [
    PRE_EXISTING_PARQUETS_OPTION,
    PARTITION_BY_OPTION,
    SORT_BY_OPTION,
];

#[derive(Debug, ThisError)]
pub enum Error {
    #[error(
        "`{option}` path `{path}` must be a plain path, not a URL: it names a directory in the database's own storage, so it carries no scheme"
    )]
    OptionPathWithScheme { option: String, path: String },
    #[error("`{option}` is not a table option; this datastore takes {known}")]
    UnknownOption { option: String, known: String },
    #[error("`{option}` column `{column}` is not a declared column of the table")]
    UnknownSpecColumn { option: String, column: String },
    #[error(transparent)]
    ParquetTable(#[from] ParquetTableError),
    #[error("table `{0}` already exists")]
    TableExists(String),
    #[error("table `{0}` does not exist")]
    TableNotFound(String),
    #[error("schema `{0}` does not exist")]
    SchemaNotFound(String),
    #[error("schema `{0}` already exists")]
    SchemaExists(String),
    #[error("datastore received a transaction created by a different backend")]
    WrongTransactionType,
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error(transparent)]
    Store(#[from] store::StoreError),
    #[error(transparent)]
    Manifest(#[from] manifest::Error),
    #[error(transparent)]
    Delta(#[from] crate::delta::Error),
    #[error("loading table footers: {0}")]
    Load(#[from] DataFlowError),
    #[error(
        "table at `{location}`: file `{file}` has no loaded row-group metadata; the copy was not synced to its manifest"
    )]
    FooterNotLoaded { location: String, file: String },
    #[error("table at `{location}` commit conflict: input file `{file}` is no longer active")]
    CommitConflict { location: String, file: String },
    #[error(
        "table at `{location}` cannot delete `{file}`: the file sits outside the table's own storage, so it belongs to whoever the table adopted it from"
    )]
    DeletingOutsideStorage { location: String, file: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// The datastore's schemas and the tables inside them, shaped like the manifest
/// it is a projection of: each schema maps its table names to identities, and
/// the tables themselves are held once, by identity.
///
/// Going through the identity is what lets a name be reassigned without
/// touching the table, and keeps one owner for each [`CatalogTable`] so a
/// commit or publish never has to update two copies.
#[derive(Clone, Default)]
struct DatastoreIndex {
    /// Every schema the datastore defines, always including
    /// [`DEFAULT_SCHEMA_NAME`](planner::DEFAULT_SCHEMA_NAME), each mapping its
    /// table names to identities. A schema with no tables in it is still a
    /// schema.
    schemas: HashMap<String, HashMap<String, Uuid>>,
    tables_by_id: HashMap<Uuid, CatalogTable>,
}

impl DatastoreIndex {
    /// Add a table under `name` and its own identity, registering the schema if
    /// the index does not already list it. A table whose schema is missing would
    /// be present but unresolvable, so the index keeps itself consistent rather
    /// than trusting its two inputs to agree. The name index is written here
    /// alone, so this is the one place a table acquires a name.
    fn insert_table(&mut self, name: SchemaQualifiedTableName, table: CatalogTable) {
        self.schemas
            .entry(name.schema)
            .or_default()
            .insert(name.table, table.id());
        self.tables_by_id.insert(table.id(), table);
    }

    /// Swap in an advanced copy of a table already in the index. Its identity is
    /// unchanged, so neither the schema nor the name index needs touching.
    fn replace_table(&mut self, table: CatalogTable) {
        self.tables_by_id.insert(table.id(), table);
    }

    /// Register a schema, reporting whether it was new.
    fn insert_schema(&mut self, schema: String) -> bool {
        !self.schemas.contains_key(&schema) && {
            self.schemas.insert(schema, HashMap::new());
            true
        }
    }

    fn contains_schema(&self, schema: &str) -> bool {
        self.schemas.contains_key(schema)
    }

    fn table_id(&self, name: &SchemaQualifiedTableName) -> Option<Uuid> {
        self.schemas.get(&name.schema)?.get(&name.table).copied()
    }

    fn get_table_by_name(&self, name: &SchemaQualifiedTableName) -> Option<&CatalogTable> {
        self.tables_by_id.get(&self.table_id(name)?)
    }

    fn get_table_by_id(&self, id: &Uuid) -> Option<&CatalogTable> {
        self.tables_by_id.get(id)
    }

    fn contains_table(&self, name: &SchemaQualifiedTableName) -> bool {
        self.table_id(name).is_some()
    }

    /// Every table with the schema-qualified name it is currently indexed under.
    fn named_tables(&self) -> impl Iterator<Item = (SchemaQualifiedTableName, &CatalogTable)> {
        self.schemas.iter().flat_map(move |(schema, tables)| {
            tables.iter().filter_map(move |(table, id)| {
                Some((
                    SchemaQualifiedTableName::new(schema.clone(), table.clone()),
                    self.tables_by_id.get(id)?,
                ))
            })
        })
    }

    fn table_names(&self) -> impl Iterator<Item = SchemaQualifiedTableName> {
        self.schemas.iter().flat_map(|(schema, tables)| {
            tables
                .keys()
                .map(move |table| SchemaQualifiedTableName::new(schema.clone(), table.clone()))
        })
    }
}

/// A Delta datastore's concurrent table index, keyed by name.
///
/// `CREATE TABLE` compiles to a dataflow that reads every data file's footer
/// **once** (in parallel over the worker pool) and stages the materialized table
/// on its transaction. The transaction's blocking commit initializes the table,
/// records it in the database index, and publishes the entry in this shared map.
/// After that, a table evolves by log commits through
/// [`commit_to_table`](Self::commit_to_table): a writer (INSERT, compaction)
/// hands its files to the datastore, which appends them to, or swaps them into,
/// the live copy it holds here and publishes the result. Copies handed out for
/// reading drift; every query resolve refreshes its copy to the latest committed
/// version, so a commit by another process becomes visible to the next query.
///
/// Cloneable (every field is an `Arc`, a `String`, or the shared dispatcher
/// handle), so a commit that writes can hand a clone to the blocking pool.
#[derive(Clone)]
pub struct DeltaDatastore {
    /// The in-memory schema and table sets. The lock guards the *index* (add on
    /// `CREATE`, swap-in on a resolve's refresh); each [`CatalogTable`] is itself
    /// a lock-free value that callers clone out and evolve independently. One
    /// lock covers both halves so the two `CREATE` paths cannot interleave their
    /// updates to the shared manifest document.
    tables_index: Arc<RwLock<DatastoreIndex>>,
    /// The database's object store: the table index, Delta logs, and tables'
    /// Parquet data. The datastore reads and writes a table's data through this
    /// one explicitly configured store: relative locations live under the
    /// database root, an absolute location at the store's own root (the
    /// filesystem root, or the bucket root).
    store: Arc<dyn ObjectStore>,
    /// The Delta Kernel engine every table's log is read and written through.
    /// Built once for the store, so a refresh, a commit, or a vacuum sweep reuses
    /// the one object-store client and task executor instead of standing up its
    /// own; each table holds a clone.
    engine: crate::delta::DeltaEngine,

    /// The worker pool every footer fetch runs on. Held by the datastore because
    /// refreshes drive their own dataflows, with no dispatcher passed in.
    dispatcher: DataFlowDispatcher,

    /// The background maintenance this datastore runs once [`start`](Datastore::start)
    /// is called: a periodic refresh sweep and optional compaction. `None` for a
    /// datastore opened without maintenance (embedded / test use).
    maintenance: Option<crate::MaintenanceConfig>,
    /// Abort handles for the maintenance tasks [`start`](Datastore::start)
    /// spawned, so [`abort`](Datastore::abort) can stop them on shutdown before
    /// the worker pool is torn down. Behind an `Arc<Mutex<..>>` so the datastore
    /// stays `Clone` (a writing commit hands a clone to the blocking pool).
    maintenance_tasks: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
}

impl std::fmt::Debug for DeltaDatastore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaDatastore")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl DeltaDatastore {
    /// Open the default datastore at an explicit local data directory. The
    /// caller owns the directory and its lifetime; pivotdb never substitutes a
    /// generated temporary path. A database with no manifest yet opens empty.
    /// No background maintenance runs (see [`from_store`](Self::from_store) for
    /// that seam).
    pub fn open_local(
        root: impl Into<PathBuf>,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Arc<Self>> {
        Self::from_store(Arc::new(LocalStore::new(root)), dispatcher, None)
    }

    /// Open a persisted database rooted at `uri`, a local directory (or
    /// `file://…`), or a remote `s3://…` object store, reloading every
    /// table the manifest records at its latest version: read each table's
    /// manifest (schema + file list) and fetch its files' footers, building the
    /// in-memory [`CatalogTable`]. A database with no manifest yet opens empty.
    /// No background maintenance runs (see [`from_store`](Self::from_store)).
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Arc<Self>> {
        let store: Arc<dyn ObjectStore> = open_store(uri)?.into();
        Self::from_store(store, dispatcher, None)
    }

    /// Open a persisted database over an already-built object store. This is the
    /// seam a metastore uses: it constructs the
    /// store (local dir / S3, with whatever credentials it holds) and hands
    /// it in, rather than having the datastore re-derive one from a URI. Reloads
    /// every table the manifest records, exactly as [`open`](Self::open) does.
    pub fn from_store(
        store: Arc<dyn ObjectStore>,
        dispatcher: &DataFlowDispatcher,
        maintenance: Option<crate::MaintenanceConfig>,
    ) -> Result<Arc<Self>> {
        let manifest = CatalogManifest::load(store.as_ref())?;
        let engine = crate::delta::DeltaEngine::new(store.as_ref())?;

        let mut index = DatastoreIndex::default();
        for schema in &manifest.schemas {
            index.insert_schema(schema.name.clone());
        }
        for entry in manifest.tables() {
            let (name, id, location) = entry?;
            let table = Self::load_table(dispatcher, &store, &engine, id, location)?;
            index.insert_table(name, table);
        }

        Ok(Arc::new(Self {
            tables_index: Arc::new(RwLock::new(index)),
            store,
            engine,
            dispatcher: dispatcher.clone(),
            maintenance,
            maintenance_tasks: Arc::new(Mutex::new(Vec::new())),
        }))
    }

    /// Build the in-memory [`CatalogTable`] for one persisted table: read its
    /// manifest (erroring if the catalog points at a table that has none), then
    /// locate its committed files under the entry's location and fetch their
    /// footers over the pool.
    fn load_table(
        dispatcher: &DataFlowDispatcher,
        store: &Arc<dyn ObjectStore>,
        engine: &crate::delta::DeltaEngine,
        id: Uuid,
        location: &ObjectPath,
    ) -> Result<CatalogTable> {
        let delta_uri = crate::delta::table_uri(&store.location_uri(), location)?;
        let state = crate::delta::load_table(&delta_uri, engine)?;
        let data_files = state
            .file_entries
            .iter()
            .map(|e| e.file.clone().into_data_file(store.as_ref(), location))
            .collect::<store::Result<Vec<DataFile>>>()?;
        let declared_columns: Arc<[planner::catalog::Column]> = state.columns.clone().into();
        // The footer fetch returns each file's row groups keyed by identity; join
        // each back to its log entry (partition tuple) by path.
        let mut footers: HashMap<ObjectPath, Vec<Arc<crate::parquet::RowGroupMetadata>>> =
            crate::parquet::load_file_row_groups(dispatcher, &data_files, declared_columns)?
                .into_iter()
                .map(|loaded| (loaded.file.path.clone(), loaded.row_groups))
                .collect();
        // A file the fetch returned nothing for would scan as an empty file, so
        // the table fails to open rather than silently serving a narrower one.
        let files = state
            .file_entries
            .into_iter()
            .map(|entry| {
                let row_groups =
                    footers
                        .remove(&entry.file.path)
                        .ok_or_else(|| Error::FooterNotLoaded {
                            location: location.as_str().to_string(),
                            file: entry.file.path.as_str().to_string(),
                        })?;
                Ok(TableFile::new(entry, row_groups))
            })
            .collect::<Result<Vec<TableFile>>>()?;
        Ok(CatalogTable::new(
            id,
            location.clone(),
            state.snapshot,
            state.columns,
            state.partition_by,
            state.sort_by,
            files,
            store.clone(),
            dispatcher.clone(),
            engine.clone(),
        ))
    }

    /// Open a transaction, typed: freeze the current table set into a
    /// [`DeltaSnapshot`] and create the injectors that receive completed INSERT
    /// files and table creations. Cheap: clones the map (the row-group metadata
    /// inside is `Arc`-shared), no I/O. The [`Datastore::begin_transaction`]
    /// trait impl delegates here.
    pub fn begin_transaction(self: Arc<Self>) -> Arc<DeltaTransaction> {
        let snapshot = Arc::new(DeltaSnapshot {
            index: self.tables_index.read().unwrap().clone(),
        });
        Arc::new(DeltaTransaction {
            snapshot,
            uploaded_files: Arc::new(Injector::new()),
            pending_table_creations: Arc::new(Injector::new()),
            pending_schema_creations: Arc::new(Injector::new()),
            datastore: self,
        })
    }

    /// Bring the in-memory schema and table sets up to date with the store: pick
    /// up schemas and tables another process registered in the database index,
    /// and advance every
    /// table to its latest committed Delta version, fetching the footers of
    /// files it doesn't hold yet. This is the **only** place the read path
    /// pays store I/O; the server drives it on a background interval, so
    /// queries always bind against an already-materialized set. Returns whether
    /// anything changed.
    ///
    /// Tables are never removed here: there is no `DROP TABLE`, and removing a
    /// map entry on a manifest miss could race a concurrent in-process create.
    ///
    /// One table failing to load or refresh (an unsupported Delta feature, a
    /// transient store error) must not starve every other table of its
    /// refresh, so per-table failures are logged and the sweep continues; only
    /// a database-index load failure aborts it.
    pub fn refresh_from_store(&self) -> Result<bool> {
        let mut changed = false;

        let manifest = CatalogManifest::load(self.store.as_ref())?;

        for schema in &manifest.schemas {
            let mut index = self.tables_index.write().unwrap();
            if !index.contains_schema(&schema.name) {
                index.insert_schema(schema.name.clone());
                changed = true;
            }
        }

        for entry in manifest.tables() {
            let (name, id, location) = entry?;
            if self.tables_index.read().unwrap().contains_table(&name) {
                continue;
            }
            let table = match Self::load_table(
                &self.dispatcher,
                &self.store,
                &self.engine,
                id,
                location,
            ) {
                Ok(table) => table,
                Err(e) => {
                    tracing::warn!(table = %name, error = %e, "catalog refresh: loading table failed");
                    continue;
                }
            };
            let mut index = self.tables_index.write().unwrap();
            // An in-process CREATE TABLE may have published it since the read.
            if !index.contains_table(&name) {
                index.insert_table(name, table);
                changed = true;
            }
        }

        // Refresh each table on a clone outside the lock (footer fetches are
        // I/O), then publish the advanced copy back.
        for (name, mut table) in self.tables() {
            match table.refresh() {
                Ok(true) => changed |= self.publish_table(table),
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(table = %name, error = %e, "catalog refresh: refreshing table failed");
                }
            }
        }

        Ok(changed)
    }

    /// Publish a committed [`CatalogTable`] copy into the shared map, so its
    /// new version is visible to the next transaction immediately (rather than
    /// after the next background refresh). INSERT and compaction call this
    /// right after their commit. A copy at or behind the map's current version
    /// is dropped (`false`); a newer one replaces it.
    pub fn publish_table(&self, table: CatalogTable) -> bool {
        let mut index = self.tables_index.write().unwrap();
        match index.get_table_by_id(&table.id()) {
            Some(existing) if existing.version() >= table.version() => false,
            _ => {
                index.replace_table(table);
                true
            }
        }
    }

    /// In one Delta commit for table `id`, add every file in `added` and remove
    /// every path in `removed`. `data_change` is `true` for a logical change such
    /// as INSERT, or `false` for a compaction rearrangement that incremental log
    /// readers can skip.
    ///
    /// This is the sole in-process write path and uses the
    /// [commit-lock protocol](field@CatalogTable::commit_lock). A caller's
    /// [`CatalogTable`] copy is a read view, never a commit base.
    pub(crate) fn commit_to_table(
        &self,
        id: Uuid,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
        data_change: bool,
    ) -> Result<()> {
        // Look the table up twice on purpose: this first lookup takes nothing
        // but the lock, because a copy read before the lock would be the version
        // the previous holder is about to supersede. The index guard is a
        // temporary in this statement, so it is released at the semicolon,
        // before the blocking log write; that only type-checks because
        // `commit_lock` hands back an owned `Arc` rather than a reference.
        let commit_lock = self
            .tables_index
            .read()
            .unwrap()
            .get_table_by_id(&id)
            .ok_or_else(|| Error::TableNotFound(id.to_string()))?
            .commit_lock();
        let _committing = commit_lock.lock().unwrap();

        let mut table = self
            .table_handle_by_id(&id)
            .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
        table.commit_files(removed, added, data_change)?;
        self.publish_table(table);
        Ok(())
    }

    /// Persist one table creation staged by a completed footer-fetch dataflow:
    /// commit Delta version 0, record the table in the database index, and
    /// publish it into the live set so the next transaction binds it. Called
    /// from [`DeltaTransaction::commit`] on the blocking pool, never from a
    /// dispatch worker.
    ///
    /// The whole step holds the table-set write lock, which serializes
    /// in-process creates and their database-index read-modify-write; the name is
    /// re-checked under it as a race backstop.
    fn finalize_table_creation(&self, pending: PendingTableCreation) -> Result<()> {
        let PendingTableCreation {
            name,
            id,
            location,
            columns,
            partition_by,
            sort_by,
            if_not_exists,
            loaded,
        } = pending;
        let mut index = self.tables_index.write().unwrap();
        if index.contains_table(&name) {
            // A concurrent create won the race. With `IF NOT EXISTS` that is
            // still a success; the fetched footers are dropped.
            if if_not_exists {
                return Ok(());
            }
            return Err(Error::TableExists(name.to_string()));
        }
        let table = CatalogTable::create_new(
            id,
            location.clone(),
            loaded,
            columns,
            partition_by,
            sort_by,
            self.store.clone(),
            self.dispatcher.clone(),
            self.engine.clone(),
        )?;
        let mut manifest = CatalogManifest::load(self.store.as_ref())?;
        manifest.upsert_table(&name, id, location)?;
        manifest.store(self.store.as_ref())?;
        index.insert_table(name, table);
        Ok(())
    }

    /// Whether this datastore defines a schema named `schema`.
    pub fn contains_schema(&self, schema: &str) -> bool {
        self.tables_index.read().unwrap().contains_schema(schema)
    }

    /// Persist one schema creation staged by a transaction: record it in the
    /// database index and publish it into the live set so the next transaction
    /// resolves it. Called from [`DeltaTransaction::commit`] on the blocking
    /// pool, never from a dispatch worker.
    ///
    /// The whole step holds the index write lock, which serializes in-process
    /// creates (schemas and tables alike) and their read-modify-write of the
    /// shared manifest; the name is re-checked under it as a race backstop.
    fn finalize_schema_creation(&self, pending: PendingSchemaCreation) -> Result<()> {
        let PendingSchemaCreation {
            name,
            if_not_exists,
        } = pending;
        let mut index = self.tables_index.write().unwrap();
        if index.contains_schema(&name) {
            // A concurrent create won the race. With `IF NOT EXISTS` that is
            // still a success.
            if if_not_exists {
                return Ok(());
            }
            return Err(Error::SchemaExists(name));
        }
        let mut manifest = CatalogManifest::load(self.store.as_ref())?;
        manifest.add_schema(name.clone());
        manifest.store(self.store.as_ref())?;
        index.insert_schema(name);
        Ok(())
    }

    /// The worker pool this datastore fetches footers on, shared with callers
    /// (the compacter) that drive their own dataflows over the same tables.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// A clone of the named table's current state, for a caller that reads it
    /// (a compaction candidate scan, introspection) or refreshes it to the
    /// latest committed version. A copy drifts as soon as anything commits, and
    /// it is never a commit base: writes go through
    /// [`commit_to_table`](Self::commit_to_table). `None` if no such table
    /// exists.
    pub fn table_handle(&self, name: &SchemaQualifiedTableName) -> Option<CatalogTable> {
        self.tables_index
            .read()
            .unwrap()
            .get_table_by_name(name)
            .cloned()
    }

    /// Whether a table named `name` exists in the datastore (a cheap membership
    /// check, no clone). Used to fail fast when a table a writer targets
    /// hasn't been created.
    pub fn contains_table(&self, name: &SchemaQualifiedTableName) -> bool {
        self.tables_index.read().unwrap().contains_table(name)
    }

    /// A snapshot clone of every table the catalog currently holds — for a sweep
    /// (e.g. the compacter) that refreshes and evolves each one independently.
    pub fn tables(&self) -> Vec<(SchemaQualifiedTableName, CatalogTable)> {
        self.tables_index
            .read()
            .unwrap()
            .named_tables()
            .map(|(name, table)| (name, table.clone()))
            .collect()
    }

    /// A clone of the table with identity `id`, or `None` if it's gone (dropped,
    /// or a stale reference outliving the table). This is how a commit resolves
    /// the *live* table its files belong to, regardless of any concurrent rename.
    pub fn table_handle_by_id(&self, id: &Uuid) -> Option<CatalogTable> {
        self.tables_index
            .read()
            .unwrap()
            .get_table_by_id(id)
            .cloned()
    }

    /// A human-readable description of where this database is rooted (local
    /// directory or object-store bucket/prefix) - for introspection. BoundTable
    /// locations are relative to this root.
    pub fn store_description(&self) -> String {
        self.store.describe()
    }

    /// BoundTable `name`'s current committed files — refreshing to the latest version
    /// first, so a commit by INSERT/compaction (in this process or another) is
    /// reflected. `None` if no such table exists.
    pub fn table_files(&self, name: &SchemaQualifiedTableName) -> Option<Vec<FileRef>> {
        let mut table = self.table_handle(name)?;
        let _ = table.refresh();
        Some(table.file_refs())
    }

    /// BoundTable `name`'s physical files as transport-neutral `(path, size)` pairs in
    /// stable metadata (manifest) order, for introspection. `None` if no such
    /// table exists. Refreshes to the latest committed version first. Sizes come
    /// from the files' `FileRef`s, correlated by path (the two orderings differ,
    /// so a map lookup rather than a zip).
    pub fn table_data_files(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> Result<Option<Vec<DataFileInfo>>> {
        let Some(mut table) = self.table_handle(name) else {
            return Ok(None);
        };
        table.refresh()?;
        let sizes: HashMap<String, u64> = table
            .file_refs()
            .into_iter()
            .map(|file| (file.path.as_str().to_string(), file.size))
            .collect();
        let files = table
            .file_partitions()
            .into_iter()
            .map(|(path, _)| {
                let path = path.as_str().to_string();
                let size = sizes.get(&path).copied().unwrap_or(0);
                DataFileInfo { path, size }
            })
            .collect();
        Ok(Some(files))
    }
}

/// One physical data file exposed by datastore introspection: its stable path
/// and byte size, transport-neutral so the HTTP layer needs no knowledge of the
/// backing store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataFileInfo {
    pub path: String,
    pub size: u64,
}

impl DeltaTransaction {
    /// Whether this transaction's frozen snapshot holds `schema`. DuckDB
    /// resolves every name through a schema lookup, so this is answered from the
    /// snapshot alone and takes no lock. A schema this transaction has staged is
    /// deliberately not visible: it does not exist until commit creates it.
    fn contains_schema(&self, schema: &str) -> bool {
        self.snapshot.contains_schema(schema)
    }

    /// Resolve a `CREATE TABLE` against this datastore: check the schema exists
    /// and the name is still free, parse the layout options, and locate the
    /// Parquet files the table adopts. Returns a [`DeltaTableCreation`] whose
    /// `compile` builds the footer-fetch-and-stage dataflow. This part runs on
    /// the coordinator; compiling needs the pool.
    ///
    /// The table's own storage is always a directory of its own under the
    /// database root, named after its identity. `WITH (with_pre_existing_parquets = '…')`
    /// only points at data to adopt; nothing is written to that directory. A
    /// statement that names none yields an empty table.
    fn bind_create(&self, request: CreateTableRequest) -> Result<DeltaTableCreation> {
        let name = request.schema_qualified_name();

        Self::reject_unknown_options(&request)?;

        // A table can only be created in a schema that exists: `CREATE TABLE`
        // never brings its schema into being as a side effect. A schema this
        // transaction staged counts, since commit creates those before any
        // table; without that, the transaction could never build a schema and
        // populate it.
        if !self.contains_schema(&name.schema) {
            return Err(Error::SchemaNotFound(name.schema));
        }

        // Reject a duplicate up front, unless `IF NOT EXISTS` makes an existing
        // table a success. This checks the transaction's frozen snapshot; the
        // datastore re-checks the name under its write lock at commit as the real
        // race backstop, honouring `IF NOT EXISTS` there too.
        if !request.if_not_exists && self.snapshot.contains_table(&name) {
            return Err(Error::TableExists(name.to_string()));
        }

        // The table's identity is minted here, before anything is written, so
        // it can name the table's storage and still be the identity the Delta
        // log records for it.
        let id = Uuid::new_v4();
        let location = ObjectPath::new(id.to_string());
        let partition_by = Self::parse_spec_columns(&request, PARTITION_BY_OPTION)?;
        let sort_by = Self::parse_spec_columns(&request, SORT_BY_OPTION)?;
        let files = self.adopted_data_files(&request)?;

        Ok(DeltaTableCreation {
            pending_table_creations: self.pending_table_creations.clone(),
            files,
            name,
            id,
            columns: request.columns,
            location,
            partition_by,
            sort_by,
            if_not_exists: request.if_not_exists,
        })
    }

    /// Resolve a `CREATE SCHEMA` against this datastore and stage it on the
    /// transaction. A schema is a pure naming construct here: it owns no store
    /// state of its own, so there is nothing to locate or load, and creating it
    /// is just recording the name. The transaction's blocking commit writes it to
    /// the database manifest and publishes it.
    ///
    /// Rejects a duplicate up front, unless `IF NOT EXISTS` makes an existing
    /// schema a success. This checks the transaction's frozen snapshot; the
    /// datastore re-checks under its write lock at commit as the real race
    /// backstop, honouring `IF NOT EXISTS` there too.
    fn bind_schema_creation(&self, request: CreateSchemaRequest) -> Result<DeltaSchemaCreation> {
        if !request.if_not_exists && self.contains_schema(&request.name) {
            return Err(Error::SchemaExists(request.name));
        }
        Ok(DeltaSchemaCreation {
            pending_schema_creations: self.pending_schema_creations.clone(),
            creation: PendingSchemaCreation {
                name: request.name,
                if_not_exists: request.if_not_exists,
            },
        })
    }

    /// Reject an option this datastore does not implement. A `CREATE TABLE` that
    /// asks for something unknown is not carried out by ignoring it: the table
    /// that appears is not the one the statement described, and the caller has no
    /// way to tell. This is what turns a retired option name, or a typo, into an
    /// error at the statement rather than a surprise at the first query.
    fn reject_unknown_options(request: &CreateTableRequest) -> Result<()> {
        let Some(option) = request
            .options
            .keys()
            .find(|option| !TABLE_OPTIONS.contains(&option.as_str()))
        else {
            return Ok(());
        };
        Err(Error::UnknownOption {
            option: option.clone(),
            known: TABLE_OPTIONS.map(|known| format!("`{known}`")).join(", "),
        })
    }

    /// The data files a new table adopts: every Parquet object directly under
    /// the `with_pre_existing_parquets` directory, located for reading and keeping its
    /// `FileRef` identity so each footer's row groups land on the right
    /// `TableFile`. No such option, or a directory holding no Parquet, gives an
    /// empty table that fills up as it is written to.
    ///
    /// Each file is recorded by its **store-root-absolute** key, not by a name
    /// under the new table's own location, since it stays where the user put it:
    /// the table writes its Delta log and every file it goes on to write under
    /// its own directory, and leaves the adopted directory untouched.
    ///
    /// The option is always a plain path, never a URL (no scheme): the data
    /// lives in the database's own storage, so the path carries none. A
    /// *relative* path is read under the database root. An *absolute* path is
    /// taken from the root of the database's storage medium: on a local
    /// database, a directory on the server's filesystem; on a remote database, a
    /// key from the **bucket root** (ignoring the prefix the database was opened
    /// at).
    fn adopted_data_files(&self, request: &CreateTableRequest) -> Result<Vec<DataFile>> {
        let Some(path) = request.options.get(PRE_EXISTING_PARQUETS_OPTION) else {
            return Ok(Vec::new());
        };
        if path.contains("://") {
            return Err(Error::OptionPathWithScheme {
                option: PRE_EXISTING_PARQUETS_OPTION.to_string(),
                path: path.clone(),
            });
        }
        let directory = ObjectPath::new(path.clone());
        let store = &self.datastore.store;
        self.list_file_refs(&directory)?
            .into_iter()
            .map(|listed| {
                let path = store.absolute_key(&directory.join(listed.path.as_str()))?;
                let file = FileRef {
                    path,
                    size: listed.size,
                };
                // An absolute key reads the same from any table location, so
                // which one it is resolved against does not matter here.
                file.into_data_file(store.as_ref(), &directory)
            })
            .collect::<store::Result<Vec<DataFile>>>()
            .map_err(Error::from)
    }

    /// Parse a comma-separated column-list option (`partition_by` / `sort_by`)
    /// into an ordered `Vec<String>`, validating each name is a declared column.
    /// Absent or empty → no columns.
    fn parse_spec_columns(request: &CreateTableRequest, option: &str) -> Result<Vec<String>> {
        let Some(raw) = request.options.get(option) else {
            return Ok(Vec::new());
        };
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|name| {
                if request.columns.iter().any(|c| c.name == name) {
                    Ok(name.to_string())
                } else {
                    Err(Error::UnknownSpecColumn {
                        option: option.to_string(),
                        column: name.to_string(),
                    })
                }
            })
            .collect()
    }

    /// List the Parquet data files directly under `directory` through the store,
    /// each named relative to it. A missing directory yields no files. Used where
    /// the *listing* is the source of truth: `CREATE TABLE`.
    fn list_file_refs(&self, directory: &ObjectPath) -> Result<Vec<FileRef>> {
        Ok(self
            .datastore
            .store
            .list(directory)?
            .into_iter()
            .map(|object| object.file)
            .filter(|file| file.path.as_str().ends_with(".parquet"))
            .collect())
    }
}

/// Commit the schema creations drained from a transaction. Each one records
/// itself in the database manifest and publishes into the live schema set.
/// Blocking store I/O, so [`DeltaTransaction::commit`] runs this on the blocking
/// pool.
fn commit_schema_creations(
    datastore: &DeltaDatastore,
    pending_schema_creations: Vec<PendingSchemaCreation>,
) -> CatalogResult<()> {
    for pending in pending_schema_creations {
        datastore.finalize_schema_creation(pending)?;
    }
    Ok(())
}

/// Commit the table creations drained from a transaction. Each one initializes
/// its Delta log, registers itself in the database manifest, and publishes into
/// the live table set. Blocking store I/O, so [`DeltaTransaction::commit`] runs
/// this on the blocking pool.
fn commit_table_creations(
    datastore: &DeltaDatastore,
    pending_table_creations: Vec<PendingTableCreation>,
) -> CatalogResult<()> {
    for pending in pending_table_creations {
        datastore.finalize_table_creation(pending)?;
    }
    Ok(())
}

/// Commit the files an INSERT drained from its transaction: group them by the
/// table they were written for and append each group through
/// [`DeltaDatastore::commit_to_table`]. The transaction's frozen snapshot is not
/// the commit base -- it is a read view, and a table it froze versions ago would
/// lose the compare-and-swap against every INSERT that committed since.
/// Blocking store I/O, so [`DeltaTransaction::commit`] runs it on the blocking
/// pool.
fn commit_uploaded_files(
    datastore: &DeltaDatastore,
    drained: Vec<insert_sink::UploadedFile>,
) -> CatalogResult<()> {
    let mut files_by_table = HashMap::<Uuid, Vec<TableFile>>::new();
    for uploaded in drained {
        let insert_sink::UploadedFile {
            table_id,
            file,
            partition,
            row_groups,
        } = uploaded;
        let stats = Some(crate::parquet::aggregate_file_stats(&row_groups));
        let entry = DeltaFileEntry {
            file,
            partition,
            stats,
        };
        files_by_table
            .entry(table_id)
            .or_default()
            .push(TableFile::new(entry, row_groups));
    }

    for (table_id, files) in files_by_table {
        datastore.commit_to_table(table_id, &[], files, true)?;
    }
    Ok(())
}

fn drain_injector<T>(injector: &Injector<T>) -> Vec<T> {
    let mut items = Vec::new();
    loop {
        match injector.steal() {
            Steal::Success(item) => items.push(item),
            Steal::Retry => continue,
            Steal::Empty => return items,
        }
    }
}

#[async_trait]
impl Datastore for DeltaDatastore {
    /// Open a transaction: freeze the table set as it stands right now. Every
    /// table the transaction binds resolves from that frozen
    /// [`DeltaSnapshot`] (pure in-memory, no I/O), so one query reads one
    /// consistent version of every table regardless of concurrent refreshes or
    /// commits. [`DatastoreTransaction::commit`] persists staged table creations
    /// and files injected by INSERT; rollback or dropping the transaction
    /// discards those pending sets.
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction> {
        DeltaDatastore::begin_transaction(self)
    }

    /// Spawn this datastore's configured maintenance onto the ambient runtime: a
    /// periodic refresh sweep, and (when configured) a compaction loop. Each task
    /// holds an `Arc` clone of the datastore and runs until [`abort`](Self::abort)
    /// stops it. Their abort handles are kept so shutdown can stop them before the
    /// worker pool is torn down.
    fn start(self: Arc<Self>) {
        let Some(maintenance) = self.maintenance.clone() else {
            return;
        };
        let mut tasks = self.maintenance_tasks.lock().unwrap();

        let refresh_datastore = Arc::clone(&self);
        let refresh_interval = maintenance.refresh_interval;
        let refresh = tokio::spawn(async move {
            let mut tick = tokio::time::interval(refresh_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // The table set was just loaded at open, so skip the interval's
            // immediate first tick and refresh one full interval from now.
            tick.tick().await;
            loop {
                tick.tick().await;
                // The refresh drives footer-fetch dataflows and blocking store
                // reads, so it runs off the reactor.
                let datastore = Arc::clone(&refresh_datastore);
                match tokio::task::spawn_blocking(move || datastore.refresh_from_store()).await {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => tracing::warn!(error = %e, "catalog refresh failed"),
                    Err(e) => tracing::warn!(error = %e, "catalog refresh panicked"),
                }
            }
        });
        tasks.push(refresh.abort_handle());

        if let Some(compaction) = maintenance.compaction {
            let compacter = Arc::new(crate::compact::Compacter::new(
                compaction.target_bytes,
                compaction.min_files,
                compaction.poll_interval,
                Arc::clone(&self),
            ));
            let compaction_task = tokio::spawn(compacter.run());
            tasks.push(compaction_task.abort_handle());
        }

        if let Some(vacuum) = maintenance.vacuum {
            let vacuumer = Arc::new(crate::vacuum::Vacuumer::new(
                vacuum.poll_interval,
                Arc::clone(&self),
            ));
            let vacuum_task = tokio::spawn(vacuumer.run());
            tasks.push(vacuum_task.abort_handle());
        }
    }

    fn abort(&self) {
        for handle in self.maintenance_tasks.lock().unwrap().drain(..) {
            handle.abort();
        }
    }

    /// One sweep with the default thresholds, or sweep-to-fixpoint under
    /// `final_sweep`: a merge changes the file list, so a sweep can leave a
    /// tail; progress is judged by the table's committed log version, and the
    /// loop stops at the first sweep that advances nothing.
    async fn compact(
        self: Arc<Self>,
        table: &SchemaQualifiedTableName,
        final_sweep: bool,
    ) -> CatalogResult<u64> {
        let version_of = |datastore: &DeltaDatastore| {
            datastore
                .tables()
                .into_iter()
                .find(|(name, _)| name == table)
                .map(|(_, found)| found.version())
        };
        if version_of(&self).is_none() {
            return Err(CatalogError::Other(
                format!("COMPACT: no table named `{table}`").into(),
            ));
        }
        let compacter = crate::compact::Compacter::new(
            crate::compact::DEFAULT_COMPACT_BYTES,
            crate::compact::DEFAULT_MIN_FILES_TO_MERGE,
            // The poll interval drives the background loop, which a manual
            // sweep never enters.
            std::time::Duration::from_secs(1),
            Arc::clone(&self),
        );
        let mut sweeps = 0;
        loop {
            let before = version_of(&self);
            compacter.sweep_table(table).await;
            sweeps += 1;
            if !final_sweep || version_of(&self) == before {
                return Ok(sweeps);
            }
        }
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}

/// One transaction's frozen view of the catalog: every table at the version it
/// held when the transaction began, with all row-group metadata already
/// materialized (the background refresh keeps the master set fully fetched).
/// Everything a query does against it (binding, scan-view construction, late
/// materialize) is pure in-memory.
pub struct DeltaSnapshot {
    /// The datastore's schemas and tables as they stood when the transaction
    /// began, so a query resolves both from one frozen view.
    index: DatastoreIndex,
}

impl std::fmt::Debug for DeltaSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaSnapshot")
            .field("tables", &self.index.table_names().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl DeltaSnapshot {
    /// A clone of the table named `name`, or `None` if this snapshot has no such
    /// table. The frozen copy a [`TableBinding`] captures at bind time, so its
    /// compile resolves the table's file set without any transaction handle.
    pub(super) fn catalog_table_by_name(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> Option<CatalogTable> {
        self.index.get_table_by_name(name).cloned()
    }

    /// Whether this frozen snapshot defines a schema named `schema`.
    fn contains_schema(&self, schema: &str) -> bool {
        self.index.contains_schema(schema)
    }

    /// The cache revision of `name` in this frozen snapshot. The Delta metadata
    /// UUID distinguishes table incarnations; the log version distinguishes
    /// every committed snapshot of one incarnation.
    fn table_revision(&self, name: &SchemaQualifiedTableName) -> Option<TableRevision> {
        let table = self.index.get_table_by_name(name)?;
        Some(TableRevision {
            identity: table.id().to_string(),
            version: table.version(),
        })
    }

    /// Whether this snapshot holds a table named `name`; the up-front duplicate
    /// check for `CREATE TABLE`.
    pub(super) fn contains_table(&self, name: &SchemaQualifiedTableName) -> bool {
        self.index.contains_table(name)
    }
}

/// One schema creation resolved by a transaction and waiting for it to commit.
/// A schema has no content to load, so it is staged as soon as it resolves.
#[derive(Clone)]
struct PendingSchemaCreation {
    name: String,
    /// Whether the statement used `IF NOT EXISTS`, so the commit treats a schema
    /// that appeared since resolution as a success rather than an error.
    if_not_exists: bool,
}

/// One table creation whose footer metadata has been loaded successfully and is
/// waiting for its transaction to commit it.
struct PendingTableCreation {
    name: SchemaQualifiedTableName,
    id: Uuid,
    location: ObjectPath,
    columns: Vec<planner::catalog::Column>,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    if_not_exists: bool,
    loaded: Vec<crate::parquet::FileRowGroups>,
}

/// The [`DatastoreTransaction`] a [`DeltaDatastore`] opens: one query's frozen
/// [`DeltaSnapshot`] plus injectors of uploaded files and completed table
/// creations awaiting commit.
pub struct DeltaTransaction {
    pub(super) snapshot: Arc<DeltaSnapshot>,
    pub(super) uploaded_files: Arc<Injector<insert_sink::UploadedFile>>,
    pending_table_creations: Arc<Injector<PendingTableCreation>>,
    pending_schema_creations: Arc<Injector<PendingSchemaCreation>>,
    /// The datastore this transaction reads and writes back to. CREATE commits
    /// publish through it so DDL is visible to the next transaction without
    /// waiting for a refresh.
    datastore: Arc<DeltaDatastore>,
}

impl std::fmt::Debug for DeltaTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaTransaction")
            .field("snapshot", &self.snapshot)
            .finish_non_exhaustive()
    }
}

impl DeltaTransaction {
    /// [`DatastoreTransaction::bind_table`], typed: the concrete
    /// [`TableBinding`] instead of the trait object. This is the single
    /// resolution path; the trait impl below only boxes its result (the planner
    /// needs a `Box<dyn BoundTable>`, while a concrete caller wants the
    /// `TableBinding`, and one function cannot return both). The binding captures
    /// the snapshot's copy of the table and shares this transaction's
    /// uploaded-files injector, so it is self-contained.
    pub fn table(&self, datastore: &str, name: &SchemaQualifiedTableName) -> Option<TableBinding> {
        Some(TableBinding::new(
            planner::catalog::TableReference {
                datastore: datastore.to_string(),
                schema: name.schema.clone(),
                table: name.table.clone(),
            },
            self.snapshot.catalog_table_by_name(name)?,
            self.uploaded_files.clone(),
        ))
    }
}

#[async_trait]
impl DatastoreTransaction for DeltaTransaction {
    fn does_schema_exist(&self, schema: &str) -> bool {
        self.contains_schema(schema)
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> Option<Box<dyn BoundTable>> {
        Some(Box::new(DeltaTransaction::table(self, datastore, name)?))
    }

    fn table_revision(&self, name: &SchemaQualifiedTableName) -> Option<TableRevision> {
        self.snapshot.table_revision(name)
    }

    fn bind_create_table(
        &self,
        request: CreateTableRequest,
    ) -> CatalogResult<Box<dyn TableCreation>> {
        Ok(Box::new(self.bind_create(request)?))
    }

    fn bind_create_schema(
        &self,
        request: CreateSchemaRequest,
    ) -> CatalogResult<Box<dyn SchemaCreation>> {
        Ok(Box::new(self.bind_schema_creation(request)?))
    }

    async fn commit(&self) -> CatalogResult<()> {
        // A read-only transaction staged nothing: an in-memory no-op, run inline
        // rather than pay a blocking-pool round trip. Only a commit that writes
        // durable state hops off the runtime.
        if self.uploaded_files.is_empty()
            && self.pending_table_creations.is_empty()
            && self.pending_schema_creations.is_empty()
        {
            return Ok(());
        }
        let datastore = self.datastore.clone();
        let pending_schema_creations = drain_injector(&self.pending_schema_creations);
        let pending_table_creations = drain_injector(&self.pending_table_creations);
        let uploaded_files = drain_injector(&self.uploaded_files);
        tokio::task::spawn_blocking(move || {
            // Schemas first: a table is registered into its schema, so the
            // schema has to be in the manifest before any table write looks for
            // it.
            commit_schema_creations(&datastore, pending_schema_creations)?;
            commit_table_creations(&datastore, pending_table_creations)?;
            commit_uploaded_files(&datastore, uploaded_files)
        })
        .await
        .map_err(|e| CatalogError::Other(format!("commit thread panicked: {e}").into()))?
    }

    fn rollback(&self) {
        drain_injector(&self.pending_schema_creations);
        drain_injector(&self.pending_table_creations);
        drain_injector(&self.uploaded_files);
    }
}

/// A resolved `CREATE SCHEMA` for a [`DeltaDatastore`]: the validated creation
/// plus the transaction-owned staging queue it will land in. Compiling it builds
/// a dataflow that stages the creation and emits no rows, so the schema appears
/// on the transaction only once that dataflow runs.
struct DeltaSchemaCreation {
    pending_schema_creations: Arc<Injector<PendingSchemaCreation>>,
    creation: PendingSchemaCreation,
}

impl SchemaCreation for DeltaSchemaCreation {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        // One nullary per worker, but only the first carries the creation; the
        // rest no-op. Handing it out here rather than racing for it at run time
        // is how the table-creation sink picks its staging worker too.
        let mut creation = Some(self.creation.clone());
        let factories: Vec<_> = (0..dispatcher.worker_count())
            .map(|_| {
                let staged = creation.take();
                let pending_schema_creations = self.pending_schema_creations.clone();
                OneShotNullaryFactory::new(move || {
                    if let Some(creation) = staged {
                        pending_schema_creations.push(creation);
                    }
                    None
                })
            })
            .collect();
        Ok(RecordBatchOperatorSpec::from_nullary(dispatcher, factories))
    }
}

/// A resolved `CREATE TABLE` for a [`DeltaDatastore`]: the validated request, a
/// transaction-owned staging injector, and the located data files, captured at
/// resolution by [`DeltaTransaction::bind_create`]. Compiling it builds the
/// dataflow that fetches the files' footers and stages the completed creation
/// for transaction commit.
struct DeltaTableCreation {
    pending_table_creations: Arc<Injector<PendingTableCreation>>,
    files: Vec<DataFile>,
    name: SchemaQualifiedTableName,
    id: Uuid,
    columns: Vec<planner::catalog::Column>,
    location: ObjectPath,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    /// Whether the statement used `IF NOT EXISTS`, so the commit treats a table
    /// that appeared since resolution as a success rather than an error.
    if_not_exists: bool,
}

impl TableCreation for DeltaTableCreation {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        // The terminal worker only stages the completed creation. Durable writes
        // and live publication happen later in the transaction's blocking commit.
        let pending_table_creations = self.pending_table_creations.clone();
        let name = self.name.clone();
        let id = self.id;
        let location = self.location.clone();
        let columns = self.columns.clone();
        let partition_by = self.partition_by.clone();
        let sort_by = self.sort_by.clone();
        let if_not_exists = self.if_not_exists;
        let declared_columns: Arc<[planner::catalog::Column]> = columns.clone().into();
        Ok(crate::parquet::create_load_and_stage_spec(
            dispatcher,
            &self.files,
            declared_columns,
            move |loaded: Vec<crate::parquet::FileRowGroups>| {
                pending_table_creations.push(PendingTableCreation {
                    name,
                    id,
                    location,
                    columns,
                    partition_by,
                    sort_by,
                    if_not_exists,
                    loaded,
                });
            },
        ))
    }
}
