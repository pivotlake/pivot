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
//! - the [`manifest`]: which tables exist (name and location);
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
mod metadata_function;
mod table;

pub use binding::TableBinding;

use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use uuid::Uuid;

use crate::manifest::{
    self, CatalogManifest, CatalogManifestTableEntry, ManifestEntry, TableManifest,
};
use crate::parquet::ParquetTableError;
use crate::store::{self, DataFile, FileRef, LocalStore, ObjectPath, ObjectStore, open_store};
use async_trait::async_trait;
use crossbeam_deque::{Injector, Steal};
use datastore::{Datastore, DatastoreTransaction};
use dispatch::{DataFlowDispatcher, DataFlowError, RecordBatchOperatorSpec};
use metadata_function::MetadataTableFunction;
use planner::TableFunction;
use planner::catalog::{
    BoundTable, CreateTableRequest, Error as CatalogError, Result as CatalogResult, TableCreation,
};
pub use table::CatalogTable;
pub use table::TableFile;
use thiserror::Error as ThisError;

const PATH_OPTION: &str = "path";
/// `WITH (partition_by = 'a, b')` — ordered, comma-separated partition columns.
const PARTITION_BY_OPTION: &str = "partition_by";
/// `WITH (sort_by = 'a, b')` — ordered, comma-separated sort columns.
const SORT_BY_OPTION: &str = "sort_by";

#[derive(Debug, ThisError)]
pub enum Error {
    #[error(
        "table path `{0}` must be a plain path, not a URL: a table's storage is the database's, so the path carries no scheme"
    )]
    TablePathWithScheme(String),
    #[error("`{option}` column `{column}` is not a declared column of the table")]
    UnknownSpecColumn { option: String, column: String },
    #[error(transparent)]
    ParquetTable(#[from] ParquetTableError),
    #[error("table `{0}` already exists")]
    TableExists(String),
    #[error("table `{0}` does not exist")]
    TableNotFound(String),
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
        "table `{table}`: file `{file}` has no loaded row-group metadata; the copy was not synced to its manifest"
    )]
    FooterNotLoaded { table: String, file: String },
    #[error("table `{table}` commit conflict: input file `{file}` is no longer active")]
    CommitConflict { table: String, file: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// The catalog's table set. A table's durable identity is its [`Uuid`], so the
/// tables are owned by `by_uuid`; `name_to_uuid` is a lightweight lookup index
/// on top. Keeping them separate means only one map owns each [`CatalogTable`]
/// (no aliasing), a rename touches only the name index, and a commit/publish
/// touches only `by_uuid` - the two never fight over the same value.
#[derive(Clone, Default)]
struct TableIndex {
    by_uuid: HashMap<Uuid, CatalogTable>,
    name_to_uuid: HashMap<String, Uuid>,
}

impl TableIndex {
    /// Insert or replace a table under both its identity and its current name.
    fn insert(&mut self, table: CatalogTable) {
        self.name_to_uuid
            .insert(table.name().to_string(), table.id());
        self.by_uuid.insert(table.id(), table);
    }

    fn get_by_name(&self, name: &str) -> Option<&CatalogTable> {
        self.by_uuid.get(self.name_to_uuid.get(name)?)
    }

    fn get_by_id(&self, id: &Uuid) -> Option<&CatalogTable> {
        self.by_uuid.get(id)
    }

    fn contains_name(&self, name: &str) -> bool {
        self.name_to_uuid.contains_key(name)
    }

    fn values(&self) -> impl Iterator<Item = &CatalogTable> {
        self.by_uuid.values()
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.name_to_uuid.keys().map(String::as_str)
    }
}

/// A Delta datastore's concurrent table index, keyed by name.
///
/// `CREATE TABLE` compiles to a single dataflow that reads every data file's
/// footer **once** (in parallel over the worker pool) and, at its terminal
/// stage, commits the table (its manifest + the database index) and publishes
/// the entry in this shared map. After that, a table evolves by manifest commits
/// on a [`CatalogTable`] *copy*: a writer takes one with
/// [`table_handle`](Self::table_handle) and appends its uploaded files or swaps
/// them (compaction), which CAS a new version into the store. Copies drift; every query resolve refreshes its copy
/// to the latest committed version, so a commit by another process (or this one)
/// becomes visible to the next query.
///
/// Cloneable (every field is an `Arc`, a `String`, or the shared dispatcher
/// handle), so a commit that writes can hand a clone to the blocking pool.
#[derive(Clone)]
pub struct DeltaDatastore {
    /// This datastore's name: the key it is registered under in the metastore
    /// and the database it is attached as in DuckDB. Stamped onto every
    /// [`TableBinding`] this datastore's transactions resolve, so a query that
    /// spans several datastores routes each table back to the one that produced
    /// it.
    name: String,
    /// The in-memory table set. The lock guards the *set* (add on `CREATE`,
    /// swap-in on a resolve's refresh); each [`CatalogTable`] is itself a
    /// lock-free value that callers clone out and evolve independently.
    tables: Arc<RwLock<TableIndex>>,
    /// The database's object store: the table index, Delta logs, and tables'
    /// Parquet data. The datastore reads and writes a table's data through this
    /// one explicitly configured store: relative locations live under the
    /// database root, an absolute location at the store's own root (the
    /// filesystem root, or the bucket root).
    store: Arc<dyn ObjectStore>,

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
            .field("name", &self.name)
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
        Self::from_store(
            "default".to_string(),
            Arc::new(LocalStore::new(root)),
            dispatcher,
            None,
        )
    }

    /// Open a persisted database rooted at `uri`, a local directory (or
    /// `file://…`), or a remote `s3://…` object store, reloading every
    /// table the manifest records at its latest version: read each table's
    /// manifest (schema + file list) and fetch its files' footers, building the
    /// in-memory [`CatalogTable`]. A database with no manifest yet opens empty.
    /// No background maintenance runs (see [`from_store`](Self::from_store)).
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Arc<Self>> {
        let store: Arc<dyn ObjectStore> = open_store(uri)?.into();
        Self::from_store("default".to_string(), store, dispatcher, None)
    }

    /// Open a persisted database over an already-built object store, under the
    /// datastore `name`. This is the seam a metastore uses: it constructs the
    /// store (local dir / S3, with whatever credentials it holds) and hands
    /// it in, rather than having the datastore re-derive one from a URI. Reloads
    /// every table the manifest records, exactly as [`open`](Self::open) does.
    pub fn from_store(
        name: String,
        store: Arc<dyn ObjectStore>,
        dispatcher: &DataFlowDispatcher,
        maintenance: Option<crate::MaintenanceConfig>,
    ) -> Result<Arc<Self>> {
        let manifest = CatalogManifest::load(store.as_ref())?;

        let mut tables = TableIndex::default();
        for entry in &manifest.tables {
            let table = Self::load_table(dispatcher, &store, entry)?;
            tables.insert(table);
        }

        Ok(Arc::new(Self {
            name,
            tables: Arc::new(RwLock::new(tables)),
            store,
            dispatcher: dispatcher.clone(),
            maintenance,
            maintenance_tasks: Arc::new(Mutex::new(Vec::new())),
        }))
    }

    /// This datastore's name, the database it is attached as in DuckDB and the
    /// key it is registered under in the metastore.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build the in-memory [`CatalogTable`] for one persisted table: read its
    /// manifest (erroring if the catalog points at a table that has none), then
    /// locate its committed files under the entry's location and fetch their
    /// footers over the pool.
    fn load_table(
        dispatcher: &DataFlowDispatcher,
        store: &Arc<dyn ObjectStore>,
        entry: &CatalogManifestTableEntry,
    ) -> Result<CatalogTable> {
        let delta_uri = crate::delta::table_uri(&store.location_uri(), &entry.location)?;
        let state = crate::delta::load_table(&delta_uri)?;
        let id = state.id;
        let manifest = TableManifest {
            version: state.version,
            columns: state.columns,
            partition_by: state.partition_by,
            sort_by: state.sort_by,
            entries: state.entries,
        };
        let files = manifest
            .entries
            .iter()
            .map(|f| {
                f.file
                    .clone()
                    .into_data_file(store.as_ref(), &entry.location)
            })
            .collect::<store::Result<Vec<DataFile>>>()?;
        let declared_columns: Arc<[planner::catalog::Column]> = manifest.columns.clone().into();
        let table_files = crate::parquet::load_table_files(dispatcher, &files, declared_columns)?;
        Ok(CatalogTable::new(
            entry.name.clone(),
            id,
            entry.location.clone(),
            manifest,
            table_files,
            store.clone(),
            dispatcher.clone(),
            delta_uri,
        ))
    }

    /// Open a transaction, typed: freeze the current table set into a
    /// [`DeltaSnapshot`] and create the injector that receives completed
    /// INSERT files. Cheap: clones the map (the row-group metadata inside is
    /// `Arc`-shared), no I/O. The [`Datastore::begin_transaction`] trait impl
    /// delegates here.
    pub fn begin_transaction(&self) -> Arc<DeltaTransaction> {
        Arc::new(DeltaTransaction {
            snapshot: Arc::new(DeltaSnapshot {
                datastore_name: self.name.clone(),
                tables: self.tables.read().unwrap().clone(),
            }),
            uploaded_files: Arc::new(Injector::new()),
            tables: self.tables.clone(),
            store: self.store.clone(),
        })
    }

    /// Bring the in-memory table set up to date with the store: pick up tables
    /// another process registered in the database index, and advance every
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

        let index = CatalogManifest::load(self.store.as_ref())?;
        for entry in &index.tables {
            if self.tables.read().unwrap().contains_name(&entry.name) {
                continue;
            }
            let table = match Self::load_table(&self.dispatcher, &self.store, entry) {
                Ok(table) => table,
                Err(e) => {
                    tracing::warn!(table = entry.name, error = %e, "catalog refresh: loading table failed");
                    continue;
                }
            };
            let mut map = self.tables.write().unwrap();
            // An in-process CREATE TABLE may have published it since the read.
            if !map.contains_name(&entry.name) {
                map.insert(table);
                changed = true;
            }
        }

        // Refresh each table on a clone outside the lock (footer fetches are
        // I/O), then publish the advanced copy back.
        for mut table in self.tables() {
            match table.refresh() {
                Ok(true) => changed |= self.publish_table(table),
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(table = table.name(), error = %e, "catalog refresh: refreshing table failed");
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
        let mut map = self.tables.write().unwrap();
        match map.get_by_id(&table.id()) {
            Some(existing) if existing.version() >= table.version() => false,
            _ => {
                map.insert(table);
                true
            }
        }
    }

    /// The worker pool this datastore fetches footers on, shared with callers
    /// (the compacter) that drive their own dataflows over the same tables.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// A clone of the named table's current state for a writer (INSERT,
    /// compaction) to evolve — appending its uploaded files or swapping them
    /// (compaction). Those commit a new version by CAS to the shared store, so
    /// this datastore's own copy may
    /// lag until its next resolve refreshes it (which is fine, the store is the
    /// source of truth). `None` if no such table exists.
    pub fn table_handle(&self, name: &str) -> Option<CatalogTable> {
        self.tables.read().unwrap().get_by_name(name).cloned()
    }

    /// Whether a table named `name` exists in the datastore (a cheap membership
    /// check, no clone). Used to fail fast when a table a writer targets
    /// hasn't been created.
    pub fn contains_table(&self, name: &str) -> bool {
        self.tables.read().unwrap().contains_name(name)
    }

    /// A snapshot clone of every table the catalog currently holds — for a sweep
    /// (e.g. the compacter) that refreshes and evolves each one independently.
    pub fn tables(&self) -> Vec<CatalogTable> {
        self.tables.read().unwrap().values().cloned().collect()
    }

    /// A clone of the table with identity `id`, or `None` if it's gone (dropped,
    /// or a stale reference outliving the table). Used by INSERT commit to resolve
    /// the *live* table its files belong to, regardless of any concurrent rename.
    pub fn table_handle_by_id(&self, id: &Uuid) -> Option<CatalogTable> {
        self.tables.read().unwrap().get_by_id(id).cloned()
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
    pub fn table_files(&self, name: &str) -> Option<Vec<FileRef>> {
        let mut table = self.table_handle(name)?;
        let _ = table.refresh();
        Some(table.file_refs())
    }

    /// BoundTable `name`'s physical files as transport-neutral `(path, size)` pairs in
    /// stable metadata (manifest) order, for introspection. `None` if no such
    /// table exists. Refreshes to the latest committed version first. Sizes come
    /// from the files' `FileRef`s, correlated by path (the two orderings differ,
    /// so a map lookup rather than a zip).
    pub fn table_data_files(&self, name: &str) -> Result<Option<Vec<DataFileInfo>>> {
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
    /// Compile a `CREATE TABLE` to the dataflow that runs it: read every Parquet
    /// footer under the table's location in parallel and, at the final stage,
    /// record the table in the manifest with the files found, and publish it in
    /// the datastore's live catalog map (shared with the datastore through this
    /// transaction's `tables` handle). Fetch and write are one spec:
    /// the caller executes it; nothing happens here but the (cheap, read-only)
    /// directory listing.
    ///
    /// The data lives at a directory/prefix: an explicit `WITH (path = '…')`,
    /// always a plain path, never an object-store URL, or `<name>` under the
    /// database root when none is given. An empty/absent location yields an
    /// empty table (registered with no row groups).
    /// Resolve a `CREATE TABLE` against this datastore: check the name is still
    /// free, parse the layout options, and locate the table's existing data
    /// files. Returns a [`DeltaTableCreation`] whose `compile` builds the
    /// footer-fetch-and-commit dataflow. This part runs on the coordinator;
    /// compiling needs the pool.
    fn bind_create(&self, request: CreateTableRequest) -> Result<DeltaTableCreation> {
        // Reject a duplicate up front, unless `IF NOT EXISTS` makes an existing
        // table a success. The commit re-checks the name under the write lock as a
        // race backstop, and likewise honours `IF NOT EXISTS` there.
        if !request.if_not_exists && self.tables.read().unwrap().contains_name(&request.name) {
            return Err(Error::TableExists(request.name));
        }

        let location = Self::get_path_for_create_table(&request)?;
        let partition_by = Self::parse_spec_columns(&request, PARTITION_BY_OPTION)?;
        let sort_by = Self::parse_spec_columns(&request, SORT_BY_OPTION)?;

        // Locate every data file under the table's directory for reading, keeping
        // its `FileRef` identity so each footer's row groups land on the right
        // `TableFile`.
        let files = self
            .list_file_refs(&location)?
            .into_iter()
            .map(|f| f.into_data_file(self.store.as_ref(), &location))
            .collect::<store::Result<Vec<DataFile>>>()?;

        Ok(DeltaTableCreation {
            tables: self.tables.clone(),
            store: self.store.clone(),
            files,
            name: request.name,
            columns: request.columns,
            location,
            partition_by,
            sort_by,
            if_not_exists: request.if_not_exists,
        })
    }

    /// The location stored for a new table: an explicit `path` (kept as given),
    /// or the table name under the database root.
    ///
    /// A table path is always a plain path, never a URL (no scheme) — where it
    /// physically lives is the database's storage, not the path's. A *relative*
    /// path lives under the database root. An *absolute* path is taken from the
    /// root of the database's storage medium: on a local database, a directory
    /// on the server's filesystem; on a remote database, a key from the **bucket
    /// root** (ignoring the prefix the database was opened at).
    fn get_path_for_create_table(request: &CreateTableRequest) -> Result<ObjectPath> {
        let Some(path) = request.options.get(PATH_OPTION) else {
            return Ok(ObjectPath::new(request.name.clone()));
        };
        if path.contains("://") {
            return Err(Error::TablePathWithScheme(path.clone()));
        }
        Ok(ObjectPath::new(path.clone()))
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

    /// List the Parquet data files at `location` through the store. An
    /// empty/absent location yields no files. Used where the *listing* is the
    /// source of truth: `CREATE TABLE`.
    fn list_file_refs(&self, location: &ObjectPath) -> Result<Vec<FileRef>> {
        Ok(self
            .store
            .list(location)?
            .into_iter()
            .filter(|file| file.path.as_str().ends_with(".parquet"))
            .collect())
    }
}

/// Publish the files an INSERT drained from its transaction: group them by table,
/// append each table's to its Delta log (a CAS over the sync object store), and
/// swap the new version into `tables`. Blocking store I/O, so
/// [`DeltaTransaction::commit`] runs it on the blocking pool.
fn publish_uploaded_files(
    tables: &Arc<RwLock<TableIndex>>,
    drained: Vec<insert_sink::UploadedFile>,
) -> CatalogResult<()> {
    let mut files_by_table = HashMap::<Uuid, Vec<_>>::new();
    for uploaded in drained {
        let insert_sink::UploadedFile {
            table_id,
            file,
            partition,
            sort_bounds,
            row_groups,
        } = uploaded;
        files_by_table.entry(table_id).or_default().push((
            ManifestEntry {
                file: file.clone(),
                partition,
                sort_bounds,
            },
            TableFile::new(file, row_groups),
        ));
    }

    for (table_id, files) in files_by_table {
        let mut table = tables
            .read()
            .unwrap()
            .get_by_id(&table_id)
            .cloned()
            .ok_or_else(|| Error::TableNotFound(table_id.to_string()))?;
        table.commit_uploaded_files(files)?;
        // Swap the committed copy into the live set unless a newer version won.
        let mut map = tables.write().unwrap();
        if map
            .get_by_id(&table.id())
            .is_none_or(|existing| existing.version() < table.version())
        {
            map.insert(table);
        }
    }
    Ok(())
}

#[async_trait]
impl Datastore for DeltaDatastore {
    fn name(&self) -> &str {
        &self.name
    }

    /// Open a transaction: freeze the table set as it stands right now. Every
    /// table the transaction binds resolves from that frozen
    /// [`DeltaSnapshot`] (pure in-memory, no I/O), so one query reads one
    /// consistent version of every table regardless of concurrent refreshes or
    /// commits. [`Datastore::commit_transaction`] publishes files injected by an
    /// INSERT; rollback or dropping the transaction discards that pending set.
    fn begin_transaction(&self) -> Arc<dyn DatastoreTransaction> {
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
    }

    fn abort(&self) {
        for handle in self.maintenance_tasks.lock().unwrap().drain(..) {
            handle.abort();
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
/// materialize, `metadata()`) is pure in-memory.
pub struct DeltaSnapshot {
    /// The datastore this snapshot belongs to, stamped onto every binding it
    /// resolves so a multi-datastore query routes each table back here.
    datastore_name: String,
    tables: TableIndex,
}

impl std::fmt::Debug for DeltaSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaSnapshot")
            .field("datastore_name", &self.datastore_name)
            .field("tables", &self.tables.names().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl DeltaSnapshot {
    /// A clone of the table named `name`, or `None` if this snapshot has no such
    /// table. The frozen copy a [`TableBinding`] captures at bind time, so its
    /// compile resolves the table's file set without any transaction handle.
    pub(super) fn catalog_table_by_name(&self, name: &str) -> Option<CatalogTable> {
        self.tables.get_by_name(name).cloned()
    }

    /// Per-file row groups (path + its row groups, manifest order) for the
    /// `metadata()` table function. Errors if the snapshot has no such table.
    pub(super) fn file_row_groups(
        &self,
        name: &str,
    ) -> Option<Vec<(String, Vec<Arc<crate::parquet::RowGroupMetadata>>)>> {
        Some(self.tables.get_by_name(name)?.file_row_groups())
    }
}

/// The [`DatastoreTransaction`] a [`DeltaDatastore`] opens: one query's frozen
/// [`DeltaSnapshot`] plus an injector of uploaded files awaiting commit.
pub struct DeltaTransaction {
    pub(super) snapshot: Arc<DeltaSnapshot>,
    pub(super) uploaded_files: Arc<Injector<insert_sink::UploadedFile>>,
    /// The datastore's live table set and object store, so a `CREATE TABLE` this
    /// transaction compiles publishes the new table into the datastore itself,
    /// not just into the frozen snapshot. Shared (`Arc`) with the datastore, so
    /// the publish is visible to later transactions.
    tables: Arc<RwLock<TableIndex>>,
    store: Arc<dyn ObjectStore>,
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
    pub fn table(&self, name: &str) -> Option<TableBinding> {
        Some(TableBinding::new(
            self.snapshot.catalog_table_by_name(name)?,
            self.uploaded_files.clone(),
        ))
    }

    fn drain_uploaded_files(&self) -> Vec<insert_sink::UploadedFile> {
        let mut files = Vec::new();
        loop {
            match self.uploaded_files.steal() {
                Steal::Success(file) => files.push(file),
                Steal::Retry => continue,
                Steal::Empty => return files,
            }
        }
    }
}

#[async_trait]
impl DatastoreTransaction for DeltaTransaction {
    fn bind_table(&self, name: &str) -> Option<Box<dyn BoundTable>> {
        Some(Box::new(DeltaTransaction::table(self, name)?))
    }

    fn bind_table_function(&self, name: &str) -> Option<Box<dyn TableFunction>> {
        // `metadata('table')` reports a table's row-group footers; it is
        // parquet-specific, so it lives here rather than in the generic
        // planner. It captures this transaction's frozen snapshot, the same view
        // the rest of the query reads.
        match name {
            "metadata" => Some(Box::new(MetadataTableFunction::new(self.snapshot.clone()))),
            _ => None,
        }
    }

    fn bind_create_table(
        &self,
        request: CreateTableRequest,
    ) -> CatalogResult<Box<dyn TableCreation>> {
        Ok(Box::new(self.bind_create(request)?))
    }

    async fn commit(&self) -> CatalogResult<()> {
        // A read-only transaction uploaded nothing: an in-memory no-op, run inline
        // rather than pay a blocking-pool round trip. Only a commit that publishes
        // files does blocking store I/O, so it (and only it) hops off the runtime.
        if self.uploaded_files.is_empty() {
            return Ok(());
        }
        let tables = self.tables.clone();
        let drained = self.drain_uploaded_files();
        tokio::task::spawn_blocking(move || publish_uploaded_files(&tables, drained))
            .await
            .map_err(|e| CatalogError::Other(format!("commit thread panicked: {e}").into()))?
    }
}

/// A resolved `CREATE TABLE` for a [`DeltaDatastore`]: the validated request, the
/// datastore's live table set and store, and the located data files, captured at
/// resolution by [`DeltaTransaction::bind_create`]. Compiling it builds the
/// dataflow that fetches the files' footers and, at its terminal, commits the new
/// table into the datastore.
struct DeltaTableCreation {
    /// The datastore's live table set (shared `Arc` with the datastore), so the
    /// terminal commit publishes the new table into the datastore itself.
    tables: Arc<RwLock<TableIndex>>,
    store: Arc<dyn ObjectStore>,
    files: Vec<DataFile>,
    name: String,
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
        // The commit runs on the dataflow's last worker once the footers are
        // fetched, under the table-set write lock (which serializes in-process
        // creates): CAS-commit the table's own manifest, record it in the database
        // index, then publish it in the in-memory map, re-checking the name as a
        // race backstop.
        let tables = self.tables.clone();
        let store = self.store.clone();
        let pool = dispatcher.clone();
        let name = self.name.clone();
        let location = self.location.clone();
        let columns = self.columns.clone();
        let partition_by = self.partition_by.clone();
        let sort_by = self.sort_by.clone();
        let if_not_exists = self.if_not_exists;
        let declared_columns: Arc<[planner::catalog::Column]> = columns.clone().into();
        Ok(crate::parquet::create_load_and_commit_spec(
            dispatcher,
            &self.files,
            declared_columns,
            move |loaded: Vec<TableFile>| {
                let mut map = tables.write().unwrap();
                if map.contains_name(&name) {
                    // A concurrent create won the race. With `IF NOT EXISTS` that
                    // is still a success; the fetched footers are dropped.
                    if if_not_exists {
                        return Ok(());
                    }
                    return Err(Box::new(Error::TableExists(name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                let table = CatalogTable::create_new(
                    name.clone(),
                    location.clone(),
                    loaded,
                    columns,
                    partition_by,
                    sort_by,
                    store.clone(),
                    pool,
                )?;
                // Record name → location in the database index.
                let mut index = CatalogManifest::load(store.as_ref())?;
                index.upsert(CatalogManifestTableEntry::new(name.clone(), location));
                index.store(store.as_ref())?;
                map.insert(table);
                Ok(())
            },
        ))
    }
}
