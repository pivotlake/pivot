//! A concrete [`planner::catalog::Catalog`] implementation backed by Parquet.
//!
//! The catalog stores tables in a `RwLock<HashMap>` keyed by name, so multiple
//! threads can resolve and create tables concurrently — many readers (lookups)
//! coexist with infrequent writers (`CREATE TABLE`, file registrations).
//! Backing the map is the database's [`ObjectStore`]: an in-memory store by
//! default ([`ParquetCatalog::new`], ephemeral), or — for a database opened with
//! [`ParquetCatalog::open`] on a local directory *or* a remote `s3://`/`gs://`
//! root — a persisted one.
//!
//! Durable state lives in two places, both in the store:
//!
//! - the [`manifest`], which tables exist (name, location);
//! - each table's Delta Lake transaction log (see [`crate::delta`]): its
//!   declared schema, partition/sort specs, and *which Parquet files* it
//!   consists of, as a sequence of versions committed with an atomic
//!   create-if-absent. The highest version is the table's current file list.
//!
//! Each table's in-memory [`table::CatalogTable`] pairs its definition with the
//! per-file row groups at one log version. The in-memory set is kept current by
//! **push, not pull**: a periodic [`refresh_catalog`](ParquetCatalog::refresh_catalog)
//! sweep (the server runs one on an interval) reloads every table to its latest
//! committed version and fetches any new files' footers, and an in-process
//! writer (ingest, compaction) [`publish_table`](ParquetCatalog::publish_table)s
//! its committed copy immediately. Queries never touch the store: a query opens
//! a transaction ([`Catalog::begin_transaction`]) whose [`CatalogSnapshot`]
//! freezes the table set as of that moment, and every binding, scan, and late
//! materialize of that query reads the frozen snapshot with zero I/O.
//!
//! Each resolve hands back a fresh [`TableBinding`], so per-query filter
//! pushdown accumulates on that binding alone.

mod binding;
mod metadata_function;
mod table;

pub use binding::TableBinding;

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub use crate::manifest::TableFile;
use crate::manifest::{self, CatalogManifest, CatalogManifestTableEntry, PartitionEqFilter};
use crate::parquet::{ParquetTable, ParquetTableError};
use crate::store::{self, DataFile, FileRef, LocalStore, ObjectPath, ObjectStore, open_store};
use dispatch::{DataFlowDispatcher, DataFlowError, RecordBatchOperatorSpec};
use metadata_function::MetadataTableFunction;
use planner::TableFunction;
use planner::catalog::{
    Catalog, CatalogTransaction, CreateTableRequest, Error as CatalogError,
    Result as CatalogResult, Table,
};
pub use table::CatalogTable;
use thiserror::Error as ThisError;

const PATH_OPTION: &str = "path";
/// `WITH (partition_by = 'a, b')` — ordered, comma-separated partition columns.
const PARTITION_BY_OPTION: &str = "partition_by";
/// `WITH (sort_by = 'a, b')` — ordered, comma-separated sort columns.
const SORT_BY_OPTION: &str = "sort_by";

#[derive(Debug, ThisError)]
pub enum Error {
    #[error(
        "table path `{0}` must be a plain path, not a URL — a table's storage is the database's, so the path carries no scheme"
    )]
    TablePathWithScheme(String),
    #[error("`{option}` column `{column}` is not a declared column of the table")]
    UnknownSpecColumn { option: String, column: String },
    #[error("`IF NOT EXISTS` is not supported")]
    IfNotExistsUnsupported,
    #[error(transparent)]
    ParquetTable(#[from] ParquetTableError),
    #[error("table `{0}` already exists")]
    TableExists(String),
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error(transparent)]
    Store(#[from] store::StoreError),
    #[error(transparent)]
    Manifest(#[from] manifest::Error),
    #[error(transparent)]
    Delta(#[from] crate::delta::Error),
    #[error(
        "writes to table `{0}` are disabled: ingest and compaction are not yet supported over the delta table format"
    )]
    WritesDisabled(String),
    #[error("loading table footers: {0}")]
    Load(#[from] DataFlowError),
    #[error(
        "table `{table}`: file `{file}` has no loaded row-group metadata; the copy was not synced to its manifest"
    )]
    FooterNotLoaded { table: String, file: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// Concurrent catalog of tables, keyed by name.
///
/// `CREATE TABLE` compiles to a single dataflow that reads every data file's
/// footer **once** (in parallel over the worker pool) and, at its terminal
/// stage, commits the table (its manifest + the database index) and publishes
/// the entry in this shared map. After that, a table evolves by manifest commits
/// on a [`CatalogTable`] *copy* — a writer takes one with
/// [`table_handle`](Self::table_handle) and calls
/// [`append_data_file`](CatalogTable::append_data_file) /
/// [`replace_data_files`](CatalogTable::replace_data_files), which CAS a new
/// version into the store. Copies drift; every query resolve refreshes its copy
/// to the latest committed version, so a commit by another process (or this one)
/// becomes visible to the next query.
pub struct ParquetCatalog {
    /// The in-memory table set. The lock guards the *set* (add on `CREATE`,
    /// swap-in on a resolve's refresh); each [`CatalogTable`] is itself a
    /// lock-free value that callers clone out and evolve independently.
    tables: Arc<RwLock<HashMap<String, CatalogTable>>>,
    /// The database's object store — both the table manifests and the tables'
    /// Parquet data. A local directory by default ([`new`], an ephemeral one
    /// under the temp dir), or the directory / S3 / GCS root a database is
    /// [`open`](Self::open)ed at. The catalog reads and writes a table's data
    /// through this one store: relative locations live under the database root,
    /// an absolute location at the store's own root (the filesystem root, or the
    /// bucket root).
    ///
    /// [`new`]: Self::new
    store: Arc<dyn ObjectStore>,

    /// The worker pool every footer fetch runs on. Held by the catalog because
    /// refreshes drive their own dataflows, with no dispatcher passed in.
    dispatcher: DataFlowDispatcher,
}

impl std::fmt::Debug for ParquetCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetCatalog")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl ParquetCatalog {
    /// An ephemeral database rooted at a fresh directory under the system temp
    /// dir: catalog and data are written there and simply abandoned on exit. For
    /// a persisted database use [`open`](Self::open).
    pub fn new(dispatcher: DataFlowDispatcher) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("catalog-{}-{}", std::process::id(), seq));
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: Arc::new(LocalStore::new(root)),
            dispatcher,
        }
    }

    /// Open a persisted database rooted at `uri` — a local directory (or
    /// `file://…`), or a remote `s3://…`/`gs://…` object store — reloading every
    /// table the manifest records at its latest version: read each table's
    /// manifest (schema + file list) and fetch its files' footers, building the
    /// in-memory [`CatalogTable`]. A database with no manifest yet opens empty.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Self> {
        let catalog = Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: open_store(uri)?.into(),
            dispatcher: dispatcher.clone(),
        };
        let manifest = CatalogManifest::load(catalog.store.as_ref())?;
        let mut tables = catalog.tables.write().unwrap();
        for entry in &manifest.tables {
            tables.insert(entry.name.clone(), catalog.load_table(entry)?);
        }
        drop(tables);
        Ok(catalog)
    }

    /// Build the in-memory [`CatalogTable`] for one persisted table: read its
    /// delta log (erroring if the catalog points at a table that has none),
    /// then locate its committed files under the entry's location and fetch
    /// their footers over the pool.
    fn load_table(&self, entry: &CatalogManifestTableEntry) -> Result<CatalogTable> {
        let (log, mut state) = crate::delta::DeltaLog::open(
            &self.store.get_config()?,
            self.store.get_absolute_url(&entry.location)?,
        )?;
        let files = state
            .entries
            .iter()
            .map(|f| {
                f.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &entry.location)
            })
            .collect::<store::Result<Vec<DataFile>>>()?;
        state.files = crate::parquet::load_table_files(&self.dispatcher, &files)?;
        Ok(CatalogTable::new(
            entry.name.clone(),
            entry.location.clone(),
            Arc::new(log),
            state,
            self.store.clone(),
            self.dispatcher.clone(),
        ))
    }

    /// Open a transaction, typed: freeze the current table set into a
    /// [`CatalogSnapshot`] and hand back the concrete [`ParquetTransaction`]
    /// over it. Cheap: clones the map (the row-group metadata inside is
    /// `Arc`-shared), no I/O. The [`Catalog::begin_transaction`] trait impl
    /// delegates here.
    pub fn begin_transaction(&self) -> Arc<ParquetTransaction> {
        Arc::new(ParquetTransaction {
            snapshot: Arc::new(CatalogSnapshot {
                tables: self.tables.read().unwrap().clone(),
            }),
        })
    }

    /// Bring the in-memory table set up to date with the store: pick up tables
    /// another process registered in the database index, and advance every
    /// table to its latest committed manifest version, fetching the footers of
    /// files it doesn't hold yet. This is the **only** place the read path
    /// pays store I/O; the server drives it on a background interval, so
    /// queries always bind against an already-materialized set. Returns whether
    /// anything changed.
    ///
    /// Tables are never removed here: there is no `DROP TABLE`, and removing a
    /// map entry on a manifest miss could race a concurrent in-process create.
    pub fn refresh_catalog(&self) -> Result<bool> {
        let mut changed = false;

        let index = CatalogManifest::load(self.store.as_ref())?;
        for entry in &index.tables {
            if self.tables.read().unwrap().contains_key(&entry.name) {
                continue;
            }
            // One unloadable table (e.g. a delta table with an unsupported
            // feature) must not stop the sweep from serving every other table;
            // it is skipped with a warning and retried next tick.
            let table = match self.load_table(entry) {
                Ok(table) => table,
                Err(e) => {
                    tracing::warn!(table = %entry.name, error = %e, "skipping unloadable table in catalog refresh");
                    continue;
                }
            };
            // No re-check under the write lock: nothing else can have inserted
            // this name meanwhile. An in-process CREATE TABLE of an
            // index-listed table always fails its create CAS (the table's log
            // already exists) before publishing, and writers only publish
            // copies of tables already in the map.
            self.tables
                .write()
                .unwrap()
                .insert(entry.name.clone(), table);
            changed = true;
        }

        // Refresh each table on a clone outside the lock (footer fetches are
        // I/O), then publish the advanced copy back. A table that fails to
        // refresh keeps serving its current version and must not block the
        // others from advancing.
        for mut table in self.tables() {
            match table.refresh() {
                Ok(true) => changed |= self.publish_table(table),
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(table = %table.name(), error = %e, "skipping failed table refresh in catalog refresh");
                }
            }
        }

        Ok(changed)
    }

    /// Publish a committed [`CatalogTable`] copy into the shared map, so its
    /// new version is visible to the next transaction immediately (rather than
    /// after the next background refresh). Ingest and compaction call this
    /// right after their commit. A copy at or behind the map's current version
    /// is dropped (`false`); a newer one replaces it.
    pub fn publish_table(&self, table: CatalogTable) -> bool {
        let mut map = self.tables.write().unwrap();
        match map.get(table.name()) {
            Some(existing) if existing.version() >= table.version() => false,
            _ => {
                map.insert(table.name().to_string(), table);
                true
            }
        }
    }

    /// The worker pool this catalog fetches footers on — shared with callers
    /// (the compacter) that drive their own dataflows over the same tables.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// A clone of the named table's current state for a writer (ingest,
    /// compaction) to evolve — [`append_data_file`](CatalogTable::append_data_file)
    /// or [`replace_data_files`](CatalogTable::replace_data_files). Those commit
    /// a new version by CAS to the shared store, so this catalog's own copy may
    /// lag until its next resolve refreshes it (which is fine — the store is the
    /// source of truth). `None` if no such table exists.
    pub fn table_handle(&self, name: &str) -> Option<CatalogTable> {
        self.tables.read().unwrap().get(name).cloned()
    }

    /// Whether a table named `name` exists in the catalog (a cheap membership
    /// check — no clone). Used by ingest to fail fast at startup when a sink's
    /// table hasn't been created.
    pub fn contains_table(&self, name: &str) -> bool {
        self.tables.read().unwrap().contains_key(name)
    }

    /// A snapshot clone of every table the catalog currently holds — for a sweep
    /// (e.g. the compacter) that refreshes and evolves each one independently.
    pub fn tables(&self) -> Vec<CatalogTable> {
        self.tables.read().unwrap().values().cloned().collect()
    }

    /// A human-readable description of where this database is rooted (local
    /// directory or object-store bucket/prefix) - for introspection. Table
    /// locations are relative to this root.
    pub fn store_description(&self) -> String {
        self.store.describe()
    }

    /// Table `name`'s current committed files — refreshing to the latest version
    /// first, so a commit by ingest/compaction (in this process or another) is
    /// reflected. `None` if no such table exists.
    pub fn table_files(&self, name: &str) -> Option<Vec<FileRef>> {
        let mut table = self.table_handle(name)?;
        let _ = table.refresh();
        Some(table.file_refs())
    }

    /// Compile a `CREATE TABLE` to the dataflow that runs it: read every Parquet
    /// footer under the table's location in parallel and, at the final stage,
    /// record the table in the manifest with the files found, and publish it in
    /// the catalog map. Fetch and write are one spec —
    /// the caller executes it; nothing happens here but the (cheap, read-only)
    /// directory listing.
    ///
    /// The data lives at a directory/prefix: an explicit `WITH (path = '…')` —
    /// always a plain path, never an object-store URL — or `<name>` under the
    /// database root when none is given. An empty/absent location yields an
    /// empty table (registered with no row groups).
    fn create(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec> {
        if request.if_not_exists {
            return Err(Error::IfNotExistsUnsupported);
        }
        // Reject a duplicate up front; the commit re-checks under the lock as a
        // race backstop.
        if self.tables.read().unwrap().contains_key(&request.name) {
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

        // The commit runs on the dataflow's last worker once the footers are
        // fetched, under the table-set write lock (which serializes in-process
        // creates): CAS-commit the table's own manifest, record it in the
        // database index, then publish it in the in-memory map — re-checking the
        // name as a race backstop.
        let tables = self.tables.clone();
        let store = self.store.clone();
        let pool = dispatcher.clone();
        Ok(crate::parquet::create_load_and_commit_spec(
            dispatcher,
            &files,
            move |loaded: Vec<TableFile>| {
                let mut map = tables.write().unwrap();
                if map.contains_key(&request.name) {
                    return Err(Box::new(Error::TableExists(request.name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                let table = CatalogTable::create_new(
                    request.name.clone(),
                    location.clone(),
                    loaded,
                    request.columns,
                    partition_by,
                    sort_by,
                    store.clone(),
                    pool,
                )?;
                // Record name → location in the database index.
                let mut index = CatalogManifest::load(store.as_ref())?;
                index.upsert(CatalogManifestTableEntry::new(
                    request.name.clone(),
                    location,
                ));
                index.store(store.as_ref())?;
                map.insert(request.name, table);
                Ok(())
            },
        ))
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

impl Catalog for ParquetCatalog {
    /// Open a transaction: freeze the table set as it stands right now. Every
    /// table the transaction binds resolves from that frozen
    /// [`CatalogSnapshot`] (pure in-memory, no I/O), so one query reads one
    /// consistent version of every table regardless of concurrent refreshes or
    /// commits. Dropping the returned transaction (the commit) releases the
    /// snapshot.
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        ParquetCatalog::begin_transaction(self)
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }
}

/// One transaction's frozen view of the catalog: every table at the version it
/// held when the transaction began, with all row-group metadata already
/// materialized (the background refresh keeps the master set fully fetched).
/// Everything a query does against it (binding, scan-view construction, late
/// materialize, `metadata()`) is pure in-memory.
pub struct CatalogSnapshot {
    tables: HashMap<String, CatalogTable>,
}

impl std::fmt::Debug for CatalogSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogSnapshot")
            .field("tables", &self.tables.keys())
            .finish_non_exhaustive()
    }
}

impl CatalogSnapshot {
    /// Resolve `name` to a fresh [`TableBinding`] over this snapshot, or `None`
    /// if the snapshot has no such table. Pure in-memory. The typed counterpart
    /// of [`CatalogTransaction::table`].
    pub fn table(&self, name: &str) -> Option<TableBinding> {
        let columns = self.tables.get(name)?.columns();
        Some(TableBinding::new(name.to_string(), columns))
    }

    /// Table `name`'s row groups for the files that can match
    /// `partition_filters` — the partition-pruned scan view. Built from the
    /// snapshot's already materialized row groups: no manifest read, no footer
    /// fetch. The snapshot is immutable, so two asks with the same filters (a
    /// scan and its late materialize) always build identical views addressing
    /// the same row-group indices. Errors if the table isn't in the snapshot
    /// (it never was, or the binding outlived its transaction into a catalog
    /// where it's gone) rather than scanning an empty file set.
    pub(super) fn parquet(
        &self,
        name: &str,
        partition_filters: impl Iterator<Item = PartitionEqFilter>,
    ) -> CatalogResult<Arc<ParquetTable>> {
        let filters: Vec<PartitionEqFilter> = partition_filters.collect();
        let table = self.tables.get(name).ok_or_else(|| {
            CatalogError::Other(format!("table {name:?} is not in this snapshot").into())
        })?;
        table
            .build_scan_view(&filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// Per-file row groups (path + its row groups, manifest order) for the
    /// `metadata()` table function. Errors if the snapshot has no such table.
    pub(super) fn file_row_groups(
        &self,
        name: &str,
    ) -> Option<Vec<(String, Vec<Arc<crate::parquet::RowGroupMetadata>>)>> {
        Some(self.tables.get(name)?.file_row_groups())
    }
}

/// The [`CatalogTransaction`] a [`ParquetCatalog`] opens: one query's handle on
/// its [`CatalogSnapshot`]. Dropped when the query finishes (the commit),
/// releasing the snapshot.
#[derive(Debug)]
pub struct ParquetTransaction {
    pub(super) snapshot: Arc<CatalogSnapshot>,
}

impl ParquetTransaction {
    /// [`CatalogTransaction::table`], typed: the concrete [`TableBinding`]
    /// instead of the trait object. This is the single resolution path; the
    /// trait impl below only boxes its result (the planner needs a
    /// `Box<dyn Table>`, while a concrete caller wants the `TableBinding`, and
    /// one function cannot return both).
    pub fn table(&self, name: &str) -> Option<TableBinding> {
        self.snapshot.table(name)
    }
}

impl CatalogTransaction for ParquetTransaction {
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        Some(Box::new(ParquetTransaction::table(self, name)?))
    }

    fn table_function(&self, name: &str) -> Option<Box<dyn TableFunction>> {
        // `metadata('table')` reports a table's row-group footers; it is
        // parquet-specific, so it lives here rather than in the generic
        // planner. It reads the transaction handed to its compile, the same
        // frozen view the rest of the query reads.
        match name {
            "metadata" => Some(Box::new(MetadataTableFunction)),
            _ => None,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
