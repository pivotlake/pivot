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
//! - the [`manifest`] — which tables exist (name, declared schema, location);
//! - the per-table [`TableManifest`](crate::manifest::TableManifest) — *which
//!   Parquet files* each table consists of, as a sequence of versions committed
//!   with the store's compare-and-swap. The highest version is the table's
//!   current file list.
//!
//! Each table's in-memory [`table::CatalogTable`] pairs its definition with the
//! per-file row groups at one log version; the flattened scan view a query sees
//! is derived from those files on demand, so there is no cached copy to drift.
//! A change swaps `version` + `files` whole. Resolving a table for a query
//! ([`Catalog::table`]) first **reloads**: one LIST of the table's log
//! directory; if a newer version exists, only the *new* files' footers are
//! fetched (over the dispatch pool) and the new file set is swapped in — so a
//! file registered by ingest, a compaction's swap, or even another process's
//! commit becomes visible to the very next query.
//!
//! Each resolve hands back a fresh [`TableBinding`], so per-query filter
//! pushdown accumulates on that binding alone.

mod binding;
mod table;

pub use binding::TableBinding;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::manifest::{
    self, CatalogManifest, CatalogManifestTableEntry, TableManifest,
};
use crate::parquet::ParquetTableError;
use crate::store::{
    self, DataFile, FileRef, LocalStore, ObjectPath, ObjectStore, open_store,
};
use dispatch::{DataFlowDispatcher, DataFlowError, RecordBatchOperatorSpec};
pub use table::CatalogTable;
pub use table::TableFile;
use planner::catalog::{
    Catalog, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use thiserror::Error as ThisError;
use tracing::warn;

const PATH_OPTION: &str = "path";

#[derive(Debug, ThisError)]
pub enum Error {
    #[error(
        "table path `{0}` must be a plain path, not a URL — a table's storage is the database's, so the path carries no scheme"
    )]
    TablePathWithScheme(String),
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
    #[error("loading table footers: {0}")]
    Load(#[from] DataFlowError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// What [`CatalogTable::register_data_file`] did with the file. The non-
/// `Registered` outcomes are not errors — a file can legitimately land before
/// its table is created — but the caller (ingest) wants to log them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterOutcome {
    /// Committed to the table's log; new binds see its rows.
    Registered,
    /// The table's log already holds this name (a replayed notification); no
    /// new version was committed.
    AlreadyRegistered,
    /// No table by that name exists yet; the file is picked up by a later
    /// `CREATE TABLE` instead.
    NoSuchTable,
    /// The table exists but its data does not live in this file's directory
    /// (or is not an absolute local directory at all); refused.
    LocationMismatch,
}

/// Concurrent catalog of tables, keyed by name.
///
/// `CREATE TABLE` compiles to a single dataflow that reads every data file's
/// footer **once** (in parallel over the worker pool) and, at its terminal
/// stage, commits the table (its manifest + the database index) and publishes
/// the entry in this shared map. After that, a table evolves by manifest commits
/// on a [`CatalogTable`] *copy* — a writer takes one with
/// [`table_handle`](Self::table_handle) and calls
/// [`register_data_file`](CatalogTable::register_data_file) /
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
    /// reloads happen at query-bind time, where no dispatcher is passed in.
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
        let root =
            std::env::temp_dir().join(format!("goose-{}-{}", std::process::id(), seq));
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
        let store: Arc<dyn ObjectStore> = open_store(uri)?.into();
        let manifest = CatalogManifest::load(store.as_ref())?;

        let mut tables = HashMap::new();
        for entry in &manifest.tables {
            let table = Self::load_table(dispatcher, &store, entry)?;
            tables.insert(entry.name.clone(), table);
        }

        Ok(Self {
            tables: Arc::new(RwLock::new(tables)),
            store,
            dispatcher: dispatcher.clone(),
        })
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
        let manifest = TableManifest::load(store.as_ref(), &entry.name)?;
        let files = manifest
            .entries
            .iter()
            .map(|f| f.clone().into_data_file(store.as_ref(), &entry.location))
            .collect::<store::Result<Vec<DataFile>>>()?;
        let table_files = crate::parquet::load_table_files(dispatcher, &files)?;
        Ok(CatalogTable::new(
            entry.name.clone(),
            entry.location.clone(),
            manifest,
            table_files,
            store.clone(),
            dispatcher.clone(),
        ))
    }

    /// Resolve `name` to a fresh [`TableBinding`] (same semantics as
    /// [`Catalog::table`] *minus the reload*, and typed). Useful for callers
    /// (and tests) that need state the [`Table`] trait does not expose.
    pub fn binding(&self, name: &str) -> Option<TableBinding> {
        self.tables
            .read()
            .unwrap()
            .get(name)
            .map(CatalogTable::binding)
    }


    /// The worker pool this catalog fetches footers on — shared with callers
    /// (the compacter) that drive their own dataflows over the same tables.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// A clone of the named table's current state for a writer (ingest,
    /// compaction) to evolve — [`register_data_file`](CatalogTable::register_data_file)
    /// or [`replace_data_files`](CatalogTable::replace_data_files). Those commit
    /// a new version by CAS to the shared store, so this catalog's own copy may
    /// lag until its next resolve refreshes it (which is fine — the store is the
    /// source of truth). `None` if no such table exists.
    pub fn table_handle(&self, name: &str) -> Option<CatalogTable> {
        self.tables.read().unwrap().get(name).cloned()
    }

    /// A snapshot clone of every table the catalog currently holds — for a sweep
    /// (e.g. the compacter) that refreshes and evolves each one independently.
    pub fn tables(&self) -> Vec<CatalogTable> {
        self.tables.read().unwrap().values().cloned().collect()
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
                    store.clone(),
                    pool,
                )?;
                // Record name → location in the database index.
                let mut index = CatalogManifest::load(store.as_ref())?;
                index.upsert(CatalogManifestTableEntry::new(request.name.clone(), location));
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
    /// Resolve `name` to a fresh, independently-mutable [`TableBinding`],
    /// **reloading first**: a copy of the table is taken and refreshed (one LIST
    /// to check the latest version; new files' footers fetched only if it
    /// advanced), so every query starts from the latest committed file list — a
    /// commit by ingest, a compaction, or another process becomes visible to the
    /// next query. An advanced copy is swapped back into the map so the next
    /// resolve starts current. Each binding is its own value, so per-query filter
    /// pushdown prunes its view without affecting other concurrent queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        let mut table = self.tables.read().unwrap().get(name)?.clone();
        match table.refresh() {
            // Advanced — publish the reloaded copy for the next resolve.
            Ok(true) => {
                self.tables.write().unwrap().insert(name.to_string(), table.clone());
            }
            Ok(false) => {}
            // Serve the version we have rather than failing the query; the next
            // resolve retries the reload.
            Err(e) => {
                warn!(table = name, error = %e, "table refresh failed; serving last known version")
            }
        }
        Some(Box::new(table.binding()))
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }
}
