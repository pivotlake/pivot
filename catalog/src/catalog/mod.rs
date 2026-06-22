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
//! - the per-table [`TableManifest`] — *which
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
mod metadata_function;
mod table;

pub use binding::TableBinding;

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::manifest::{
    self, CatalogManifest, CatalogManifestTableEntry, PartitionEqFilter, TableManifest,
};
use crate::parquet::{ParquetTable, ParquetTableError};
use crate::store::{self, DataFile, FileRef, LocalStore, ObjectPath, ObjectStore, open_store};
use dispatch::{DataFlowDispatcher, DataFlowError, RecordBatchOperatorSpec};
use metadata_function::MetadataTableFunction;
use planner::TableFunction;
use planner::catalog::{
    Catalog, CreateTableRequest, Error as CatalogError, QueryContext, Result as CatalogResult,
    Table,
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
            .map(|f| {
                f.file
                    .clone()
                    .into_data_file(store.as_ref(), &entry.location)
            })
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
        let columns = self.tables.read().unwrap().get(name)?.columns();
        Some(TableBinding::new(name.to_string(), columns))
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
    /// Resolve `name` to a [`TableBinding`]: it carries the table's schema and
    /// (after pushdown) this query's predicates, but no file set. The files are
    /// read from the query's [`ParquetQueryContext`] when it compiles, so a
    /// reused (cached) plan always scans the latest committed files. Resolving
    /// here is pure in-memory and does no I/O. Each binding is its own value, so
    /// per-query filter pushdown prunes its view without affecting other queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        Some(Box::new(self.binding(name)?))
    }

    fn query_context(&self) -> Box<dyn QueryContext> {
        Box::new(ParquetQueryContext {
            tables: self.tables.clone(),
            pinned: Mutex::new(HashMap::new()),
        })
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }

    fn table_function(&self, name: &str) -> Option<Box<dyn TableFunction>> {
        // `metadata('table')` reports a table's row-group footers; it is
        // parquet-specific, so it lives here rather than in the generic planner.
        match name {
            "metadata" => Some(Box::new(MetadataTableFunction)),
            _ => None,
        }
    }
}

/// One pinned scan view per `(table name, partition filter)`. The filter
/// half is each `(column, JSON constant)` sorted, the hashable form of the
/// predicates.
type PinnedScanViews = HashMap<(String, Vec<(String, String)>), Arc<ParquetTable>>;

/// One query's [`QueryContext`]: the concrete context a [`TableBinding`]
/// downcasts to. The first time a scan asks for a table it reloads the master to
/// the latest committed version and pins the resulting row groups; later asks
/// (the same table's late materialize, or a self-join) return that same pinned
/// `Arc`. So each table reloads at most once per query and every scan of it sees
/// one consistent snapshot.
pub(super) struct ParquetQueryContext {
    tables: Arc<RwLock<HashMap<String, CatalogTable>>>,
    /// One pinned scan view per `(table, its partition filter)`. The key groups
    /// exactly the asks that must agree: a scan and its later materialize share a
    /// `(name, filters)`, so they pin the same view and address the same row-group
    /// indices. Different filters (a self-join's branches) get their own views,
    /// which is correct — they never share an index space. The filter half is each
    /// `(column, JSON constant)` sorted, the hashable form of the predicates.
    pinned: Mutex<PinnedScanViews>,
}

impl ParquetQueryContext {
    /// Table `name`'s committed row groups for the files that can match
    /// `partition_filters` — the partition-pruned scan view, pinned on first ask
    /// and returned from the pin thereafter. Footers are fetched lazily here, only
    /// for the surviving partitions, so a one-partition query never reads every
    /// other partition's footer. Errors if the table no longer exists (dropped
    /// since planning) or the reload fails, rather than scanning a stale or empty
    /// file set.
    pub(super) fn parquet(
        &self,
        name: &str,
        partition_filters: impl Iterator<Item = PartitionEqFilter>,
    ) -> CatalogResult<Arc<ParquetTable>> {
        let filters: Vec<PartitionEqFilter> = partition_filters.collect();
        let mut key_filters: Vec<(String, String)> = filters
            .iter()
            .map(|f| (f.column.clone(), f.value.to_string()))
            .collect();
        key_filters.sort(); // filter order doesn't change the view
        let key = (name.to_string(), key_filters);
        if let Some(parquet) = self.pinned.lock().unwrap().get(&key) {
            return Ok(parquet.clone());
        }

        // Build the view: take the master copy, advance its manifest (no footer
        // I/O), then fetch footers for just the surviving partitions.
        let mut table = self
            .tables
            .read()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| {
                CatalogError::Other(
                    format!("table {name:?} no longer exists (dropped since planning?)").into(),
                )
            })?;
        table
            .reload_manifest_only()
            .map_err(|e| CatalogError::Other(Box::new(e)))?;
        let parquet = table
            .parquet(&filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))?;
        // Publish the warmed copy (advanced version + the footers just fetched) so
        // a later query starts current and re-fetches nothing it already holds.
        self.tables.write().unwrap().insert(name.to_string(), table);
        self.pinned.lock().unwrap().insert(key, parquet.clone());
        Ok(parquet)
    }
}

impl QueryContext for ParquetQueryContext {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
