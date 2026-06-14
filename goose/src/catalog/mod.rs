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
//! - the per-table [`table_log`] — *which Parquet files* each table consists
//!   of, as an append-only sequence of versions committed with the store's
//!   compare-and-swap. The highest version is the table's current file list.
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

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::manifest::{self, CatalogManifestTableEntry};
use crate::parquet::{ParquetTableError, RowGroupMetadata};
use crate::store::{
    self, DataFile, DataFileSource, FileRef, LocalStore, ObjectStore, location_key,
};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
pub use table::{CatalogTable};
pub use table::TableFile;
use planner::catalog::{
    Catalog, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use thiserror::Error as ThisError;
use tracing::warn;
use crate::manifest::CatalogManifest;

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
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// What [`ParquetCatalog::register_data_file`] did with the file. The non-
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
/// stage, commits the table to the manifest, seeds its [`table_log`], and
/// publishes the entry in this shared map. After that, the table evolves by
/// log commits ([`register_data_file`](Self::register_data_file),
/// [`replace_data_files`](Self::replace_data_files)) and every query resolve
/// reloads the entry up to the latest committed version.
pub struct ParquetCatalog {
    tables: Arc<RwLock<HashMap<String, CatalogTable>>>,
    /// The database's object store — both the table [`manifest`]/[`table_log`]s
    /// and the tables' Parquet data. A local directory by default ([`new`], an
    /// ephemeral one under the temp dir), or the directory / S3 / GCS root a
    /// database is [`open`](Self::open)ed at. The catalog reads and writes a
    /// table's data through this one store: relative locations live under the
    /// database root, an absolute location at the store's own root (the
    /// filesystem root, or the bucket root).
    ///
    /// [`new`]: Self::new
    store: Arc<dyn ObjectStore>,

    manifest: CatalogManifest,
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
        todo!()
    }

    /// Open a persisted database rooted at `uri` — a local directory (or
    /// `file://…`), or a remote `s3://…`/`gs://…` object store — reloading
    /// every table the manifest records at its latest [`table_log`] version.
    /// A table without a log (created before logs existed) falls back to
    /// listing its data location; its first commit seeds the log.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Self> {
        // TODO: Load or Create a CatalogManifest, then run per CatalogManifestTableEntry and create
        // a CatalogTable (by getting last TableManifest)
        todo!()
        // Ok(catalog)
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

    /// Atomically swap a set of table `name`'s data files for another — the
    /// compaction commit. Reads the `added` files' footers, commits a log
    /// version whose list is `latest − removed + added` (retrying past CAS
    /// conflicts), and swaps the in-memory entry. One version, so no binding
    /// ever sees the rows doubled or missing; queries already bound keep
    /// their snapshot (their open handles keep even a deleted file's bytes
    /// readable until they finish). Returns `false` when no table named
    /// `name` exists.
    pub fn replace_data_files(
        &self,
        name: &str,
        removed: &[String],
        added: &[FileRef],
    ) -> Result<bool> {
        // TODO
        Ok(true)
    }

    /// Compile a `CREATE TABLE` to the dataflow that runs it: read every Parquet
    /// footer under the table's location in parallel and, at the final stage,
    /// record the table in the manifest, seed its [`table_log`] with the files
    /// found, and publish it in the catalog map. Fetch and write are one spec —
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
        let manifest_entry = CatalogManifestTableEntry::new(request.name);

        // Locate every data file under the table's directory for reading, keeping
        // its `FileRef` identity so each footer's row groups land on the right
        // `TableFile`.
        let files = self
            .list_file_refs(&location)?
            .into_iter()
            .map(|f| self.store.data_file(&location_key(&location, &f.name), f.size))
            .collect::<store::Result<Vec<DataFile>>>()?;

        // The commit runs on the dataflow's last worker once the footers are
        // fetched: record the table durably (manifest + seed log version), then
        // in the in-memory map (re-checking the name under the lock as a race
        // backstop).
        let tables = self.tables.clone();
        Ok(crate::parquet::create_load_and_commit_spec(
            dispatcher,
            &files,
            move |_loaded: Vec<TableFile>| {
                let mut map = tables.write().unwrap();
                if map.contains_key(&manifest_entry.name) {
                    return Err(Box::new(Error::TableExists(manifest_entry.name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                // todo: Create new CatalogTable from `_loaded`, update manifest.
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
    fn get_path_for_create_table(request: &CreateTableRequest) -> Result<String> {
        let Some(path) = request.options.get(PATH_OPTION) else {
            return Ok(request.name.clone());
        };
        if path.contains("://") {
            return Err(Error::TablePathWithScheme(path.clone()));
        }
        Ok(path.clone())
    }

    /// List the Parquet data files at `location` through the store. An
    /// empty/absent location yields no files. Used where the *listing* is the
    /// source of truth: `CREATE TABLE`, and reloading a legacy table with no log.
    fn list_file_refs(&self, location: &str) -> Result<Vec<FileRef>> {
        Ok(self
            .store
            .list(location)?
            .into_iter()
            .filter(|file| file.name.ends_with(".parquet"))
            .collect())
    }
}


/// Whether `file` sits directly in the table data directory `location`. Only an
/// absolute local location qualifies — a relative one lives inside the
/// database's object store, which is not where a locally-written file is.
/// Both sides are canonicalized so spelling differences don't refuse a
/// legitimate registration.
fn file_in_table_dir(file: &Path, location: &str) -> bool {
    let dir = Path::new(location);
    if !dir.is_absolute() {
        return false;
    }
    match (std::fs::canonicalize(dir), file.parent()) {
        (Ok(dir), Some(parent)) => std::fs::canonicalize(parent).is_ok_and(|p| p == dir),
        _ => false,
    }
}

impl Catalog for ParquetCatalog {
    /// Resolve `name` to a fresh, independently-mutable [`TableBinding`],
    /// **reloading first**: the table log is checked (one LIST) and any newer
    /// committed version is pulled in, so every query starts from the latest
    /// file list. Each binding is its own value, so per-query filter pushdown
    /// prunes its view without affecting the master entry or other concurrent
    /// queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        // if let Err(e) = self.refresh(name) {
        //     // Serve the version we have rather than failing the query; the
        //     // next bind retries the reload.
        //     warn!(table = name, error = %e, "table reload failed; serving last known version");
        // }
        self.binding(name).map(|t| Box::new(t) as _)
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }
}
