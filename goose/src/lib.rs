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
//! Each table's in-memory entry tracks the log version it reflects, with row
//! groups grouped per file. Resolving a table for a query
//! ([`Catalog::table`]) first **reloads**: one LIST of the table's log
//! directory; if a newer version exists, only the *new* files' footers are
//! fetched (over the dispatch pool) and the entry is swapped to the new
//! version — so a file registered by ingest, a compaction's swap, or even
//! another process's commit becomes visible to the very next query.
//!
//! Each `Catalog::table` lookup hands back a fresh [`Box<dyn Table>`] cloned
//! from the master entry, so per-binding filter pushdown accumulates on that
//! binding alone. [`Table::pushdown_filter`] just *records* each single-column
//! predicate; the pruning (dropping row groups whose min/max can't match, and
//! dictionary-pruning for equality) happens in [`Table::compile`], once the
//! row-group metadata is in hand. Pushdown always reports `false` because
//! per-row evaluation is still required on the survivors.

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here — we're not a
// published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

pub mod manifest;
pub mod parquet;
mod sql_type;
pub mod store;
pub mod table_log;

pub use manifest::ManifestEntry;
pub use table_log::LoggedFile;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetTable, ParquetTableError, ScanEqualityPredicate, materialize, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use crate::parquet::{RowGroupMetadata, load};
use crate::store::{DataFile, DataFileSource, LocalStore, MemoryStore, ObjectStore};
use crate::table_log::TableVersion;
use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::expression::{CompareType, Expression, TableFilter};
use thiserror::Error;
use tracing::warn;

const PATH_OPTION: &str = "path";

#[derive(Debug, Error)]
pub enum Error {
    #[error("path `{0}` does not exist")]
    PathNotFound(String),
    #[error("path `{0}` is not a directory")]
    PathNotDirectory(String),
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
    Manifest(#[from] manifest::ManifestError),
    #[error(transparent)]
    TableLog(#[from] table_log::TableLogError),
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

/// One data file of an in-memory table entry: its logged identity paired with
/// its materialized row groups (file-local order; global indices are assigned
/// when the entry's files are flattened into the scan view).
#[derive(Clone)]
struct TableFileEntry {
    file: LoggedFile,
    row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// The master record of one table: the binding template every query clones,
/// plus the versioned per-file state the template's `parquet` is derived from.
#[derive(Clone)]
struct TableEntry {
    /// The table's name — the catalog map's key.
    name: String,
    /// What [`Catalog::table`] clones for each binding.
    table: ParquetCatalogTable,
    /// The [`table_log`] version this entry reflects (`0` = a log-less legacy
    /// table, loaded by directory listing; its first commit seeds the log).
    version: u64,
    /// The version's files with their row groups, in log order.
    files: Vec<TableFileEntry>,
}

impl TableEntry {
    fn new(manifest: ManifestEntry, version: u64, files: Vec<TableFileEntry>) -> Self {
        let mut entry = Self {
            name: manifest.name,
            table: ParquetCatalogTable {
                columns: manifest.columns,
                location: manifest.location,
                parquet: Arc::new(ParquetTable::new(Vec::new())),
                predicates: Vec::new(),
            },
            version,
            files,
        };
        entry.reflatten();
        entry
    }

    /// Rebuild the binding template's scan view from the per-file state:
    /// concatenate every file's row groups in log order and (re)assign global
    /// indices. Row groups whose index already matches are shared, not cloned.
    fn reflatten(&mut self) {
        let rows = self
            .files
            .iter()
            .flat_map(|f| f.row_groups.iter())
            .enumerate()
            .map(|(global_idx, rg)| {
                if rg.global_row_group_idx == global_idx {
                    rg.clone()
                } else {
                    let mut renumbered = (**rg).clone();
                    renumbered.global_row_group_idx = global_idx;
                    Arc::new(renumbered)
                }
            })
            .collect();
        self.table.parquet = Arc::new(ParquetTable::new(rows));
    }

    /// The entry's file list in logged form — the base a commit builds on when
    /// the table has no log yet (version 0).
    fn logged_files(&self) -> Vec<LoggedFile> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }
}

/// Concurrent catalog of [`ParquetCatalogTable`]s, keyed by table name.
///
/// `CREATE TABLE` compiles to a single dataflow that reads every data file's
/// footer **once** (in parallel over the worker pool) and, at its terminal
/// stage, commits the table to the manifest, seeds its [`table_log`], and
/// publishes the entry in this shared map. After that, the table evolves by
/// log commits ([`register_data_file`](Self::register_data_file),
/// [`replace_data_files`](Self::replace_data_files)) and every query resolve
/// reloads the entry up to the latest committed version.
pub struct ParquetCatalog {
    tables: Arc<RwLock<HashMap<String, TableEntry>>>,
    /// The database's object store: an in-memory [`MemoryStore`] by default
    /// (ephemeral), or a local directory / S3 / GCS for one opened with
    /// [`open`](Self::open). It holds the table [`manifest`], the per-table
    /// [`table_log`]s, and the data of tables that live under the database
    /// root; `WITH (path = …)` tables read their own local directory directly.
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
    /// An ephemeral, in-memory database: tables vanish on restart and must
    /// each name a `WITH (path = …)`.
    pub fn new(dispatcher: DataFlowDispatcher) -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: Arc::new(MemoryStore::new()),
            dispatcher,
        }
    }

    /// Open a persisted database rooted at `uri` — a local directory (or
    /// `file://…`), or a remote `s3://…`/`gs://…` object store — reloading
    /// every table the manifest records at its latest [`table_log`] version.
    /// A table without a log (created before logs existed) falls back to
    /// listing its data location; its first commit seeds the log.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Self> {
        let catalog = Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: Arc::from(store::open_store(uri)?),
            dispatcher: dispatcher.clone(),
        };
        for manifest_entry in manifest::load(&*catalog.store)? {
            // Reload on the coordinator (open runs there, before serving).
            let (version, logged) = match table_log::latest(&*catalog.store, &manifest_entry.name)?
            {
                Some(current) => (current.version, current.files),
                None => (
                    0,
                    listed_files(&catalog.data_files(&manifest_entry.location)?),
                ),
            };
            let files = catalog.fetch_files(&manifest_entry.location, &logged)?;
            let entry = TableEntry::new(manifest_entry, version, files);
            catalog
                .tables
                .write()
                .unwrap()
                .insert(entry.name.clone(), entry);
        }
        Ok(catalog)
    }

    /// Resolve `name` to a fresh [`ParquetCatalogTable`] (same semantics as
    /// [`Catalog::table`] *minus the reload*, and typed). Useful for callers
    /// (and tests) that need state the [`Table`] trait does not expose.
    ///
    /// NOTE: Performance wise, cloning all the row groups for every new query resolve
    /// might be expensive for very large dataset (10/100tb+). If this ever becomes
    /// a bottleneck, we can consider using another mechanism.
    pub fn parquet_table(&self, name: &str) -> Option<ParquetCatalogTable> {
        self.tables
            .read()
            .unwrap()
            .get(name)
            .map(|entry| entry.table.clone())
    }

    /// Every table this catalog knows, by name. What a maintenance sweep (a
    /// compacter) iterates.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.read().unwrap().keys().cloned().collect()
    }

    /// The data files of `name`'s current in-memory version (logged name +
    /// size each). What a compacter scans for merge candidates.
    pub fn table_files(&self, name: &str) -> Option<Vec<LoggedFile>> {
        self.tables
            .read()
            .unwrap()
            .get(name)
            .map(TableEntry::logged_files)
    }

    /// Read/write access to `name`'s data location, wherever it lives — see
    /// [`TableData`]. `None` when no such table exists.
    pub fn table_data(&self, name: &str) -> Option<TableData> {
        let map = self.tables.read().unwrap();
        let location = &map.get(name)?.table.location;
        Some(if Path::new(location).is_absolute() {
            TableData {
                store: Arc::new(LocalStore::new(location)),
                prefix: String::new(),
            }
        } else {
            TableData {
                store: self.store.clone(),
                prefix: location.clone(),
            }
        })
    }

    /// Resolve logged files of table `name` into fetchable [`DataFile`]s (e.g.
    /// for a compaction read). `None` when no such table exists.
    pub fn resolve_data_files(
        &self,
        name: &str,
        files: &[LoggedFile],
    ) -> Result<Option<Vec<DataFile>>> {
        let Some(location) = self
            .tables
            .read()
            .unwrap()
            .get(name)
            .map(|e| e.table.location.clone())
        else {
            return Ok(None);
        };
        Ok(Some(self.resolve_files(&location, files)?))
    }

    /// The worker pool this catalog fetches footers on — shared with callers
    /// (the compacter) that drive their own dataflows over the same tables.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// Reload `name` up to its latest [`table_log`] version: one LIST; if the
    /// log is ahead of the in-memory entry, GET the version document, fetch
    /// footers for the files the entry doesn't have yet (over the dispatch
    /// pool, outside any lock), and swap the entry. Runs at every query bind,
    /// so a commit by ingest, a compacter, or another process is visible to
    /// the next query.
    pub fn refresh(&self, name: &str) -> Result<()> {
        let Some(latest_version) = table_log::latest_version(&*self.store, name)? else {
            return Ok(());
        };
        let (location, known) = {
            let map = self.tables.read().unwrap();
            let Some(entry) = map.get(name) else {
                return Ok(());
            };
            if entry.version >= latest_version {
                return Ok(());
            }
            let known: HashSet<String> = entry.files.iter().map(|f| f.file.name.clone()).collect();
            (entry.table.location.clone(), known)
        };
        let target = table_log::read(&*self.store, name, latest_version)?;
        let missing: Vec<LoggedFile> = target
            .files
            .iter()
            .filter(|f| !known.contains(&f.name))
            .cloned()
            .collect();
        let fetched = self.fetch_files(&location, &missing)?;
        self.apply_version(name, target, fetched);
        Ok(())
    }

    /// Register one newly-written local Parquet data file with table `name`:
    /// read its footer (over the dispatch pool), commit a new log version
    /// whose file list is `latest + this file` — retrying past CAS conflicts
    /// with other writers — and swap the new version into the in-memory entry.
    /// Registering a name the log already holds is a no-op, so a replayed
    /// notification can't double-count rows.
    ///
    /// The file must live in the table's data directory — and only a table
    /// over an absolute local directory can accept a *path* (a store-relative
    /// location is not where local files land); anything else is reported as
    /// [`RegisterOutcome::LocationMismatch`] and nothing is committed.
    pub fn register_data_file(&self, name: &str, path: &Path) -> Result<RegisterOutcome> {
        {
            let map = self.tables.read().unwrap();
            let Some(entry) = map.get(name) else {
                return Ok(RegisterOutcome::NoSuchTable);
            };
            if !file_in_table_dir(path, &entry.table.location) {
                return Ok(RegisterOutcome::LocationMismatch);
            }
        }
        let logged = LoggedFile {
            name: file_name_of(path),
            size: std::fs::metadata(path)?.len(),
        };
        let data_file = DataFile {
            name: logged.name.clone(),
            size: logged.size,
            source: DataFileSource::Local(path.to_path_buf()),
        };
        let row_groups = self.load_per_file(&[data_file])?;

        let (target, outcome) = self.commit_change(name, |files| {
            if files.iter().any(|f| f.name == logged.name) {
                return None;
            }
            files.push(logged.clone());
            Some(())
        })?;
        let fetched = vec![logged.name.clone()]
            .into_iter()
            .zip(row_groups)
            .map(|(name, groups)| TableFileEntry {
                file: logged_of(&target, &name),
                row_groups: groups,
            })
            .collect();
        self.apply_version(name, target, fetched);
        Ok(match outcome {
            Committed::NewVersion => RegisterOutcome::Registered,
            Committed::AlreadyApplied => RegisterOutcome::AlreadyRegistered,
        })
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
        added: &[LoggedFile],
    ) -> Result<bool> {
        let resolved = match self.resolve_data_files(name, added)? {
            Some(resolved) => resolved,
            None => return Ok(false),
        };
        let row_groups = self.load_per_file(&resolved)?;

        let (target, _) = self.commit_change(name, |files| {
            files.retain(|f| !removed.contains(&f.name));
            for add in added {
                if !files.iter().any(|f| f.name == add.name) {
                    files.push(add.clone());
                }
            }
            Some(())
        })?;
        let fetched = added
            .iter()
            .zip(row_groups)
            .map(|(file, groups)| TableFileEntry {
                file: file.clone(),
                row_groups: groups,
            })
            .collect();
        self.apply_version(name, target, fetched);
        Ok(true)
    }

    /// Commit one change to `name`'s log: read the latest version (or, for a
    /// log-less legacy table, the in-memory entry's files as the seed base),
    /// let `change` rewrite the file list, and CAS-commit it as the next
    /// version. A lost race re-reads and retries on top of the winner;
    /// `change` returning `None` means the latest list already reflects the
    /// change (nothing to commit).
    fn commit_change(
        &self,
        name: &str,
        mut change: impl FnMut(&mut Vec<LoggedFile>) -> Option<()>,
    ) -> Result<(TableVersion, Committed)> {
        loop {
            let latest = table_log::latest(&*self.store, name)?;
            let mut files = match &latest {
                Some(version) => version.files.clone(),
                None => self
                    .tables
                    .read()
                    .unwrap()
                    .get(name)
                    .map(TableEntry::logged_files)
                    .unwrap_or_default(),
            };
            if change(&mut files).is_none() {
                let version = latest.unwrap_or(TableVersion { version: 0, files });
                return Ok((version, Committed::AlreadyApplied));
            }
            let version = TableVersion::next_after(latest.as_ref());
            if table_log::commit(&*self.store, name, version, &files)? {
                return Ok((TableVersion { version, files }, Committed::NewVersion));
            }
            // Lost the CAS: someone committed this version first. Re-read and
            // retry the change on top of theirs.
        }
    }

    /// Bring `name`'s in-memory entry up to `target`, reusing the row groups
    /// of files the entry already has and taking new files' row groups from
    /// `fetched`. A no-op if the entry already reflects `target` (or newer).
    /// If a yet-newer concurrent version added files we didn't fetch, the
    /// entry is left as is — the next reload converges.
    fn apply_version(&self, name: &str, target: TableVersion, fetched: Vec<TableFileEntry>) {
        let mut map = self.tables.write().unwrap();
        let Some(entry) = map.get_mut(name) else {
            return;
        };
        if entry.version >= target.version {
            return;
        }
        let mut available: HashMap<&str, &Vec<Arc<RowGroupMetadata>>> = entry
            .files
            .iter()
            .map(|f| (f.file.name.as_str(), &f.row_groups))
            .collect();
        available.extend(
            fetched
                .iter()
                .map(|f| (f.file.name.as_str(), &f.row_groups)),
        );

        let mut files = Vec::with_capacity(target.files.len());
        for file in &target.files {
            let Some(row_groups) = available.get(file.name.as_str()) else {
                return;
            };
            files.push(TableFileEntry {
                file: file.clone(),
                row_groups: (*row_groups).clone(),
            });
        }
        entry.files = files;
        entry.version = target.version;
        entry.reflatten();
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

        let location = self.locate(&request)?;
        let files = self.data_files(&location)?;
        let logged = listed_files(&files);
        let manifest_entry = ManifestEntry {
            name: request.name,
            columns: request.columns,
            location,
        };

        // The commit runs on the dataflow's last worker once the footers are
        // fetched: record the table durably (manifest + seed log version), then
        // in the in-memory map (re-checking the name under the lock as a race
        // backstop).
        let tables = self.tables.clone();
        let store = self.store.clone();
        Ok(crate::parquet::create_load_and_commit_spec(
            dispatcher,
            &files,
            move |loaded| {
                let mut map = tables.write().unwrap();
                if map.contains_key(&manifest_entry.name) {
                    return Err(Box::new(Error::TableExists(manifest_entry.name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                // Seed the log with the files the listing found — the listing
                // is this create's truth, whatever any stale log says. Bump
                // past conflicting versions (an external writer) rather than
                // merging with them.
                let mut version =
                    table_log::latest_version(&*store, &manifest_entry.name)?.map_or(1, |v| v + 1);
                while !table_log::commit(&*store, &manifest_entry.name, version, &logged)? {
                    version += 1;
                }
                manifest::insert(&*store, &manifest_entry)?;

                let files = logged
                    .iter()
                    .cloned()
                    .zip(loaded.into_per_file())
                    .map(|(file, row_groups)| TableFileEntry {
                        file,
                        row_groups: row_groups.into_iter().map(Arc::new).collect(),
                    })
                    .collect();
                let entry = TableEntry::new(manifest_entry, version, files);
                map.insert(entry.name.clone(), entry);
                Ok(())
            },
        ))
    }

    /// The location stored for a new table: an explicit `path` (kept as given),
    /// or the table name under the database root.
    ///
    /// A table path is always a plain path, never a URL (no scheme) — where it
    /// physically lives is the database's storage, not the path's. An *absolute*
    /// path names an external directory on the server's local filesystem
    /// (whatever the database's store) and must already exist; a relative one
    /// lives in the store under the database root.
    fn locate(&self, request: &CreateTableRequest) -> Result<String> {
        let Some(path) = request.options.get(PATH_OPTION) else {
            return Ok(request.name.clone());
        };
        if is_url(path) {
            return Err(Error::TablePathWithScheme(path.clone()));
        }
        let dir = Path::new(path);
        if dir.is_absolute() {
            if !dir.exists() {
                return Err(Error::PathNotFound(path.clone()));
            }
            if !dir.is_dir() {
                return Err(Error::PathNotDirectory(path.clone()));
            }
        }
        Ok(path.clone())
    }

    /// List the Parquet data files at `location`, ready for the metadata
    /// fetch. An absolute path is an external local directory, read from the
    /// filesystem; a relative location lives under the database root, in the
    /// store (which yields a local path or presigned remote per object). An
    /// empty/absent location yields no files. Used where the *directory* is
    /// the source of truth: `CREATE TABLE`, and reloading a legacy table with
    /// no log.
    fn data_files(&self, location: &str) -> Result<Vec<DataFile>> {
        if Path::new(location).is_absolute() {
            return Ok(store::local_parquet_files(Path::new(location))?);
        }
        Ok(self
            .store
            .list(location)?
            .into_iter()
            .filter(|object| object.key.ends_with(".parquet"))
            .map(|object| self.store.data_file(&object.key, object.size))
            .collect::<crate::store::Result<Vec<_>>>()?)
    }

    /// Resolve logged files (name + size) at `location` into fetchable
    /// [`DataFile`]s. The mirror of [`data_files`](Self::data_files) for when
    /// the *log* is the source of truth.
    fn resolve_files(&self, location: &str, files: &[LoggedFile]) -> Result<Vec<DataFile>> {
        if Path::new(location).is_absolute() {
            return Ok(files
                .iter()
                .map(|f| DataFile {
                    name: f.name.clone(),
                    size: f.size,
                    source: DataFileSource::Local(Path::new(location).join(&f.name)),
                })
                .collect());
        }
        files
            .iter()
            .map(|f| {
                Ok(self
                    .store
                    .data_file(&join_location(location, &f.name), f.size)?)
            })
            .collect()
    }

    /// Fetch the footers of `files` at `location` (in parallel over the
    /// dispatch pool — a pipeline breaker, coordinator-only) into per-file
    /// entries.
    fn fetch_files(&self, location: &str, files: &[LoggedFile]) -> Result<Vec<TableFileEntry>> {
        let resolved = self.resolve_files(location, files)?;
        let row_groups = self.load_per_file(&resolved)?;
        Ok(files
            .iter()
            .zip(row_groups)
            .map(|(file, row_groups)| TableFileEntry {
                file: file.clone(),
                row_groups,
            })
            .collect())
    }

    /// Read `files`' footers over the dispatch pool, one `Vec` per input file.
    fn load_per_file(&self, files: &[DataFile]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let loaded = load(&self.dispatcher, files)
            .map_err(|e| ParquetTableError::Materialize(e.to_string()))?;
        Ok(loaded
            .into_per_file()
            .into_iter()
            .map(|groups| groups.into_iter().map(Arc::new).collect())
            .collect())
    }
}

/// Whether [`ParquetCatalog::commit_change`] wrote a new log version or found
/// the change already reflected in the latest one.
enum Committed {
    NewVersion,
    AlreadyApplied,
}

/// Read/write access to one table's data files, wherever the table's location
/// puts them: an absolute local directory is wrapped in a [`LocalStore`]
/// rooted at it, a store-relative location addresses the database's own store
/// under that prefix. Callers (the compacter) write and delete data files
/// through this one interface — local and remote are the same code path.
pub struct TableData {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

impl TableData {
    fn key(&self, name: &str) -> String {
        join_location(&self.prefix, name)
    }

    /// Write a data file (whole). On a local directory this lands via a temp
    /// file + rename; on a remote store a PUT is atomic per object. Either
    /// way a reader never sees a partial file — and an unfinished file is
    /// invisible regardless, because only log-committed names are read.
    pub fn put(&self, name: &str, bytes: &[u8]) -> store::Result<()> {
        self.store.put(&self.key(name), bytes)
    }

    /// Delete a data file (idempotent).
    pub fn delete(&self, name: &str) -> store::Result<()> {
        self.store.delete(&self.key(name))
    }
}

/// `dir/name` as a store key (or just `name` for an empty prefix).
fn join_location(location: &str, name: &str) -> String {
    if location.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", location.trim_end_matches('/'), name)
    }
}

/// The logged form of a directory listing — `CREATE TABLE`'s and the legacy
/// fallback's file list.
fn listed_files(files: &[DataFile]) -> Vec<LoggedFile> {
    files
        .iter()
        .map(|f| LoggedFile {
            name: f.name.clone(),
            size: f.size,
        })
        .collect()
}

/// A path's final component as a logged file name.
fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The logged entry for `name` within `version` (present by construction).
fn logged_of(version: &TableVersion, name: &str) -> LoggedFile {
    version
        .files
        .iter()
        .find(|f| f.name == name)
        .cloned()
        .unwrap_or_else(|| LoggedFile {
            name: name.to_string(),
            size: 0,
        })
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

/// Whether `s` carries a URL scheme (`s3://`, `gs://`, `file://`, …). A table
/// `path` never may — it must be a bare path within the database.
fn is_url(s: &str) -> bool {
    s.contains("://")
}

impl Catalog for ParquetCatalog {
    /// Resolve `name` to a fresh, independently-mutable [`ParquetCatalogTable`],
    /// **reloading first**: the table log is checked (one LIST) and any newer
    /// committed version is pulled in, so every query starts from the latest
    /// file list. Each binding gets its own clone so per-query filter pushdown
    /// can prune row groups without affecting the master entry or other
    /// concurrent queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        if let Err(e) = self.refresh(name) {
            // Serve the version we have rather than failing the query; the
            // next bind retries the reload.
            warn!(table = name, error = %e, "table reload failed; serving last known version");
        }
        self.parquet_table(name).map(|t| Box::new(t) as _)
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }
}

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile`](Table::compile)
/// time after the row-group metadata exists — min/max stats prune row groups,
/// and equality additionally prunes by dictionary contents in the decoder. The
/// upstream `Filter` always runs, so this is a pure optimization.
#[derive(Clone, Debug)]
struct PushedPredicate {
    column_idx: usize,
    compare_type: CompareType,
    value: Scalar<ArrayRef>,
}

/// A catalog table backed by a directory of Parquet files. Cloned per-binding
/// so each query accumulates its own pushed-down predicates via
/// [`Table::pushdown_filter`] without affecting others.
#[derive(Clone, Debug)]
pub struct ParquetCatalogTable {
    pub columns: Vec<Column>,
    /// Where this table's Parquet data lives (a directory/key prefix in the
    /// database store). Kept so a future `refresh()` can re-read the latest files.
    pub location: String,
    /// The table's row-group metadata, read once when the table was defined and
    /// shared across every binding/query (cheap `Arc` clone). Pruning a binding's
    /// predicates clones the row-group `Vec` and filters it — no footer re-read.
    pub parquet: Arc<ParquetTable>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl Table for ParquetCatalogTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Equality predicates additionally let the decoder skip row groups whose
        // dictionary for that column excludes the constant.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                value: p.value.clone(),
            })
            .collect();

        // Prune the (already-materialized) row groups by the pushed-down
        // predicates' stats. No footer re-read — the row groups were built once
        // when the table was defined.
        let parquet = Arc::new(self.pruned_parquet());
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> RecordBatchOperatorSpec {
        // Late materialization re-reads rows by their *global* row-group index,
        // so it uses the full table, not the pruned scan view.
        materialize(input, self.parquet.clone(), projection)
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        let TableFilter::Expression(expr) = filter else {
            return Ok(false);
        };
        let Expression::Compare(compare) = expr.as_ref() else {
            return Ok(false);
        };
        let (reference, constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (Expression::Ref(r), Expression::Constant(k))
            | (Expression::Constant(k), Expression::Ref(r)) => (r, k),
            _ => return Ok(false),
        };

        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates.push(PushedPredicate {
            column_idx: reference.column_idx,
            compare_type: compare.compare_type,
            value: constant.clone(),
        });

        Ok(false)
    }
}

impl ParquetCatalogTable {
    /// Clone this table's row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`Table::compile`] actually
    /// scans. A min/max stat that proves no row in a group can match drops it; a
    /// stats-comparison error means "can't prune" (kept) — never wrong, just
    /// unoptimized. No footer I/O: the row groups were materialized once when the
    /// table was defined. Exposed so pruning can be asserted directly.
    pub fn pruned_parquet(&self) -> ParquetTable {
        let mut parquet = (*self.parquet).clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        parquet
    }
}
