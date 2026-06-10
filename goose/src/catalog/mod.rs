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
//! Each table's in-memory [`entry`] pairs its definition with an immutable
//! [`entry::TableState`] — the per-file row groups at one log version, with
//! the flattened scan view derived at construction (the two cannot disagree;
//! a change swaps in a whole new state). Resolving a table for a query
//! ([`Catalog::table`]) first **reloads**: one LIST of the table's log
//! directory; if a newer version exists, only the *new* files' footers are
//! fetched (over the dispatch pool) and a new state is swapped in — so a file
//! registered by ingest, a compaction's swap, or even another process's
//! commit becomes visible to the very next query.
//!
//! Each resolve hands back a fresh [`TableBinding`], so per-query filter
//! pushdown accumulates on that binding alone.

mod binding;
mod data;
mod entry;

pub use binding::TableBinding;
pub use data::TableStore;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::manifest::{self, ManifestEntry};
use crate::parquet::{LoadedFiles, ParquetTableError, RowGroupMetadata};
use crate::store::{
    self, DataFile, DataFileSource, LocalStore, MemoryStore, ObjectStore, join_prefix,
};
use crate::table_log::{self, FIRST_VERSION, LoggedFile, TableLog, TableVersion};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use entry::{TableEntry, TableFile, TableState};
use planner::catalog::{
    Catalog, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use thiserror::Error as ThisError;
use tracing::warn;

const PATH_OPTION: &str = "path";

#[derive(Debug, ThisError)]
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

/// Row groups fetched for a version swap, keyed by their file's logged name —
/// what [`ParquetCatalog::apply_version`] consumes for files the entry doesn't
/// already hold.
type FetchedFiles = HashMap<String, Vec<Arc<RowGroupMetadata>>>;

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
            let (version, logged) =
                match TableLog::new(&*catalog.store, &manifest_entry.name).read_latest()? {
                    Some(current) => (current.version, current.files),
                    None => {
                        let listed = catalog.data_files(&manifest_entry.location)?;
                        (0, listed.iter().map(LoggedFile::from).collect())
                    }
                };
            let files = catalog.fetch_files(&manifest_entry.location, &logged)?;
            let entry = TableEntry::new(manifest_entry, TableState::new(version, files));
            catalog
                .tables
                .write()
                .unwrap()
                .insert(entry.name.clone(), entry);
        }
        Ok(catalog)
    }

    /// Resolve `name` to a fresh [`TableBinding`] (same semantics as
    /// [`Catalog::table`] *minus the reload*, and typed). Useful for callers
    /// (and tests) that need state the [`Table`] trait does not expose.
    pub fn binding(&self, name: &str) -> Option<TableBinding> {
        self.tables
            .read()
            .unwrap()
            .get(name)
            .map(TableEntry::binding)
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
            .map(|entry| entry.state.logged_files())
    }

    /// Write access to `name`'s data location, wherever it lives — see
    /// [`TableStore`]. `None` when no such table exists.
    pub fn table_store(&self, name: &str) -> Option<TableStore> {
        let map = self.tables.read().unwrap();
        let location = &map.get(name)?.location;
        Some(if Path::new(location).is_absolute() {
            TableStore {
                store: Arc::new(LocalStore::new(location)),
                prefix: String::new(),
            }
        } else {
            TableStore {
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
            .map(|e| e.location.clone())
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
        let log = TableLog::new(&*self.store, name);
        let Some(latest_version) = log.latest_version()? else {
            return Ok(());
        };
        let (location, known) = {
            let map = self.tables.read().unwrap();
            let Some(entry) = map.get(name) else {
                return Ok(());
            };
            if entry.state.version() >= latest_version {
                return Ok(());
            }
            let known: HashSet<String> = entry
                .state
                .files()
                .iter()
                .map(|f| f.logged.name.clone())
                .collect();
            (entry.location.clone(), known)
        };
        let target = log.read(latest_version)?;
        let missing: Vec<LoggedFile> = target
            .files
            .iter()
            .filter(|f| !known.contains(&f.name))
            .cloned()
            .collect();
        let resolved = self.resolve_files(&location, &missing)?;
        let fetched = missing
            .iter()
            .map(|f| f.name.clone())
            .zip(self.load_per_file(&resolved)?)
            .collect();
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
            if !file_in_table_dir(path, &entry.location) {
                return Ok(RegisterOutcome::LocationMismatch);
            }
        }
        let logged = LoggedFile {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            size: std::fs::metadata(path)?.len(),
        };
        let data_file = DataFile {
            name: logged.name.clone(),
            size: logged.size,
            source: DataFileSource::Local(path.to_path_buf()),
        };
        let row_groups = self.load_per_file(&[data_file])?.pop().unwrap_or_default();

        let (target, committed) = self.commit_change(name, |files| {
            if files.iter().any(|f| f.name == logged.name) {
                return None;
            }
            files.push(logged.clone());
            Some(())
        })?;
        let fetched = FetchedFiles::from([(logged.name, row_groups)]);
        self.apply_version(name, target, fetched);
        Ok(match committed {
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
            .map(|f| f.name.clone())
            .zip(row_groups)
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
        let log = TableLog::new(&*self.store, name);
        loop {
            let latest = log.read_latest()?;
            let mut files = match &latest {
                Some(version) => version.files.clone(),
                None => self
                    .tables
                    .read()
                    .unwrap()
                    .get(name)
                    .map(|entry| entry.state.logged_files())
                    .unwrap_or_default(),
            };
            if change(&mut files).is_none() {
                let version = latest.unwrap_or(TableVersion { version: 0, files });
                return Ok((version, Committed::AlreadyApplied));
            }
            let version = latest.as_ref().map_or(FIRST_VERSION, TableVersion::next);
            if log.commit(version, &files)? {
                return Ok((TableVersion { version, files }, Committed::NewVersion));
            }
            // Lost the CAS: someone committed this version first. Re-read and
            // retry the change on top of theirs.
        }
    }

    /// Bring `name`'s in-memory entry up to `target`: build the new
    /// [`TableState`], reusing the row groups of files the current state
    /// already has and taking new files' row groups from `fetched`, then swap
    /// it in. A no-op if the entry already reflects `target` (or newer). If a
    /// yet-newer concurrent version added files we didn't fetch, the entry is
    /// left as is — the next reload converges.
    fn apply_version(&self, name: &str, target: TableVersion, fetched: FetchedFiles) {
        let mut map = self.tables.write().unwrap();
        let Some(entry) = map.get_mut(name) else {
            return;
        };
        if entry.state.version() >= target.version {
            return;
        }
        let mut available: HashMap<&str, &Vec<Arc<RowGroupMetadata>>> = entry
            .state
            .files()
            .iter()
            .map(|f| (f.logged.name.as_str(), &f.row_groups))
            .collect();
        available.extend(fetched.iter().map(|(name, groups)| (name.as_str(), groups)));

        let mut files = Vec::with_capacity(target.files.len());
        for file in &target.files {
            let Some(row_groups) = available.get(file.name.as_str()) else {
                return;
            };
            files.push(TableFile {
                logged: file.clone(),
                row_groups: (*row_groups).clone(),
            });
        }
        entry.state = TableState::new(target.version, files);
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
        let logged: Vec<LoggedFile> = files.iter().map(LoggedFile::from).collect();
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
            move |loaded: LoadedFiles| {
                let mut map = tables.write().unwrap();
                if map.contains_key(&manifest_entry.name) {
                    return Err(Box::new(Error::TableExists(manifest_entry.name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                // Seed the log with the files the listing found — the listing
                // is this create's truth, whatever any stale log says. Bump
                // past conflicting versions (an external writer) rather than
                // merging with them.
                let log = TableLog::new(&*store, &manifest_entry.name);
                let mut version = log.latest_version()?.map_or(FIRST_VERSION, |v| v + 1);
                while !log.commit(version, &logged)? {
                    version += 1;
                }
                manifest::insert(&*store, &manifest_entry)?;

                let files = logged
                    .iter()
                    .cloned()
                    .zip(loaded.into_per_file())
                    .map(|(logged, row_groups)| TableFile {
                        logged,
                        row_groups: row_groups.into_iter().map(Arc::new).collect(),
                    })
                    .collect();
                let entry = TableEntry::new(manifest_entry, TableState::new(version, files));
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
        if path.contains("://") {
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
                    .data_file(&join_prefix(location, &f.name), f.size)?)
            })
            .collect()
    }

    /// Fetch the footers of `files` at `location` (in parallel over the
    /// dispatch pool — a pipeline breaker, coordinator-only) into per-file
    /// state, in logged order.
    fn fetch_files(&self, location: &str, files: &[LoggedFile]) -> Result<Vec<TableFile>> {
        let resolved = self.resolve_files(location, files)?;
        let row_groups = self.load_per_file(&resolved)?;
        Ok(files
            .iter()
            .zip(row_groups)
            .map(|(logged, row_groups)| TableFile {
                logged: logged.clone(),
                row_groups,
            })
            .collect())
    }

    /// Read `files`' footers over the dispatch pool, one `Vec` per input file.
    fn load_per_file(&self, files: &[DataFile]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let loaded = LoadedFiles::load(&self.dispatcher, files)
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
        if let Err(e) = self.refresh(name) {
            // Serve the version we have rather than failing the query; the
            // next bind retries the reload.
            warn!(table = name, error = %e, "table reload failed; serving last known version");
        }
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
