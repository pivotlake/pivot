//! The catalog's master record of one table: its definition plus its current
//! content — the files at one manifest version.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::catalog::TableBinding;
use crate::manifest::{FIRST_VERSION, TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectStore, location_key};
use dispatch::DataFlowDispatcher;
use planner::catalog::Column;
use crate::{Error, RegisterOutcome};

/// One data file of a table: its identity ([`FileRef`]) paired with its
/// materialized row groups (file-local order).
#[derive(Clone)]
pub struct TableFile {
    pub(super) file: FileRef,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's identity with the row groups read from its footer. Built by
    /// the metadata fetcher (one per file) and the catalog's table assembly.
    pub(crate) fn new(file: FileRef, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        Self { file, row_groups }
    }

    /// This file's row groups, in file-local order.
    pub(crate) fn row_groups(&self) -> &[Arc<RowGroupMetadata>] {
        &self.row_groups
    }
}

/// The catalog's master record of one table: its [`TableManifest`] (declared
/// columns + committed file list, at one version) and its current content — the
/// per-file row groups (`files`). The version lives in the manifest, not
/// alongside it.
///
/// It is a plain **value** — `Clone`, no interior locks. A writer (ingest,
/// compaction) clones one out of the catalog, mutates its own copy, and lets the
/// durable manifest be the source of truth: every mutation
/// ([`register_data_file`](Self::register_data_file),
/// [`replace_data_files`](Self::replace_data_files)) commits a new manifest
/// version by compare-and-swap, retrying past a concurrent writer. Copies drift
/// freely; [`refresh`](Self::refresh) reconciles any copy to the latest version
/// (re-reading only the footers it doesn't already hold). `store`, `name`, and
/// `location` are kept so a copy can persist and reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    name: String,
    /// Where the table's Parquet data lives (a store location — relative to the
    /// database root, or an absolute store path).
    location: String,
    pub(super) manifest: TableManifest,
    pub(super) files: Vec<TableFile>,
    store: Arc<dyn ObjectStore>,
}

impl CatalogTable {
    /// Reassemble a persisted table from its loaded `manifest` and the per-file
    /// row groups (`files`) just fetched for it — the reopen path. Does not
    /// persist anything; the manifest it was loaded from is already durable.
    pub(super) fn new(
        name: String,
        location: String,
        manifest: TableManifest,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            name,
            location,
            manifest,
            files,
            store,
        }
    }

    /// Create a brand-new table from freshly-read footers: build its
    /// [`TableManifest`] (declared `columns` + the loaded files, at
    /// [`FIRST_VERSION`]) and commit it. The compare-and-swap fails with
    /// [`Error::TableExists`] if another writer already committed this table's
    /// first version. The `CREATE TABLE` commit path.
    pub(super) fn create_new(
        name: String,
        location: String,
        files: Vec<TableFile>,
        columns: Vec<Column>,
        store: Arc<dyn ObjectStore>,
    ) -> crate::Result<Self> {
        let entries = files.iter().map(|f| f.file.clone()).collect();
        let manifest = TableManifest {
            version: FIRST_VERSION,
            columns,
            entries,
        };
        if !manifest.commit(store.as_ref(), &name)? {
            return Err(Error::TableExists(name));
        }
        Ok(Self {
            name,
            location,
            manifest,
            files,
            store,
        })
    }

    /// Reload this copy to the latest committed version, if any writer has moved
    /// past the version it holds. Cheap when current (one `list` to check the
    /// latest version); when behind, loads the latest manifest and fetches only
    /// the footers of files it doesn't already hold. Returns whether it advanced.
    pub fn refresh(&mut self, dispatcher: &DataFlowDispatcher) -> crate::Result<bool> {
        let latest = match TableManifest::latest_version(self.store.as_ref(), &self.name)? {
            Some(version) if version > self.manifest.version => version,
            _ => return Ok(false),
        };
        let manifest = TableManifest::load_version(self.store.as_ref(), &self.name, latest)?;
        self.files = self.assemble_files(dispatcher, &manifest)?;
        self.manifest = manifest;
        Ok(true)
    }

    /// Register one newly-written local Parquet data file: commit a new manifest
    /// version whose file list is `latest + this file`, retrying past concurrent
    /// commits. Registering a name the manifest already holds is a no-op
    /// ([`RegisterOutcome::AlreadyRegistered`]), so a replayed notification can't
    /// double-count rows.
    ///
    /// The file must sit directly in the table's data directory, which only an
    /// absolute local location can be — a store-relative location is not where a
    /// locally-written file lands; otherwise [`RegisterOutcome::LocationMismatch`]
    /// and nothing is committed.
    pub fn register_data_file(
        &mut self,
        dispatcher: &DataFlowDispatcher,
        path: &Path,
    ) -> crate::Result<RegisterOutcome> {
        if !super::file_in_table_dir(path, &self.location) {
            return Ok(RegisterOutcome::LocationMismatch);
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return Ok(RegisterOutcome::LocationMismatch);
        };
        let new_file = FileRef {
            name: name.to_string(),
            size: std::fs::metadata(path)?.len(),
        };

        loop {
            if self.manifest.entries.iter().any(|e| e.name == new_file.name) {
                return Ok(RegisterOutcome::AlreadyRegistered);
            }
            let mut entries = self.manifest.entries.clone();
            entries.push(new_file.clone());
            if self.try_commit(dispatcher, entries)? {
                return Ok(RegisterOutcome::Registered);
            }
            // Lost the race — rebase onto the winner's version and retry.
            self.refresh(dispatcher)?;
        }
    }

    /// Atomically swap a set of this table's files for another — the compaction
    /// commit. Commits a new version whose file list is `latest − removed +
    /// added`, retrying past concurrent commits, so no reader ever sees the rows
    /// doubled or missing.
    pub fn replace_data_files(
        &mut self,
        dispatcher: &DataFlowDispatcher,
        removed: &[String],
        added: &[FileRef],
    ) -> crate::Result<()> {
        loop {
            let mut entries: Vec<FileRef> = self
                .manifest
                .entries
                .iter()
                .filter(|e| !removed.contains(&e.name))
                .cloned()
                .collect();
            entries.extend(added.iter().cloned());
            if self.try_commit(dispatcher, entries)? {
                return Ok(());
            }
            self.refresh(dispatcher)?;
        }
    }

    /// Try to commit `entries` as the next version via compare-and-swap. On
    /// success, swaps this copy's manifest + files (fetching only the new
    /// footers) and returns `true`; on a CAS conflict returns `false` without
    /// touching this copy.
    fn try_commit(
        &mut self,
        dispatcher: &DataFlowDispatcher,
        entries: Vec<FileRef>,
    ) -> crate::Result<bool> {
        let manifest = TableManifest {
            version: self.manifest.version + 1,
            columns: self.manifest.columns.clone(),
            entries,
        };
        if !manifest.commit(self.store.as_ref(), &self.name)? {
            return Ok(false);
        }
        self.files = self.assemble_files(dispatcher, &manifest)?;
        self.manifest = manifest;
        Ok(true)
    }

    /// The [`TableFile`]s for `manifest`'s entries, in entry order: reuse the
    /// footers this copy already holds and fetch only the rest over the pool.
    fn assemble_files(
        &self,
        dispatcher: &DataFlowDispatcher,
        manifest: &TableManifest,
    ) -> crate::Result<Vec<TableFile>> {
        let held: HashMap<&str, &TableFile> =
            self.files.iter().map(|f| (f.file.name.as_str(), f)).collect();
        let to_fetch: Vec<DataFile> = manifest
            .entries
            .iter()
            .filter(|e| !held.contains_key(e.name.as_str()))
            .map(|e| self.store.data_file(&location_key(&self.location, &e.name), e.size))
            .collect::<store::Result<_>>()?;
        let mut fetched: HashMap<String, TableFile> = crate::parquet::load_table_files(dispatcher, &to_fetch)?
            .into_iter()
            .map(|f| (f.file.name.clone(), f))
            .collect();

        manifest
            .entries
            .iter()
            .map(|e| {
                held.get(e.name.as_str())
                    .map(|f| (*f).clone())
                    .or_else(|| fetched.remove(&e.name))
                    .ok_or_else(|| Error::MissingFile(e.name.clone()))
            })
            .collect()
    }

    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// A fresh per-query [`TableBinding`] over a flat scan view of the current
    /// files: every file's row groups concatenated in manifest order, where a row
    /// group's global index is simply its position. Derived here (cheap `Arc`
    /// clones) rather than cached, so the view can never disagree with `files`.
    pub(super) fn binding(&self) -> TableBinding {
        let row_groups = self
            .files
            .iter()
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        TableBinding::new(self.manifest.columns.clone(), Arc::new(ParquetTable::new(row_groups)))
    }
}
