//! The catalog's master record of one table: its definition plus its current
//! content — the files at one manifest version.

use std::collections::HashSet;
use std::sync::Arc;

use crate::catalog::TableBinding;
use crate::manifest::{FIRST_VERSION, TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use dispatch::DataFlowDispatcher;
use planner::catalog::Column;
use crate::Error;

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
/// ([`append_data_file`](Self::append_data_file),
/// [`replace_data_files`](Self::replace_data_files)) commits a new manifest
/// version by compare-and-swap, retrying past a concurrent writer. Copies drift
/// freely; [`refresh`](Self::refresh) reconciles any copy to the latest version
/// (re-reading only the footers it doesn't already hold). `store`, `name`, and
/// `location` are kept so a copy can persist and reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    name: String,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    pub(super) manifest: TableManifest,
    pub(super) files: Vec<TableFile>,
    store: Arc<dyn ObjectStore>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
}

impl CatalogTable {
    /// Reassemble a persisted table from its loaded `manifest` and the per-file
    /// row groups (`files`) just fetched for it — the reopen path. Does not
    /// persist anything; the manifest it was loaded from is already durable.
    pub(super) fn new(
        name: String,
        location: ObjectPath,
        manifest: TableManifest,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            name,
            location,
            manifest,
            files,
            store,
            dispatcher,
        }
    }

    /// Create a brand-new table from freshly-read footers: build its
    /// [`TableManifest`] (declared `columns` + the loaded files, at
    /// [`FIRST_VERSION`]) and commit it. The compare-and-swap fails with
    /// [`Error::TableExists`] if another writer already committed this table's
    /// first version. The `CREATE TABLE` commit path.
    pub(super) fn create_new(
        name: String,
        location: ObjectPath,
        files: Vec<TableFile>,
        columns: Vec<Column>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
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
            dispatcher,
        })
    }

    /// Reload this copy to the latest committed version. Errors if the table has
    /// no manifest at all (a corrupt catalog). Returns whether it advanced;
    /// `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let manifest = TableManifest::load(self.store.as_ref(), &self.name)?;
        if manifest.version <= self.manifest.version {
            return Ok(false);
        }
        self.manifest = manifest;
        self.sync_files_to_manifest()?;
        Ok(true)
    }

    /// Write `bytes` as a new data file at `path` (under the table's location)
    /// and commit it into the table — the ingest sink's append, one call for the
    /// write and the manifest commit. Idempotent on `path`.
    pub fn append_data_file(&mut self, path: ObjectPath, bytes: &[u8]) -> crate::Result<()> {
        let file = self.write_data_file(path, bytes)?;
        self.commit_added_file(file)
    }

    /// CAS-commit one already-written file into the manifest, retrying past a
    /// concurrent writer (refresh + retry). A file the manifest already holds is
    /// an idempotent no-op, so a replayed append can't double-count rows.
    fn commit_added_file(&mut self, file: FileRef) -> crate::Result<()> {
        loop {
            if self.manifest.entries.iter().any(|e| e.path == file.path) {
                return Ok(());
            }
            let mut entries = self.manifest.entries.clone();
            entries.push(file.clone());
            if self.try_commit(entries)? {
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// Atomically swap a set of this table's files for another — the compaction
    /// commit. Commits a new version whose file list is `latest − removed +
    /// added`, retrying past concurrent commits, so no reader ever sees the rows
    /// doubled or missing.
    ///
    /// Returns whether the swap was committed. It is **aborted** (`Ok(false)`,
    /// nothing committed) if any of `removed` is no longer in the latest
    /// version: that means another writer already swapped these inputs out, so
    /// re-adding `added` (which holds their rows) would double-count. A
    /// concurrent compacter that loses this race must discard its `added` files
    /// as orphans and leave the inputs alone — they belong to the swap that won.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileRef],
    ) -> crate::Result<bool> {
        loop {
            if !removed
                .iter()
                .all(|p| self.manifest.entries.iter().any(|e| &e.path == p))
            {
                return Ok(false);
            }
            let mut entries: Vec<FileRef> = self
                .manifest
                .entries
                .iter()
                .filter(|e| !removed.contains(&e.path))
                .cloned()
                .collect();
            entries.extend(added.iter().cloned());
            if self.try_commit(entries)? {
                return Ok(true);
            }
            self.refresh()?;
        }
    }

    /// Try to commit `entries` as the next version via compare-and-swap. On
    /// success, swaps in the new manifest + files (fetching only the new footers)
    /// and returns `true`; on a CAS conflict returns `false` without touching
    /// this copy.
    fn try_commit(&mut self, entries: Vec<FileRef>) -> crate::Result<bool> {
        let manifest = TableManifest {
            version: self.manifest.version + 1,
            columns: self.manifest.columns.clone(),
            entries,
        };
        if !manifest.commit(self.store.as_ref(), &self.name)? {
            return Ok(false);
        }
        self.manifest = manifest;
        self.sync_files_to_manifest()?;
        // GC superseded versions. Best-effort: the commit already stands, so a
        // prune failure is logged, not propagated.
        if let Err(e) =
            TableManifest::prune_old_versions(self.store.as_ref(), &self.name, self.manifest.version)
        {
            tracing::warn!(table = %self.name, error = %e, "pruning old manifest versions failed");
        }
        Ok(true)
    }

    /// Reconcile `files` to the current `manifest`: drop the files no longer in
    /// it, then fetch and append the ones not yet held.
    fn sync_files_to_manifest(&mut self) -> crate::Result<()> {
        let kept: HashSet<&str> = self.manifest.entries.iter().map(|e| e.path.as_str()).collect();
        self.files.retain(|f| kept.contains(f.file.path.as_str()));
        let missing = self.retrieve_missing_table_files()?;
        self.files.extend(missing);
        Ok(())
    }

    /// Fetch the [`TableFile`]s for this table's manifest entries that aren't
    /// already held — the footers this copy is missing (disjoint from `files`).
    fn retrieve_missing_table_files(&self) -> crate::Result<Vec<TableFile>> {
        let to_fetch: Vec<DataFile> = self
            .manifest
            .entries
            .iter()
            .filter(|e| !self.files.iter().any(|f| f.file.path == e.path))
            .map(|e| e.clone().into_data_file(self.store.as_ref(), &self.location))
            .collect::<store::Result<_>>()?;
        Ok(crate::parquet::load_table_files(&self.dispatcher, &to_fetch)?)
    }

    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a [`replace_data_files`](Self::replace_data_files) swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }

    /// A scannable [`ParquetTable`] over just the `wanted` files (matched by
    /// path) — the compaction read view: feed it to
    /// [`table_input`](crate::parquet::table_input) to decode their rows.
    pub fn parquet_table_for(&self, wanted: &[FileRef]) -> Arc<ParquetTable> {
        let want: HashSet<&ObjectPath> = wanted.iter().map(|f| &f.path).collect();
        let row_groups = self
            .files
            .iter()
            .filter(|f| want.contains(&f.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    /// Write `bytes` as a new data file at `path` (resolved against the table's
    /// location like any [`FileRef`] path), returning its [`FileRef`]. The
    /// compaction writer's output, committed with
    /// [`replace_data_files`](Self::replace_data_files).
    pub fn write_data_file(&self, path: ObjectPath, bytes: &[u8]) -> crate::Result<FileRef> {
        self.store.put(&self.location.resolve(&path), bytes)?;
        Ok(FileRef {
            path,
            size: bytes.len() as u64,
        })
    }

    /// Delete a data file (a compaction input swapped out of the manifest).
    /// `path` resolves against the table's location like any [`FileRef`] path.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        self.store.delete(&self.location.resolve(path))?;
        Ok(())
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
