//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::manifest::{ManifestEntry, PartitionEqFilter, SortBounds, TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use arrow_array::{ArrayRef, Scalar};
use dispatch::DataFlowDispatcher;
use planner::catalog::Column;

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
    /// The Delta table root loaded by the catalog sync.
    delta_uri: url::Url,
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
        delta_uri: url::Url,
    ) -> Self {
        Self {
            name,
            location,
            manifest,
            files,
            store,
            dispatcher,
            delta_uri,
        }
    }

    /// Create a brand-new table from freshly-read footers by atomically writing
    /// Delta version 0 with its protocol, metadata, and initial `Add` actions.
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn create_new(
        name: String,
        location: ObjectPath,
        files: Vec<TableFile>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
    ) -> crate::Result<Self> {
        // Files discovered at CREATE TABLE carry no partition metadata (opaque
        // paths, footers not parsed for it); the partitioning sink records it on
        // the files it appends later.
        let entries = files
            .iter()
            .map(|f| ManifestEntry::new(f.file.clone()))
            .collect();
        let file_refs = files
            .iter()
            .map(|file| file.file.clone())
            .collect::<Vec<_>>();
        let delta_uri = crate::delta::initialize_table(
            store.as_ref(),
            &location,
            &columns,
            &partition_by,
            &file_refs,
        )?;
        let manifest = TableManifest {
            version: 0,
            columns,
            partition_by,
            sort_by,
            entries,
        };
        Ok(Self {
            name,
            location,
            manifest,
            files,
            store,
            dispatcher,
            delta_uri,
        })
    }

    /// Reload this copy through Delta Kernel at the latest committed version.
    /// Returns whether it advanced;
    /// `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let state = crate::delta::load_table(&self.delta_uri)?;
        if state.version <= self.manifest.version {
            return Ok(false);
        }
        self.manifest = TableManifest {
            version: state.version,
            columns: state.columns,
            partition_by: state.partition_by,
            sort_by: Vec::new(),
            entries: state.entries,
        };
        self.sync_files_to_manifest()?;
        Ok(true)
    }

    /// Write `bytes` as a new data file at `path` (under the table's location)
    /// and commit it into the table — the ingest sink's append, one call for the
    /// write and the manifest commit. Idempotent on `path`.
    pub fn append_data_file(
        &mut self,
        _path: ObjectPath,
        _bytes: &[u8],
        _partition: Option<HashMap<String, Scalar<ArrayRef>>>,
        _sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        Err(Error::DeltaWritesUnsupported(self.name.clone()))
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
        _removed: &[ObjectPath],
        _added: &[ManifestEntry],
    ) -> crate::Result<bool> {
        Err(Error::DeltaWritesUnsupported(self.name.clone()))
    }

    /// Garbage-collect this table's superseded manifest versions, keeping only a
    /// recent tail (see [`TableManifest::prune_old_versions`]). Each commit
    /// leaves a new version file behind; this reclaims the old ones so a busy
    /// table's version directory stays small and the `list` every bind does
    /// stays cheap. The compacter drives it — it's the background sweep that
    /// already polls every table.
    pub fn prune_old_versions(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Record the just-swapped-out `removed` inputs (paths as they appear in the
    /// manifest) for *deferred* deletion at this table's current manifest version.
    /// Their objects are deleted only when that version is pruned
    /// ([`prune_old_versions`](Self::prune_old_versions)) — so a query still
    /// reading the prior version never has a file deleted out from under it.
    pub fn record_deletions(&self, _removed: &[ObjectPath]) -> crate::Result<()> {
        Err(Error::DeltaWritesUnsupported(self.name.clone()))
    }

    /// Reconcile `files` to the current `manifest`: drop the files no longer in
    /// it, then fetch and append the ones not yet held.
    fn sync_files_to_manifest(&mut self) -> crate::Result<()> {
        let kept: HashSet<&str> = self
            .manifest
            .entries
            .iter()
            .map(|e| e.file.path.as_str())
            .collect();
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
            .filter(|e| !self.files.iter().any(|f| f.file.path == e.file.path))
            .map(|e| {
                e.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &self.location)
            })
            .collect::<store::Result<_>>()?;
        Ok(crate::parquet::load_table_files(
            &self.dispatcher,
            &to_fetch,
        )?)
    }

    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        &self.manifest.partition_by
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        &self.manifest.sort_by
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a [`replace_data_files`](Self::replace_data_files) swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`). Reads the manifest, so it reflects the current committed
    /// version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<HashMap<String, Scalar<ArrayRef>>>)> {
        self.manifest
            .entries
            .iter()
            .map(|e| (e.file.path.clone(), e.partition.clone()))
            .collect()
    }

    /// Each committed file paired with its recorded sort-key bounds (the
    /// `sort_by` columns at the file's first/last row, or `None`).
    pub fn file_sort_bounds(&self) -> Vec<(ObjectPath, Option<SortBounds>)> {
        self.manifest
            .entries
            .iter()
            .map(|e| (e.file.path.clone(), e.sort_bounds.clone()))
            .collect()
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
    pub fn write_data_file(&self, _path: ObjectPath, _bytes: &[u8]) -> crate::Result<FileRef> {
        Err(Error::DeltaWritesUnsupported(self.name.clone()))
    }

    /// Delete a data file (a compaction input swapped out of the manifest).
    /// `path` resolves against the table's location like any [`FileRef`] path.
    pub fn delete_data_file(&self, _path: &ObjectPath) -> crate::Result<()> {
        Err(Error::DeltaWritesUnsupported(self.name.clone()))
    }

    /// The manifest version this copy is at. Monotonic per table; used to
    /// decide whether a published copy is newer than the catalog's.
    pub fn version(&self) -> u64 {
        self.manifest.version
    }

    /// A flat scan view of the files whose recorded partition tuple can still
    /// match `filters` — every surviving file's row groups concatenated in
    /// manifest order, where a row group's global index is simply its position.
    /// An empty `filters`, or one naming no partition column, keeps every file.
    ///
    /// Read-only: it is built entirely from the row groups this copy already
    /// holds, so it does **no** I/O. Every surviving file's footer must have
    /// been fetched (the refresh path keeps `files` synced to the manifest); a
    /// missing one is an error, never a silently narrower scan.
    ///
    /// The row group's global index is its position in this returned flat list, so
    /// a scan and its materialize must build it from the *same* `filters` (they
    /// do: both go through the binding's predicates) to address the same groups.
    pub fn build_scan_view(
        &self,
        filters: &[PartitionEqFilter],
    ) -> crate::Result<Arc<ParquetTable>> {
        let by_path: HashMap<&ObjectPath, &TableFile> =
            self.files.iter().map(|f| (&f.file.path, f)).collect();
        let mut row_groups = Vec::new();
        for entry in self
            .manifest
            .entries
            .iter()
            .filter(|e| e.maybe_matches_partition(&self.manifest.partition_by, filters))
        {
            let file = by_path
                .get(&entry.file.path)
                .ok_or_else(|| Error::FooterNotLoaded {
                    table: self.name.clone(),
                    file: entry.file.path.as_str().to_string(),
                })?;
            row_groups.extend(file.row_groups.iter().cloned());
        }
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.manifest.columns.clone()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root
    /// (see [`ParquetCatalog::store_description`](crate::ParquetCatalog::store_description))
    /// to know the physical location.
    pub fn location(&self) -> &str {
        self.location.as_str()
    }

    /// Each committed file's manifest path paired with its loaded row groups, in
    /// manifest order - the source for the `metadata()` table function, where
    /// each row group reports the file it belongs to. Only files whose footers
    /// are loaded contribute (the refresh path keeps them synced to the
    /// manifest).
    pub(super) fn file_row_groups(&self) -> Vec<(String, Vec<Arc<RowGroupMetadata>>)> {
        let by_path: HashMap<&ObjectPath, &TableFile> =
            self.files.iter().map(|f| (&f.file.path, f)).collect();
        self.manifest
            .entries
            .iter()
            .filter_map(|e| {
                by_path
                    .get(&e.file.path)
                    .map(|f| (e.file.path.as_str().to_string(), f.row_groups.to_vec()))
            })
            .collect()
    }
}
