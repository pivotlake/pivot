//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::manifest::{ManifestEntry, PartitionEqFilter, SortBounds, TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use arrow_array::{ArrayRef, Scalar};
use crossbeam_deque::{Injector, Steal};
use dispatch::{DataFlowDispatcher, Projection};
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
    /// The table's durable identity, from the Delta `metaData.id` (minted once at
    /// creation, stable across renames and every commit). The catalog indexes by
    /// this; `name` is only the user-facing label.
    id: uuid::Uuid,
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
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn new(
        name: String,
        id: uuid::Uuid,
        location: ObjectPath,
        manifest: TableManifest,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        delta_uri: url::Url,
    ) -> Self {
        Self {
            name,
            id,
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
        let (delta_uri, id) = crate::delta::initialize_table(
            store.as_ref(),
            &location,
            &columns,
            &partition_by,
            &sort_by,
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
            id,
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
        // The Delta log does not persist per-file sort bounds, so a reloaded
        // entry carries none; keep the bounds this copy already recorded for
        // files it still holds, or sort-key pruning would silently degrade on
        // every refresh.
        let mut entries = state.entries;
        let known_bounds: HashMap<&str, &SortBounds> = self
            .manifest
            .entries
            .iter()
            .filter_map(|e| Some((e.file.path.as_str(), e.sort_bounds.as_ref()?)))
            .collect();
        for entry in &mut entries {
            if entry.sort_bounds.is_none() {
                entry.sort_bounds = known_bounds.get(entry.file.path.as_str()).cloned().cloned();
            }
        }
        self.manifest = TableManifest {
            version: state.version,
            columns: state.columns,
            partition_by: state.partition_by,
            sort_by: state.sort_by,
            entries,
        };
        self.sync_files_to_manifest()?;
        Ok(true)
    }

    /// Write `bytes` as a new data file at `path` (under the table's location)
    /// and commit it into the table — the ingest sink's append, one call for the
    /// write and the manifest commit. Idempotent on `path`.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<HashMap<String, Scalar<ArrayRef>>>,
        sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        if self
            .manifest
            .entries
            .iter()
            .any(|entry| entry.file.path == path)
        {
            return Ok(());
        }
        let file = self.write_data_file(path, bytes)?;
        let data_file = file
            .clone()
            .into_data_file(self.store.as_ref(), &self.location)?;
        let table_file = crate::parquet::load_table_files(&self.dispatcher, &[data_file])?
            .pop()
            .expect("one uploaded file yields one footer");
        self.commit_uploaded_files(vec![(
            ManifestEntry {
                file,
                partition,
                sort_bounds,
            },
            table_file,
        )])
    }

    /// The commit primitive behind every data-file change: atomically remove
    /// `removed` and add `added` (each with its already-built row groups) in one
    /// Delta version, retrying past concurrent commits. `added`'s row groups come
    /// from what the caller holds, so no footer is re-read. `data_change` labels
    /// the commit (see [`crate::delta::replace_files`]).
    ///
    /// Returns whether it committed. It is **aborted** (`Ok(false)`, nothing
    /// committed) when any `removed` path is no longer in the latest version:
    /// another writer already removed it, so re-adding `added` (which holds those
    /// rows) would double-count. The caller then discards its `added` files as
    /// orphans and leaves the inputs alone — they belong to the commit that won.
    fn cas_commit(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<(ManifestEntry, TableFile)>,
        data_change: bool,
    ) -> crate::Result<bool> {
        let entries = added
            .iter()
            .map(|(entry, _)| entry.clone())
            .collect::<Vec<_>>();
        loop {
            if !removed
                .iter()
                .all(|p| self.manifest.entries.iter().any(|e| &e.file.path == p))
            {
                return Ok(false);
            }
            let next_version = self.manifest.version + 1;
            if crate::delta::replace_files(
                self.store.as_ref(),
                &self.location,
                next_version,
                removed,
                &entries,
                data_change,
            )? {
                self.manifest.version = next_version;
                self.manifest
                    .entries
                    .retain(|e| !removed.contains(&e.file.path));
                self.manifest.entries.extend(entries.iter().cloned());
                self.files.retain(|f| !removed.contains(&f.file.path));
                self.files
                    .extend(added.iter().map(|(_, table_file)| table_file.clone()));
                return Ok(true);
            }
            // A concurrent writer took this version. The footer refresh below is
            // itself a dataflow and must never be launched recursively from a
            // terminal worker; both commit paths (INSERT, compaction) run
            // off-worker (`WORKER_IDX == usize::MAX`), so a caller committing from
            // a worker (which would deadlock on the refresh) fails cleanly here.
            if dispatch::worker::WORKER_IDX.get() != usize::MAX {
                return Err(Error::ConcurrentInsert(self.name.clone()));
            }
            self.refresh()?;
        }
    }

    /// Commit files whose data and footer metadata are already complete — the
    /// INSERT / ingest append. A plain add (`data_change = true`), no removes.
    pub(crate) fn commit_uploaded_files(
        &mut self,
        uploaded: Vec<(ManifestEntry, TableFile)>,
    ) -> crate::Result<()> {
        self.cas_commit(&[], uploaded, true).map(|_| ())
    }

    /// Atomically swap a set of this table's files for another as a raw manifest
    /// edit — `added` given as bare manifest entries, their footers read only on
    /// success (via [`sync_files_to_manifest`](Self::sync_files_to_manifest)), so
    /// a losing swap aborts before reading its output. Compaction, which already
    /// holds the footer metadata, commits through
    /// [`compact_files`](Self::compact_files) instead. `data_change = false`: a
    /// swap rearranges bytes without changing rows.
    ///
    /// Returns whether the swap committed. It is **aborted** (`Ok(false)`) when
    /// any `removed` path is no longer in the latest version: another writer
    /// already swapped these inputs out, so re-adding `added` (which holds their
    /// rows) would double-count. The loser then discards its `added` as orphans.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[ManifestEntry],
    ) -> crate::Result<bool> {
        loop {
            if !removed
                .iter()
                .all(|p| self.manifest.entries.iter().any(|e| &e.file.path == p))
            {
                return Ok(false);
            }
            let next_version = self.manifest.version + 1;
            if crate::delta::replace_files(
                self.store.as_ref(),
                &self.location,
                next_version,
                removed,
                added,
                false,
            )? {
                self.manifest.version = next_version;
                self.manifest
                    .entries
                    .retain(|e| !removed.contains(&e.file.path));
                self.manifest.entries.extend(added.iter().cloned());
                self.sync_files_to_manifest()?;
                return Ok(true);
            }
            self.refresh()?;
        }
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

    /// The table's durable identity (Delta `metaData.id`), stable across renames
    /// and commits. The catalog indexes by this.
    pub fn id(&self) -> uuid::Uuid {
        self.id
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

    pub(crate) fn object_location(&self) -> &ObjectPath {
        &self.location
    }

    pub(crate) fn store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    /// Merge `inputs` into fresh target-sized files and atomically swap them in
    /// for the inputs, one Delta version labelled as a rearrangement
    /// (`data_change = false`, so incremental log readers skip it). The merged
    /// files are written over the shared io_uring ring by the same upload
    /// operators an INSERT uses, and their row groups come straight from the
    /// writer's own footer metadata — no footer is re-read.
    ///
    /// Returns the merged files, or an empty vec when another writer already
    /// swapped these inputs out (this merge is discarded, its output deleted).
    /// `inputs` must share one partition tuple; the caller batches them so.
    pub fn compact_files(
        &mut self,
        inputs: &[FileRef],
        target_rows: usize,
        target_row_groups: usize,
    ) -> crate::Result<Vec<FileRef>> {
        // Scan only the inputs and re-encode their rows. They share one partition
        // tuple, so re-applying the table's partition/sort spec reproduces that
        // tuple and recomputes the merged files' sort bounds.
        let parquet = self.parquet_table_for(inputs);
        let columns = parquet.schema().fields().len();
        let scan = crate::parquet::table_input(
            &self.dispatcher,
            &parquet,
            Projection::all(columns),
            false,
        );
        let commit_queue = Arc::new(Injector::new());
        let spec = super::insert_sink::upload_spec(
            self.store(),
            self.location.clone(),
            self.id(),
            commit_queue.clone(),
            scan,
            Arc::from(self.partition_by()),
            Arc::from(self.sort_by()),
            target_rows,
            target_row_groups,
            &self.dispatcher,
        );
        // Drive scan → encode → upload to completion. Its emitted row-count batch
        // is ignored here; the uploaded files arrive on `commit_queue`.
        spec.collect()?;

        let mut added = Vec::new();
        loop {
            match commit_queue.steal() {
                Steal::Success(file) => added.push((file.entry, file.table_file)),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        let removed: Vec<ObjectPath> = inputs.iter().map(|f| f.path.clone()).collect();
        let merged: Vec<FileRef> = added.iter().map(|(entry, _)| entry.file.clone()).collect();
        if self.cas_commit(&removed, added, false)? {
            Ok(merged)
        } else {
            // Another writer won the swap; our merged output is now an orphan.
            for file in &merged {
                if let Err(e) = self.delete_data_file(&file.path) {
                    tracing::warn!(error = %e, file = %file.path, "compaction: deleting discarded merge output failed (orphan left)");
                }
            }
            Ok(Vec::new())
        }
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
