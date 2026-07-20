//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::manifest::{
    ManifestEntry, PartitionEqFilter, PartitionValues, SortBounds, TableManifest,
};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
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
/// It is a plain **value** — `Clone`, no interior locks. A writer (INSERT,
/// compaction) clones one out of the catalog, mutates its own copy, and lets the
/// durable manifest be the source of truth: every mutation
/// (an append of freshly-uploaded files, or a compaction swap) commits a new
/// manifest version by compare-and-swap, retrying past a concurrent writer. Copies drift
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
        // paths, footers not parsed for it); a partitioned INSERT records it on
        // the files it writes later.
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
        //
        // TODO: persist the bounds in the Delta `Add` action (its `stats`, or a
        // `tags` blob) so a reload carries them and this keep-across-refresh
        // workaround can go away. That also fixes the bounds being lost entirely
        // to a fresh process or a restart, which reload with none.
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

    /// The compare-and-swap loop every manifest commit runs: check each `removed`
    /// path is still in the latest version (else return
    /// [`Error::CommitConflict`] -- a concurrent writer swapped it out), write
    /// version N+1 with the removes plus `entries` adds, and on a lost CAS reload
    /// to the latest and retry.
    /// `apply_committed` reconciles the in-memory `files` once the version lands
    /// -- with row groups the caller already holds, or by reading footers.
    ///
    /// Runs off the dispatch workers: the retry's `refresh` drives a footer-fetch
    /// dataflow, and every commit path commits from a blocking thread, never a
    /// pinned worker (which would deadlock driving a dataflow from inside one).
    pub(crate) fn commit_manifest_version(
        &mut self,
        removed: &[ObjectPath],
        entries: &[ManifestEntry],
        data_change: bool,
        mut apply_committed: impl FnMut(&mut Self) -> crate::Result<()>,
    ) -> crate::Result<()> {
        loop {
            if let Some(missing) = removed.iter().find(|path| {
                !self
                    .manifest
                    .entries
                    .iter()
                    .any(|entry| entry.file.path.as_str() == path.as_str())
            }) {
                return Err(Error::CommitConflict {
                    table: self.name.clone(),
                    file: missing.to_string(),
                });
            }
            let next_version = self.manifest.version + 1;
            if crate::delta::commit_file_changes(
                self.store.as_ref(),
                &self.location,
                next_version,
                removed,
                entries,
                data_change,
            )? {
                self.manifest.version = next_version;
                self.manifest
                    .entries
                    .retain(|e| !removed.contains(&e.file.path));
                self.manifest.entries.extend(entries.iter().cloned());
                apply_committed(self)?;
                return Ok(());
            }
            // A concurrent writer took this version; reload to the latest and
            // retry the commit on top of it.
            self.refresh()?;
        }
    }

    /// Commit an append (`removed` empty) or swap whose files' row groups are
    /// already built (from the writer's own footer metadata), so no footer is
    /// re-read. `data_change` labels the Delta actions. A `removed` path already
    /// swapped out by another writer returns [`Error::CommitConflict`].
    pub(crate) fn cas_commit(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<(ManifestEntry, TableFile)>,
        data_change: bool,
    ) -> crate::Result<()> {
        let entries = added
            .iter()
            .map(|(entry, _)| entry.clone())
            .collect::<Vec<_>>();
        let table_files = added
            .into_iter()
            .map(|(_, table_file)| table_file)
            .collect::<Vec<_>>();
        self.commit_manifest_version(removed, &entries, data_change, |table| {
            table.files.retain(|f| !removed.contains(&f.file.path));
            table.files.extend(table_files.iter().cloned());
            Ok(())
        })
    }

    /// Commit files whose data and footer metadata are already complete — the
    /// INSERT append. A plain add (`data_change = true`), no removes.
    pub(crate) fn commit_uploaded_files(
        &mut self,
        uploaded: Vec<(ManifestEntry, TableFile)>,
    ) -> crate::Result<()> {
        self.cas_commit(&[], uploaded, true)
    }

    /// Reconcile `files` to the current `manifest`: drop the files no longer in
    /// it, then fetch and append the ones not yet held.
    pub(crate) fn sync_files_to_manifest(&mut self) -> crate::Result<()> {
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
            self.declared_columns(),
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
    /// merge candidates, and names in a compaction swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`). Reads the manifest, so it reflects the current committed
    /// version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<PartitionValues>)> {
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
    /// location like any [`FileRef`] path), returning its [`FileRef`] to be
    /// committed into the manifest.
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

    /// The declared schema as one shareable slice: what a footer load
    /// reconciles each file's parsed schema against.
    fn declared_columns(&self) -> Arc<[Column]> {
        self.manifest.columns.clone().into()
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
    /// Returns the merged files. If the commit fails for any reason, including
    /// another writer already having swapped the inputs out, the uncommitted
    /// merge outputs are deleted before the error is returned.
    /// `inputs` must share one partition tuple; the caller batches them so.
    pub fn compact_files(
        &mut self,
        inputs: &[FileRef],
        target_rows_per_group: usize,
        target_row_groups_per_file: usize,
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
        let uploaded_files = Arc::new(Injector::new());
        let spec = super::insert_sink::encode_and_upload_spec(
            self.store(),
            self.location.clone(),
            self.id(),
            self.declared_columns(),
            uploaded_files.clone(),
            scan,
            Arc::from(self.partition_by()),
            Arc::from(self.sort_by()),
            target_rows_per_group,
            target_row_groups_per_file,
            &self.dispatcher,
        );
        // Drive encode → upload to completion; the emitted row-count batch is
        // ignored, and the uploaded files arrive on `uploaded_files`.
        spec.collect()?;

        let mut added = Vec::new();
        loop {
            match uploaded_files.steal() {
                Steal::Success(uploaded) => {
                    // table_id is ignored: this table is the commit target.
                    let super::insert_sink::UploadedFile {
                        file,
                        partition,
                        sort_bounds,
                        row_groups,
                        ..
                    } = uploaded;
                    added.push((
                        ManifestEntry {
                            file: file.clone(),
                            partition,
                            sort_bounds,
                        },
                        TableFile::new(file, row_groups),
                    ));
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        let files: Vec<FileRef> = added.iter().map(|(entry, _)| entry.file.clone()).collect();
        let removed: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
        // A compaction swap rearranges bytes without changing rows.
        if let Err(error) = self.cas_commit(&removed, added, false) {
            for file in &files {
                if let Err(cleanup_error) = self.delete_data_file(&file.path) {
                    tracing::warn!(
                        commit_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %file.path,
                        "compaction: deleting output after commit failure failed (orphan left)"
                    );
                }
            }
            return Err(error);
        }
        Ok(files)
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
