//! The catalog's master record of one table: its definition plus its current
//! content, the files at one Delta Lake log version.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::delta;
use crate::manifest::{ManifestEntry, PartitionEqFilter, SortBounds, TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, DeltaTableTarget, FileRef, ObjectPath, ObjectStore};
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

/// The catalog's in-memory record of one table: its [`TableManifest`] extracted
/// from Delta Lake and the loaded row groups for its active files.
///
/// It is a plain value with no interior locks. [`refresh`](Self::refresh)
/// reconciles a copy to the latest Delta log version and reads only missing
/// Parquet footers. Catalog writes are intentionally unsupported while the
/// Delta integration is read-only.
#[derive(Clone)]
pub struct CatalogTable {
    name: String,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    /// How delta-rs addresses the transaction log at this table's location.
    target: DeltaTableTarget,
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
        target: DeltaTableTarget,
        manifest: TableManifest,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            name,
            location,
            target,
            manifest,
            files,
            store,
            dispatcher,
        }
    }

    /// Reject `CREATE TABLE` until the catalog has a Delta writer path.
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
        let _ = (
            name,
            location,
            files,
            columns,
            partition_by,
            sort_by,
            store,
            dispatcher,
        );
        Err(Error::DeltaWritesUnsupported)
    }

    /// Reload this copy to the latest committed Delta version.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(manifest) = delta::load_manifest_after(&self.target, self.manifest.version)?
        else {
            return Ok(false);
        };
        self.manifest = manifest;
        self.sync_files_to_manifest()?;
        Ok(true)
    }

    /// Reject ingest writes until they can commit a matching Delta transaction.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<serde_json::Value>,
        sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        let _ = (path, bytes, partition, sort_bounds);
        Err(Error::DeltaWritesUnsupported)
    }

    /// Reject compaction swaps until they can commit a matching Delta transaction.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[ManifestEntry],
    ) -> crate::Result<bool> {
        let _ = (removed, added);
        Err(Error::DeltaWritesUnsupported)
    }

    /// Reject legacy manifest pruning for Delta-backed tables.
    pub fn prune_old_versions(&self) -> crate::Result<()> {
        Err(Error::DeltaWritesUnsupported)
    }

    /// Reject legacy deferred-deletion records for Delta-backed tables.
    pub fn record_deletions(&self, removed: &[ObjectPath]) -> crate::Result<()> {
        let _ = removed;
        Err(Error::DeltaWritesUnsupported)
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

    /// Each committed file paired with the partition tuple recorded for it (the
    /// one-row arrow-json object a partitioning writer stamped, or `None`). Reads
    /// the manifest, so it reflects the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<serde_json::Value>)> {
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
