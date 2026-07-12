//! The catalog's master record of one table: its definition plus its current
//! content, the files at one delta log version.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::delta::{self, DeltaLog};
use crate::manifest::{ManifestEntry, PartitionEqFilter, SortBounds, TableFile, TableState};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use dispatch::DataFlowDispatcher;
use planner::catalog::Column;

/// The catalog's master record of one table: its [`TableState`] (declared
/// columns + committed file list, at one delta log version) and its current
/// content, the per-file row groups (`files`). The version lives in the
/// manifest, not alongside it.
///
/// It is a plain **value** — `Clone`, no interior locks. The table's delta log
/// is the source of truth; copies drift freely and [`refresh`](Self::refresh)
/// reconciles any copy to the latest committed version (re-reading only the
/// footers it doesn't already hold). The data-write mutators a writer (ingest,
/// compaction) would evolve a copy through are currently **disabled** — they
/// keep their signatures but always error (see
/// [`append_data_file`](Self::append_data_file)); the one write the catalog
/// performs is `CREATE TABLE`'s first commit. `store`, `name`, and `location`
/// are kept so a copy can reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    name: String,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    /// The table's delta transaction log: a kernel-backed handle whose cached
    /// snapshot advances incrementally, shared by every copy of this table.
    log: Arc<DeltaLog>,
    pub(super) state: TableState,
    store: Arc<dyn ObjectStore>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
}

impl CatalogTable {
    /// Reassemble a persisted table from its loaded `state` and the per-file
    /// row groups (`files`) just fetched for it — the reopen path. Does not
    /// persist anything; the state it was loaded from is already durable.
    pub(super) fn new(
        name: String,
        location: ObjectPath,
        log: Arc<DeltaLog>,
        state: TableState,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            name,
            location,
            log,
            state,
            store,
            dispatcher,
        }
    }

    /// Create a brand-new table from freshly-read footers: create its delta
    /// table (declared `columns`, the partition/sort specs, and the loaded
    /// files, as the log's first version). Fails with [`Error::TableExists`]
    /// if a delta log already exists at the location (including a concurrent
    /// creator winning the first commit). The `CREATE TABLE` commit path.
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
        let (log, mut state) = DeltaLog::create(
            &store.get_config()?,
            store.get_absolute_url(&location)?,
            columns,
            partition_by,
            sort_by,
            entries,
        )
        .map_err(|e| match e {
            delta::Error::TableExists => Error::TableExists(name.clone()),
            other => Error::Delta(other),
        })?;
        state.files = files;
        Ok(Self {
            name,
            location,
            log: Arc::new(log),
            state,
            store,
            dispatcher,
        })
    }

    /// Reload this copy to the latest committed delta log version. Errors if
    /// the table has no log at all (a corrupt catalog). Returns whether it
    /// advanced; `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(mut state) = self.log.load_after(self.state.version)? else {
            return Ok(false);
        };
        self.attach_files(&mut state)?;
        self.state = state;
        Ok(true)
    }

    /// The ingest sink's append: write `bytes` as a new data file and commit it
    /// into the table. **Disabled** while the write paths are being reworked
    /// for the delta format: always errors without writing anything, so a
    /// misconfigured ingest fails loudly instead of committing through an
    /// unfinished path. The signature is kept so ingest keeps compiling.
    pub fn append_data_file(
        &mut self,
        _path: ObjectPath,
        _bytes: &[u8],
        _partition: Option<serde_json::Value>,
        _sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        Err(Error::WritesDisabled(self.name.clone()))
    }

    /// The compaction commit: atomically swap a set of this table's files for
    /// another. **Disabled** while the write paths are being reworked for the
    /// delta format: always errors without committing anything. The signature
    /// is kept so the compacter keeps compiling.
    pub fn replace_data_files(
        &mut self,
        _removed: &[ObjectPath],
        _added: &[ManifestEntry],
    ) -> crate::Result<bool> {
        Err(Error::WritesDisabled(self.name.clone()))
    }

    /// Background upkeep of this table's delta log (checkpoint, expired-log
    /// cleanup, vacuum). **Disabled** while the write paths are being reworked
    /// for the delta format: always errors without touching the log. The
    /// signature is kept so the compacter keeps compiling.
    pub fn maintain(&self) -> crate::Result<()> {
        Err(Error::WritesDisabled(self.name.clone()))
    }

    /// Attach each of `state`'s entries' footers to it before it is installed:
    /// carry over the footers this copy already holds and fetch only the ones
    /// it doesn't, so a state and its footers always change together.
    fn attach_files(&self, state: &mut TableState) -> crate::Result<()> {
        let mut files: Vec<TableFile> = self
            .state
            .files
            .iter()
            .filter(|f| state.entries.iter().any(|e| e.file.path == f.file.path))
            .cloned()
            .collect();
        let to_fetch: Vec<DataFile> = state
            .entries
            .iter()
            .filter(|e| !files.iter().any(|f| f.file.path == e.file.path))
            .map(|e| {
                e.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &self.location)
            })
            .collect::<store::Result<_>>()?;
        files.extend(crate::parquet::load_table_files(
            &self.dispatcher,
            &to_fetch,
        )?);
        state.files = files;
        Ok(())
    }

    pub fn files(&self) -> &[TableFile] {
        &self.state.files
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        &self.state.partition_by
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        &self.state.sort_by
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a [`replace_data_files`](Self::replace_data_files) swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.state.files.iter().map(|f| f.file.clone()).collect()
    }

    /// Each committed file paired with the partition tuple recorded for it (the
    /// one-row arrow-json object a partitioning writer stamped, or `None`). Reads
    /// the manifest, so it reflects the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<serde_json::Value>)> {
        self.state
            .entries
            .iter()
            .map(|e| (e.file.path.clone(), e.partition.clone()))
            .collect()
    }

    /// Each committed file paired with its recorded sort-key bounds (the
    /// `sort_by` columns at the file's first/last row, or `None`).
    pub fn file_sort_bounds(&self) -> Vec<(ObjectPath, Option<SortBounds>)> {
        self.state
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
            .state
            .files
            .iter()
            .filter(|f| want.contains(&f.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    /// The compaction writer's output write. **Disabled** while the write
    /// paths are being reworked for the delta format: always errors before
    /// putting any object, so the compacter can't litter a table's directory
    /// with output whose commit would be refused anyway. The signature is kept
    /// so the compacter keeps compiling.
    pub fn write_data_file(&self, _path: ObjectPath, _bytes: &[u8]) -> crate::Result<FileRef> {
        Err(Error::WritesDisabled(self.name.clone()))
    }

    /// Delete a data file (a compaction input swapped out of the table).
    /// **Disabled** like every other write; the signature is kept so the
    /// compacter keeps compiling.
    pub fn delete_data_file(&self, _path: &ObjectPath) -> crate::Result<()> {
        Err(Error::WritesDisabled(self.name.clone()))
    }

    /// The manifest version this copy is at. Monotonic per table; used to
    /// decide whether a published copy is newer than the catalog's.
    pub fn version(&self) -> u64 {
        self.state.version
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
            self.state.files.iter().map(|f| (&f.file.path, f)).collect();
        let mut row_groups = Vec::new();
        for entry in self
            .state
            .entries
            .iter()
            .filter(|e| e.maybe_matches_partition(&self.state.partition_by, filters))
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
        self.state.columns.clone()
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
            self.state.files.iter().map(|f| (&f.file.path, f)).collect();
        self.state
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
