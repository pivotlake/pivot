//! The catalog's master record of one table: its definition plus its current
//! content, the files at one delta log version.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::delta::{DeltaLog, ManifestEntry, PartitionEqFilter, SortBounds, TableState};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
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

/// The catalog's master record of one table: its committed [`TableState`]
/// (declared columns + file list, at one delta log version) and its current
/// content, the per-file row groups (`files`).
///
/// It is a plain **value** — `Clone`, no interior locks. A writer (an insert,
/// compaction) clones one out of the catalog, mutates its own copy, and lets
/// the delta log be the source of truth: every mutation
/// ([`commit_added_files`](Self::commit_added_files),
/// [`replace_data_files`](Self::replace_data_files)) commits a new log version
/// by compare-and-swap, retrying past a concurrent writer. Copies drift
/// freely; [`refresh`](Self::refresh) reconciles any copy to the latest version
/// (re-reading only the footers it doesn't already hold). The shared
/// [`DeltaLog`] handle, `store`, `name`, and `location` are kept so a copy can
/// persist and reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    name: String,
    /// Where the table's Parquet data (and so its delta log) lives in the
    /// object store.
    location: ObjectPath,
    pub(super) state: TableState,
    pub(super) files: Vec<TableFile>,
    store: Arc<dyn ObjectStore>,
    /// The table's delta transaction log, shared by every copy.
    log: Arc<DeltaLog>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
}

impl CatalogTable {
    /// Reassemble a persisted table from its loaded `state` and the per-file
    /// row groups (`files`) just fetched for it — the reopen path. Does not
    /// persist anything; the log version it was loaded from is already durable.
    pub(super) fn new(
        name: String,
        location: ObjectPath,
        log: Arc<DeltaLog>,
        state: TableState,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            name,
            location,
            state,
            files,
            store,
            log,
            dispatcher,
        }
    }

    /// Create a brand-new table from freshly-read footers: create its delta
    /// table (declared `columns`, partition/sort specs, and the loaded files
    /// as the first commit). Fails with [`Error::LocationHasTable`] if a delta
    /// table already exists at this location (another table's data, or a log
    /// left behind by a previous database). The `CREATE TABLE` commit path.
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
        let target = store.delta_table_target(&location)?;
        let (log, state) = DeltaLog::create(
            target.uri,
            target.storage_options,
            columns,
            partition_by,
            sort_by,
            entries,
        )
        .map_err(|e| match e {
            crate::delta::Error::TableExists => Error::LocationHasTable {
                name: name.clone(),
                location: location.as_str().to_string(),
            },
            other => Error::Delta(other),
        })?;
        Ok(Self {
            name,
            location,
            state,
            files,
            store,
            log: Arc::new(log),
            dispatcher,
        })
    }

    /// Reload this copy to the latest committed version. Returns whether it
    /// advanced; `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(state) = self.log.load_after(self.state.version)? else {
            return Ok(false);
        };
        self.state = state;
        self.sync_files_to_state()?;
        Ok(true)
    }

    /// Reload to the latest version *without* fetching any footers: advance the
    /// state and drop cached files no longer in it, but leave fetching the new
    /// files' footers to [`parquet`](Self::parquet), which fetches only the
    /// partitions a query actually scans. The read path uses this; a writer still
    /// uses [`refresh`](Self::refresh) (it needs every file's row groups to pick
    /// compaction candidates). Returns whether it advanced.
    pub fn reload_manifest_only(&mut self) -> crate::Result<bool> {
        let Some(state) = self.log.load_after(self.state.version)? else {
            return Ok(false);
        };
        self.set_state(state);
        Ok(true)
    }

    /// Write `bytes` as a new data file at `path` (under the table's location)
    /// and commit it into the table — an external writer's single-file append,
    /// one call for the write and the log commit. Idempotent on `path`.
    /// (An `INSERT` writes its files first and commits them all at once via
    /// [`commit_added_files`](Self::commit_added_files).)
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<serde_json::Value>,
        sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        let file = self.write_data_file(path, bytes)?;
        self.commit_added_files(vec![ManifestEntry {
            file,
            partition,
            sort_bounds,
        }])
    }

    /// Commit a batch of already-written files into the delta log as **one**
    /// new version (one `INSERT` = one commit, however many files it produced),
    /// retrying past concurrent writers. Entries whose path the log already
    /// holds are skipped, so a replayed commit can't double-count rows.
    ///
    /// Touches only the log: it never fetches footers, so it is safe to call
    /// from a dispatch worker (an `INSERT`'s terminal sink), where a footer
    /// fetch would drive a nested dataflow on the same pool and deadlock the
    /// worker driving it. Readers fetch the new files' footers lazily via
    /// [`parquet`](Self::parquet).
    pub fn commit_added_files(&mut self, entries: Vec<ManifestEntry>) -> crate::Result<()> {
        let state = self.log.commit_added(entries)?;
        self.set_state(state);
        Ok(())
    }

    /// Swap in `state` as this copy's current version, dropping cached footers
    /// for files no longer in it. No footer fetch, so it is safe on a dispatch
    /// worker.
    fn set_state(&mut self, state: TableState) {
        self.state = state;
        self.retain_state_files();
    }

    /// Drop cached footers for files that are no longer committed entries.
    fn retain_state_files(&mut self) {
        let kept: HashSet<&str> = self
            .state
            .entries
            .iter()
            .map(|e| e.file.path.as_str())
            .collect();
        self.files.retain(|f| kept.contains(f.file.path.as_str()));
    }

    /// Atomically swap a set of this table's files for another — the compaction
    /// commit. Commits a new log version holding the removed files' tombstones
    /// and the added files, retrying past concurrent commits, so no reader ever
    /// sees the rows doubled or missing. The removed files' objects are deleted
    /// later, by [`maintain`](Self::maintain), once their tombstones age past
    /// the retention horizon and no live reader can still reference them.
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
        added: &[ManifestEntry],
    ) -> crate::Result<bool> {
        let Some(state) = self.log.replace(removed.to_vec(), added.to_vec())? else {
            return Ok(false);
        };
        self.state = state;
        self.sync_files_to_state()?;
        Ok(true)
    }

    /// Background upkeep of this table's delta log and data directory:
    /// checkpoint the log, drop log entries older than the retention horizon,
    /// and physically delete the data files whose tombstones have aged past it
    /// (no live reader still resolves a snapshot that old). Each commit leaves
    /// a new log entry behind; this keeps the log listing every reload does
    /// cheap. The compacter drives it; it's the background sweep that already
    /// polls every table.
    pub fn maintain(&self) -> crate::Result<()> {
        self.log.maintain()?;
        Ok(())
    }

    /// Reconcile `files` to the current `state`: drop the files no longer in
    /// it, then fetch and append the ones not yet held.
    fn sync_files_to_state(&mut self) -> crate::Result<()> {
        self.retain_state_files();
        let missing = self.retrieve_missing_table_files()?;
        self.files.extend(missing);
        Ok(())
    }

    /// Fetch the [`TableFile`]s for this table's committed entries that aren't
    /// already held — the footers this copy is missing (disjoint from `files`).
    fn retrieve_missing_table_files(&self) -> crate::Result<Vec<TableFile>> {
        let to_fetch: Vec<DataFile> = self
            .state
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
        self.files.iter().map(|f| f.file.clone()).collect()
    }

    /// Each committed file paired with the partition tuple recorded for it (the
    /// one-row arrow-json object a partitioning writer stamped, or `None`).
    /// Reads the committed state, so it reflects the current log version.
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

    /// Delete a data file (a compaction output orphaned by a lost swap race).
    /// `path` resolves against the table's location like any [`FileRef`] path.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        self.store.delete(&self.location.resolve(path))?;
        Ok(())
    }

    /// A flat scan view of the files whose recorded partition tuple can still
    /// match `filters` — every surviving file's row groups concatenated in
    /// committed order, where a row group's global index is simply its position.
    /// Footers are fetched here, and only for the surviving files, so a query
    /// touching one partition never pays the HTTP to read every other partition's
    /// footer. Fetched footers are cached in `files`, so a second call in the same
    /// query (the late materialize after the scan) re-fetches nothing. An empty
    /// `filters`, or one naming no partition column, keeps every file.
    ///
    /// The row group's global index is its position in this returned flat list, so
    /// a scan and its materialize must build it from the *same* `filters` (they
    /// do: both go through the binding's predicates) to address the same groups.
    pub fn parquet(&mut self, filters: &[PartitionEqFilter]) -> crate::Result<Arc<ParquetTable>> {
        let matching: Vec<ManifestEntry> = self
            .state
            .entries
            .iter()
            .filter(|e| e.maybe_matches_partition(&self.state.partition_by, filters))
            .cloned()
            .collect();

        let to_fetch: Vec<DataFile> = matching
            .iter()
            .filter(|e| !self.files.iter().any(|f| f.file.path == e.file.path))
            .map(|e| {
                e.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &self.location)
            })
            .collect::<store::Result<_>>()?;
        let fetched = crate::parquet::load_table_files(&self.dispatcher, &to_fetch)?;
        self.files.extend(fetched);

        let by_path: HashMap<&ObjectPath, &TableFile> =
            self.files.iter().map(|f| (&f.file.path, f)).collect();
        let row_groups = matching
            .iter()
            .filter_map(|e| by_path.get(&e.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.state.columns.clone()
    }

    /// The table's physical arrow schema: each declared column at the physical
    /// type it is stored and scanned as (e.g. `Utf8View` for a `Utf8` column).
    /// What a writer conforms incoming batches to. The fields are declared
    /// nullable even though stored columns are required: a null in insert data
    /// (e.g. from an overflowing cast) must reach the write pipeline's typed
    /// `NullsInRequiredColumn` error, not fail batch construction with a panic.
    pub fn physical_arrow_schema(&self) -> arrow_schema::SchemaRef {
        let fields: Vec<arrow_schema::Field> = self
            .state
            .columns
            .iter()
            .map(|column| {
                arrow_schema::Field::new(
                    column.name.clone(),
                    planner::types::physical_arrow_type(&column.col_type),
                    true,
                )
            })
            .collect();
        Arc::new(arrow_schema::Schema::new(fields))
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root
    /// (see [`ParquetCatalog::store_description`](crate::ParquetCatalog::store_description))
    /// to know the physical location.
    pub fn location(&self) -> &str {
        self.location.as_str()
    }

    /// Each committed file's path paired with its loaded row groups, in
    /// committed order - the source for the `metadata()` table function, where
    /// each row group reports the file it belongs to. Only files whose footers
    /// are loaded contribute (a [`parquet`](Self::parquet) call warms them).
    pub(super) fn file_row_groups(&self) -> Vec<(String, Vec<Arc<RowGroupMetadata>>)> {
        let by_path: HashMap<&ObjectPath, &TableFile> =
            self.files.iter().map(|f| (&f.file.path, f)).collect();
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
