//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::manifest::{
    ColumnStatFilter, DeltaFileEntry, FileStats, PartitionEqFilter, PartitionValues, scalar_equal,
};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use arrow_array::{ArrayRef, Datum, Scalar};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use crossbeam_deque::{Injector, Steal};
use datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use dispatch::{DataFlowDispatcher, Projection};
use planner::catalog::{Column, SchemaQualifiedTableName};

/// One data file of a table, materialized: its Delta log entry
/// ([`DeltaFileEntry`] -- identity, partition tuple, stats) paired with the row
/// groups read from its footer (file-local order). This is the single per-file
/// record the catalog holds: the entry is the metadata that commits to the log,
/// the row groups the content that scans.
#[derive(Clone)]
pub struct TableFile {
    pub(super) entry: DeltaFileEntry,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's Delta log entry with the row groups read from its footer.
    /// The footer fetcher builds one with a bare [`DeltaFileEntry::new`] (identity
    /// only); the catalog then joins in the log entry's partition/stats.
    ///
    /// Derive the file's stats from its footers when the entry carries none, so
    /// file-level range pruning always has min/max bounds. Usually the entry
    /// already has them -- a fresh INSERT records them, and a reload reads the
    /// log's persisted stats -- so this covers only a file with neither: one
    /// adopted at CREATE, or whose log entry recorded no stats.
    pub(crate) fn new(mut entry: DeltaFileEntry, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        if entry.stats.is_none() && !row_groups.is_empty() {
            entry.stats = Some(Arc::new(crate::parquet::aggregate_file_stats(&row_groups)));
        }
        Self { entry, row_groups }
    }

    /// The file's store identity (path and size), for a caller that names the
    /// file outside the catalog module (reporting a merge's outputs, deleting
    /// one whose commit failed).
    pub(crate) fn file_ref(&self) -> &FileRef {
        &self.entry.file
    }

    /// The partition tuple this file belongs to, or `None` for an unpartitioned
    /// table (or a file that does not sit in one partition).
    pub(crate) fn partition(&self) -> &Option<PartitionValues> {
        &self.entry.partition
    }

    /// This file's stored bytes, summed over its column chunks. Leaving out the
    /// footer, which is most of what a tiny file weighs, is what makes this a
    /// usable estimate of the same rows inside a merged file.
    pub(crate) fn compressed_size(&self) -> u64 {
        self.row_groups
            .iter()
            .flat_map(|row_group| row_group.columns.iter())
            .map(|chunk| chunk.total_compressed_size.max(0) as u64)
            .sum()
    }

    /// What this file's bytes hold decoded, summed over its column chunks.
    fn uncompressed_size(&self) -> u64 {
        self.row_groups
            .iter()
            .flat_map(|row_group| row_group.columns.iter())
            .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
            .sum()
    }
}

/// The catalog's master record of one table: its declared schema, partitioning,
/// version, and active [`TableFile`]s, each pairing a Delta log entry with its
/// materialized row groups.
///
/// This is a cloneable snapshot value. Copies handed out are read views and may
/// drift as commits land; [`refresh`](Self::refresh) reconciles one to the latest
/// version. Writers instead go through
/// [`DeltaDatastore::commit_to_table`](crate::DeltaDatastore::commit_to_table),
/// following the table's
/// [commit-lock protocol](field@Self::commit_lock). `store` and `location`
/// let any copy persist and reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    /// The table's durable identity from the Pivot manifest, minted once at
    /// creation and stable across renames and every commit. The catalog indexes
    /// by this; Delta Kernel independently owns the log's `metaData.id`.
    id: uuid::Uuid,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    /// The Delta snapshot this copy is at: the version its `files` were built
    /// from, and the base every commit from this copy is written on top of. It is
    /// this copy's alone: a refresh advances it, a commit replaces it with the one
    /// the commit produced, and neither reads a log another copy is holding.
    snapshot: Arc<delta_kernel::Snapshot>,
    /// The table's declared columns (schema).
    columns: Vec<Column>,
    /// Partition columns, in order; empty = unpartitioned.
    partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    sort_by: Vec<String>,
    /// The active files: each a log entry (identity + partition + stats) paired
    /// with its materialized footers.
    pub(super) files: Vec<TableFile>,
    store: Arc<dyn ObjectStore>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
    /// The Kernel engine this table's log is read and written through, shared
    /// with every other table of the datastore that opened it.
    engine: crate::delta::DeltaEngine,
    /// This exists only for throughput: Delta log compare-and-swap already ensures
    /// correctness, but concurrent in-process writers can all start at `V` and race
    /// for `V + 1`, forcing every loser to replay the log and fetch the winner's
    /// footers before retrying.
    commit_lock: Arc<std::sync::Mutex<()>>,
}

impl CatalogTable {
    /// Reassemble a persisted table from its table-level metadata and the per-file
    /// [`TableFile`]s (log entry + footers) just built for it — the reopen path.
    /// Does not persist anything; the log it was loaded from is already durable.
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn new(
        id: uuid::Uuid,
        location: ObjectPath,
        snapshot: Arc<delta_kernel::Snapshot>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        engine: crate::delta::DeltaEngine,
    ) -> Self {
        Self {
            id,
            location,
            snapshot,
            columns,
            partition_by,
            sort_by,
            files,
            store,
            dispatcher,
            engine,
            commit_lock: Arc::default(),
        }
    }

    /// Create a brand-new table from freshly-read footers by atomically writing
    /// Delta version 0 with its protocol, metadata, and initial `Add` actions.
    /// The `files` arrive with bare entries (identity only); files discovered at
    /// CREATE carry no partition metadata, and a partitioned INSERT records it on
    /// the files it writes later.
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn create_new(
        id: uuid::Uuid,
        location: ObjectPath,
        loaded: Vec<crate::parquet::FileRowGroups>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        engine: crate::delta::DeltaEngine,
    ) -> crate::Result<Self> {
        // Pair each discovered file with a bare log entry; `TableFile::new`
        // derives its stats from the footers.
        let mut files: Vec<TableFile> = loaded
            .into_iter()
            .map(|file| TableFile::new(DeltaFileEntry::new(file.file), file.row_groups))
            .collect();
        // For a partitioned table, a discovered file whose partition columns are
        // constant belongs to that partition; stamp it from the file's stats so it
        // is recorded in the log and prunes like a written file.
        if !partition_by.is_empty() {
            for file in &mut files {
                let partition = file
                    .entry
                    .stats
                    .as_ref()
                    .and_then(|stats| constant_partition(stats, &partition_by));
                file.entry.partition = partition;
            }
        }
        let entries: Vec<DeltaFileEntry> = files.iter().map(|file| file.entry.clone()).collect();
        // Version 0's commit yields the snapshot the new table starts at, so the
        // log it just wrote is not read back to open it.
        let snapshot = crate::delta::initialize_table(
            &engine,
            store.as_ref(),
            &location,
            &columns,
            &partition_by,
            &sort_by,
            &entries,
        )?;
        Ok(Self {
            id,
            location,
            snapshot,
            columns,
            partition_by,
            sort_by,
            files,
            store,
            dispatcher,
            engine,
            commit_lock: Arc::default(),
        })
    }

    /// An owned handle to this table's
    /// [commit lock](field@Self::commit_lock).
    pub(super) fn commit_lock(&self) -> Arc<std::sync::Mutex<()>> {
        self.commit_lock.clone()
    }

    /// Advance this copy to the latest committed version, reconciling its files
    /// to it. Returns whether it advanced; `Ok(false)` means this copy was
    /// already current, which costs one log listing and no file reads (Kernel
    /// updates the snapshot in hand rather than rebuilding it).
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(state) = crate::delta::refresh_table(&self.snapshot, &self.engine)? else {
            return Ok(false);
        };
        let crate::delta::DeltaTableState {
            snapshot,
            columns,
            partition_by,
            sort_by,
            file_entries,
        } = state;
        // Reconcile the files first: it is the only step that can fail, and it
        // leaves this copy untouched when it does, so a copy never ends up at a
        // version whose files it does not hold.
        self.rebuild_files(file_entries, &columns)?;
        self.snapshot = snapshot;
        self.columns = columns;
        self.partition_by = partition_by;
        self.sort_by = sort_by;
        Ok(true)
    }

    /// Commit file changes against this copy's snapshot. Each attempt first
    /// verifies that every `removed` path is still active, returning
    /// [`Error::CommitConflict`] if another commit removed one. After a successful
    /// log write, the already-loaded `added` files fold directly into this copy.
    /// A version conflict refreshes the copy and retries.
    ///
    /// In-process callers follow the table's
    /// [commit-lock protocol](field@Self::commit_lock), so this retry path
    /// normally handles contention from another process on a shared remote
    /// store.
    ///
    /// Runs off the dispatch workers: the retry's `refresh` drives a footer-fetch
    /// dataflow, and every commit path commits from a blocking thread, never a
    /// pinned worker (which would deadlock driving a dataflow from inside one).
    pub(crate) fn commit_files(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            if let Some(missing) = removed.iter().find(|path| {
                !self
                    .files
                    .iter()
                    .any(|f| f.entry.file.path.as_str() == path.as_str())
            }) {
                return Err(Error::CommitConflict {
                    location: self.location.as_str().to_string(),
                    file: missing.to_string(),
                });
            }
            let removed_entries: Vec<DeltaFileEntry> = removed
                .iter()
                .map(|path| {
                    self.files
                        .iter()
                        .find(|file| file.entry.file.path.as_str() == path.as_str())
                        .expect("removed paths were validated above")
                        .entry
                        .clone()
                })
                .collect();
            let entries: Vec<DeltaFileEntry> = added.iter().map(|f| f.entry.clone()).collect();
            if let Some(committed) = crate::delta::commit_file_changes(
                &self.engine,
                &self.snapshot,
                &removed_entries,
                &entries,
                data_change,
            )? {
                self.snapshot = committed;
                self.files.retain(|f| !removed.contains(&f.entry.file.path));
                self.files.extend(added);
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// Commit bare log `entries` whose footers are not yet in hand: like
    /// [`commit_files`](Self::commit_files), but the added files' footers are read
    /// only if the commit wins (a losing swap errors before touching its output).
    /// Used by test/seed paths; production writers (INSERT, compaction) already
    /// hold the footers and use `commit_files`. Only its `test-support` caller
    /// ([`test_support`](crate::test_support)) needs it, so it is gated with that
    /// feature to stay out of a plain library build.
    #[cfg(feature = "test-support")]
    pub(crate) fn commit_entries(
        &mut self,
        removed: &[ObjectPath],
        added: &[DeltaFileEntry],
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            if let Some(missing) = removed.iter().find(|path| {
                !self
                    .files
                    .iter()
                    .any(|f| f.entry.file.path.as_str() == path.as_str())
            }) {
                return Err(Error::CommitConflict {
                    location: self.location.as_str().to_string(),
                    file: missing.to_string(),
                });
            }
            let removed_entries: Vec<DeltaFileEntry> = removed
                .iter()
                .map(|path| {
                    self.files
                        .iter()
                        .find(|file| file.entry.file.path.as_str() == path.as_str())
                        .expect("removed paths were validated above")
                        .entry
                        .clone()
                })
                .collect();
            if let Some(committed) = crate::delta::commit_file_changes(
                &self.engine,
                &self.snapshot,
                &removed_entries,
                added,
                data_change,
            )? {
                // We won; only now read the added files' footers. They are built
                // whole before this copy moves onto the committed version, so a
                // failed read leaves it where it was rather than at a version
                // whose files it does not hold.
                let mut footers: HashMap<ObjectPath, Vec<Arc<RowGroupMetadata>>> = self
                    .fetch_footers(added, &self.columns)?
                    .into_iter()
                    .map(|f| (f.file.path.clone(), f.row_groups))
                    .collect();
                let mut committed_files = Vec::with_capacity(added.len());
                for entry in added {
                    let row_groups =
                        footers
                            .remove(&entry.file.path)
                            .ok_or_else(|| Error::FooterNotLoaded {
                                location: self.location.as_str().to_string(),
                                file: entry.file.path.as_str().to_string(),
                            })?;
                    committed_files.push(TableFile::new(entry.clone(), row_groups));
                }
                self.snapshot = committed;
                self.files.retain(|f| !removed.contains(&f.entry.file.path));
                self.files.extend(committed_files);
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// Reconcile `self.files` to the log `entries` of a version whose schema is
    /// `columns`: keep the footers this copy already read (carrying forward their
    /// sort bounds, which the log does not persist), fetch the footers for files
    /// not yet held, and drop files no longer present.
    ///
    /// The new file list is built whole before `self.files` is touched, so a
    /// footer fetch that fails leaves this copy exactly as it was. A copy left
    /// holding a version's files only partially would never recover: its next
    /// refresh finds that version already current and reconciles nothing.
    fn rebuild_files(
        &mut self,
        entries: Vec<DeltaFileEntry>,
        columns: &[Column],
    ) -> crate::Result<()> {
        let held: HashMap<&ObjectPath, &TableFile> = self
            .files
            .iter()
            .map(|file| (&file.entry.file.path, file))
            .collect();
        let to_fetch: Vec<DeltaFileEntry> = entries
            .iter()
            .filter(|entry| !held.contains_key(&entry.file.path))
            .cloned()
            .collect();
        let mut fetched: HashMap<ObjectPath, Vec<Arc<RowGroupMetadata>>> = self
            .fetch_footers(&to_fetch, columns)?
            .into_iter()
            .map(|f| (f.file.path.clone(), f.row_groups))
            .collect();

        let mut files = Vec::with_capacity(entries.len());
        for entry in entries {
            let row_groups = match held.get(&entry.file.path) {
                Some(held_file) => held_file.row_groups.clone(),
                None => fetched
                    .remove(&entry.file.path)
                    .ok_or_else(|| Error::FooterNotLoaded {
                        location: self.location.as_str().to_string(),
                        file: entry.file.path.as_str().to_string(),
                    })?,
            };
            files.push(TableFile::new(entry, row_groups));
        }
        self.files = files;
        Ok(())
    }

    /// Fetch the footers for `entries` — the row groups of the files this copy
    /// does not yet hold, keyed by their store identity for the caller to join.
    /// Each file's parsed schema is reconciled against `columns`, the schema
    /// declared by the version the entries come from.
    fn fetch_footers(
        &self,
        entries: &[DeltaFileEntry],
        columns: &[Column],
    ) -> crate::Result<Vec<crate::parquet::FileRowGroups>> {
        let to_fetch: Vec<DataFile> = entries
            .iter()
            .map(|e| {
                e.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &self.location)
            })
            .collect::<store::Result<_>>()?;
        Ok(crate::parquet::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            columns.to_vec().into(),
        )?)
    }

    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// The table's durable identity from the Pivot manifest, stable across
    /// renames and commits. The catalog indexes by this.
    pub fn id(&self) -> uuid::Uuid {
        self.id
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        &self.partition_by
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a compaction swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.entry.file.clone()).collect()
    }

    /// The current committed file entries, including the partition tuple and
    /// file-level statistics used to choose compaction work.
    pub(crate) fn file_entries(&self) -> impl Iterator<Item = &DeltaFileEntry> {
        self.files.iter().map(|file| &file.entry)
    }

    /// This table as the cross-datastore catalog describes it, under the `name`
    /// it is currently indexed by: the columns it declares with what they cost,
    /// and the files it holds.
    pub(crate) fn metadata(&self, name: SchemaQualifiedTableName) -> DatastoreTableMetadata {
        let column_bytes = self.column_bytes();
        let columns = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let (bytes, bytes_uncompressed) = column_bytes[position];
                DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by.contains(&column.name),
                    is_sort_key: self.sort_by.contains(&column.name),
                }
            })
            .collect();

        let files = self
            .files
            .iter()
            .map(|file| DatastoreFileMetadata {
                path: file.entry.file.path.as_str().to_string(),
                bytes: file.entry.file.size,
                bytes_uncompressed: file.uncompressed_size(),
                partition: self.format_partition(&file.entry.partition),
            })
            .collect();

        DatastoreTableMetadata {
            name,
            id: self.id.to_string(),
            columns,
            sort_by: self.sort_by.clone(),
            partition_by: self.partition_by.clone(),
            total_rows: self
                .files
                .iter()
                .flat_map(|file| file.row_groups.iter())
                .map(|row_group| row_group.num_rows.max(0) as u64)
                .sum(),
            bytes: self.files.iter().map(|file| file.entry.file.size).sum(),
            bytes_uncompressed: self.files.iter().map(TableFile::uncompressed_size).sum(),
            files,
        }
    }

    /// What each declared column costs across the table's committed files: the
    /// bytes it occupies in storage, and what they hold decoded. Chunks are per
    /// leaf, so a nested column is charged for every leaf it spans; a file that
    /// does not carry a column contributes nothing to it.
    fn column_bytes(&self) -> Vec<(u64, u64)> {
        let position_by_name: HashMap<&str, usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();

        let mut totals = vec![(0, 0); self.columns.len()];
        for row_group in self.files.iter().flat_map(|file| file.row_groups.iter()) {
            let mut leaf = 0;
            for field in row_group.schema.fields() {
                let leaves = crate::parquet::types::leaves::leaf_count(field);
                if let Some(&position) = position_by_name.get(field.name().as_str()) {
                    for chunk in &row_group.columns[leaf..leaf + leaves] {
                        totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                        totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
                    }
                }
                leaf += leaves;
            }
        }
        totals
    }

    /// A file's partition as the catalog reports it: `column=value` pairs in the
    /// table's partition order, comma-separated. Empty for an unpartitioned
    /// table, and for a file adopted before the table was partitioned, which
    /// carries no tuple (or none for some of the columns).
    fn format_partition(&self, partition: &Option<PartitionValues>) -> String {
        let Some(values) = partition else {
            return String::new();
        };
        self.partition_by
            .iter()
            .filter_map(|column| {
                let (array, _) = values.get(column)?.get();
                let formatter = ArrayFormatter::try_new(array, &FormatOptions::default()).ok()?;
                Some(format!("{column}={}", formatter.value(0)))
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`), reflecting the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<PartitionValues>)> {
        self.files
            .iter()
            .map(|f| (f.entry.file.path.clone(), f.entry.partition.clone()))
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
            .filter(|f| want.contains(&f.entry.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    /// Delete a data file (a compaction input swapped out of the manifest).
    /// `path` resolves against the table's location like any [`FileRef`] path,
    /// and must land inside it: that directory is the only storage the table
    /// owns.
    ///
    /// A file the table *adopted* (`with_pre_existing_parquets`) lives wherever its owner
    /// put it, so it is never the table's to delete: compaction may copy its rows
    /// into the table's own storage and drop it from the log, but the object
    /// itself stays. The test is where the file lands rather than how the path is
    /// spelled, so a path that names the table's own directory outright is still
    /// the table's to delete.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        let resolved = self.location.resolve(path);
        if !self.is_in_own_storage(&resolved)? {
            return Err(crate::Error::DeletingOutsideStorage {
                location: self.location.as_str().to_string(),
                file: path.as_str().to_string(),
            });
        }
        self.store.delete(&resolved)?;
        Ok(())
    }

    /// Whether `key` names an object under the table's own location. Both sides
    /// are taken to their store-root-absolute form first, so a key that resolves
    /// into the table's directory counts however it was written.
    fn is_in_own_storage(&self, key: &ObjectPath) -> crate::Result<bool> {
        let owned = self.store.absolute_key(&self.location)?;
        let owned = format!("{}/", owned.as_str().trim_end_matches('/'));
        Ok(self
            .store
            .absolute_key(key)?
            .as_str()
            .starts_with(owned.as_str()))
    }

    /// How long a file this table no longer references is kept before vacuum may
    /// delete it: the table's `delta.deletedFileRetentionDuration`, or Delta's
    /// default when unset. Vacuum ages unreferenced files against this window.
    pub fn deleted_file_retention(&self) -> std::time::Duration {
        crate::delta::deleted_file_retention(&self.snapshot)
    }

    /// Delete commit JSONs from this table's `_delta_log` that a checkpoint has
    /// made redundant and are past the table's `delta.logRetentionDuration`
    /// (evaluated against `now_ms`). The checkpoints themselves are written inline
    /// on the commit path; this is the periodic cleanup the vacuum sweep drives.
    /// The `_delta_log` naming convention lives with the rest of the Delta-format
    /// code in [`crate::delta`]; returns how many files were deleted.
    pub fn cleanup_log(&self, now_ms: u64) -> crate::Result<usize> {
        Ok(crate::delta::cleanup_log(
            self.store.as_ref(),
            &self.location,
            &self.snapshot,
            now_ms,
        )?)
    }

    /// The Parquet data objects physically present under the table location,
    /// each with its storage modification time (Unix ms) — what vacuum's orphan
    /// sweep lists to find files that no log action references. Data files are
    /// flat under the location (partition values live in the log, not the path),
    /// so a one-level listing is complete; non-`.parquet` objects (the
    /// `_delta_log`) are excluded. Paths are relative to the location, matching
    /// [`file_refs`](Self::file_refs) and what [`delete_data_file`](Self::delete_data_file)
    /// expects.
    pub fn list_data_files(&self) -> crate::Result<Vec<(ObjectPath, u64)>> {
        Ok(self
            .store
            .list(&self.location)?
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(".parquet"))
            .map(|object| (object.file.path, object.modified_unix_ms))
            .collect())
    }

    /// The Delta version this copy is at, read off the snapshot it holds.
    /// Monotonic per table; used to decide whether a published copy is newer than
    /// the catalog's.
    pub fn version(&self) -> u64 {
        self.snapshot.version()
    }

    /// A flat scan view of the files a query's pushed-down predicates cannot rule
    /// out — every surviving file's row groups concatenated in manifest order,
    /// where a row group's global index is simply its position. A file survives
    /// when its recorded partition tuple can still match every `partition_filters`
    /// *and* its Parquet stats can still match every `stat_filters`. Empty filters
    /// keep every file.
    ///
    /// Read-only: it is built entirely from the row groups this copy already
    /// holds, so it does **no** I/O. Every surviving file's footer must have
    /// been fetched (the refresh path keeps `files` synced to the manifest); a
    /// missing one is an error, never a silently narrower scan.
    ///
    /// The row group's global index is its position in this returned flat list, so
    /// a scan and its materialize must build it from the *same* filters (they do:
    /// both go through the binding's predicates) to address the same groups.
    pub fn build_scan_view(
        &self,
        partition_filters: &[PartitionEqFilter],
        stat_filters: &[ColumnStatFilter],
    ) -> crate::Result<Arc<ParquetTable>> {
        let mut row_groups = Vec::new();
        for file in self
            .files
            .iter()
            .filter(|f| {
                f.entry
                    .maybe_matches_partition(&self.partition_by, partition_filters)
            })
            .filter(|f| f.entry.maybe_matches_stats(stat_filters))
        {
            row_groups.extend(file.row_groups.iter().cloned());
        }
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    /// Whether each column (in [`columns`](Self::columns) order) can hold SQL
    /// NULLs, computed from the loaded footers rather than the declared schema:
    /// the planner uses the flags to route between the branch-free and the
    /// null-aware execution paths, so they must reflect whether the data can
    /// actually hold NULLs. An empty table reports every column nullable.
    pub fn nullability(&self) -> Vec<bool> {
        self.columns
            .iter()
            .map(|column| self.files.is_empty() || self.column_may_hold_nulls(&column.name))
            .collect()
    }

    /// Whether any loaded row group can hold a NULL in `name`. A column that is
    /// `REQUIRED` everywhere cannot; an `OPTIONAL` one (writers like DuckDB mark
    /// every column `OPTIONAL` even when no value is ever NULL) is refined by
    /// the chunk's `null_count` statistic when the schema is flat enough to map
    /// fields to leaves; a file missing the column entirely reads as all-NULL.
    fn column_may_hold_nulls(&self, name: &str) -> bool {
        self.files
            .iter()
            .flat_map(|file| file.row_groups.iter())
            .any(|rg| {
                let Ok(field_idx) = rg.schema.index_of(name) else {
                    return true;
                };
                if !rg.schema.field(field_idx).is_nullable() {
                    return false;
                }
                // Nested fields (e.g. a shredded variant) span several leaves,
                // so the field-to-chunk mapping below does not hold; stay
                // conservative for the whole row group.
                if rg.columns.len() != rg.schema.fields().len() {
                    return true;
                }
                rg.columns[field_idx]
                    .statistics
                    .as_ref()
                    .and_then(|stats| stats.null_count)
                    .is_none_or(|null_count| null_count > 0)
            })
    }

    /// The declared schema as one shareable slice: what a footer load
    /// reconciles each file's parsed schema against.
    fn declared_columns(&self) -> Arc<[Column]> {
        self.columns.clone().into()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root
    /// (see [`DeltaDatastore::store_description`](crate::DeltaDatastore::store_description))
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

    /// Re-encode `inputs`' rows into fresh files of at most `max_file_bytes` and
    /// return them, uploaded but not yet part of the table: the read half of a
    /// compaction, which [`compact_table_files`](crate::compact_table_files) then
    /// swaps in for the inputs. Merged rows past that size continue in the next
    /// file rather than growing this one, so a sorted table's outputs cover
    /// successive key ranges. The merged files are written over the
    /// shared io_uring ring by the same upload operators an INSERT uses, and
    /// their row groups come straight from the writer's own footer metadata, so
    /// no footer is re-read.
    ///
    /// Nothing here touches the log, so it runs off any read copy that holds the
    /// inputs, and a merge of a large batch never stalls another writer's commit.
    /// `inputs` must share one partition tuple; the caller batches them so.
    pub(crate) fn merge_files(
        &self,
        inputs: &[FileRef],
        target_rows_per_group: usize,
        max_file_bytes: u64,
    ) -> crate::Result<Vec<TableFile>> {
        // The selector guarantees that all inputs share one partition, so its
        // recorded tuple can key every decoded batch without repartitioning it.
        let parquet = self.parquet_table_for(inputs);
        let columns = parquet.schema().fields().len();
        let scan = crate::parquet::table_input(
            &self.dispatcher,
            &parquet,
            Projection::all(columns),
            false,
        );
        let partition = inputs.first().and_then(|input| {
            self.files
                .iter()
                .find(|file| file.entry.file.path == input.path)
                .expect("a compaction input belongs to this table")
                .entry
                .partition
                .clone()
        });
        let uploaded_files = Arc::new(Injector::new());
        let encoded = crate::parquet::writing::encode_compaction_batches_spec(
            scan,
            parquet.schema().clone(),
            partition,
            Arc::from(self.partition_by()),
            Arc::from(self.sort_by()),
            target_rows_per_group,
            usize::try_from(max_file_bytes).unwrap_or(usize::MAX),
        );
        let spec = super::insert_sink::upload_files_spec(
            self.store(),
            self.location.clone(),
            self.id(),
            self.declared_columns(),
            uploaded_files.clone(),
            encoded,
        );
        // Drive encode → upload to completion; the emitted row-count batch is
        // ignored, and the uploaded files arrive on `uploaded_files`.
        spec.collect()?;

        let mut added: Vec<TableFile> = Vec::new();
        loop {
            match uploaded_files.steal() {
                Steal::Success(uploaded) => {
                    // table_id is ignored: this table is the commit target.
                    let super::insert_sink::UploadedFile {
                        file,
                        partition,
                        row_groups,
                        ..
                    } = uploaded;
                    let stats = Some(std::sync::Arc::new(crate::parquet::aggregate_file_stats(
                        &row_groups,
                    )));
                    let entry = DeltaFileEntry {
                        file,
                        partition,
                        stats,
                    };
                    added.push(TableFile::new(entry, row_groups));
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        Ok(added)
    }
}

/// The partition a file with these `stats` belongs to: each partition column
/// mapped to the value it holds when that value is constant across the file (its
/// min equals its max). Returns `None` if any partition column is absent from the
/// stats or varies within the file, so a file that does not sit in a single
/// partition is left unstamped rather than mislabeled.
fn constant_partition(stats: &FileStats, partition_by: &[String]) -> Option<PartitionValues> {
    let mut values = PartitionValues::new();
    for column in partition_by {
        let min = Scalar::new(ArrayRef::clone(stats.min_values.get(column)?));
        let max = Scalar::new(ArrayRef::clone(stats.max_values.get(column)?));
        if scalar_equal(&min, &max) != Some(true) {
            return None;
        }
        values.insert(column.clone(), min);
    }
    Some(values)
}
