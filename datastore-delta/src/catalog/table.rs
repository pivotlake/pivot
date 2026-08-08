//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::Error;
use crate::manifest::{
    ColumnStatFilter, DeltaFileEntry, FileStats, PartitionEqFilter, PartitionValues, scalar_equal,
};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use arrow_array::{ArrayRef, Scalar};
use crossbeam_deque::{Injector, Steal};
use dispatch::{DataFlowDispatcher, Projection};
use planner::catalog::Column;

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
            entry.stats = Some(crate::parquet::aggregate_file_stats(&row_groups));
        }
        Self { entry, row_groups }
    }
}

/// One committed version of a table: the Delta snapshot, the schema and layout
/// declared by it, and the [`TableFile`]s active at it (each a log entry paired
/// with its materialized row groups).
///
/// The snapshot and the files are one indivisible unit. A snapshot paired with a
/// file list built at a different version would name files its holder does not
/// hold, and nothing would heal that: the holder's next refresh finds its version
/// already current and reconciles nothing. So every step that moves one moves the
/// other, and builds the whole value before publishing it.
struct TableVersion {
    /// The Delta snapshot this version is at: the version its `files` were built
    /// from, and the base a commit riding it is written on top of.
    snapshot: Arc<delta_kernel::Snapshot>,
    /// The table's declared columns (schema), as of this version. Shared rather
    /// than owned, so the versions a table passes through hand the schema and
    /// layout along by pointer instead of re-copying them per commit.
    columns: Arc<[Column]>,
    /// Partition columns, in order; empty = unpartitioned.
    partition_by: Arc<[String]>,
    /// Sort columns, in order; empty = unsorted.
    sort_by: Arc<[String]>,
    /// The active files: each a log entry (identity + partition + stats) paired
    /// with its materialized footers.
    files: Vec<TableFile>,
}

impl TableVersion {
    /// One version, from the schema and layout a log read reports and the files
    /// materialized for it.
    fn new(
        snapshot: Arc<delta_kernel::Snapshot>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        files: Vec<TableFile>,
    ) -> Self {
        Self {
            snapshot,
            columns: columns.into(),
            partition_by: partition_by.into(),
            sort_by: sort_by.into(),
            files,
        }
    }

    /// This version plus the commit just written on top of it: `snapshot` is the
    /// version that commit produced, and the files it changed are folded onto the
    /// ones this version holds. Schema and layout carry over, since a data-file
    /// commit does not touch them.
    ///
    /// A path is carried by exactly one file, so an added path displaces the file
    /// already holding it rather than joining it. The log reads the same way (the
    /// later `Add` wins), and a file listed twice would have its rows scanned, and
    /// counted, twice.
    fn advance(
        &self,
        snapshot: Arc<delta_kernel::Snapshot>,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Self {
        let replaced: HashSet<&ObjectPath> =
            added.iter().map(|file| &file.entry.file.path).collect();
        let mut files: Vec<TableFile> = self
            .files
            .iter()
            .filter(|f| {
                !removed.contains(&f.entry.file.path) && !replaced.contains(&f.entry.file.path)
            })
            .cloned()
            .collect();
        files.extend(added);
        Self {
            snapshot,
            columns: self.columns.clone(),
            partition_by: self.partition_by.clone(),
            sort_by: self.sort_by.clone(),
            files,
        }
    }
}

/// Replace `target` with `candidate` when `candidate` is at a later version, so
/// that of two views of one table the newer wins. Both directions matter: a
/// writer takes up the version the table's previous writer left, and a copy that
/// read a newer version off the log leaves it for the table's next writer.
fn keep_newer(target: &mut Arc<TableVersion>, candidate: &Arc<TableVersion>) {
    if candidate.snapshot.version() > target.snapshot.version() {
        *target = candidate.clone();
    }
}

/// What every copy of one table shares: the identity and storage that are fixed
/// for the table's life, and the latest version committed in this process, behind
/// the lock a writer commits under.
struct SharedTable {
    /// The table's durable identity, from the Delta `metaData.id` (minted once at
    /// creation, stable across renames and every commit). The catalog indexes by
    /// this.
    id: uuid::Uuid,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    store: Arc<dyn ObjectStore>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
    /// The Kernel engine this table's log is read and written through, shared
    /// with every other table of the datastore that opened it.
    engine: crate::delta::DeltaEngine,
    /// The newest version this process holds, and the lock a commit takes for its
    /// whole duration. Writers of one table therefore take it in turn, each
    /// building on the version the one before produced, so two in-process writers
    /// never aim at the same log version: the compare-and-swap is left to settle
    /// races against *other* processes only. Without it, every concurrent writer
    /// but one loses its swap and pays a full reload (a log listing plus the
    /// footers of the winner's files) before retrying, which is quadratic in the
    /// number of writers committing at once.
    committed: Mutex<Arc<TableVersion>>,
}

impl SharedTable {
    /// Take the commit lock, waiting for the writer that holds it.
    ///
    /// Poisoning is recovered from rather than propagated: the slot only ever
    /// holds a whole committed version, installed as the last step of a commit
    /// that already won its swap, so a panic mid-commit leaves the value intact
    /// and the next writer can build on it. Failing every later commit and
    /// refresh of the table instead would turn one panicking write into a table
    /// that never advances again.
    fn lock_committed(&self) -> std::sync::MutexGuard<'_, Arc<TableVersion>> {
        self.committed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take the commit lock only if no writer holds it. A refresh syncs with the
    /// shared version through this: reading the log is what a refresh is for, and
    /// it is never worth parking a reader (or a runtime worker) behind a commit's
    /// object-store I/O for a step that only saves work.
    fn try_lock_committed(&self) -> Option<std::sync::MutexGuard<'_, Arc<TableVersion>>> {
        match self.committed.try_lock() {
            Ok(committed) => Some(committed),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }
}

/// The catalog's master record of one table: its declared schema, partitioning,
/// version, and its active [`TableFile`]s.
///
/// A copy reads a **frozen** version: cloning one is two pointer bumps and never
/// blocks, and the version it reads cannot move under it, so a query binds,
/// scans, and late-materializes against one consistent file set. Writing is the
/// other half: a writer (INSERT, compaction) clones one out of the catalog and
/// commits its uploaded files or a compaction swap, which advances the version
/// this copy reads and the one [`SharedTable::committed`] hands the table's next
/// writer. [`refresh`](Self::refresh) advances a copy onto the latest committed
/// version, re-reading only the footers it doesn't already hold.
#[derive(Clone)]
pub struct CatalogTable {
    shared: Arc<SharedTable>,
    /// The version this copy reads: whatever the table stood at when the copy was
    /// made, held frozen until this copy refreshes or commits.
    version: Arc<TableVersion>,
}

impl CatalogTable {
    /// Reassemble a persisted table from its table-level metadata and the per-file
    /// [`TableFile`]s (log entry + footers) just built for it -- the reopen path.
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
        Self::at_version(
            id,
            location,
            TableVersion::new(snapshot, columns, partition_by, sort_by, files),
            store,
            dispatcher,
            engine,
        )
    }

    /// Open a table at `version`, minting the state its copies share. Every later
    /// copy comes from cloning this one, so they all commit through the one lock;
    /// calling this twice for the same table would give the two families separate
    /// locks, which is why only the load and create paths do.
    fn at_version(
        id: uuid::Uuid,
        location: ObjectPath,
        version: TableVersion,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        engine: crate::delta::DeltaEngine,
    ) -> Self {
        let version = Arc::new(version);
        Self {
            shared: Arc::new(SharedTable {
                id,
                location,
                store,
                dispatcher,
                engine,
                committed: Mutex::new(version.clone()),
            }),
            version,
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
        Ok(Self::at_version(
            id,
            location,
            TableVersion::new(snapshot, columns, partition_by, sort_by, files),
            store,
            dispatcher,
            engine,
        ))
    }

    /// Advance this copy to the latest committed version, reconciling its files
    /// to it. Returns whether it advanced; `Ok(false)` means this copy was
    /// already current, which costs one log listing and no file reads (Kernel
    /// updates the snapshot in hand rather than rebuilding it).
    ///
    /// Syncs with the version the table's writers share, in both directions:
    /// starting from a version a writer in this process already committed spares
    /// the log read that version's files (they are materialized, sort bounds and
    /// all), and a version read off the log is left for the table's next writer,
    /// so a commit another process wrote is picked up here rather than by that
    /// writer losing a swap to discover it. Both steps skip a table under an
    /// in-flight commit rather than wait for it: they only save work, and the log
    /// read below reaches the same version either way.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let mut advanced = false;
        if let Some(committed) = self.shared.try_lock_committed() {
            let before = self.version();
            keep_newer(&mut self.version, &committed);
            advanced = self.version() > before;
        }
        if !self.advance_to_latest()? {
            return Ok(advanced);
        }
        if let Some(mut committed) = self.shared.try_lock_committed() {
            keep_newer(&mut committed, &self.version);
        }
        Ok(true)
    }

    /// Advance this copy alone onto the latest committed version, leaving the
    /// shared state untouched. The commit loop refreshes through this, since it
    /// already holds the commit lock and publishes what it commits.
    fn advance_to_latest(&mut self) -> crate::Result<bool> {
        let Some(state) = crate::delta::refresh_table(&self.version.snapshot, &self.shared.engine)?
        else {
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
        let files = self.reconcile_files(file_entries, &columns)?;
        self.version = Arc::new(TableVersion::new(
            snapshot,
            columns,
            partition_by,
            sort_by,
            files,
        ));
        Ok(true)
    }

    /// The commit loop every mutation runs, under the table's commit lock: check
    /// each `removed` path is still present (else [`Error::CommitConflict`] -- a
    /// concurrent writer swapped it out), then commit the removes plus `added`'s
    /// log entries through Delta Kernel on top of the newest version this process
    /// holds. A win therefore lands exactly one version above the state that
    /// version's files describe, so `added` (each a log entry *and* its footers)
    /// folds straight onto them -- no footer re-read, and no other writer's files
    /// to miss. A writer in *another* process that got there first conflicts
    /// instead, and this copy reloads onto the newer version and retries; so does
    /// a removal such a writer already swapped out.
    ///
    /// Holding the lock across the whole commit is what keeps in-process writers
    /// off each other (see [`SharedTable::committed`]).
    ///
    /// Runs off the dispatch workers: a retry drives a footer-fetch dataflow, and
    /// every commit path commits from a blocking thread, never a pinned worker
    /// (which would deadlock driving a dataflow from inside one).
    pub(crate) fn commit_files(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
        data_change: bool,
    ) -> crate::Result<()> {
        let entries: Vec<DeltaFileEntry> = added.iter().map(|f| f.entry.clone()).collect();
        self.commit_and_publish(removed, &entries, data_change, |_| Ok(added))
    }

    /// The one commit protocol both entry points run: take the table's commit
    /// lock, build on the version its last writer left, retry the swap until it
    /// lands, then install what it produced. `files_of` supplies the added files
    /// paired with their footers, and runs only once the swap has won.
    fn commit_and_publish(
        &mut self,
        removed: &[ObjectPath],
        entries: &[DeltaFileEntry],
        data_change: bool,
        files_of: impl FnOnce(&Self) -> crate::Result<Vec<TableFile>>,
    ) -> crate::Result<()> {
        let shared = self.shared.clone();
        let mut committed = shared.lock_committed();
        keep_newer(&mut self.version, &committed);
        let snapshot = loop {
            self.check_still_present(removed)?;
            if let Some(snapshot) = crate::delta::commit_file_changes(
                &shared.engine,
                &self.version.snapshot,
                removed,
                entries,
                data_change,
            )? {
                break snapshot;
            }
            // The swap did not land, so the log must have moved for the retry to
            // stand a different chance. It always has when another writer took
            // the version; a swap refused with the log where we left it would
            // otherwise have this loop reissue the same commit forever, holding
            // the table's commit lock while it spun.
            if !self.advance_to_latest()? {
                return Err(Error::CommitStalled {
                    location: shared.location.as_str().to_string(),
                    version: self.version(),
                });
            }
        };
        let added = files_of(self)?;
        self.version = Arc::new(self.version.advance(snapshot, removed, added));
        *committed = self.version.clone();
        Ok(())
    }

    /// Fail with [`Error::CommitConflict`] if any of `removed` is no longer among
    /// this copy's files: a writer this commit cannot win against already swapped
    /// it out, so the removal has nothing to remove.
    fn check_still_present(&self, removed: &[ObjectPath]) -> crate::Result<()> {
        let Some(missing) = removed.iter().find(|path| {
            !self
                .version
                .files
                .iter()
                .any(|f| f.entry.file.path.as_str() == path.as_str())
        }) else {
            return Ok(());
        };
        Err(Error::CommitConflict {
            location: self.shared.location.as_str().to_string(),
            file: missing.to_string(),
        })
    }

    /// Commit freshly-uploaded files whose footers are already built — the INSERT
    /// append. A plain add (`data_change = true`), no removes.
    pub(crate) fn commit_uploaded_files(&mut self, uploaded: Vec<TableFile>) -> crate::Result<()> {
        self.commit_files(&[], uploaded, true)
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
        // The footers are read by `files_of`, so they are read only once the swap
        // has won, and are built whole before this copy moves onto the committed
        // version: a failed read leaves it where it was rather than at a version
        // whose files it does not hold.
        self.commit_and_publish(removed, added, data_change, |table| {
            let mut footers: HashMap<ObjectPath, Vec<Arc<RowGroupMetadata>>> = table
                .fetch_footers(added, &table.version.columns)?
                .into_iter()
                .map(|f| (f.file.path.clone(), f.row_groups))
                .collect();
            let mut committed_files = Vec::with_capacity(added.len());
            for entry in added {
                let row_groups =
                    footers
                        .remove(&entry.file.path)
                        .ok_or_else(|| Error::FooterNotLoaded {
                            location: table.shared.location.as_str().to_string(),
                            file: entry.file.path.as_str().to_string(),
                        })?;
                committed_files.push(TableFile::new(entry.clone(), row_groups));
            }
            Ok(committed_files)
        })
    }

    /// The file list of a version whose log `entries` and schema are given,
    /// reconciled against the one this copy holds: keep the footers it already
    /// read (carrying forward their sort bounds, which the log does not persist),
    /// fetch the footers for files not yet held, and drop files no longer present.
    ///
    /// Returned rather than assigned, so a footer fetch that fails leaves this
    /// copy exactly as it was. A copy left holding a version's files only
    /// partially would never recover: its next refresh finds that version already
    /// current and reconciles nothing.
    fn reconcile_files(
        &self,
        entries: Vec<DeltaFileEntry>,
        columns: &[Column],
    ) -> crate::Result<Vec<TableFile>> {
        let held: HashMap<&ObjectPath, &TableFile> = self
            .version
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
        for mut entry in entries {
            let row_groups = match held.get(&entry.file.path) {
                Some(held_file) => {
                    // The log does not persist sort bounds, so carry forward the
                    // ones this copy already recorded for a file it still holds.
                    entry.sort_bounds = held_file.entry.sort_bounds.clone();
                    held_file.row_groups.clone()
                }
                None => fetched
                    .remove(&entry.file.path)
                    .ok_or_else(|| Error::FooterNotLoaded {
                        location: self.shared.location.as_str().to_string(),
                        file: entry.file.path.as_str().to_string(),
                    })?,
            };
            files.push(TableFile::new(entry, row_groups));
        }
        Ok(files)
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
                    .into_data_file(self.shared.store.as_ref(), &self.shared.location)
            })
            .collect::<store::Result<_>>()?;
        Ok(crate::parquet::load_file_row_groups(
            &self.shared.dispatcher,
            &to_fetch,
            columns.to_vec().into(),
        )?)
    }

    pub fn files(&self) -> &[TableFile] {
        &self.version.files
    }

    /// The table's durable identity (Delta `metaData.id`), stable across renames
    /// and commits. The catalog indexes by this.
    pub fn id(&self) -> uuid::Uuid {
        self.shared.id
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        &self.version.partition_by
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        &self.version.sort_by
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a compaction swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.version
            .files
            .iter()
            .map(|f| f.entry.file.clone())
            .collect()
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`), reflecting the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<PartitionValues>)> {
        self.version
            .files
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
            .version
            .files
            .iter()
            .filter(|f| want.contains(&f.entry.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    /// Write `bytes` as a new data file at `path` (resolved against the table's
    /// location like any [`FileRef`] path), returning its [`FileRef`] to be
    /// committed into the manifest.
    pub fn write_data_file(&self, path: ObjectPath, bytes: &[u8]) -> crate::Result<FileRef> {
        self.shared
            .store
            .put(&self.shared.location.resolve(&path), bytes)?;
        Ok(FileRef {
            path,
            size: bytes.len() as u64,
        })
    }

    /// Delete a data file (a compaction input swapped out of the manifest).
    /// `path` resolves against the table's location like any [`FileRef`] path.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        self.shared
            .store
            .delete(&self.shared.location.resolve(path))?;
        Ok(())
    }

    /// How long a file this table no longer references is kept before vacuum may
    /// delete it: the table's `delta.deletedFileRetentionDuration`, or Delta's
    /// default when unset. Vacuum ages unreferenced files against this window.
    pub fn deleted_file_retention(&self) -> std::time::Duration {
        crate::delta::deleted_file_retention(&self.version.snapshot)
    }

    /// Delete commit JSONs from this table's `_delta_log` that a checkpoint has
    /// made redundant and are past the table's `delta.logRetentionDuration`
    /// (evaluated against `now_ms`). The checkpoints themselves are written inline
    /// on the commit path; this is the periodic cleanup the vacuum sweep drives.
    /// The `_delta_log` naming convention lives with the rest of the Delta-format
    /// code in [`crate::delta`]; returns how many files were deleted.
    pub fn cleanup_log(&self, now_ms: u64) -> crate::Result<usize> {
        Ok(crate::delta::cleanup_log(
            self.shared.store.as_ref(),
            &self.shared.location,
            &self.version.snapshot,
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
            .shared
            .store
            .list(&self.shared.location)?
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(".parquet"))
            .map(|object| (object.file.path, object.modified_unix_ms))
            .collect())
    }

    /// The Delta version this copy is at, read off the snapshot it holds.
    /// Monotonic per table; used to decide whether a published copy is newer than
    /// the catalog's.
    pub fn version(&self) -> u64 {
        self.version.snapshot.version()
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
            .version
            .files
            .iter()
            .filter(|f| {
                f.entry
                    .maybe_matches_partition(&self.version.partition_by, partition_filters)
            })
            .filter(|f| f.entry.maybe_matches_stats(stat_filters))
        {
            row_groups.extend(file.row_groups.iter().cloned());
        }
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.version.columns.to_vec()
    }

    /// Whether each column (in [`columns`](Self::columns) order) can hold SQL
    /// NULLs, computed from the loaded footers rather than the declared schema:
    /// the planner uses the flags to route between the branch-free and the
    /// null-aware execution paths, so they must reflect whether the data can
    /// actually hold NULLs. An empty table reports every column nullable.
    pub fn nullability(&self) -> Vec<bool> {
        self.version
            .columns
            .iter()
            .map(|column| self.version.files.is_empty() || self.column_may_hold_nulls(&column.name))
            .collect()
    }

    /// Whether any loaded row group can hold a NULL in `name`. A column that is
    /// `REQUIRED` everywhere cannot; an `OPTIONAL` one (writers like DuckDB mark
    /// every column `OPTIONAL` even when no value is ever NULL) is refined by
    /// the chunk's `null_count` statistic when the schema is flat enough to map
    /// fields to leaves; a file missing the column entirely reads as all-NULL.
    fn column_may_hold_nulls(&self, name: &str) -> bool {
        self.version
            .files
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
        self.version.columns.clone()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root
    /// (see [`DeltaDatastore::store_description`](crate::DeltaDatastore::store_description))
    /// to know the physical location.
    pub fn location(&self) -> &str {
        self.shared.location.as_str()
    }

    pub(crate) fn object_location(&self) -> &ObjectPath {
        &self.shared.location
    }

    pub(crate) fn store(&self) -> Arc<dyn ObjectStore> {
        self.shared.store.clone()
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
            &self.shared.dispatcher,
            &parquet,
            Projection::all(columns),
            false,
        );
        let uploaded_files = Arc::new(Injector::new());
        let spec = super::insert_sink::encode_and_upload_spec(
            self.store(),
            self.shared.location.clone(),
            self.id(),
            self.declared_columns(),
            uploaded_files.clone(),
            scan,
            self.version.partition_by.clone(),
            self.version.sort_by.clone(),
            target_rows_per_group,
            target_row_groups_per_file,
            &self.shared.dispatcher,
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
                        sort_bounds,
                        row_groups,
                        ..
                    } = uploaded;
                    let stats = Some(crate::parquet::aggregate_file_stats(&row_groups));
                    let entry = DeltaFileEntry {
                        file,
                        partition,
                        sort_bounds,
                        stats,
                    };
                    added.push(TableFile::new(entry, row_groups));
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        let files: Vec<FileRef> = added.iter().map(|f| f.entry.file.clone()).collect();
        let removed: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
        // A compaction swap rearranges bytes without changing rows.
        if let Err(error) = self.commit_files(&removed, added, false) {
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
