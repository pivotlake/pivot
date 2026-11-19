//! The catalog's in-memory handle to one table at one committed version,
//! whichever format keeps it: what every reader, writer and maintenance sweep
//! holds and evolves, over the format's own [`Table`].

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::Error;
use crate::file_entry::FileEntry;
use crate::table_format::{RemovedFiles, Table, VisitPartition};
use catalog::datastore::DatastoreTableMetadata;
use crossbeam_deque::{Injector, Steal};
use dispatch::DataFlowDispatcher;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::writing::upload::UploadedFiles;
use parquet_engine::{ParquetTable, RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use table_file_index::PartitionFiles;
use uuid::Uuid;

/// One data file of a table, materialized: its metadata entry ([`FileEntry`]:
/// identity, partition tuple, stats) paired with the row groups read from its
/// footer (file-local order). What a writer hands a commit: the entry is the
/// metadata the commit records, the row groups the content the next scan
/// reads.
#[derive(Clone)]
pub struct TableFile {
    pub(crate) entry: FileEntry,
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's entry with the row groups read from its footer.
    ///
    /// Derive the file's stats from its footers when the entry carries none, so
    /// file-level range pruning always has min/max bounds. Usually the entry
    /// already has them (a fresh INSERT records them, and a reload reads the
    /// persisted stats), so this covers only a file with neither: one adopted
    /// at CREATE, or whose metadata recorded no stats.
    pub(crate) fn new(mut entry: FileEntry, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        if entry.stats.is_none() && !row_groups.is_empty() {
            entry.stats = Some(Arc::new(parquet_engine::aggregate_file_stats(&row_groups)));
        }
        Self { entry, row_groups }
    }

    /// The file's store identity (path and size), for a caller that names the
    /// file outside the table's own module (reporting a merge's outputs,
    /// deleting one whose commit failed).
    pub(crate) fn file_ref(&self) -> &FileRef {
        &self.entry.file
    }
}

/// A table's own storage: the directory under the database root its data and
/// metadata live in, and the store it is read and deleted through. The one
/// place a table's objects are listed and deleted, so a deletion never reaches
/// outside the directory the table owns.
#[derive(Clone)]
pub(crate) struct TableStorage {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
}

impl TableStorage {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, location: ObjectPath) -> Self {
        Self { store, location }
    }

    /// The Parquet data objects physically present under the table location,
    /// each with its storage modification time (Unix ms): what a vacuum sweep
    /// lists to find files no metadata references. Data files are flat under
    /// the location (partition values live in the metadata, not the path), so a
    /// one-level listing is complete; non-`.parquet` objects (the metadata) are
    /// excluded. Paths are relative to the location, matching the table's
    /// [`FileRef`]s and what [`delete_data_file`](Self::delete_data_file)
    /// expects.
    pub(crate) fn list_data_files(&self) -> crate::Result<Vec<(ObjectPath, u64)>> {
        Ok(self
            .store
            .list(&self.location)?
            .objects
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(".parquet"))
            .map(|object| (object.file.path, object.modified_unix_ms))
            .collect())
    }

    /// Delete a data file. `path` resolves against the table's location like
    /// any [`FileRef`] path, and must land inside it: that directory is the
    /// only storage the table owns.
    ///
    /// A file the table *adopted* (`with_pre_existing_parquets`) lives wherever
    /// its owner put it, so it is never the table's to delete: compaction may
    /// copy its rows into the table's own storage and drop it from the
    /// metadata, but the object itself stays. The test is where the file lands
    /// rather than how the path is spelled, so a path that names the table's
    /// own directory outright is still the table's to delete.
    pub(crate) fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        let resolved = self.location.resolve(path);
        if !self.is_in_own_storage(&resolved)? {
            return Err(Error::DeletingOutsideStorage {
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
    pub(crate) fn is_in_own_storage(&self, key: &ObjectPath) -> crate::Result<bool> {
        let owned = self.store.absolute_key(&self.location)?;
        let owned = format!("{}/", owned.as_str().trim_end_matches('/'));
        Ok(self
            .store
            .absolute_key(key)?
            .as_str()
            .starts_with(owned.as_str()))
    }
}

/// The commits in-process writers wait to make to one table, in the order
/// they arrived, and the lock the writer making the next of them holds.
///
/// A writer queues its commit, then takes the lock. Holding it, the writer
/// makes the commits at the front of the queue, its own or not, until its own
/// is made: the inserts waiting one after another go into the table as one
/// version, and a commit that removes files is a version of its own. A writer
/// that gets the lock after another made its commit finds the outcome waiting
/// and writes nothing. So the store is asked for one version per group of
/// waiting inserts, however many writers wait, and a rewrite waits behind at
/// most the commits queued before it.
#[derive(Default)]
pub(super) struct CommitQueue {
    committing: Mutex<()>,
    waiting: Mutex<VecDeque<QueuedCommit>>,
}

/// One writer's commit, waiting to be made.
struct QueuedCommit {
    /// The partition of the files the commit removes, as a walk names it.
    removed_partition: Vec<u8>,
    removed_paths: Vec<ObjectPath>,
    added: Vec<TableFile>,
    data_change: bool,
    outcome: Arc<CommitOutcome>,
}

/// Where a queued commit's writer finds how the commit went, once a writer
/// holding the lock has made it.
#[derive(Default)]
pub(super) struct CommitOutcome {
    committed: Mutex<Option<Result<(), Arc<Error>>>>,
}

impl CommitOutcome {
    /// How the commit went, when it has been made. A failure of a version
    /// that held other writers' commits too is shared with them
    /// ([`Error::SharedCommit`]); the only writer of a version gets the
    /// failure itself.
    pub(super) fn take(&self) -> Option<crate::Result<()>> {
        let committed = self.committed.lock().unwrap().take()?;
        Some(committed.map_err(|error| Arc::try_unwrap(error).unwrap_or_else(Error::SharedCommit)))
    }
}

/// The queued commits that go into a table as one version.
pub(super) struct CommitGroup {
    removed_partition: Vec<u8>,
    removed_paths: Vec<ObjectPath>,
    added: Vec<TableFile>,
    data_change: bool,
    outcomes: Vec<Arc<CommitOutcome>>,
}

impl CommitGroup {
    /// Commit the group's files against `table` as one version.
    pub(super) fn commit_to(&mut self, table: &mut CatalogTable) -> crate::Result<()> {
        let removed = RemovedFiles {
            partition: &self.removed_partition,
            paths: &self.removed_paths,
        };
        table.commit_files(removed, std::mem::take(&mut self.added), self.data_change)
    }

    /// Tell every writer of the group how its commit went. The last writer
    /// told is handed the outcome itself, so the only writer of a version
    /// holds its failure alone.
    pub(super) fn report(mut self, committed: crate::Result<()>) {
        let committed = committed.map_err(Arc::new);
        let last = self
            .outcomes
            .pop()
            .expect("a group holds at least one commit");
        for outcome in self.outcomes {
            *outcome.committed.lock().unwrap() = Some(committed.clone());
        }
        *last.committed.lock().unwrap() = Some(committed);
    }
}

impl CommitQueue {
    /// Queue a commit behind those already waiting. The writer then takes
    /// the lock ([`lock_to_commit`](Self::lock_to_commit)) and looks at the
    /// outcome handed back.
    pub(super) fn enqueue(
        &self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
        data_change: bool,
    ) -> Arc<CommitOutcome> {
        let outcome = Arc::new(CommitOutcome::default());
        self.waiting.lock().unwrap().push_back(QueuedCommit {
            removed_partition: removed.partition.to_vec(),
            removed_paths: removed.paths.to_vec(),
            added,
            data_change,
            outcome: Arc::clone(&outcome),
        });
        outcome
    }

    /// Wait to be the one writer committing to the table.
    pub(super) fn lock_to_commit(&self) -> MutexGuard<'_, ()> {
        self.committing.lock().unwrap()
    }

    /// Take the commits at the front of the queue that make the next
    /// version: one that removes files alone, else every insert waiting
    /// before the next commit that removes files. `None` when nothing waits.
    pub(super) fn take_next_group(&self) -> Option<CommitGroup> {
        let mut waiting = self.waiting.lock().unwrap();
        let first = waiting.pop_front()?;
        let mut group = CommitGroup {
            removed_partition: first.removed_partition,
            removed_paths: first.removed_paths,
            added: first.added,
            data_change: first.data_change,
            outcomes: vec![first.outcome],
        };
        if !group.removed_paths.is_empty() {
            return Some(group);
        }
        while let Some(next) = waiting.front()
            && next.removed_paths.is_empty()
            && next.data_change == group.data_change
        {
            let next = waiting.pop_front().unwrap();
            group.added.extend(next.added);
            group.outcomes.push(next.outcome);
        }
        Some(group)
    }
}

/// A table at one committed version: its durable identity and storage, the
/// format's own view of that version, and the queue in-process writers commit
/// on top of it through.
///
/// A copy handed out for reading drifts as soon as anything commits; it is
/// never a commit base, since writes go through the datastore, which commits
/// on the live copy under the lock and publishes the result.
#[derive(Clone)]
pub struct CatalogTable {
    /// The table's durable identity from the database index, minted once at
    /// creation and stable across renames and every commit. The catalog
    /// indexes by this; the format's own metadata independently owns whatever
    /// id it mints.
    id: Uuid,
    storage: TableStorage,
    /// The version this copy is at, as its format holds it. Replaced whole by
    /// a refresh or a commit.
    table: Arc<dyn Table>,
    /// The pool a merge decodes on, so the mutators need no dispatcher passed
    /// in.
    dispatcher: DataFlowDispatcher,
    /// This exists only for throughput: the format's own conditional write
    /// already ensures correctness, but concurrent in-process writers can all
    /// start at `V` and race for `V + 1`, forcing every loser to reload and
    /// retry, and each would write a version of its own.
    commit_queue: Arc<CommitQueue>,
}

impl CatalogTable {
    pub(super) fn new(
        id: Uuid,
        storage: TableStorage,
        table: Arc<dyn Table>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            id,
            storage,
            table,
            dispatcher,
            commit_queue: Arc::default(),
        }
    }

    /// An owned handle to this table's
    /// [commit queue](field@Self::commit_queue).
    pub(super) fn commit_queue(&self) -> Arc<CommitQueue> {
        self.commit_queue.clone()
    }

    /// Advance this copy to the latest committed version. Returns whether it
    /// advanced; `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(table) = self.table.refresh()? else {
            return Ok(false);
        };
        self.table = table;
        Ok(true)
    }

    /// Commit file changes against this copy's version. Each attempt first
    /// verifies that every `removed` path is still live, returning
    /// [`Error::CommitConflict`] if another commit removed one. After a
    /// successful commit, the already-loaded `added` files fold directly into
    /// this copy. A version conflict refreshes the copy and retries. A commit
    /// whose version is current but could not be loaded has succeeded: the
    /// copy is refreshed onto it, or stays where it was for the next refresh.
    ///
    /// In-process callers follow the table's
    /// [commit queue](field@Self::commit_queue), so this retry path
    /// normally handles contention from another process on a shared remote
    /// store.
    ///
    /// Runs off the dispatch workers: the retry's `refresh` may drive a
    /// dataflow that reads the table's metadata, and every commit path
    /// commits from a blocking
    /// thread, never a pinned worker (which would deadlock driving a dataflow
    /// from inside one).
    pub(crate) fn commit_files(
        &mut self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            self.check_files_are_live(removed)?;
            match self.table.commit(removed, &added, data_change) {
                Ok(Some(table)) => {
                    self.table = table;
                    return Ok(());
                }
                Ok(None) => {
                    self.refresh()?;
                }
                Err(Error::Iceberg(
                    error @ iceberg_tables::Error::CommittedTableUnloaded { .. },
                )) => {
                    tracing::warn!(error = %error, "a commit succeeded without its version loading; refreshing onto it");
                    if let Err(refresh_error) = self.refresh() {
                        tracing::warn!(
                            error = %refresh_error,
                            "refreshing onto a committed version failed; the next refresh loads it"
                        );
                    }
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// [`Error::CommitConflict`] naming the first of `files` this version no
    /// longer holds.
    fn check_files_are_live(&self, files: RemovedFiles<'_>) -> crate::Result<()> {
        let found = self.table.find_file_entries(files.partition, files.paths)?;
        match files
            .paths
            .iter()
            .zip(&found)
            .find(|(_, file)| file.is_none())
        {
            Some((path, _)) => Err(self.conflict_at(path)),
            None => Ok(()),
        }
    }

    fn conflict_at(&self, path: &ObjectPath) -> Error {
        Error::CommitConflict {
            location: self.storage.location.as_str().to_string(),
            file: path.to_string(),
        }
    }

    /// Commit bare `entries` whose footers are not yet in hand: read the
    /// added files' footers, then commit the files with them as
    /// [`commit_files`](Self::commit_files) does, an entry that carries no
    /// stats recording those its footer holds. Used by test/seed paths;
    /// production writers (INSERT, compaction) already hold the footers and
    /// use `commit_files`. Only its `test-support` caller
    /// ([`test_support`](crate::test_support)) needs it, so it is gated with
    /// that feature to stay out of a plain library build.
    #[cfg(feature = "test-support")]
    pub(crate) fn commit_entries(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileEntry],
        data_change: bool,
    ) -> crate::Result<()> {
        let footers = self.fetch_footers(added)?;
        let added: Vec<TableFile> = added
            .iter()
            .zip(footers)
            .map(|(entry, footer)| TableFile::new(entry.clone(), footer.row_groups))
            .collect();
        let removed_partition = match removed.first() {
            Some(path) => self
                .find_partition_of(path)?
                .ok_or_else(|| self.conflict_at(path))?,
            None => Vec::new(),
        };
        let removed = RemovedFiles {
            partition: &removed_partition,
            paths: removed,
        };
        self.commit_files(removed, added, data_change)
    }

    /// Read the footers of `entries`, files under this table's location that
    /// its metadata does not hold yet, one per entry in order, each reconciled
    /// by name against the declared columns.
    #[cfg(feature = "test-support")]
    fn fetch_footers(
        &self,
        entries: &[FileEntry],
    ) -> crate::Result<Vec<parquet_engine::FileRowGroups>> {
        let to_fetch: Vec<object_storage::DataFile> = entries
            .iter()
            .map(|entry| {
                entry
                    .file
                    .clone()
                    .into_data_file(self.storage.store.as_ref(), &self.storage.location)
            })
            .collect::<object_storage::Result<_>>()?;
        Ok(parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            parquet_engine::TableColumns::by_name(self.columns()),
        )?)
    }

    /// Whether this copy's version has anything an expiry would expire as
    /// of `now_ms`, with nothing read; see `Table::has_expired_history`.
    pub(crate) fn has_expired_history(&self, now_ms: u64) -> crate::Result<bool> {
        self.table.has_expired_history(now_ms)
    }

    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the commit lock the caller holds. When that committed a version, this
    /// copy advances to it and what the expiry dropped is handed back, for
    /// the caller to [delete the objects of](Self::delete_snapshot_objects)
    /// once it has released the lock.
    pub(crate) fn expire_snapshots(
        &mut self,
        now_ms: u64,
    ) -> crate::Result<Option<ExpiredSnapshots>> {
        let Some(expiry) = self.table.expire_snapshots(now_ms)? else {
            return Ok(None);
        };
        self.table = expiry.table;
        Ok(Some(expiry.expired))
    }

    /// Delete what the snapshots `expired` left unused in this copy's
    /// version, the one their expiry committed, logging what could not be
    /// deleted; see `Table::delete_snapshot_objects`.
    pub(crate) fn delete_snapshot_objects(&self, expired: &ExpiredSnapshots) {
        self.table.delete_snapshot_objects(&self.storage, expired)
    }

    /// Delete what this version retired longer ago than the retention window,
    /// as of `now_ms`.
    pub(crate) fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        now_ms: u64,
    ) -> crate::Result<()> {
        self.table
            .delete_expired_objects(name, &self.storage, now_ms)
    }

    /// The table's durable identity from the database index, stable across
    /// renames and commits. The catalog indexes by this.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The terms the table is partitioned by, in order (empty =
    /// unpartitioned), each as `CREATE TABLE` spells it: a column, or a
    /// transform of one (`day(ts)`). A partitioning writer routes each row
    /// to a file by the value of every term.
    pub fn partition_by(&self) -> &[String] {
        self.table.partition_by()
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        self.table.sort_by()
    }

    /// Visit every partition this version has a file in with its files;
    /// see `Table::visit_partitions`.
    pub(crate) fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> crate::Result<()> {
        self.table.visit_partitions(visit)
    }

    /// Every partition this version has a file in; see
    /// `Table::list_partitions`.
    pub(crate) fn list_partitions(&self) -> crate::Result<Vec<Vec<u8>>> {
        self.table.list_partitions()
    }

    /// Where this version stands among the table's commits; see
    /// `Table::commit_position`.
    pub(crate) fn commit_position(&self) -> u64 {
        self.table.commit_position()
    }

    /// The partitions holding a file added after the commit standing at
    /// `commit_position`; see `Table::list_partitions_added_after`.
    pub(crate) fn list_partitions_added_after(
        &self,
        commit_position: u64,
    ) -> crate::Result<Vec<Vec<u8>>> {
        self.table.list_partitions_added_after(commit_position)
    }

    /// Of `partitions`, those whose files a rewrite may replace; see
    /// `Table::select_rewritable_partitions`.
    pub(crate) fn select_rewritable_partitions(
        &self,
        partitions: Vec<Vec<u8>>,
    ) -> crate::Result<Vec<Vec<u8>>> {
        self.table.select_rewritable_partitions(partitions)
    }

    /// Index each of `partitions` together; see `Table::index_partitions`.
    pub(crate) fn index_partitions(&self, partitions: &[Vec<u8>]) -> crate::Result<()> {
        self.table.index_partitions(partitions)
    }

    /// The live data files of `partition`; see
    /// `Table::load_partition_files`.
    pub(crate) fn load_partition_files(
        &self,
        partition: &[u8],
    ) -> crate::Result<Option<Arc<dyn PartitionFiles>>> {
        self.table.load_partition_files(partition)
    }

    /// The entry of each of `paths`, the live file of `partition` at that
    /// path, with its partition tuple and its bounds on the sort columns;
    /// see `Table::find_file_entries`.
    pub(crate) fn find_file_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> crate::Result<Vec<Option<FileEntry>>> {
        self.table.find_file_entries(partition, paths)
    }

    /// The partition the live file at `path` sits in, as a walk names it,
    /// or `None` when this version has no such file. Visits the table's
    /// partitions until one holds the path: for a caller that names a file
    /// without having walked to it.
    pub(crate) fn find_partition_of(&self, path: &ObjectPath) -> crate::Result<Option<Vec<u8>>> {
        let mut holding = None;
        self.visit_partitions(&mut |partition, files| {
            if files.find_file(path).is_none() {
                return ControlFlow::Continue(());
            }
            holding = Some(partition.to_vec());
            ControlFlow::Break(())
        })?;
        Ok(holding)
    }

    /// The row groups of each of `files` (live files of this very version),
    /// in the order given: what a merge and a layout measurement read.
    pub(crate) fn row_groups(
        &self,
        files: &[FileRef],
    ) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        self.table.row_groups(files)
    }

    /// This table as the cross-datastore catalog describes it, under the `name`
    /// it is currently indexed by and its durable id: the columns it declares
    /// with what they cost, and the files it holds.
    pub(crate) fn metadata(
        &self,
        name: SchemaQualifiedTableName,
    ) -> crate::Result<DatastoreTableMetadata> {
        self.table.describe(name, self.id.to_string())
    }

    /// Delete a data file (a compaction input swapped out of the metadata, or a
    /// merge output whose commit failed). See
    /// `TableStorage::delete_data_file`.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        self.storage.delete_data_file(path)
    }

    /// How long a file this table no longer references is kept before vacuum may
    /// delete it. Vacuum ages unreferenced files against this window.
    pub fn deleted_file_retention(&self) -> crate::Result<Duration> {
        self.table.deleted_file_retention()
    }

    /// The committed version this copy is at. Monotonic per table; used to
    /// decide whether a published copy is newer than the catalog's.
    pub fn version(&self) -> u64 {
        self.table.version()
    }

    /// Whether each column (in [`columns`](Self::columns) order) may hold a
    /// NULL in a file, as the table's metadata declares it with nothing
    /// read.
    pub(crate) fn declared_nullability(&self) -> Vec<bool> {
        self.table.declared_nullability()
    }

    /// A flat scan view of the files a query's pushed-down predicates cannot
    /// rule out: every surviving file's row groups concatenated in metadata
    /// order, where a row group's global index is simply its position. Empty
    /// `filters` keep every file. A scan and its materialize must build the
    /// view from the same filters (they do: both go through the binding's
    /// predicates) to address the same groups. See `Table::select_scan`.
    ///
    /// With it, whether each column (in [`columns`](Self::columns) order)
    /// can hold SQL NULLs in the files the scan reads, as the data shows it
    /// rather than as the schema declares it: the planner uses the flags to
    /// route between the branch-free and the null-aware execution paths.
    pub(crate) fn load_scan_selection(
        &self,
        filters: &[Expression],
    ) -> crate::Result<ScanSelection> {
        self.table.select_scan(filters)
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.table.columns().to_vec()
    }

    /// The rows of every live data file at this version, with no file's
    /// metadata read.
    pub fn total_rows(&self) -> crate::Result<u64> {
        self.table.total_rows()
    }

    /// The rows a scan of the whole table returns, when this version's
    /// metadata counts them exactly; `None` when only a scan can.
    pub fn count_rows(&self) -> crate::Result<Option<u64>> {
        self.table.count_rows()
    }

    /// The field id each column is written under in new files, for a format
    /// that resolves columns by id.
    pub(crate) fn field_ids(&self) -> Option<Vec<i32>> {
        self.table.field_ids()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root to
    /// know the physical location.
    pub fn location(&self) -> &str {
        self.storage.location.as_str()
    }

    pub(crate) fn store(&self) -> Arc<dyn ObjectStore> {
        self.storage.store.clone()
    }

    /// Re-encode `inputs`' rows into fresh target-sized files and return them,
    /// uploaded but not yet part of the table: the read half of a compaction,
    /// which [`compact_table_files`](crate::compact_table_files) then swaps in
    /// for the inputs. The inputs are merged a few row groups at a time and
    /// each output row group is written as soon as it is encoded, so the
    /// merge's memory does not grow with the size of the inputs. The output
    /// files' row groups come straight from the writer's own footer metadata,
    /// so no footer is re-read.
    ///
    /// Nothing here touches the metadata, so it runs off any read copy that
    /// holds the inputs, and a merge of a large batch never stalls another
    /// writer's commit. An input the copy no longer lists is a commit conflict:
    /// another writer swapped it out first.
    /// `inputs` are files of `partition`, as a walk names it: the caller
    /// batches them so. A partition whose files the table's format does not
    /// rewrite is refused: the outputs take the partition of the inputs.
    pub(crate) fn merge_files(
        &self,
        partition: &[u8],
        inputs: &[FileRef],
        target_rows_per_group: usize,
        max_output_file_size: u64,
    ) -> crate::Result<Vec<TableFile>> {
        let rewritable = self.select_rewritable_partitions(vec![partition.to_vec()])?;
        if rewritable.is_empty() {
            return Err(Error::PartitionNotRewritable {
                location: self.storage.location.as_str().to_string(),
            });
        }
        let input_paths: Vec<ObjectPath> = inputs.iter().map(|input| input.path.clone()).collect();
        let input_entries = input_paths
            .iter()
            .zip(self.table.find_file_entries(partition, &input_paths)?)
            .map(|(path, entry)| entry.ok_or_else(|| self.conflict_at(path)))
            .collect::<crate::Result<Vec<FileEntry>>>()?;
        let row_groups = self
            .table
            .row_groups(inputs)?
            .into_iter()
            .flatten()
            .collect();
        let parquet = Arc::new(ParquetTable::new(row_groups));
        if parquet.total_rows() == 0 {
            return Ok(Vec::new());
        }
        // The selector guarantees that all inputs share one partition, whose
        // tuple the table records for each of them. The outputs hold the
        // inputs' rows and no other, so the tuple is theirs too.
        let output_partition = input_entries
            .first()
            .and_then(|first| first.partition.clone());
        // The merge checks that each input's rows are in the order of the
        // table's sort columns and fails otherwise, so its outputs are in
        // that order whatever the inputs were recorded as: a file adopted as
        // it lay is recorded as in no order.
        let merged_by: Arc<[String]> = self.sort_by().into();
        // A format that resolves columns by field id gets its outputs written
        // under the declared schema, which carries the ids: an adopted file
        // may carry none. Otherwise the outputs keep the inputs' own schema.
        let output_schema = match self.table.field_ids() {
            None => parquet.schema().clone(),
            Some(field_ids) => iceberg_tables::io::sink::write_schema(
                self.table.columns(),
                &self.schema_nullability(),
                Some(&field_ids),
            ),
        };
        let max_file_size = usize::try_from(max_output_file_size).unwrap_or(usize::MAX);

        // The first dataflow settles the output's layout from a sample of the
        // input; the second merges the files and writes the result.
        let shredding = parquet_engine::writing::plan_compaction_shredding(
            &self.dispatcher,
            &parquet,
            output_schema,
        )
        .map_err(crate::Error::Merge)?;
        let compaction = parquet_engine::writing::compact_files_spec(
            &self.dispatcher,
            parquet,
            shredding.clone(),
            &merged_by,
            target_rows_per_group,
        )
        .map_err(crate::Error::Merge)?;
        let stored_files = Arc::new(Injector::new());
        let started_uploads = Arc::new(Injector::new());
        let uploaded_files = Arc::new(Injector::new());
        let spec = super::compaction_sink::write_row_groups_spec(
            super::compaction_sink::CompactionOutput {
                store: self.store(),
                location: self.storage.location.clone(),
                declared_columns: self.columns().into(),
                schema: shredding.schema().clone(),
                partition: output_partition,
                max_file_size,
                stored_files: stored_files.clone(),
                started_uploads: started_uploads.clone(),
                uploaded_files: uploaded_files.clone(),
            },
            compaction,
        );
        let outcome = spec.collect().map(|_| ()).map_err(crate::Error::Merge);

        // A merge that failed part way has put some of its outputs in
        // storage. They will never be committed, so delete them rather than
        // leave orphans.
        if let Err(error) = outcome {
            // The parts of an upload that was never completed stay stored
            // until it is aborted. Aborting one that was completed changes
            // nothing, and its object is deleted below.
            for upload in drain_injector(&started_uploads) {
                let aborted = self
                    .storage
                    .store
                    .as_s3()
                    .expect("a multipart upload is only started in an S3 store")
                    .abort_multipart_upload(&upload.key, &upload.upload_id);
                if let Err(cleanup_error) = aborted {
                    tracing::warn!(
                        merge_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %upload.key,
                        "compaction: aborting a multipart upload of a merge that will not commit failed (its parts are left)"
                    );
                }
            }
            for path in drain_injector(&stored_files) {
                if let Err(cleanup_error) = self.delete_data_file(&path) {
                    tracing::warn!(
                        merge_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %path,
                        "compaction: deleting an output of a merge that will not commit failed (orphan left)"
                    );
                }
            }
            return Err(error);
        }

        let added: Vec<TableFile> = drain_injector(&uploaded_files)
            .into_iter()
            .map(|uploaded| {
                let super::compaction_sink::CompactedFile {
                    file,
                    partition,
                    row_groups,
                } = uploaded;
                let stats = Some(std::sync::Arc::new(parquet_engine::aggregate_file_stats(
                    &row_groups,
                )));
                let entry = FileEntry {
                    file,
                    partition,
                    stats,
                    sorted_by: merged_by.clone(),
                };
                TableFile::new(entry, row_groups)
            })
            .collect();
        Ok(added)
    }

    /// Whether each column (in [`columns`](Self::columns) order) may be
    /// written a NULL: every one, as this datastore declares every column
    /// optional whatever the files written so far hold.
    fn schema_nullability(&self) -> Vec<bool> {
        vec![true; self.table.columns().len()]
    }

    /// What an INSERT into this table writes: files directly in the table's
    /// directory, in the declared columns, with the field ids the format
    /// assigns when it resolves columns by id, cut by the keys of the
    /// table's partition terms; or why the table's format cannot compute
    /// one of those keys.
    pub(crate) fn plan_insert_target(
        &self,
    ) -> crate::Result<iceberg_tables::io::sink::InsertTarget> {
        Ok(iceberg_tables::io::sink::InsertTarget {
            uploaded_files: UploadedFiles::new(),
            store: self.store(),
            directory: self.storage.location.clone(),
            columns: self.columns(),
            nullability: self.schema_nullability(),
            field_ids: self.field_ids(),
            partition_keys: self.table.plan_partition_keys()?,
            sort_by: self.sort_by().to_vec(),
        })
    }
}

/// Take everything queued on `injector`.
fn drain_injector<T>(injector: &Injector<T>) -> Vec<T> {
    let mut items = Vec::new();
    loop {
        match injector.steal() {
            Steal::Success(item) => items.push(item),
            Steal::Retry => continue,
            Steal::Empty => return items,
        }
    }
}
