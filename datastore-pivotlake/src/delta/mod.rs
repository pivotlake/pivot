//! The Delta Lake format: a `_delta_log` per table, read and written through
//! Delta Kernel.
//!
//! # What a version holds
//!
//! A version holds the snapshot it was read at, which is the log's position,
//! and what its files amount to as a whole ([`VersionSummary`]): how many
//! there are, their rows and bytes, and the partitions they sit in. It holds
//! no file and no footer.
//!
//! The log, its checkpoint and the commits after it, lists every file of
//! every partition, like one manifest for the whole table. A partition's
//! files are indexed when a scan, a commit or a maintenance pass first asks
//! for the partition, by replaying the log and keeping the files of the
//! partitions asked for: a replay keeps the files of several partitions at
//! once, so a pass over many partitions replays the log once, or a few
//! times for a table whose files do not fit in what a replay may keep. The
//! indexes of every table, of whichever format, live in one cache
//! ([`table_file_index::FileMetadataCache`]), which keeps them within a
//! budget of memory: an index the cache let go of is built again by another
//! replay. The same cache, within the same budget, keeps the footers the
//! table's files were found to have.
//!
//! The open of a table replays its log once, for the summary, and a table
//! whose files' records are few is indexed by that same replay. The open of
//! a database of few files also selects a scan of each table, so the cache
//! holds every index and every footer before the first query.
//!
//! # From one version to the next
//!
//! A commit made here derives the next version from the one it was made on:
//! the summary by the files it removed and added, and the cached index of
//! each partition it touched likewise, with no replay. What each such
//! commit changed is remembered ([`CommitHistory`]), so a copy of the table
//! standing at another version than the one the cache holds an index for,
//! a compaction round's or a running query's, has the index brought to its
//! own version rather than reading the log.
//!
//! A refresh that finds a version committed elsewhere replays the log at
//! that version, once: for the summary, and for the files of the partitions
//! the cache holds an index for, each built on the index held, so a file
//! both versions list keeps its record.
//!
//! # Compaction rounds
//!
//! A version stands among its table's commits at its log version, which
//! every writer numbers alike. The partitions a file was added to since a
//! version are read from what the commits made here changed
//! ([`CommitHistory`]), which is what a routine compaction round examines.
//! A version committed elsewhere in between, or one the history no longer
//! holds, leaves them unknown: every partition then counts. A file's
//! `modificationTime` does not serve for this: another writer records its
//! file's own time there, which may lie before a commit made earlier.
//!
//! # A file's place among its partition's files
//!
//! A partition's files stand in the order the shared index keeps
//! ([`table_file_index::file_index`]). The commit a file belongs to is read
//! from the `modificationTime` its `Add` action carries: a commit made here
//! numbers its files from the clock's reading, or from just past the latest
//! file when the clock has not moved past it, one millisecond apart in the
//! order it adds them. A replay therefore reads back the order the files
//! were committed in, checkpoint or not, which the log's versions do not
//! give: a checkpoint keeps no trace of the version that added a file.

mod commit_history;
mod describe;
mod file_record;
pub(crate) mod log;
mod partition_key;
mod version_summary;

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use dispatch::DataFlowDispatcher;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use object_storage::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use parquet_engine::file_entry::partition_maybe_matches;
use parquet_engine::{ParquetTable, PushedPredicate, RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::{CompareType, Expression};
use table_file_index::{
    FileMetadataCache, IndexedRecord, LookupUse, PartitionFiles, TableCacheScope, VersionOrder,
};
use tracing::{info, warn};
use uuid::Uuid;

use self::commit_history::{CommitHistory, PartitionChange};
use self::file_record::{DeltaFile, DeltaIndex, RecordBuilder};
use self::log::TableShape;
use self::version_summary::{CountedFile, VersionSummary};
use crate::Result;
use crate::file_entry::{ColumnStatFilter, FileEntry, PartitionEqFilter, constant_partition};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{RemovedFiles, SnapshotExpiry, Table, TableFactory, VisitPartition};

/// How many partitions a walk or a scan of many has indexed at a time.
const PARTITIONS_PER_LOAD: usize = 64;

/// How much of the files a replay met ahead of the partitions a pass will
/// index it keeps, by [`IndexedRecord::estimate_heap_bytes`]; past it the
/// files kept ahead are let go of, and the log is replayed again when
/// their partitions' turn comes.
const REPLAYED_AHEAD_HELD_BYTES: u64 = 256 * 1024 * 1024;

/// How much the records of a table's files may take, by
/// [`IndexedRecord::estimate_heap_bytes`], for the replay that opens the
/// table to index every partition as it goes.
const INDEXED_AT_OPEN_HELD_BYTES: u64 = 8 * 1024 * 1024;

/// What creates and loads the Delta tables of one database, which every
/// table holds a copy of: the Kernel engine its logs are read and written
/// through (built once for the store, so a refresh, a commit or a vacuum
/// sweep reuses one object-store client and task executor), the store, and
/// the pool footers are fetched on.
#[derive(Clone)]
pub(crate) struct DeltaTableFactory {
    engine: log::DeltaEngine,
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
}

impl DeltaTableFactory {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, dispatcher: DataFlowDispatcher) -> Result<Self> {
        let engine = log::DeltaEngine::new(store.as_ref())?;
        Ok(Self {
            engine,
            store,
            dispatcher,
        })
    }
}

impl TableFactory for DeltaTableFactory {
    /// Write Delta version 0 with its protocol, metadata and initial `Add`
    /// actions. An adopted file whose partition columns are constant is
    /// stamped with that partition from its stats, so it is recorded in the log
    /// and prunes like a written file; one whose values vary is adopted
    /// without a tuple. The adopted files are indexed and their footers kept
    /// as the caller read them, so the log just written is not read back.
    fn create_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
        _id: Uuid,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        mut adopted: Vec<TableFile>,
    ) -> Result<Arc<dyn Table>> {
        if !partition_by.is_empty() {
            for file in &mut adopted {
                file.entry.partition = file
                    .entry
                    .stats
                    .as_ref()
                    .and_then(|stats| constant_partition(stats, &partition_by));
            }
        }
        let entries: Vec<FileEntry> = adopted.iter().map(|file| file.entry.clone()).collect();
        // Version 0's commit yields the snapshot the new table starts at.
        let snapshot = log::initialize_table(
            &self.engine,
            self.store.as_ref(),
            directory,
            &columns,
            &partition_by,
            &sort_by,
            &entries,
        )?;
        let shape = Arc::new(log::read_table_shape(&snapshot)?);
        let lineage = Lineage::begin(&shape)?;
        // The adopted files are recorded in the order given, numbered from
        // the first.
        let records = lineage.build_records(&adopted, 0)?;
        let mut summary = VersionSummary::default();
        let mut files_by_partition: HashMap<Arc<[u8]>, Vec<Arc<DeltaFile>>> = HashMap::new();
        for record in records {
            summary.add(CountedFile::of(&record));
            files_by_partition
                .entry(Arc::clone(record.partition()))
                .or_default()
                .push(record);
        }
        let table = DeltaTable {
            factory: self.clone(),
            location: directory.clone(),
            snapshot,
            shape,
            summary: Arc::new(summary),
            tombstones: Arc::default(),
            lineage,
            swapped: None,
        };
        for (partition, files) in files_by_partition {
            table.store_index(&partition, build_index(files));
        }
        table.keep_footers(&adopted);
        table.lineage.remember_version(&table);
        Ok(Arc::new(table))
    }

    /// Read the table's log at its latest version: its files are counted,
    /// and indexed when their records are few. No footer is read.
    fn load_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
    ) -> Result<Arc<dyn Table>> {
        let delta_uri = log::table_uri(&self.store.location_uri(), directory)?;
        let snapshot = log::open_snapshot(&delta_uri, &self.engine)?;
        let shape = Arc::new(log::read_table_shape(&snapshot)?);
        let lineage = Lineage::begin(&shape)?;
        let opened = DeltaTable {
            factory: self.clone(),
            location: directory.clone(),
            snapshot,
            shape,
            summary: Arc::default(),
            tombstones: Arc::default(),
            lineage,
            swapped: None,
        };
        Ok(Arc::new(opened.read_from_log(
            Kept::WholeTableWithin(INDEXED_AT_OPEN_HELD_BYTES),
            &HashMap::new(),
        )?))
    }
}

/// What the versions of a table that read a file alike share: of one schema,
/// partitioned and sorted by the same columns, each derived from another by
/// a refresh, a commit or a swap.
struct Lineage {
    /// Where the versions' partition indexes and the footers of their files
    /// are cached.
    scope: TableCacheScope<DeltaIndex>,
    /// Builds the record of a file of the table.
    records: Mutex<RecordBuilder>,
    /// What the commits made here changed.
    commits: Mutex<CommitHistory>,
    /// The newest version a copy of the table in this process read from
    /// the log or committed, for another copy's refresh to take as it is.
    newest: Mutex<Option<ReadVersion>>,
    /// Held while partitions are indexed, so scans that want one partition
    /// at once wait for one replay of the log rather than each replaying
    /// it.
    indexing: Mutex<()>,
}

impl Lineage {
    /// A lineage no version belongs to yet, of tables of `shape`, its
    /// partition indexes and footers kept in the process's cache.
    fn begin(shape: &TableShape) -> Result<Arc<Self>> {
        let records = RecordBuilder::new(&shape.columns, &shape.partition_by, &shape.sort_by)?;
        Ok(Arc::new(Self {
            scope: TableCacheScope::new(FileMetadataCache::process_wide()),
            records: Mutex::new(records),
            commits: Mutex::default(),
            newest: Mutex::default(),
            indexing: Mutex::new(()),
        }))
    }

    /// Remember `version` when no newer one is remembered.
    fn remember_version(&self, version: &DeltaTable) {
        let mut newest = self.newest.lock().unwrap();
        let is_newer = newest
            .as_ref()
            .is_none_or(|held| held.snapshot.version() < version.snapshot.version());
        if is_newer {
            *newest = Some(ReadVersion {
                snapshot: Arc::clone(&version.snapshot),
                summary: Arc::clone(&version.summary),
                tombstones: Arc::clone(&version.tombstones),
            });
        }
    }

    /// What is remembered of the version numbered `version`, when it is the
    /// newest one remembered.
    fn find_version(&self, version: u64) -> Option<ReadVersion> {
        let newest = self.newest.lock().unwrap();
        newest
            .as_ref()
            .filter(|held| held.snapshot.version() == version)
            .cloned()
    }

    /// The record of each of `files`, in the order given, the first
    /// standing at `first_commit_sequence` among the table's commits and
    /// each of the others one past the file before it.
    fn build_records(
        &self,
        files: &[TableFile],
        first_commit_sequence: u64,
    ) -> Result<Vec<Arc<DeltaFile>>> {
        let mut records = self.records.lock().unwrap();
        files
            .iter()
            .zip(first_commit_sequence..)
            .map(|(file, commit_sequence)| {
                Ok(Arc::new(
                    records.build(file.entry.clone(), commit_sequence)?,
                ))
            })
            .collect()
    }
}

/// What a committed version holds, as it was read from the log or derived
/// by the commit that wrote it.
#[derive(Clone)]
struct ReadVersion {
    snapshot: Arc<delta_kernel::Snapshot>,
    summary: Arc<VersionSummary>,
    tombstones: Arc<HashMap<ObjectPath, u64>>,
}

/// The files a local view swapped, by the key of their partition.
type SwappedFiles = HashMap<Arc<[u8]>, PartitionSwap>;

/// What a local view swapped in one partition.
#[derive(Clone, Default)]
struct PartitionSwap {
    /// The committed files swapped out.
    removed: Vec<ObjectPath>,
    /// The files swapped in.
    added: Vec<Arc<DeltaFile>>,
}

/// Which of the files a replay meets it keeps.
#[derive(Clone, Copy)]
enum Kept<'k> {
    /// Those of the partitions `now`, and those of the partitions `ahead`
    /// while what is kept of them stays within
    /// [`REPLAYED_AHEAD_HELD_BYTES`].
    Partitions {
        now: &'k HashSet<Arc<[u8]>>,
        ahead: &'k HashSet<Arc<[u8]>>,
    },
    /// Every file, as long as they all fit within the bytes given; none
    /// when they do not.
    WholeTableWithin(u64),
}

/// What one replay of the log read.
struct Replayed {
    /// What the version's files amount to.
    summary: VersionSummary,
    /// Every file of each partition the replay kept the files of.
    files: FilesByPartition,
}

/// Files by the key of the partition they sit in.
type FilesByPartition = HashMap<Arc<[u8]>, Vec<Arc<DeltaFile>>>;

/// What a walk over a table's partitions does with each partition, by its
/// key, and its files: `Break` ends the walk.
type VisitIndex<'v> = dyn FnMut(&Arc<[u8]>, &Arc<DeltaIndex>) -> Result<ControlFlow<()>> + 'v;

/// What a scan over some filters selected of a table version.
struct SelectedFiles {
    /// The keys of the partitions whose tuple the filters keep, which the
    /// files were selected from.
    partitions: Vec<Arc<[u8]>>,
    /// The files, in the order they are scanned.
    files: Vec<Arc<DeltaFile>>,
}

/// Who counts the indexes a walk loads as used.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexUse {
    /// The walk, as it loads them.
    Counted,
    /// The caller, once it has taken what it reads through them.
    LeftToCaller,
}

/// What a taker of footers leaves of itself in the cache.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FooterUse {
    /// The footers found count as used, and those read are kept.
    Kept,
    /// The cache is left as it was: nothing counts as used and nothing
    /// read is kept.
    Unrecorded,
}

/// One Delta table at one log version: the snapshot it was read at and that
/// every commit from it is written on top of, what its files amount to, and
/// the retired files the log still remembers.
struct DeltaTable {
    factory: DeltaTableFactory,
    location: ObjectPath,
    snapshot: Arc<delta_kernel::Snapshot>,
    /// The columns the version declares, and those its files are
    /// partitioned and sorted by.
    shape: Arc<TableShape>,
    /// What the committed version's files amount to.
    summary: Arc<VersionSummary>,
    /// The files the log retired but still remembers, each by the Unix ms its
    /// commit retired it: what vacuum ages a retired file by, so the retention
    /// window runs from the moment the file stopped being referenced rather
    /// than from its storage mtime. Rebuilt from the log on a reload, which
    /// bounds it to what the log still carries, and extended by every commit
    /// made from this value's lineage, which drops the entries a window older
    /// than itself.
    tombstones: Arc<HashMap<ObjectPath, u64>>,
    /// What this version shares with the versions it was derived from and
    /// those derived from it.
    lineage: Arc<Lineage>,
    /// For a local view of file swaps this process committed through
    /// another copy of the table: per partition, the files swapped out of
    /// the committed ones and those swapped in, which the log at this
    /// version does not record. The next refresh reconciles the copy to the
    /// committed log.
    swapped: Option<Arc<SwappedFiles>>,
}

/// Whether `schema` holds each of `columns` at the place `positions` gives
/// for it. A column `positions` gives no place for may sit anywhere in
/// `schema`, so only a schema that holds every column where expected is
/// vouched for.
fn holds_columns_at(
    schema: &arrow_schema::Schema,
    columns: &[Column],
    positions: &[Option<usize>],
) -> bool {
    let fields = schema.fields();
    columns.iter().zip(positions).all(|(column, position)| {
        position.is_some_and(|position| {
            fields
                .get(position)
                .is_some_and(|field| field.name() == &column.name)
        })
    })
}

/// The index of one partition whose files are `files`.
fn build_index(files: Vec<Arc<DeltaFile>>) -> DeltaIndex {
    let mut index = DeltaIndex::default();
    for file in files {
        index.insert(file);
    }
    index
}

impl DeltaTable {
    /// Where this version stands among the versions of its table, for the
    /// cache.
    fn version_order(&self) -> VersionOrder {
        VersionOrder {
            sequence: self.snapshot.version() as i64,
            updated_ms: 0,
        }
    }

    /// Hold `index` in the cache as this version's index of `partition`.
    fn store_index(&self, partition: &Arc<[u8]>, index: DeltaIndex) -> Arc<DeltaIndex> {
        let index = Arc::new(index);
        self.lineage
            .scope
            .store_index(partition, &index, self.version_order());
        index
    }

    /// Hold the footers of `files` in the cache, as their writer or reader
    /// produced them. A file handed over without its row groups is left to
    /// be read when something first needs them.
    fn keep_footers(&self, files: &[TableFile]) {
        let footers = files
            .iter()
            .filter(|file| !file.row_groups.is_empty())
            .map(|file| {
                let path = Arc::from(file.entry.file.path.as_str());
                (path, file.row_groups.clone())
            })
            .collect();
        self.lineage.scope.store_footers(footers);
    }

    /// The shared key of `partition`, when this version or a local swap
    /// has a file in it.
    fn find_partition_key(&self, partition: &[u8]) -> Option<Arc<[u8]>> {
        if let Some((key, _)) = self.summary.partitions.get_key_value(partition) {
            return Some(Arc::clone(key));
        }
        let swapped = self.swapped.as_ref()?;
        swapped
            .get_key_value(partition)
            .map(|(key, _)| Arc::clone(key))
    }

    /// Every partition this version, or a local swap, has a file in, in
    /// the order of the bytes they are named by.
    fn list_partition_keys(&self) -> Vec<Arc<[u8]>> {
        let committed = self.summary.partitions.keys();
        let Some(swapped) = &self.swapped else {
            return committed.cloned().collect();
        };
        let mut keys: Vec<Arc<[u8]>> = committed.chain(swapped.keys()).cloned().collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Replay the log at this version, keeping the files `kept` names, each
    /// as the record `held` has for it when it has one: `held` is the index
    /// the cache holds for a partition at another version.
    fn replay_log(
        &self,
        kept: Kept<'_>,
        held: &HashMap<Arc<[u8]>, Arc<DeltaIndex>>,
    ) -> Result<Replayed> {
        let mut records = self.lineage.records.lock().unwrap();
        let mut summary = VersionSummary::default();
        let mut files = FilesByPartition::new();
        // The files kept that the replay may yet let go of: how much they
        // may take, how much they do, and whether any more is kept.
        let droppable_bytes_bound = match kept {
            Kept::Partitions { .. } => REPLAYED_AHEAD_HELD_BYTES,
            Kept::WholeTableWithin(bound) => bound,
        };
        let mut droppable_bytes = 0_u64;
        let mut keeps_droppable = true;
        log::replay_files(
            &self.snapshot,
            &self.factory.engine,
            &self.shape,
            &mut |file: log::ReplayedFile<'_>| -> Result<()> {
                let partition = records.encode_partition(&file.entry)?;
                summary.add(CountedFile {
                    partition: &partition,
                    values: file.entry.partition.as_ref(),
                    size: file.entry.file.size,
                    rows: file.rows,
                    commit_sequence: file.commit_sequence,
                });
                let is_droppable = match kept {
                    Kept::Partitions { now, .. } if now.contains(&partition) => false,
                    Kept::Partitions { ahead, .. } if ahead.contains(&partition) => true,
                    Kept::Partitions { .. } => return Ok(()),
                    Kept::WholeTableWithin(_) => true,
                };
                if is_droppable && !keeps_droppable {
                    return Ok(());
                }
                let record = match held
                    .get(&partition)
                    .and_then(|index| index.find(&file.entry.file.path))
                {
                    Some(record) => Arc::clone(record),
                    None => {
                        let stats = file.read_stats().map(Arc::new);
                        let entry = FileEntry {
                            stats,
                            ..file.entry
                        };
                        Arc::new(records.build(entry, file.commit_sequence)?)
                    }
                };
                if is_droppable {
                    droppable_bytes += record.estimate_heap_bytes();
                    if droppable_bytes > droppable_bytes_bound {
                        keeps_droppable = false;
                        match kept {
                            Kept::Partitions { ahead, .. } => {
                                files.retain(|partition, _| !ahead.contains(partition));
                            }
                            Kept::WholeTableWithin(_) => files.clear(),
                        }
                        return Ok(());
                    }
                }
                files.entry(partition).or_default().push(record);
                Ok(())
            },
        )?;
        Ok(Replayed { summary, files })
    }

    /// This table as the log at its snapshot has it: its files counted, the
    /// files `kept` names indexed and their indexes held in the cache, and
    /// the retired files the log remembers read. `held` is as
    /// [`replay_log`](Self::replay_log) takes it. A partition `kept` named
    /// that the version has no file in is indexed as empty, so the cache
    /// does not go on holding another version's files for it.
    fn read_from_log(
        &self,
        kept: Kept<'_>,
        held: &HashMap<Arc<[u8]>, Arc<DeltaIndex>>,
    ) -> Result<Self> {
        let named: Vec<Arc<[u8]>> = match &kept {
            Kept::Partitions { now, .. } => now.iter().cloned().collect(),
            Kept::WholeTableWithin(_) => Vec::new(),
        };
        let Replayed { summary, mut files } = self.replay_log(kept, held)?;
        // The files this version no longer references but the log still dates, off
        // the same segment the replay read, so the two always agree.
        let tombstones = log::read_remove_tombstones(&self.snapshot, &self.factory.engine)?;
        let read = Self {
            factory: self.factory.clone(),
            location: self.location.clone(),
            snapshot: Arc::clone(&self.snapshot),
            shape: Arc::clone(&self.shape),
            summary: Arc::new(summary),
            tombstones: Arc::new(tombstones),
            lineage: Arc::clone(&self.lineage),
            swapped: None,
        };
        for partition in named {
            let partition_files = files.remove(&partition).unwrap_or_default();
            read.store_index(&partition, build_index(partition_files));
        }
        for (partition, partition_files) in files {
            read.store_index(&partition, build_index(partition_files));
        }
        read.lineage.remember_version(&read);
        Ok(read)
    }

    /// The index the cache holds for `partition`, built at `built_at`, as
    /// this version's: itself when it was built at this version, or brought
    /// across the commits made here between the two. `None` when a commit
    /// between them was made elsewhere. An index brought forward is this
    /// version's from now on, and is held as such.
    fn bring_index(
        &self,
        partition: &Arc<[u8]>,
        index: Arc<DeltaIndex>,
        built_at: VersionOrder,
    ) -> Option<Arc<DeltaIndex>> {
        let version = self.snapshot.version();
        let built_at = built_at.sequence as u64;
        if built_at == version {
            return Some(index);
        }
        let commits = self.lineage.commits.lock().unwrap();
        let brought = commits.bring_index(&index, partition, built_at, version)?;
        drop(commits);
        if built_at < version {
            return Some(self.store_index(partition, brought));
        }
        Some(Arc::new(brought))
    }

    /// The committed files of each of `partitions` at this version, in the
    /// order given, each partition one this version has a file in.
    ///
    /// The index the cache holds for a partition is taken when it is this
    /// version's, or can be brought to it. The others are built from the
    /// files `replayed` holds for them, and when it holds none for some,
    /// the log is replayed for those, keeping as well, as far as the bound
    /// allows, the files of the partitions `ahead` whose indexes the cache
    /// does not hold: the partitions the caller will ask for next, which
    /// are looked at only when the log is replayed. An index taken from the
    /// cache is not counted as used here.
    fn load_committed_indexes(
        &self,
        partitions: &[Arc<[u8]>],
        ahead: &[Arc<[u8]>],
        replayed: &mut FilesByPartition,
    ) -> Result<Vec<Arc<DeltaIndex>>> {
        let _indexing = self.lineage.indexing.lock().unwrap();
        let scope = &self.lineage.scope;
        let mut indexes: Vec<Option<Arc<DeltaIndex>>> = Vec::with_capacity(partitions.len());
        let mut stale: HashMap<Arc<[u8]>, Arc<DeltaIndex>> = HashMap::new();
        let held = scope.find_indexes(
            partitions.iter().map(|partition| &**partition),
            LookupUse::Uncounted,
        );
        let version = self.version_order();
        for (partition, held) in partitions.iter().zip(held) {
            let current = match held {
                // The common case: the index held is this version's.
                Some((index, built_at)) if built_at == version => Some(index),
                Some((index, built_at)) => {
                    let brought = self.bring_index(partition, Arc::clone(&index), built_at);
                    if brought.is_none() {
                        stale.insert(Arc::clone(partition), index);
                    }
                    brought
                }
                None => None,
            };
            indexes.push(current);
        }
        let unbuilt: Vec<&Arc<[u8]>> = partitions
            .iter()
            .zip(&indexes)
            .filter(|(_, index)| index.is_none())
            .map(|(partition, _)| partition)
            .collect();
        let unread: HashSet<Arc<[u8]>> = unbuilt
            .into_iter()
            .filter(|partition| !replayed.contains_key(*partition))
            .cloned()
            .collect();
        if !unread.is_empty() {
            // A partition only a local swap has a file in has no committed
            // file to replay.
            let ahead: Vec<&Arc<[u8]>> = ahead
                .iter()
                .filter(|partition| self.summary.partitions.contains_key(*partition))
                .collect();
            let held_ahead = scope.find_indexes(
                ahead.iter().map(|partition| &***partition),
                LookupUse::Uncounted,
            );
            let mut unread_ahead: HashSet<Arc<[u8]>> = HashSet::new();
            for (partition, held) in ahead.into_iter().zip(held_ahead) {
                if replayed.contains_key(partition) {
                    continue;
                }
                match held {
                    Some((index, built_at)) => {
                        if self
                            .bring_index(partition, Arc::clone(&index), built_at)
                            .is_none()
                        {
                            stale.insert(Arc::clone(partition), index);
                            unread_ahead.insert(Arc::clone(partition));
                        }
                    }
                    None => {
                        unread_ahead.insert(Arc::clone(partition));
                    }
                }
            }
            let kept = Kept::Partitions {
                now: &unread,
                ahead: &unread_ahead,
            };
            replayed.extend(self.replay_log(kept, &stale)?.files);
        }
        partitions
            .iter()
            .zip(indexes)
            .map(|(partition, index)| match index {
                Some(index) => Ok(index),
                None => {
                    let files = replayed.remove(partition).unwrap_or_default();
                    self.check_replayed_file_count(partition, files.len() as u64)?;
                    Ok(self.store_index(partition, build_index(files)))
                }
            })
            .collect()
    }

    /// An error unless `replayed`, how many files a replay of the log met in
    /// `partition`, is how many this version counts in it: a partition's
    /// index is its files exactly, or it is not built.
    fn check_replayed_file_count(&self, partition: &[u8], replayed: u64) -> Result<()> {
        let counted = self
            .summary
            .partitions
            .get(partition)
            .map_or(0, |partition| partition.file_count);
        if replayed == counted {
            return Ok(());
        }
        Err(log::Error::ReplayedFileCount {
            location: self.location.as_str().to_string(),
            replayed,
            counted,
        }
        .into())
    }

    /// Visit each of `partitions`, partitions this version or a local swap
    /// has a file in, with its files at this version, a local swap's
    /// swapped, a few partitions indexed at a time; `Break` ends the walk.
    /// With [`IndexUse::Counted`] the indexes count as used as they are
    /// loaded.
    fn visit_indexes(
        &self,
        partitions: &[Arc<[u8]>],
        index_use: IndexUse,
        visit: &mut VisitIndex<'_>,
    ) -> Result<()> {
        let mut replayed = FilesByPartition::new();
        let mut visited = 0;
        for chunk in partitions.chunks(PARTITIONS_PER_LOAD) {
            visited += chunk.len();
            let indexes = self.load_indexes(chunk, &partitions[visited..], &mut replayed)?;
            if index_use == IndexUse::Counted {
                self.lineage.scope.mark_indexes_used(chunk);
            }
            for (partition, index) in chunk.iter().zip(&indexes) {
                if visit(partition, index)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// The files of each of `partitions` at this version, in the order
    /// given, with a local swap's files swapped; see
    /// [`load_committed_indexes`](Self::load_committed_indexes). A
    /// partition only a local swap has a file in has no committed file. An
    /// index taken from the cache is not counted as used here.
    fn load_indexes(
        &self,
        partitions: &[Arc<[u8]>],
        ahead: &[Arc<[u8]>],
        replayed: &mut FilesByPartition,
    ) -> Result<Vec<Arc<DeltaIndex>>> {
        // Without a local swap every partition asked for is a committed one,
        // and its files are the committed ones.
        if self.swapped.is_none() {
            return self.load_committed_indexes(partitions, ahead, replayed);
        }
        let is_committed = |partition: &Arc<[u8]>| self.summary.partitions.contains_key(partition);
        let committed: Vec<Arc<[u8]>> = partitions
            .iter()
            .filter(|partition| is_committed(partition))
            .cloned()
            .collect();
        let mut committed_indexes = self
            .load_committed_indexes(&committed, ahead, replayed)?
            .into_iter();
        Ok(partitions
            .iter()
            .map(|partition| {
                let index = if is_committed(partition) {
                    committed_indexes
                        .next()
                        .expect("an index is loaded for every committed partition")
                } else {
                    Arc::default()
                };
                self.swap_files(partition, index)
            })
            .collect())
    }

    /// The files of `partition` at this version, with a local swap's files
    /// swapped; `None` when neither the version nor a swap has a file in
    /// it.
    fn load_index(&self, partition: &[u8]) -> Result<Option<Arc<DeltaIndex>>> {
        let Some(key) = self.find_partition_key(partition) else {
            return Ok(None);
        };
        let partitions = [key];
        let mut indexes = self.load_indexes(&partitions, &[], &mut FilesByPartition::new())?;
        self.lineage.scope.mark_indexes_used(&partitions);
        Ok(indexes.pop())
    }

    /// The committed files of `partition` at this version, a local swap
    /// left out; `None` when the version has no file in the partition.
    fn load_committed_index(&self, partition: &[u8]) -> Result<Option<Arc<DeltaIndex>>> {
        let Some((key, _)) = self.summary.partitions.get_key_value(partition) else {
            return Ok(None);
        };
        let partitions = std::slice::from_ref(key);
        let mut indexes =
            self.load_committed_indexes(partitions, &[], &mut FilesByPartition::new())?;
        self.lineage.scope.mark_indexes_used(partitions);
        Ok(indexes.pop())
    }

    /// `index`, the committed files of `partition`, with what a local swap
    /// swapped in the partition.
    fn swap_files(&self, partition: &[u8], index: Arc<DeltaIndex>) -> Arc<DeltaIndex> {
        let swap = self
            .swapped
            .as_ref()
            .and_then(|swapped| swapped.get(partition));
        let Some(swap) = swap else {
            return index;
        };
        let mut view = DeltaIndex::clone(&index);
        for path in &swap.removed {
            view.remove(path);
        }
        for file in &swap.added {
            view.insert(Arc::clone(file));
        }
        Arc::new(view)
    }

    /// The local swaps of this view with `removed`, files of the partition
    /// `removed_partition`, dropped and `added` swapped in.
    fn swap_more_files(
        &self,
        removed_partition: &[u8],
        removed: &[ObjectPath],
        added: Vec<Arc<DeltaFile>>,
    ) -> Arc<SwappedFiles> {
        let mut swapped = self.swapped.as_deref().cloned().unwrap_or_default();
        if !removed.is_empty() {
            let key = self
                .find_partition_key(removed_partition)
                .unwrap_or_else(|| Arc::from(removed_partition));
            let swap = swapped.entry(key).or_default();
            for path in removed {
                let swapped_in = swap
                    .added
                    .iter()
                    .position(|file| &file.facts().file().path == path);
                match swapped_in {
                    Some(position) => {
                        swap.added.remove(position);
                    }
                    None => swap.removed.push(path.clone()),
                }
            }
        }
        for file in added {
            let swap = swapped.entry(Arc::clone(file.partition())).or_default();
            swap.added
                .retain(|earlier| earlier.facts().file().path != file.facts().file().path);
            swap.added.push(file);
        }
        Arc::new(swapped)
    }

    /// The row groups of each of `files`, distinct files of the table, in
    /// the order given, shared with the cache and with whatever else took
    /// them. A file whose footer the cache holds is neither opened nor
    /// read; the others' footers are read and parsed in one pass over the
    /// pool, which is not run when the cache holds them all, and, with
    /// [`FooterUse::Kept`], cached.
    fn open_footers(
        &self,
        files: &[&FileRef],
        footer_use: FooterUse,
    ) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let scope = &self.lineage.scope;
        let paths = files.iter().map(|file| file.path.as_str());
        let lookup_use = match footer_use {
            FooterUse::Kept => LookupUse::Counted,
            FooterUse::Unrecorded => LookupUse::Uncounted,
        };
        let held = scope.find_footers(paths, lookup_use);
        let store = self.factory.store.as_ref();
        let unheld: Vec<DataFile> = files
            .iter()
            .zip(&held)
            .filter(|(_, held)| held.is_none())
            .map(|(file, _)| FileRef::clone(file).into_data_file(store, &self.location))
            .collect::<object_storage::Result<_>>()?;
        let read = parquet_engine::load_file_row_groups(
            &self.factory.dispatcher,
            &unheld,
            parquet_engine::TableColumns::by_name(self.shape.columns.clone()),
        )?;
        let unheld_paths = files
            .iter()
            .zip(&held)
            .filter(|(_, held)| held.is_none())
            .map(|(file, _)| &file.path);
        if footer_use == FooterUse::Kept {
            let kept = unheld_paths
                .zip(&read)
                .map(|(path, footer)| (Arc::from(path.as_str()), footer.row_groups.clone()))
                .collect();
            scope.store_footers(kept);
        }
        let mut read = read.into_iter();
        Ok(held
            .into_iter()
            .map(|held| match held {
                Some(row_groups) => row_groups,
                None => {
                    read.next()
                        .expect("every file whose footer is not held was read")
                        .row_groups
                }
            })
            .collect())
    }

    /// Where the first file of the next commit stands among the table's
    /// commits when the commit is made at `now_ms`: at that reading, or
    /// just after the latest file when the clock has not moved past it,
    /// so the files of a later commit always stand after an earlier one's.
    fn next_commit_sequence(&self, now_ms: u64) -> u64 {
        now_ms.max(self.summary.latest_commit_sequence + 1)
    }

    /// The tombstones after a commit at `retired_at_ms` retired `removed`: a
    /// clock reading taken just before the transaction was opened, back to back
    /// with the one Kernel stamps their `Remove` tombstones with, so they are
    /// dated within milliseconds of what a reload from the log would date them.
    /// The same reading retires the tombstones a full retention window older
    /// than this commit: any sweep run after it sees them past the window, so
    /// they date nothing, and dropping them is what bounds a writer's copy,
    /// which only ever extends its tombstones between reloads, to one window's
    /// worth of retirements.
    fn tombstones_after(
        &self,
        removed: &[ObjectPath],
        retired_at_ms: u64,
    ) -> Arc<HashMap<ObjectPath, u64>> {
        if removed.is_empty() {
            return Arc::clone(&self.tombstones);
        }
        let mut tombstones = HashMap::clone(&self.tombstones);
        let cutoff = retired_at_ms
            .saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        tombstones.retain(|_, at| *at > cutoff);
        tombstones.extend(removed.iter().map(|path| (path.clone(), retired_at_ms)));
        Arc::new(tombstones)
    }

    /// The equality predicates among `predicates` as partition-filter
    /// candidates: the column name and the typed scalar. A file's entry
    /// intersects these with the table's partition columns, so yielding every
    /// equality (not just ones on partition columns) is fine; a non-partition
    /// column prunes no files.
    fn partition_filter_candidates(
        &self,
        predicates: &[PushedPredicate],
    ) -> Vec<PartitionEqFilter> {
        predicates
            .iter()
            .filter(|predicate| matches!(predicate.compare_type, CompareType::Equal))
            .filter_map(|predicate| {
                Some(PartitionEqFilter {
                    column: self.shape.columns.get(predicate.column_idx)?.name.clone(),
                    value: predicate.value.clone(),
                })
            })
            .collect()
    }

    /// The predicates among `predicates` that bound a file's stats: a plain
    /// top-level column comparison (no variant path) paired with its typed
    /// constant. Variant paths are excluded: their stats live in a shredded
    /// leaf, pruned per row group, not in the file's column stats. A column the
    /// file's stats don't bound simply prunes no files.
    fn stat_filter_candidates(&self, predicates: &[PushedPredicate]) -> Vec<ColumnStatFilter> {
        predicates
            .iter()
            .filter(|predicate| predicate.path.is_empty())
            .filter_map(|predicate| {
                Some(ColumnStatFilter {
                    column: self.shape.columns.get(predicate.column_idx)?.name.clone(),
                    compare_type: predicate.compare_type,
                    value: predicate.value.clone(),
                })
            })
            .collect()
    }

    /// Whether each column can hold a NULL in `row_groups`, those of the
    /// files a scan reads: a column is free of NULLs when every row group
    /// has it and says so; a file missing the column reads as all-NULL. A
    /// scan of no file reports every column nullable. Where each column
    /// sits in a row group is looked up once for the row groups laid out
    /// alike: those of one file share a schema, and the files of one table
    /// mostly hold the same columns in the same places.
    fn find_nullable_columns(&self, row_groups: &[Arc<RowGroupMetadata>]) -> Vec<bool> {
        let columns = &self.shape.columns;
        if row_groups.is_empty() {
            return vec![true; columns.len()];
        }
        let mut may_hold_nulls = vec![false; columns.len()];
        // Where each declared column sits in the schema the positions were
        // last looked up in.
        let mut positions: Vec<Option<usize>> = Vec::new();
        let mut positions_schema = None;
        for row_group in row_groups {
            let schema = &row_group.schema;
            let is_resolved = positions_schema.is_some_and(|resolved| {
                Arc::ptr_eq(resolved, schema) || holds_columns_at(schema, columns, &positions)
            });
            if !is_resolved {
                positions.clear();
                positions.extend(
                    columns
                        .iter()
                        .map(|column| schema.index_of(&column.name).ok()),
                );
            }
            positions_schema = Some(schema);
            for (may_hold_nulls, position) in may_hold_nulls.iter_mut().zip(&positions) {
                *may_hold_nulls = *may_hold_nulls
                    || position.is_none_or(|position| row_group.column_may_hold_nulls(position));
            }
        }
        may_hold_nulls
    }

    /// What a scan over `filters` reads of the table's files: the
    /// partitions whose tuple the filters keep, and of their files those
    /// whose recorded partition tuple and column bounds admit a row the
    /// filters keep, partition by partition in the order a partition's
    /// files stand in. Only the kept partitions are indexed, and their
    /// indexes are left for the caller to count as used.
    fn select_scan_files(&self, filters: &[Expression]) -> Result<SelectedFiles> {
        let predicates: Vec<PushedPredicate> = filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let partition_by = &self.shape.partition_by;
        let partition_filters = self.partition_filter_candidates(&predicates);
        let stat_filters = self.stat_filter_candidates(&predicates);
        let committed = &self.summary.partitions;
        let kept = committed.iter().filter(|(_, partition)| {
            partition_maybe_matches(
                partition.values.as_deref(),
                partition_by,
                &partition_filters,
            )
        });
        let mut partitions: Vec<Arc<[u8]>> = kept.map(|(key, _)| Arc::clone(key)).collect();
        // A partition only a local swap has a file in is kept: its tuple is
        // read from its files.
        if let Some(swapped) = &self.swapped {
            let swapped_only = swapped.keys().filter(|key| !committed.contains_key(*key));
            partitions.extend(swapped_only.cloned());
            partitions.sort_unstable();
        }
        let mut files: Vec<Arc<DeltaFile>> = Vec::new();
        self.visit_indexes(&partitions, IndexUse::LeftToCaller, &mut |_, index| {
            let admitted = index.files().filter(|file| {
                file.entry
                    .maybe_matches_partition(partition_by, &partition_filters)
                    && file.entry.maybe_matches_stats(&stat_filters)
            });
            files.extend(admitted.cloned());
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(SelectedFiles { partitions, files })
    }

    /// The row groups of the files a scan over `filters` reads, in the
    /// order [`select_scan_files`](Self::select_scan_files) selects them.
    /// The footers are taken from the cache, where those read are kept;
    /// once they are taken, the indexes the files were selected from count
    /// as used again, so the cache lets go of a partition's footers before
    /// its index.
    fn load_scan_row_groups(&self, filters: &[Expression]) -> Result<Vec<Arc<RowGroupMetadata>>> {
        let selected = self.select_scan_files(filters)?;
        let files: Vec<&FileRef> = selected.files.iter().map(|file| &file.entry.file).collect();
        let footers = self.open_footers(&files, FooterUse::Kept)?;
        self.lineage.scope.mark_indexes_used(&selected.partitions);
        Ok(footers.into_iter().flatten().collect())
    }
}

impl Table for DeltaTable {
    fn version(&self) -> u64 {
        self.snapshot.version()
    }

    fn columns(&self) -> &[Column] {
        &self.shape.columns
    }

    fn partition_by(&self) -> &[String] {
        &self.shape.partition_by
    }

    fn sort_by(&self) -> &[String] {
        &self.shape.sort_by
    }

    /// The partitions in the order of the bytes they are named by, a few
    /// indexed at a time.
    fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> Result<()> {
        let partitions = self.list_partition_keys();
        self.visit_indexes(&partitions, IndexUse::Counted, &mut |partition, index| {
            Ok(visit(partition, &**index))
        })
    }

    /// Read from what the version counted of its files: no log is read.
    fn list_partitions(&self) -> Result<Vec<Vec<u8>>> {
        Ok(self
            .list_partition_keys()
            .iter()
            .map(|key| key.to_vec())
            .collect())
    }

    /// The version's number in the log.
    fn commit_position(&self) -> u64 {
        self.version()
    }

    /// Read from what the commits made here since that version changed,
    /// with no log read. A version committed elsewhere since, or one the
    /// history let go of, leaves what was added unknown: every partition is
    /// then listed.
    fn list_partitions_added_after(&self, commit_position: u64) -> Result<Vec<Vec<u8>>> {
        let added = self
            .lineage
            .commits
            .lock()
            .unwrap()
            .list_partitions_added_between(commit_position, self.version());
        match added {
            Some(partitions) => Ok(partitions.iter().map(|key| key.to_vec()).collect()),
            None => self.list_partitions(),
        }
    }

    /// Every one: a rewrite's output sits in the partition of its inputs.
    fn select_rewritable_partitions(&self, partitions: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        Ok(partitions)
    }

    /// The partitions are indexed together: the log is replayed once for
    /// those the cache holds no index of, as far as their files fit in
    /// what a replay keeps.
    fn index_partitions(&self, partitions: &[Vec<u8>]) -> Result<()> {
        let keys: Vec<Arc<[u8]>> = partitions
            .iter()
            .filter_map(|partition| self.find_partition_key(partition))
            .collect();
        self.visit_indexes(&keys, IndexUse::Counted, &mut |_, _| {
            Ok(ControlFlow::Continue(()))
        })
    }

    /// The partition's index; the log is replayed when the cache does not
    /// hold it.
    fn load_partition_files(&self, partition: &[u8]) -> Result<Option<Arc<dyn PartitionFiles>>> {
        Ok(self
            .load_index(partition)?
            .map(|index| index as Arc<dyn PartitionFiles>))
    }

    /// A record holds everything the log recorded for its file.
    fn find_file_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>> {
        let Some(index) = self.load_index(partition)? else {
            return Ok(vec![None; paths.len()]);
        };
        Ok(paths
            .iter()
            .map(|path| index.find(path).map(|file| file.entry.clone()))
            .collect())
    }

    /// Read from the files' records a few partitions at a time, with their
    /// footers.
    fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> Result<DatastoreTableMetadata> {
        self.describe_as(name, id)
    }

    /// Counted from the log's entries.
    fn total_rows(&self) -> Result<u64> {
        self.summary.count_rows().ok_or_else(|| {
            log::Error::UncountedRows {
                location: self.location.as_str().to_string(),
            }
            .into()
        })
    }

    /// Counted from the log's entries when every file's records its rows:
    /// a Delta file here has no deleted row. A local swap changed the files
    /// without the log, so only a scan counts its rows.
    fn count_rows(&self) -> Result<Option<u64>> {
        if self.swapped.is_some() {
            return Ok(None);
        }
        Ok(self.summary.count_rows())
    }

    /// Counted while the log was replayed; `None` for a local swap, which
    /// changed the files without the log.
    fn count_files(&self) -> Option<u64> {
        if self.swapped.is_some() {
            return None;
        }
        Some(self.summary.file_count)
    }

    /// Selects the files of an unfiltered scan and takes their footers:
    /// the partition indexes built for it and the row groups of the footers
    /// read stay in the cache of the tables' file metadata, where the next
    /// scan takes them from.
    fn preload_scan_metadata(&self) -> Result<()> {
        self.load_scan_row_groups(&[])?;
        Ok(())
    }

    /// Delta files resolve columns by name.
    fn field_ids(&self) -> Option<Vec<i32>> {
        None
    }

    /// The row groups the cache of the tables' file metadata holds for each
    /// file, shared with it; a footer is read only when the cache does not
    /// hold it, and is then kept there.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let files: Vec<&FileRef> = files.iter().collect();
        self.open_footers(&files, FooterUse::Kept)
    }

    /// Every column is declared nullable: the footers tell which columns
    /// hold a NULL, the log's metadata does not.
    fn declared_nullability(&self) -> Vec<bool> {
        vec![true; self.shape.columns.len()]
    }

    /// The row groups of the files whose recorded partition tuple and
    /// column bounds admit a row the filters keep, with the footers the
    /// cache holds; see
    /// [`load_scan_row_groups`](Self::load_scan_row_groups). Nullability
    /// is answered for the files the scan reads.
    fn select_scan(&self, filters: &[Expression]) -> Result<ScanSelection> {
        let row_groups = self.load_scan_row_groups(filters)?;
        let nullability = self.find_nullable_columns(&row_groups);
        Ok(ScanSelection {
            row_groups: Arc::new(ParquetTable::new(row_groups)),
            nullability,
        })
    }

    /// One log listing when nothing changed (Kernel updates the snapshot in
    /// hand rather than rebuilding it). A version another copy of the table
    /// in this process read or committed is taken as that copy left it.
    /// Otherwise the log is replayed at the new version, once: its files
    /// are counted, and those of the partitions the cache holds an index
    /// for are indexed again, each on the index held. No footer is read.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>> {
        let engine = &self.factory.engine;
        let Some(snapshot) = log::advance_snapshot(&self.snapshot, engine)? else {
            return Ok(None);
        };
        if let Some(known) = self.lineage.find_version(snapshot.version()) {
            return Ok(Some(Arc::new(Self {
                factory: self.factory.clone(),
                location: self.location.clone(),
                snapshot: known.snapshot,
                shape: Arc::clone(&self.shape),
                summary: known.summary,
                tombstones: known.tombstones,
                lineage: Arc::clone(&self.lineage),
                swapped: None,
            })));
        }
        let shape = log::read_table_shape(&snapshot)?;
        let (shape, lineage) = if shape == *self.shape {
            (Arc::clone(&self.shape), Arc::clone(&self.lineage))
        } else {
            let lineage = Lineage::begin(&shape)?;
            (Arc::new(shape), lineage)
        };
        let advanced = Self {
            factory: self.factory.clone(),
            location: self.location.clone(),
            snapshot,
            shape,
            summary: Arc::default(),
            tombstones: Arc::default(),
            lineage,
            swapped: None,
        };
        let indexing = advanced.lineage.indexing.lock().unwrap();
        let scope = &advanced.lineage.scope;
        let indexed: Vec<Arc<[u8]>> = scope.list_indexed_partitions();
        let held = scope.find_indexes(
            indexed.iter().map(|partition| &**partition),
            LookupUse::Counted,
        );
        let held: HashMap<Arc<[u8]>, Arc<DeltaIndex>> = indexed
            .iter()
            .cloned()
            .zip(held)
            .filter_map(|(partition, held)| Some((partition, held?.0)))
            .collect();
        let indexed: HashSet<Arc<[u8]>> = held.keys().cloned().collect();
        let kept = Kept::Partitions {
            now: &indexed,
            ahead: &HashSet::new(),
        };
        let refreshed = advanced.read_from_log(kept, &held)?;
        drop(indexing);
        Ok(Some(Arc::new(refreshed)))
    }

    /// The next version is derived from this one: its counts by the files
    /// removed and added, and the index the cache holds for each partition
    /// they sit in likewise. The log is not read, unless the partition of
    /// the removed files is not indexed.
    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        data_change: bool,
    ) -> Result<Option<Arc<dyn Table>>> {
        let now_ms = crate::vacuum::now_unix_ms();
        let removed_index = match removed.paths {
            [] => None,
            _ => self.load_committed_index(removed.partition)?,
        };
        let removed_records: Option<Vec<Arc<DeltaFile>>> = removed
            .paths
            .iter()
            .map(|path| removed_index.as_ref()?.find(path).cloned())
            .collect();
        // A file this view holds that the log at its version does not was
        // swapped in by a commit made through another copy: the log has
        // moved, and the caller refreshes onto it.
        let Some(removed_records) = removed_records else {
            return Ok(None);
        };
        let removed_entries: Vec<FileEntry> = removed_records
            .iter()
            .map(|file| file.entry.clone())
            .collect();
        let added_entries: Vec<FileEntry> = added.iter().map(|file| file.entry.clone()).collect();
        let first_commit_sequence = self.next_commit_sequence(now_ms);
        let Some(committed) = log::commit_file_changes(
            &self.factory.engine,
            &self.snapshot,
            &removed_entries,
            &added_entries,
            data_change,
            first_commit_sequence,
        )?
        else {
            return Ok(None);
        };
        let added_records = self.lineage.build_records(added, first_commit_sequence)?;
        let mut summary = VersionSummary::clone(&self.summary);
        for file in &removed_records {
            summary.remove(CountedFile::of(file));
        }
        for file in &added_records {
            summary.add(CountedFile::of(file));
        }
        let swapped = match &self.swapped {
            Some(_) if !removed.paths.is_empty() => {
                Some(self.swap_more_files(removed.partition, removed.paths, Vec::new()))
            }
            swapped => swapped.clone(),
        };
        let next = Self {
            factory: self.factory.clone(),
            location: self.location.clone(),
            snapshot: committed,
            shape: Arc::clone(&self.shape),
            summary: Arc::new(summary),
            tombstones: self.tombstones_after(removed.paths, now_ms),
            lineage: Arc::clone(&self.lineage),
            swapped,
        };
        next.record_commit(removed_records, added_records);
        next.keep_footers(added);
        next.lineage.remember_version(&next);
        Ok(Some(Arc::new(next)))
    }

    /// The files swapped in stand after every file of the version, in the
    /// order given, and their footers are kept as their writer produced
    /// them.
    fn with_files_swapped(
        &self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
    ) -> Result<Arc<dyn Table>> {
        let first_commit_sequence = self.summary.latest_commit_sequence + 1;
        let added_records = self.lineage.build_records(&added, first_commit_sequence)?;
        let swapped = self.swap_more_files(removed.partition, removed.paths, added_records);
        self.keep_footers(&added);
        Ok(Arc::new(Self {
            factory: self.factory.clone(),
            location: self.location.clone(),
            snapshot: Arc::clone(&self.snapshot),
            shape: Arc::clone(&self.shape),
            summary: Arc::clone(&self.summary),
            tombstones: Arc::clone(&self.tombstones),
            lineage: Arc::clone(&self.lineage),
            swapped: Some(swapped),
        }))
    }

    /// The table's `delta.deletedFileRetentionDuration`, or the datastore's
    /// 4-hour default when unset.
    fn deleted_file_retention(&self) -> Result<Duration> {
        Ok(log::deleted_file_retention(&self.snapshot))
    }

    /// Never: a Delta table has no snapshot to expire.
    fn has_expired_history(&self, _now_ms: u64) -> Result<bool> {
        Ok(false)
    }

    /// A Delta table has nothing to expire here: its checkpoints are written
    /// on the commit path, and what a checkpoint supersedes is deleted by
    /// [`delete_expired_objects`](Self::delete_expired_objects).
    fn expire_snapshots(&self, _now_ms: u64) -> Result<Option<SnapshotExpiry>> {
        Ok(None)
    }

    /// Nothing: a Delta table expires no snapshot, so there is never one to
    /// delete for.
    fn delete_snapshot_objects(&self, _storage: &TableStorage, _expired: &ExpiredSnapshots) {}

    /// Delete every data file this version does not reference once it has been
    /// unreferenced for the retention window: a retired file's window runs from
    /// the tombstone held for it, an orphan's (an upload no commit adopted) from
    /// its storage mtime. Then delete the commits, checkpoints and checksums a
    /// later checkpoint has made redundant, once past the log-retention window.
    /// Failed deletions are retried by the next sweep.
    ///
    /// The paths of the version's files are gathered a few partitions at a
    /// time and held for the sweep, as the listing of the table's directory
    /// is. A footer the cache holds keeps its file open, so the cache lets
    /// go of a file's before the file is deleted.
    fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()> {
        let cutoff =
            now_ms.saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        let mut live: HashSet<ObjectPath> = HashSet::new();
        let partitions = self.list_partition_keys();
        self.visit_indexes(&partitions, IndexUse::Counted, &mut |_, index| {
            live.extend(index.files().map(|file| file.entry.file.path.clone()));
            Ok(ControlFlow::Continue(()))
        })?;

        let mut deleted = 0u64;
        let mut first_error = None;
        for (path, modified_ms) in storage.list_data_files()? {
            if live.contains(&path) {
                continue;
            }
            // A retired file is dated by the commit that retired it, which keeps
            // it whatever the storage mtime says: it may have been live for days
            // before a merge or delete retired it, and a reader on the prior
            // snapshot still needs the bytes. A file with no tombstone, an
            // orphan upload or one retired before the window, is dated the only
            // way left, by when its bytes landed in storage.
            let unreferenced_since_ms = self.tombstones.get(&path).copied().unwrap_or(modified_ms);
            if unreferenced_since_ms > cutoff {
                continue;
            }
            self.lineage
                .scope
                .remove_footers(std::iter::once(path.as_str()));
            if let Err(error) = storage.delete_data_file(&path) {
                warn!(table = %name, file = %path, error = %error, "vacuum: deleting file failed");
                first_error.get_or_insert(error);
                continue;
            }
            deleted += 1;
        }
        if deleted > 0 {
            info!(table = %name, files = deleted, "vacuumed files");
        }

        match log::cleanup_log(
            self.factory.store.as_ref(),
            &self.location,
            &self.snapshot,
            now_ms,
        ) {
            Ok(cleaned) if cleaned > 0 => {
                info!(table = %name, log_files = cleaned, "cleaned up superseded log files");
            }
            Ok(_) => {}
            Err(error) => {
                warn!(table = %name, error = %error, "vacuum: log cleanup failed");
                first_error.get_or_insert(error.into());
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl DeltaTable {
    /// Remember what the commit that wrote this version changed, the files
    /// `removed` from the version it was made on and those `added`, and
    /// bring the index the cache holds for each partition they sit in to
    /// this version by them.
    fn record_commit(&self, removed: Vec<Arc<DeltaFile>>, added: Vec<Arc<DeltaFile>>) {
        let mut changes: HashMap<Arc<[u8]>, PartitionChange> = HashMap::new();
        for file in removed {
            find_change(&mut changes, &file).removed.push(file);
        }
        for file in added {
            find_change(&mut changes, &file).added.push(file);
        }
        let partitions: Vec<Arc<[u8]>> = changes.keys().cloned().collect();
        let version = self.snapshot.version();
        self.lineage
            .commits
            .lock()
            .unwrap()
            .record(version, changes.into_values().collect());
        let scope = &self.lineage.scope;
        let held = scope.find_indexes(
            partitions.iter().map(|partition| &**partition),
            LookupUse::Counted,
        );
        for (partition, held) in partitions.iter().zip(held) {
            if let Some((index, built_at)) = held {
                self.bring_index(partition, index, built_at);
            }
        }
    }
}

/// What a commit changed in the partition of `file`, among `changes`.
fn find_change<'c>(
    changes: &'c mut HashMap<Arc<[u8]>, PartitionChange>,
    file: &DeltaFile,
) -> &'c mut PartitionChange {
    let partition = file.partition();
    changes
        .entry(Arc::clone(partition))
        .or_insert_with(|| PartitionChange {
            partition: Arc::clone(partition),
            removed: Vec::new(),
            added: Vec::new(),
        })
}
