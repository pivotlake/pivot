//! The Delta Lake format: a `_delta_log` per table, read and written through
//! Delta Kernel.
//!
//! # What a version holds
//!
//! A version holds the snapshot it was read at, which is the log's position,
//! and every live file of the table at that version ([`TableFiles`]): each
//! with the entry the log records for it, its partition tuple and its
//! statistics, and the row groups of its footer. The log, its checkpoint and
//! the commits after it, lists every file of every partition, like one
//! manifest for the whole table, so one read of it builds the whole list,
//! and the version holds the list for as long as it lives. A scan, a merge,
//! a layout measurement and a vacuum sweep are answered from it, with
//! nothing read.
//!
//! # From one version to the next
//!
//! A commit made here derives the next version from the one it was made on,
//! by the files it removed and added, with no log read. A refresh that finds
//! a version committed elsewhere reads the log at that version, once, and
//! keeps the row groups of the files the version it started from holds, so
//! only the footers of the files new to it are read.
//!
//! # Compaction rounds
//!
//! A version stands among its table's commits at its log version. Which
//! partitions a commit since an earlier version added a file to is not
//! kept: a round is told every partition, and examines each off the files
//! held here.
//!
//! # A file's place among its partition's files
//!
//! A partition's files stand in the order the shared index keeps
//! ([`table_file_index::file_index`]). The commit a file belongs to is read
//! from the `modificationTime` its `Add` action carries: a commit made here
//! numbers its files from the clock's reading, or from just past the latest
//! file when the clock has not moved past it, one millisecond apart in the
//! order it adds them. A read of the log therefore gives back the order the
//! files were committed in, checkpoint or not, which the log's versions do
//! not give: a checkpoint keeps no trace of the version that added a file.

mod describe;
mod file_record;
pub(crate) mod log;
mod partition_key;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use dispatch::DataFlowDispatcher;
use iceberg_tables::PartitionKeys;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use imbl::OrdMap;
use object_storage::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, PushedPredicate, RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, PARTITION_BY_OPTION, SchemaQualifiedTableName};
use planner::expression::{CompareType, Expression};
use table_file_index::{IndexedRecord, PartitionFiles};
use tracing::{info, warn};
use uuid::Uuid;

use self::file_record::{DeltaFile, DeltaIndex, RecordBuilder};
use self::log::TableShape;
use crate::file_entry::{ColumnStatFilter, FileEntry, PartitionEqFilter, constant_partition};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{RemovedFiles, SnapshotExpiry, Table, TableFactory, VisitPartition};
use crate::{Error, Result};

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

    /// The table at `location` as the log at `snapshot`, which declares
    /// `shape`, has it: every live file with its row groups, and the retired
    /// files the log remembers. A file `held` has keeps the row groups held
    /// for it; the footers of the others are read in one pass over the pool,
    /// each reconciled by name against the columns `shape` declares. Nothing
    /// is handed back unless every footer was read.
    fn read_version(
        &self,
        location: &ObjectPath,
        snapshot: Arc<delta_kernel::Snapshot>,
        shape: Arc<TableShape>,
        held: &TableFiles,
    ) -> Result<DeltaTable> {
        let logged = log::read_files(&snapshot, &self.engine, &shape)?;
        let unheld: Vec<DataFile> = logged
            .iter()
            .filter(|file| held.find(&file.entry.file.path).is_none())
            .map(|file| {
                let file = file.entry.file.clone();
                file.into_data_file(self.store.as_ref(), location)
            })
            .collect::<object_storage::Result<_>>()?;
        // One footer per file not held, in the order the log lists them.
        let mut read = parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &unheld,
            parquet_engine::TableColumns::by_name(shape.columns.clone()),
        )?
        .into_iter();

        let records = RecordBuilder::new(&shape.columns, &shape.partition_by, &shape.sort_by)?;
        let mut files = TableFiles::default();
        for file in logged {
            let row_groups = match held.find(&file.entry.file.path) {
                Some(held_file) => held_file.row_groups.clone(),
                None => {
                    read.next()
                        .expect("a footer is read for every file not held")
                        .row_groups
                }
            };
            let (partition, record) =
                records.build(TableFile::new(file.entry, row_groups), file.commit_sequence)?;
            files.insert(&partition, record);
        }
        // The files this version no longer references but the log still
        // dates, off the same segment the files were read from, so the two
        // always agree.
        let tombstones = log::read_remove_tombstones(&snapshot, &self.engine)?;
        Ok(DeltaTable {
            factory: self.clone(),
            location: location.clone(),
            snapshot,
            shape,
            files,
            tombstones: Arc::new(tombstones),
        })
    }
}

impl TableFactory for DeltaTableFactory {
    /// A Delta table is partitioned by columns alone: every term is a
    /// declared column, and a transform of one is refused as what Delta
    /// Lake does not have.
    fn check_partition_by(
        &self,
        _name: &SchemaQualifiedTableName,
        columns: &[Column],
        partition_by: &[String],
        _adopts_files: bool,
    ) -> Result<()> {
        for term in partition_by {
            if columns.iter().any(|column| &column.name == term) {
                continue;
            }
            return Err(if term.contains('(') {
                Error::PartitionTransformInDelta { term: term.clone() }
            } else {
                Error::UnknownSpecColumn {
                    option: PARTITION_BY_OPTION.to_string(),
                    column: term.clone(),
                }
            });
        }
        Ok(())
    }

    /// Write Delta version 0 with its protocol, metadata and initial `Add`
    /// actions. An adopted file whose partition columns are constant is
    /// stamped with that partition from its stats, so it is recorded in the log
    /// and prunes like a written file; one whose values vary is adopted
    /// without a tuple. The adopted files are held with the footers the
    /// caller read, so the log just written is not read back.
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
        // The adopted files stand in the order given, numbered from the
        // first, as the log numbers them.
        let mut files = TableFiles::default();
        for (partition, record) in build_records(&shape, adopted, 0)? {
            files.insert(&partition, record);
        }
        Ok(Arc::new(DeltaTable {
            factory: self.clone(),
            location: directory.clone(),
            snapshot,
            shape,
            files,
            tombstones: Arc::default(),
        }))
    }

    /// Read the table's log at its latest version and the footer of every
    /// file it lists.
    fn load_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
    ) -> Result<Arc<dyn Table>> {
        let delta_uri = log::table_uri(&self.store.location_uri(), directory)?;
        let snapshot = log::open_snapshot(&delta_uri, &self.engine)?;
        let shape = Arc::new(log::read_table_shape(&snapshot)?);
        let loaded = self.read_version(directory, snapshot, shape, &TableFiles::default())?;
        Ok(Arc::new(loaded))
    }

    /// Nothing: a Delta table's files and footers are held by its versions,
    /// which go with the table, and by no cache.
    fn forget_cached_table(&self, _directory: &ObjectPath) -> Result<()> {
        Ok(())
    }
}

/// The record of each of `files`, files of a table of `shape`, in the order
/// given, beside the bytes its partition is named by. The first stands at
/// `first_commit_sequence` among the table's commits and each of the others
/// one past the file before it.
fn build_records(
    shape: &TableShape,
    files: Vec<TableFile>,
    first_commit_sequence: u64,
) -> Result<Vec<(Vec<u8>, DeltaFile)>> {
    let records = RecordBuilder::new(&shape.columns, &shape.partition_by, &shape.sort_by)?;
    files
        .into_iter()
        .zip(first_commit_sequence..)
        .map(|(file, commit_sequence)| records.build(file, commit_sequence))
        .collect()
}

/// Every live file of one table version. Persistent: a clone shares every
/// file and every partition, and an edit copies only what it touches, so
/// the version a commit derives costs what the commit changed.
#[derive(Clone, Default)]
struct TableFiles {
    /// The files of each partition the version has a file in, by the bytes
    /// the partition is named by.
    partitions: OrdMap<Arc<[u8]>, Arc<DeltaIndex>>,
    /// Every file of the table, for whoever names a file by its path alone.
    by_path: DeltaIndex,
    /// The rows the files' row groups hold.
    row_count: u64,
    /// Where the latest file added stands among the table's commits; 0 for
    /// a table no file was added to.
    latest_commit_sequence: u64,
}

impl TableFiles {
    /// The file at `path`, whichever partition it sits in.
    fn find(&self, path: &ObjectPath) -> Option<&Arc<DeltaFile>> {
        self.by_path.find(path)
    }

    /// The files of `partition`; `None` when the version has no file in it.
    fn find_partition(&self, partition: &[u8]) -> Option<&Arc<DeltaIndex>> {
        self.partitions.get(partition)
    }

    /// Every partition the version has a file in with its files, in the
    /// order of the bytes the partitions are named by.
    fn list_partitions(&self) -> impl Iterator<Item = (&Arc<[u8]>, &Arc<DeltaIndex>)> {
        self.partitions.iter()
    }

    /// Every file, partition by partition, each partition's in the order
    /// its files stand in.
    fn list_files(&self) -> impl Iterator<Item = &Arc<DeltaFile>> {
        self.partitions.values().flat_map(|index| index.files())
    }

    fn count_files(&self) -> u64 {
        self.by_path.len() as u64
    }

    fn count_rows(&self) -> u64 {
        self.row_count
    }

    /// Add `file` to `partition`, in place of a file of the partition
    /// already at its path.
    fn insert(&mut self, partition: &[u8], file: DeltaFile) {
        let file = Arc::new(file);
        let facts = file.facts();
        self.remove(partition, &facts.file().path);
        self.row_count += file.count_rows();
        self.latest_commit_sequence = self.latest_commit_sequence.max(facts.commit_sequence());
        let (key, mut index) = match self.partitions.get_key_value(partition) {
            Some((key, index)) => (Arc::clone(key), DeltaIndex::clone(index)),
            None => (Arc::from(partition), DeltaIndex::default()),
        };
        index.insert(Arc::clone(&file));
        self.partitions.insert(key, Arc::new(index));
        self.by_path.insert(file);
    }

    /// Drop the file of `partition` at `path`, if the version has one. A
    /// partition left without a file is no longer one of the version's.
    fn remove(&mut self, partition: &[u8], path: &ObjectPath) {
        let Some((key, index)) = self.partitions.get_key_value(partition) else {
            return;
        };
        let (key, mut index) = (Arc::clone(key), DeltaIndex::clone(index));
        let Some(file) = index.remove(path) else {
            return;
        };
        if index.is_empty() {
            self.partitions.remove(&key);
        } else {
            self.partitions.insert(key, Arc::new(index));
        }
        self.by_path.remove(path);
        self.row_count -= file.count_rows();
    }
}

/// One Delta table at one log version: the snapshot it was read at and that
/// every commit from it is written on top of, every live file of the version
/// with its row groups, and the retired files the log still remembers.
#[derive(Clone)]
struct DeltaTable {
    factory: DeltaTableFactory,
    location: ObjectPath,
    snapshot: Arc<delta_kernel::Snapshot>,
    /// The columns the version declares, and those its files are
    /// partitioned and sorted by.
    shape: Arc<TableShape>,
    /// The version's files. A local view of file swaps this process
    /// committed through another copy of the table holds the files swapped
    /// in and not those swapped out, which the log at its snapshot does not
    /// record; the next refresh reconciles the copy to the committed log.
    files: TableFiles,
    /// The files the log retired but still remembers, each by the Unix ms its
    /// commit retired it: what vacuum ages a retired file by, so the retention
    /// window runs from the moment the file stopped being referenced rather
    /// than from its storage mtime. Read from the log alongside the files,
    /// which bounds it to what the log still carries, and extended by every
    /// commit made on this version, which drops the entries a window older
    /// than itself.
    tombstones: Arc<HashMap<ObjectPath, u64>>,
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

impl DeltaTable {
    /// The error of a caller that names `path`, a file this version does
    /// not hold.
    fn conflict_at(&self, path: &ObjectPath) -> Error {
        Error::CommitConflict {
            location: self.location.as_str().to_string(),
            file: path.to_string(),
        }
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

    /// Every partition column is a key the rows carry.
    fn plan_partition_keys(&self) -> Result<PartitionKeys> {
        Ok(PartitionKeys::by_columns(&self.shape.partition_by))
    }

    fn sort_by(&self) -> &[String] {
        &self.shape.sort_by
    }

    /// The partitions in the order of the bytes they are named by.
    fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> Result<()> {
        for (partition, index) in self.files.list_partitions() {
            if visit(partition, &**index).is_break() {
                break;
            }
        }
        Ok(())
    }

    fn list_partitions(&self) -> Result<Vec<Vec<u8>>> {
        Ok(self
            .files
            .list_partitions()
            .map(|(partition, _)| partition.to_vec())
            .collect())
    }

    /// The version's number in the log.
    fn commit_position(&self) -> u64 {
        self.version()
    }

    /// Every partition: what a commit added is not kept.
    fn list_partitions_added_after(&self, _commit_position: u64) -> Result<Vec<Vec<u8>>> {
        self.list_partitions()
    }

    /// Every one: a rewrite's output sits in the partition of its inputs.
    fn select_rewritable_partitions(&self, partitions: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        Ok(partitions)
    }

    /// Nothing: the version holds every partition's files.
    fn index_partitions(&self, _partitions: &[Vec<u8>]) -> Result<()> {
        Ok(())
    }

    fn load_partition_files(&self, partition: &[u8]) -> Result<Option<Arc<dyn PartitionFiles>>> {
        Ok(self
            .files
            .find_partition(partition)
            .map(|index| Arc::clone(index) as Arc<dyn PartitionFiles>))
    }

    /// A file's record holds everything the log recorded for it.
    fn find_file_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>> {
        let index = self.files.find_partition(partition);
        Ok(paths
            .iter()
            .map(|path| Some(index?.find(path)?.entry.clone()))
            .collect())
    }

    fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> Result<DatastoreTableMetadata> {
        Ok(self.describe_as(name, id))
    }

    /// The rows the footers of the version's files count.
    fn total_rows(&self) -> Result<u64> {
        Ok(self.files.count_rows())
    }

    /// The rows the footers of the version's files count: a scan reads
    /// exactly those row groups, and a Delta file here has no deleted row.
    fn count_rows(&self) -> Result<Option<u64>> {
        Ok(Some(self.files.count_rows()))
    }

    fn count_files(&self) -> Option<u64> {
        Some(self.files.count_files())
    }

    /// Nothing: the version holds what a scan reads of its metadata and of
    /// its files' footers.
    fn preload_scan_metadata(&self) -> Result<()> {
        Ok(())
    }

    /// Delta files resolve columns by name.
    fn field_ids(&self) -> Option<Vec<i32>> {
        None
    }

    /// The row groups the version holds for each file, shared with it. A
    /// file the version does not hold is a commit conflict: another writer
    /// swapped it out.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        files
            .iter()
            .map(|file| match self.files.find(&file.path) {
                Some(held) => Ok(held.row_groups.clone()),
                None => Err(self.conflict_at(&file.path)),
            })
            .collect()
    }

    /// Every column is declared nullable: the footers tell which columns
    /// hold a NULL, the log's metadata does not.
    fn declared_nullability(&self) -> Vec<bool> {
        vec![true; self.shape.columns.len()]
    }

    /// The row groups of the files whose recorded partition tuple and
    /// column bounds admit a row the filters keep, partition by partition
    /// in the order a partition's files stand in. Nullability is answered
    /// for the files the scan reads.
    fn select_scan(&self, filters: &[Expression]) -> Result<ScanSelection> {
        let predicates: Vec<PushedPredicate> = filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let partition_by = &self.shape.partition_by;
        let partition_filters = self.partition_filter_candidates(&predicates);
        let stat_filters = self.stat_filter_candidates(&predicates);
        let row_groups: Vec<Arc<RowGroupMetadata>> = self
            .files
            .list_files()
            .filter(|file| {
                file.entry
                    .maybe_matches_partition(partition_by, &partition_filters)
                    && file.entry.maybe_matches_stats(&stat_filters)
            })
            .flat_map(|file| file.row_groups.iter().cloned())
            .collect();
        let nullability = self.find_nullable_columns(&row_groups);
        Ok(ScanSelection {
            row_groups: Arc::new(ParquetTable::new(row_groups)),
            nullability,
        })
    }

    /// One log listing when nothing changed (Kernel updates the snapshot in
    /// hand rather than rebuilding it). Otherwise the log is read at the
    /// new version, and the footers of the files this version does not hold.
    /// A footer that cannot be read fails the refresh, so no copy is handed
    /// back at a version whose files it does not hold.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>> {
        let factory = &self.factory;
        let Some(snapshot) = log::advance_snapshot(&self.snapshot, &factory.engine)? else {
            return Ok(None);
        };
        let shape = Arc::new(log::read_table_shape(&snapshot)?);
        let refreshed = factory.read_version(&self.location, snapshot, shape, &self.files)?;
        Ok(Some(Arc::new(refreshed)))
    }

    /// The next version is this one's files less those removed, with those
    /// added standing after every file of this version, in the order given:
    /// the log is written and not read.
    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        data_change: bool,
    ) -> Result<Option<Arc<dyn Table>>> {
        let now_ms = crate::vacuum::now_unix_ms();
        let removed_index = self.files.find_partition(removed.partition);
        let removed_entries: Vec<FileEntry> = removed
            .paths
            .iter()
            .map(|path| {
                removed_index
                    .and_then(|index| index.find(path))
                    .expect("a file to remove is live in the version it is removed from")
                    .entry
                    .clone()
            })
            .collect();
        let added_entries: Vec<FileEntry> = added.iter().map(|file| file.entry.clone()).collect();
        // Where the first file added stands among the table's commits: at
        // the clock's reading, or just after the latest file when the clock
        // has not moved past it, so the files of a later commit always stand
        // after an earlier one's.
        let first_commit_sequence = now_ms.max(self.files.latest_commit_sequence + 1);
        let added_records = build_records(&self.shape, added.to_vec(), first_commit_sequence)?;
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
        let mut files = self.files.clone();
        for path in removed.paths {
            files.remove(removed.partition, path);
        }
        for (partition, record) in added_records {
            files.insert(&partition, record);
        }
        Ok(Some(Arc::new(Self {
            snapshot: committed,
            files,
            tombstones: self.tombstones_after(removed.paths, now_ms),
            ..self.clone()
        })))
    }

    /// The files swapped in stand after every file of the version, in the
    /// order given, with the footers their writer produced.
    fn with_files_swapped(
        &self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
    ) -> Result<Arc<dyn Table>> {
        let first_commit_sequence = self.files.latest_commit_sequence + 1;
        let mut files = self.files.clone();
        for path in removed.paths {
            files.remove(removed.partition, path);
        }
        for (partition, record) in build_records(&self.shape, added, first_commit_sequence)? {
            files.insert(&partition, record);
        }
        Ok(Arc::new(Self {
            files,
            ..self.clone()
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
    fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()> {
        let cutoff =
            now_ms.saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        let mut deleted = 0u64;
        let mut first_error = None;
        for (path, modified_ms) in storage.list_data_files()? {
            if self.files.find(&path).is_some() {
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
