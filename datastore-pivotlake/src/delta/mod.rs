//! The Delta Lake format: a `_delta_log` per table, read and written through
//! Delta Kernel.
//!
//! # What a version holds
//!
//! A version holds the snapshot it was read at, which is the log's position,
//! and every live file of the table at that version ([`TableFiles`]), once,
//! by its path: each with the entry the log records for it, its partition
//! tuple and its statistics, and the row groups of its footer. The log, its checkpoint and
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
//! A Delta table keeps no manifests, so one version of it is described as
//! a single manifest, named by the version: the log lists
//! every file of every partition, and keeps no trace of which commit a file
//! came with. The compacter, which keeps a table's files between its
//! rounds, therefore takes every file of a version it has not seen, off the
//! records held here, and keeps what it measured of the ones it already
//! had. The version a commit made here holds the records of the one it was
//! made on, less those removed. The compacter tells the files that stayed
//! by their paths, and takes out and in only what the commit removed and
//! added.
//!
//! # The order of a version's files
//!
//! A version lists its files in the order of their paths, to a scan, to
//! the catalog and to the compacter alike: what a version lists depends on
//! what it holds and on nothing else, so one version lists alike every
//! time. The order maintenance works a partition's files in, by where each
//! starts on the table's sort key and then by its commit
//! (`compare_file_order`), is the compacter's to make of the
//! records it takes. The commit a file belongs to is read
//! from the `modificationTime` its `Add` action carries: a commit made here
//! numbers its files from the clock's reading, or from just past the latest
//! file when the clock has not moved past it, one millisecond apart in the
//! order it adds them. A read of the log therefore gives back the order the
//! files were committed in, checkpoint or not, which the log's versions do
//! not give: a checkpoint keeps no trace of the version that added a file.

mod describe;
mod file_record;
pub(crate) mod log;
mod partition;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use dispatch::DataFlowDispatcher;
use iceberg_tables::PartitionKeys;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use imbl::{OrdMap, OrdSet};
use object_storage::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, PushedPredicate, RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::{CompareType, Expression};
use tracing::{info, warn};
use uuid::Uuid;

use self::file_record::{DeltaFile, FileByPath, RecordBuilder};
use self::log::TableShape;
use self::partition::DeltaPartition;
use crate::file_entry::{ColumnStatFilter, FileEntry, PartitionEqFilter};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{
    CommitOutcome, CompactionFile, ManifestId, PartitionFiles, PartitionHandle, RemovedFile,
    SnapshotExpiry, Table, TableFactory, TablePartition,
};
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
            let record =
                records.build(TableFile::new(file.entry, row_groups), file.commit_sequence)?;
            files.insert(record);
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

/// Refuse `partition_by`, the terms a new table of `columns` is to be
/// partitioned by, unless every term is a declared column: a Delta table is
/// partitioned by columns alone, and a transform of one is refused as what
/// Delta Lake does not have.
fn check_partition_by(columns: &[Column], partition_by: &[String]) -> Result<()> {
    for term in partition_by {
        if columns.iter().any(|column| &column.name == term) {
            continue;
        }
        return Err(if term.contains('(') {
            Error::PartitionTransformInDelta { term: term.clone() }
        } else {
            Error::UnknownPartitionColumn {
                column: term.clone(),
            }
        });
    }
    Ok(())
}

impl TableFactory for DeltaTableFactory {
    /// Write Delta version 0 with its protocol, metadata and initial `Add`
    /// actions. A partition term that is not a declared column is refused
    /// first, with nothing written. An adopted file whose partition columns are constant is
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
        check_partition_by(&columns, &partition_by)?;
        if !partition_by.is_empty() {
            for file in &mut adopted {
                file.entry.partition = file
                    .entry
                    .stats
                    .as_ref()
                    .and_then(|stats| stats.find_constant_partition(&partition_by));
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
        // The adopted files are numbered from the first, in the order
        // given, as the log numbers them.
        let mut files = TableFiles::default();
        for record in build_records(&shape, adopted, 0)? {
            files.insert(record);
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
        _id: Uuid,
    ) -> Result<Arc<dyn Table>> {
        let delta_uri = log::table_uri(&self.store.location_uri(), directory)?;
        let snapshot = log::open_snapshot(&delta_uri, &self.engine)?;
        let shape = Arc::new(log::read_table_shape(&snapshot)?);
        let loaded = self.read_version(directory, snapshot, shape, &TableFiles::default())?;
        Ok(Arc::new(loaded))
    }

    /// Nothing: a Delta table's files and footers are held by its versions,
    /// which go with the table, and by no cache.
    fn forget_cached_table(&self, _id: Uuid) {}
}

/// The record of each of `files`, files of a table of `shape`, in the order
/// given. The first stands at `first_commit_sequence` among the table's
/// commits and each of the others one past the file before it.
fn build_records(
    shape: &TableShape,
    files: Vec<TableFile>,
    first_commit_sequence: u64,
) -> Result<Vec<DeltaFile>> {
    let records = RecordBuilder::new(&shape.columns, &shape.partition_by, &shape.sort_by)?;
    files
        .into_iter()
        .zip(first_commit_sequence..)
        .map(|(file, commit_sequence)| records.build(file, commit_sequence))
        .collect()
}

/// Every live file of one table version, held once, by its path.
/// Persistent: a clone shares every file, and an edit copies only what it
/// touches, so the version a commit derives costs what the commit changed.
///
/// The files stand in the order of their paths and in no other: a listing
/// of a version depends on what the version holds alone, so the same
/// version lists alike every time, and whoever needs its files in another
/// order orders them.
#[derive(Clone, Default)]
struct TableFiles {
    /// The files, each found by its path.
    files: OrdSet<FileByPath>,
    /// How many files each partition the version has a file in holds. The
    /// partition held here is the one copy every file of it shares.
    partitions: OrdMap<Arc<DeltaPartition>, usize>,
    /// The rows the files' row groups hold.
    row_count: u64,
    /// Where the latest file added stands among the table's commits; 0 for
    /// a table no file was added to.
    latest_commit_sequence: u64,
}

impl TableFiles {
    /// The file at `path`, whichever partition it sits in.
    fn find(&self, path: &ObjectPath) -> Option<&Arc<DeltaFile>> {
        let FileByPath(file) = self.files.get(path.as_str())?;
        Some(file)
    }

    /// The file of `partition` at `path`; `None` when the version has no
    /// file at the path, or has it in another partition.
    fn find_in_partition(
        &self,
        partition: &DeltaPartition,
        path: &ObjectPath,
    ) -> Option<&Arc<DeltaFile>> {
        self.find(path)
            .filter(|file| file.partition.as_ref() == partition)
    }

    /// Every partition the version has a file in, with how many files it
    /// holds, in the partitions' order.
    fn list_partitions(&self) -> impl Iterator<Item = (&Arc<DeltaPartition>, usize)> {
        let partitions = self.partitions.iter();
        partitions.map(|(partition, files)| (partition, *files))
    }

    /// Every file, in the order of the paths.
    fn list_files(&self) -> impl Iterator<Item = &Arc<DeltaFile>> {
        self.files.iter().map(|FileByPath(file)| file)
    }

    fn count_files(&self) -> u64 {
        self.files.len() as u64
    }

    fn count_rows(&self) -> u64 {
        self.row_count
    }

    /// Add `file`, in place of a file already at its path.
    fn insert(&mut self, mut file: DeltaFile) {
        self.remove(&file.entry.file.path);
        self.row_count += file.count_rows();
        self.latest_commit_sequence = self.latest_commit_sequence.max(file.commit_sequence());
        let held = self.partitions.get_key_value(file.partition.as_ref());
        let files_held = match held {
            Some((partition, files)) => {
                file.partition = Arc::clone(partition);
                *files
            }
            None => 0,
        };
        self.partitions
            .insert(Arc::clone(&file.partition), files_held + 1);
        self.files.insert(FileByPath(Arc::new(file)));
    }

    /// Drop the file at `path`, if the version has one. A partition left
    /// without a file is no longer one of the version's.
    fn remove(&mut self, path: &ObjectPath) {
        let Some(FileByPath(file)) = self.files.remove(path.as_str()) else {
            return;
        };
        self.row_count -= file.count_rows();
        let files_held = self.partitions[file.partition.as_ref()];
        if files_held == 1 {
            self.partitions.remove(file.partition.as_ref());
        } else {
            self.partitions
                .insert(Arc::clone(&file.partition), files_held - 1);
        }
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
    /// The version's files.
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
    /// The version's files a partition at a time, in the partitions'
    /// order, each partition's in the order of their paths: the
    /// version's one set of files gone through once, each file put with its
    /// partition's.
    fn list_files_by_partition(&self) -> Vec<PartitionFiles> {
        let mut partitions: Vec<PartitionFiles> = Vec::new();
        let mut positions: HashMap<&DeltaPartition, usize> = HashMap::new();
        for (partition, files_held) in self.files.list_partitions() {
            positions.insert(partition.as_ref(), partitions.len());
            let shared: Arc<dyn TablePartition> = Arc::clone(partition) as _;
            partitions.push(PartitionFiles {
                partition: PartitionHandle::new(shared),
                files: Vec::with_capacity(files_held),
            });
        }
        for file in self.files.list_files() {
            let position = positions[file.partition.as_ref()];
            let record: Arc<dyn CompactionFile> = Arc::clone(file) as _;
            partitions[position].files.push(record);
        }
        partitions
    }

    /// What names the one manifest this version's files come in.
    fn name_manifest(&self) -> ManifestId {
        ManifestId::new(Arc::from(format!("version {}", self.version())))
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
        removed: &[RemovedFile<'_>],
        retired_at_ms: u64,
    ) -> Arc<HashMap<ObjectPath, u64>> {
        if removed.is_empty() {
            return Arc::clone(&self.tombstones);
        }
        let mut tombstones = HashMap::clone(&self.tombstones);
        let cutoff = retired_at_ms
            .saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        tombstones.retain(|_, at| *at > cutoff);
        tombstones.extend(
            removed
                .iter()
                .map(|file| (file.path.clone(), retired_at_ms)),
        );
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

    /// Every partition column is a key the rows carry.
    fn plan_partition_keys(&self) -> Result<PartitionKeys> {
        Ok(PartitionKeys::by_columns(&self.shape.partition_by))
    }

    fn sort_by(&self) -> &[String] {
        &self.shape.sort_by
    }

    /// The one manifest of the version, named by the version's number in
    /// the log.
    fn list_manifest_ids(&self) -> Vec<ManifestId> {
        vec![self.name_manifest()]
    }

    /// Every file of the version, a partition at a time in the
    /// partitions' order, each partition's in the order of their paths
    /// and each as the record the version holds of it: nothing is read.
    fn load_manifest_files(&self, manifests: &[ManifestId]) -> Result<Vec<Vec<PartitionFiles>>> {
        let own_manifest = self.name_manifest();
        Ok(manifests
            .iter()
            .map(|manifest| {
                assert_eq!(
                    manifest, &own_manifest,
                    "a version is asked for a manifest it lists"
                );
                self.list_files_by_partition()
            })
            .collect())
    }

    /// A file's record holds everything the log recorded for it.
    fn find_file_entries(
        &self,
        partition: &PartitionHandle,
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>> {
        let partition: &DeltaPartition = partition.read_as();
        Ok(paths
            .iter()
            .map(|path| {
                let file = self.files.find_in_partition(partition, path)?;
                Some(file.entry.clone())
            })
            .collect())
    }

    fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> Result<DatastoreTableMetadata> {
        Ok(self.describe_as(name, id))
    }

    /// The rows the footers of the version's files count: a Delta file here
    /// has no deleted row, so the stored rows are the live ones.
    fn count_stored_rows(&self) -> Result<u64> {
        Ok(self.files.count_rows())
    }

    /// The rows the footers of the version's files count: a scan reads
    /// exactly those row groups, and a Delta file here has no deleted row.
    fn count_live_rows(&self) -> Option<u64> {
        Some(self.files.count_rows())
    }

    fn count_files(&self) -> Option<u64> {
        Some(self.files.count_files())
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
                None => Err(Error::CommitConflict {
                    location: self.location.as_str().to_string(),
                    file: file.path.to_string(),
                }),
            })
            .collect()
    }

    /// Every column is declared nullable: the footers tell which columns
    /// hold a NULL, the log's metadata does not.
    fn declared_nullability(&self) -> Vec<bool> {
        vec![true; self.shape.columns.len()]
    }

    /// The row groups of the files whose recorded partition tuple and
    /// column bounds admit a row the filters keep, in the order of the
    /// files' paths. Nullability is answered for the files the scan reads.
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
    /// added, each standing after every file of this version among the
    /// table's commits, in the order given: the log is written and not
    /// read. It holds this version's records of the files that stay.
    fn commit(
        &self,
        removed: &[RemovedFile<'_>],
        added: &[TableFile],
        data_change: bool,
    ) -> Result<CommitOutcome> {
        let now_ms = crate::vacuum::now_unix_ms();
        let mut removed_entries: Vec<FileEntry> = Vec::with_capacity(removed.len());
        for file in removed {
            let partition: &DeltaPartition = file.partition.read_as();
            match self.files.find_in_partition(partition, file.path) {
                Some(held) => removed_entries.push(held.entry.clone()),
                None => return Ok(CommitOutcome::MissingFile(file.path.clone())),
            }
        }
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
            return Ok(CommitOutcome::OutOfDate);
        };
        let mut files = self.files.clone();
        for file in removed {
            files.remove(file.path);
        }
        for record in added_records {
            files.insert(record);
        }
        let next = Self {
            snapshot: committed,
            files,
            tombstones: self.tombstones_after(removed, now_ms),
            ..self.clone()
        };
        Ok(CommitOutcome::Committed(Arc::new(next)))
    }

    /// The table's `delta.deletedFileRetentionDuration`, or the datastore's
    /// 4-hour default when unset.
    fn deleted_file_retention(&self) -> Result<Duration> {
        Ok(log::deleted_file_retention(&self.snapshot))
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
