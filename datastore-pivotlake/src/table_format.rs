//! The table format a database keeps its tables in, and the seam each format
//! implements so the datastore above it is written once.
//!
//! A format owns only what differs between Delta Lake and Iceberg: how a
//! table's current version is read and advanced, how a version's files
//! are found in its metadata and selected for a scan, which groups its
//! files come in for whoever keeps them from one version to the next, how
//! a set of file additions and removals becomes one committed version, and
//! what its metadata lets maintenance delete. Everything else, the
//! database index, the transaction and DDL, the insert sink, the compaction
//! planner and actor, the vacuum driver and the query binding, lives above
//! this seam and never asks which format it is working on.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use iceberg_tables::PartitionKeys;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Result;
use crate::compact::sort_key::{BoundSide, SortValue};
use crate::file_entry::FileEntry;
use crate::table::{TableFile, TableStorage};

/// The format every table of one database is kept in, Delta Lake or
/// Iceberg, recorded once in the database's index when the database is
/// created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TableFormat {
    /// Delta Lake: a `_delta_log` per table, read and written through Delta
    /// Kernel.
    #[serde(alias = "deltalake")]
    Delta,
    /// Iceberg: numbered metadata files per table, each version committed with a
    /// conditional write, readable by any Iceberg reader from the path of a
    /// version's metadata file.
    Iceberg,
}

impl TableFormat {
    /// The format a database created without naming one gets.
    pub const DEFAULT_FOR_NEW_DATABASES: Self = Self::Iceberg;
    /// The format of a database whose index records none: every database
    /// created before the index carried the field is Delta Lake.
    pub const RECORDED_BY_OMISSION: Self = Self::Delta;

    /// The format's name as config, the shell and the index spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Delta => "delta",
            Self::Iceberg => "iceberg",
        }
    }
}

impl fmt::Display for TableFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A format name that is neither `delta` (or its alias `deltalake`) nor
/// `iceberg`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownTableFormat(pub String);

impl fmt::Display for UnknownTableFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown table format `{}`; the formats are `delta` and `iceberg`",
            self.0
        )
    }
}

impl std::error::Error for UnknownTableFormat {}

impl FromStr for TableFormat {
    type Err = UnknownTableFormat;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        match name {
            "delta" | "deltalake" => Ok(Self::Delta),
            "iceberg" => Ok(Self::Iceberg),
            other => Err(UnknownTableFormat(other.to_string())),
        }
    }
}

/// What names one group of a table's files: files the table's format lists
/// together and never changes once it has, as a manifest of an Iceberg
/// table does, or one version of a Delta table. Two versions of a table
/// that have a group under one name hold the same files in it, so whoever
/// keeps a table's files from one version to the next catches up with a
/// version by the groups it does not hold.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FileGroupId(Arc<str>);

impl FileGroupId {
    /// The group a format names `name`.
    pub(crate) fn new(name: Arc<str>) -> Self {
        Self(name)
    }

    /// The name its format gave the group.
    pub(crate) fn name(&self) -> &Arc<str> {
        &self.0
    }
}

/// The live files a group lists of one partition.
pub(crate) struct PartitionGroupFiles {
    /// The partition's key: the bytes the table's format names the
    /// partition by.
    pub partition: Vec<u8>,
    pub files: Vec<Arc<dyn CompactionFile>>,
}

/// What a commit did to the groups of the version it was made on, as a
/// format that need not load a group to say so tells it. Whoever keeps that
/// version's files, and knows which of them the commit removed, brings them
/// to the committed version by this alone.
pub(crate) struct CarriedFileGroups {
    /// Each group the commit replaced, beside the group of the committed
    /// version that lists what the commit left of its files. Every other
    /// group of the earlier version is one of the committed version's, as
    /// it was.
    pub replaced: Vec<(FileGroupId, FileGroupId)>,
    /// The group of the committed version that lists the files the commit
    /// added: one of those that replaced a group, or one of its own.
    pub added_in: FileGroupId,
    /// The files the commit added, a partition at a time, each as the
    /// committed version holds it.
    pub added: Vec<PartitionGroupFiles>,
}

/// What a commit on top of a version came to.
pub(crate) enum CommitOutcome {
    /// The commit made the next version.
    Committed(CommittedVersion),
    /// Another writer committed a version first: the caller refreshes and
    /// retries.
    OutOfDate,
    /// The version holds no file of the partition at this path, one the
    /// commit was to remove: another writer removed it first.
    MissingFile(ObjectPath),
}

/// The version a commit made.
pub(crate) struct CommittedVersion {
    pub table: Arc<dyn Table>,
    /// What the commit did to the groups of the version it was made on;
    /// `None` for a format whose commit writes groups that are loaded to
    /// learn what they list.
    pub carried_groups: Option<CarriedFileGroups>,
}

/// One live data file as the compacter keeps it between its rounds: what
/// the record a format keeps of the file tells of it. A format holds a
/// record in a form of its own, and what it tells where that suits the
/// format: the compacter reads it here, with no value copied out, so it
/// keeps the files of any format alike.
pub(crate) trait CompactionFile: Send + Sync + 'static {
    /// The file's path as its table names it, and its size.
    fn file(&self) -> &FileRef;

    /// How many rows the file holds, deleted ones included, when the
    /// table's metadata records it.
    fn row_count(&self) -> Option<u64>;

    /// Where the file's commit stands among the table's commits: a later
    /// commit's is not below an earlier one's.
    fn commit_sequence(&self) -> u64;

    /// How many columns the file's table is sorted by.
    fn count_sort_columns(&self) -> usize;

    /// The file's bound on `side` of the sort column at `position`, which
    /// is below [`count_sort_columns`](Self::count_sort_columns), read
    /// where the record holds it, as a value of the column's current type.
    /// The same value for as long as the record lives: the order the
    /// compacter holds the file in depends on it.
    fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_>;

    /// The file's entry, which a layout measurement compares the file's
    /// row groups against, its statistics holding its row count and its
    /// bounds on the columns the table is sorted by, as the table's
    /// metadata recorded them. A format that keeps no more of the file
    /// fills in nothing else of the entry.
    fn build_entry(&self) -> FileEntry;
}

/// What creates and loads tables in one format on one database's storage.
pub(crate) trait TableFactory: Send + Sync {
    /// Refuse `partition_by`, the terms `CREATE TABLE` gave for the new
    /// table `name` of `columns`, each a column or a transform of one
    /// (`day(ts)`), unless the format can partition the table by every one
    /// of them. `adopts_files` says the table adopts files as its initial
    /// data, which no partition transform cut.
    fn check_partition_by(
        &self,
        name: &SchemaQualifiedTableName,
        columns: &[Column],
        partition_by: &[String],
        adopts_files: bool,
    ) -> Result<()>;

    /// Write the initial version of a new table at `directory` (its own
    /// directory under the database root) holding `adopted`, whose footers
    /// the caller already read, and return the table at that version.
    /// `partition_by` passed [`check_partition_by`](Self::check_partition_by).
    #[allow(clippy::too_many_arguments)] // one call per table creation; each field is needed
    fn create_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        id: Uuid,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        adopted: Vec<TableFile>,
    ) -> Result<Arc<dyn Table>>;

    /// Read the table at `directory` at its latest committed version.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
    ) -> Result<Arc<dyn Table>>;

    /// Let go of what this process caches of the dropped table at
    /// `directory`, whose storage is deleted whole once its tombstone ages
    /// out: a cached footer holds its file open, and a deleted local file
    /// keeps its blocks for as long as it is open.
    fn forget_cached_table(&self, directory: &ObjectPath) -> Result<()>;
}

/// One table at one committed version, as its format holds it. Immutable: a
/// refresh or a commit hands back the next version as a new value, so a
/// copy handed to a query never changes under it.
pub(crate) trait Table: Send + Sync {
    /// The committed version this value is at: monotonic per table, so a
    /// published copy is newer than the catalog's exactly when its version is
    /// higher.
    fn version(&self) -> u64;
    fn columns(&self) -> &[Column];
    /// The terms a new file of the table is partitioned by, in order, each
    /// as `CREATE TABLE` spells it: a column, or a transform of one.
    fn partition_by(&self) -> &[String];
    /// The key columns a new file of the table is cut by, one per term of
    /// [`partition_by`](Self::partition_by): which of them the rows carry
    /// as columns, and how the others are computed from a row.
    fn plan_partition_keys(&self) -> Result<PartitionKeys>;
    fn sort_by(&self) -> &[String];

    /// Of `partitions`, each by its key, those whose files a rewrite may
    /// replace, in the order given. A format leaves
    /// out a partition it cannot write the replacement of, and logs that
    /// it did.
    fn select_rewritable_partitions(&self, partitions: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>>;
    /// The groups the live files of this version come in, in no order: a
    /// live file is in exactly one of them. Nothing is read.
    fn list_file_groups(&self) -> Vec<FileGroupId>;
    /// The live files each of `groups` lists, groups of this version, in
    /// the order given: for each group, the files of every partition it
    /// lists one of, a partition's in no order the caller may rely on but
    /// that of the version's own listing. A format reads what it does not hold of a group's
    /// metadata; one whose version holds every file reads nothing.
    fn load_file_groups(&self, groups: &[FileGroupId]) -> Result<Vec<Vec<PartitionGroupFiles>>>;
    /// The entry of each of `paths`, in the order given: the live file of
    /// `partition` at that path, with its partition tuple and the columns
    /// its rows are sorted by, as the metadata recorded them; `None` for a
    /// path this version has no file of the partition at. A format may
    /// leave an entry's statistics out.
    fn find_file_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>>;
    /// This version as the cross-datastore catalog describes it under
    /// `name` and `id`: the columns it declares with what they cost, and
    /// the files it holds, each with its partition and the bounds its
    /// metadata recorded.
    fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> Result<DatastoreTableMetadata>;
    /// The rows of every live data file at this version, counted without
    /// reading the files' metadata.
    fn total_rows(&self) -> Result<u64>;
    /// The rows a scan of the whole table returns, when the metadata this
    /// version holds counts them exactly; `None` when only a scan can.
    fn count_rows(&self) -> Option<u64>;
    /// The live data files at this version, counted from the metadata the
    /// version holds, with nothing read; `None` when that metadata does not
    /// count them.
    fn count_files(&self) -> Option<u64>;
    /// Load what a scan of the whole table reads of the table's metadata
    /// and of its files' footers, so the scan reads none of it.
    fn preload_scan_metadata(&self) -> Result<()>;
    /// The field id each column (in `columns` order) is written under in new
    /// Parquet files, for a format that resolves columns by id; `None` for one
    /// that resolves them by name.
    fn field_ids(&self) -> Option<Vec<i32>>;
    /// The row groups of each of `files` (live files of this very version),
    /// in the order given: what a merge and a layout measurement read. A
    /// format that keeps its footers in a cache reads those the cache does
    /// not hold, and the caller keeps what it is handed only for as long as
    /// it works on it.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>>;
    /// Whether each column (in `columns` order) may hold a NULL in a file
    /// of this version, as the metadata held declares it with nothing read;
    /// a format whose metadata declares no column free of NULLs reports
    /// every one nullable.
    fn declared_nullability(&self) -> Vec<bool>;
    /// What a scan over `filters`, the predicates pushed into it, reads:
    /// the row groups of the files this version's metadata cannot rule out,
    /// in an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization. A format may
    /// answer nullability for more files than the scan reads.
    fn select_scan(&self, filters: &[Expression]) -> Result<ScanSelection>;
    /// The latest committed version, or `None` when this one is still current.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>>;
    /// Commit, on top of this version, the removal of `removed` and the
    /// addition of `added`, whose footers the writer holds, as one new
    /// version. `data_change` is `false` for a rearrangement (compaction)
    /// that leaves the table's rows as they were. Nothing is committed
    /// when this version does not hold one of `removed`.
    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        data_change: bool,
    ) -> Result<CommitOutcome>;
    /// How long a file this table no longer references is kept before it may
    /// be deleted.
    fn deleted_file_retention(&self) -> Result<Duration>;
    /// Whether this version has anything
    /// [`expire_snapshots`](Self::expire_snapshots) would expire as of
    /// `now_ms`, answered from the metadata the version holds with nothing
    /// read: what a caller asks before it takes the table's commit lock for
    /// an expiry.
    fn has_expired_history(&self, now_ms: u64) -> Result<bool>;
    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the table's commit lock, and hand back the version that wrote with
    /// what it expired. `None` when there was nothing to expire, which is
    /// learned with nothing read; a race with another writer is retried
    /// inside. It lists nothing and deletes nothing. A table the format
    /// does not expire, as an Iceberg table with a branch or a tag besides
    /// `main`, is refused with an error before anything is committed.
    fn expire_snapshots(&self, now_ms: u64) -> Result<Option<SnapshotExpiry>>;
    /// Delete, through `storage`, what the snapshots `expired` left unused
    /// in this version, the one their expiry committed, logging
    /// what it could not delete: it fails nothing. It reads manifests and
    /// deletes objects, which no writer should wait for, so a caller runs
    /// it once it has released the table's commit lock.
    fn delete_snapshot_objects(&self, storage: &TableStorage, expired: &ExpiredSnapshots);
    /// Delete, through `storage`, what this version retired longer ago than
    /// the retention window as of `now_ms`, for a format whose retirements
    /// are not snapshots an expiry drops.
    fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()>;
}

/// The live files a commit removes, all of one partition: a rewrite's
/// inputs share one. The partition is named by its key, so a format that
/// holds its files a partition at a time finds them without looking
/// through the others.
#[derive(Clone, Copy)]
pub(crate) struct RemovedFiles<'a> {
    pub partition: &'a [u8],
    pub paths: &'a [ObjectPath],
}

impl RemovedFiles<'static> {
    /// No file: what a commit that only adds files removes.
    pub const NONE: Self = Self {
        partition: &[],
        paths: &[],
    };
}

/// A snapshot expiry a table committed.
pub(crate) struct SnapshotExpiry {
    /// The version the expiry wrote.
    pub table: Arc<dyn Table>,
    /// What the expiry dropped from the table's metadata, whose objects
    /// [`Table::delete_snapshot_objects`] of [`table`](Self::table) deletes.
    pub expired: ExpiredSnapshots,
}
