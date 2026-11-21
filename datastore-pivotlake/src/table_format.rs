//! The table format a database keeps its tables in, and the seam each format
//! implements so the datastore above it is written once.
//!
//! A format owns only what differs between Delta Lake and Iceberg: how a
//! table's current version is read and advanced, how a version's files
//! are found in its metadata and selected for a scan, which manifests its
//! files come in for whoever keeps them from one version to the next, how
//! a set of file additions and removals becomes one committed version, and
//! what its metadata lets maintenance delete. Everything else, the
//! database index, the transaction and DDL, the insert sink, the compaction
//! planner and actor, the vacuum driver and the query binding, lives above
//! this seam and never asks which format it is working on.

use parquet_engine::{BoundSide, SortValue};
use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
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

/// What names one manifest of a table: a list of some of the table's files
/// that never changes once it is written, as a manifest of an Iceberg
/// table is. One whole version of a Delta table is a single manifest.
/// Two versions of a table that have a manifest under one name hold the
/// same files in it, so whoever keeps a table's files from one version to
/// the next catches up with a version by the manifests it does not hold.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ManifestId(Arc<str>);

impl ManifestId {
    /// The manifest a format names `name`.
    pub(crate) fn new(name: Arc<str>) -> Self {
        Self(name)
    }

    /// The name its format gave the manifest.
    pub(crate) fn name(&self) -> &Arc<str> {
        &self.0
    }
}

/// One partition of a table, as its format holds it. A format tells its
/// partitions apart and orders them by what it holds of them; whoever keeps
/// a table's files a partition at a time holds each through a
/// [`PartitionHandle`], and hands it back to the table to name the
/// partition.
pub(crate) trait TablePartition: Send + Sync + 'static {
    /// How this partition orders against `other`, a partition of the same
    /// table: `Equal` for the same partition alone, and the same answer for
    /// as long as the two live.
    fn compare(&self, other: &dyn TablePartition) -> Ordering;

    /// Feed what tells this partition from another of its table into
    /// `hasher`: the same for two partitions that compare `Equal`.
    fn hash_into(&self, hasher: &mut dyn Hasher);

    /// This partition as its format's own type, which the format reads it
    /// back as.
    fn as_any(&self) -> &dyn Any;
}

/// A partition of a table, held by whoever keeps the table's files: equal
/// to, ordered against and hashed like another of the same table as the
/// table's format tells ([`TablePartition`]). A clone shares the partition.
#[derive(Clone)]
pub(crate) struct PartitionHandle(Arc<dyn TablePartition>);

impl PartitionHandle {
    /// The handle on `partition`.
    pub(crate) fn new(partition: Arc<dyn TablePartition>) -> Self {
        Self(partition)
    }

    /// The partition as `Format`, the type its table's format holds its
    /// partitions as. A table is handed the partitions it listed and no
    /// other format's.
    pub(crate) fn read_as<Format: TablePartition>(&self) -> &Format {
        downcast_partition(self.0.as_ref())
    }
}

/// `partition` as `Format`, the type the format asking holds its partitions
/// as: the partitions of one table are all of its format.
pub(crate) fn downcast_partition<Format: TablePartition>(
    partition: &dyn TablePartition,
) -> &Format {
    let downcast = partition.as_any().downcast_ref::<Format>();
    downcast.expect("a table's format is handed a partition of another format")
}

impl PartialEq for PartitionHandle {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PartitionHandle {}

impl PartialOrd for PartitionHandle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PartitionHandle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.compare(other.0.as_ref())
    }
}

impl Hash for PartitionHandle {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.0.hash_into(hasher);
    }
}

/// The live files one manifest lists of one partition.
pub(crate) struct PartitionFiles {
    pub partition: PartitionHandle,
    pub files: Vec<Arc<dyn CompactionFile>>,
}

/// What a commit on top of a version came to.
pub(crate) enum CommitOutcome {
    /// The commit made the next version, this one.
    Committed(Arc<dyn Table>),
    /// Another writer committed a version first: the caller refreshes and
    /// retries.
    OutOfDate,
    /// The version holds no file of the partition at this path, one the
    /// commit was to remove: another writer removed it first.
    MissingFile(ObjectPath),
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
    /// Write the initial version of a new table at `directory` (its own
    /// directory under the database root) holding `adopted`, whose footers
    /// the caller already read, and return the table at that version.
    /// `partition_by` holds the terms `CREATE TABLE` gave, each a column or
    /// a transform of one (`day(ts)`): a term the format cannot partition
    /// the table by is refused before anything is written.
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

    /// Read the table at `directory`, which the database knows by `id`, at
    /// its latest committed version.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        id: Uuid,
    ) -> Result<Arc<dyn Table>>;

    /// Let go of what this process caches of the dropped table the
    /// database knew by `id`, whose storage is deleted whole once its
    /// tombstone ages out: a cached footer holds its file open, and a
    /// deleted local file keeps its blocks for as long as it is open.
    fn forget_cached_table(&self, id: Uuid);
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
    /// The key columns a new file of the table is cut by, one per term the
    /// table is partitioned by, each a column or a transform of one as
    /// `CREATE TABLE` spells it: which of them the rows carry as columns,
    /// and how the others are computed from a row.
    fn plan_partition_keys(&self) -> Result<PartitionKeys>;
    fn sort_by(&self) -> &[String];

    /// What names the manifests the live files of this version come in, in
    /// no order: a live file is in exactly one of them. Nothing is read.
    fn list_manifest_ids(&self) -> Vec<ManifestId>;
    /// The live files each of `manifests` lists, manifests of this version,
    /// in the order given: for each manifest, the files of every partition
    /// it lists one of, a partition's in no order the caller may rely on
    /// but that of the version's own listing. A format reads what it does
    /// not hold of a manifest; one whose version holds every file reads
    /// nothing. A format may leave out the files of a partition it cannot
    /// write the replacement of, which no rewrite then takes.
    fn load_manifest_files(&self, manifests: &[ManifestId]) -> Result<Vec<Vec<PartitionFiles>>>;
    /// The entry of each of `paths`, in the order given: the live file of
    /// `partition` at that path, with its partition tuple and the columns
    /// its rows are sorted by, as the metadata recorded them; `None` for a
    /// path this version has no file of the partition at. A format may
    /// leave an entry's statistics out.
    fn find_file_entries(
        &self,
        partition: &PartitionHandle,
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
    /// The stored rows: those of every live data file at this version,
    /// deleted ones included, counted without reading the files' metadata.
    fn count_stored_rows(&self) -> Result<u64>;
    /// The live rows: those a scan of the whole table returns, deleted ones
    /// excluded, when the metadata this version holds counts them exactly;
    /// `None` when only a scan can.
    fn count_live_rows(&self) -> Option<u64>;
    /// The live data files at this version, counted from the metadata the
    /// version holds, with nothing read; `None` when that metadata does not
    /// count them.
    fn count_files(&self) -> Option<u64>;
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
    /// Commit, on top of this version, the removal of `removed`, each a
    /// file of the partition it names, and the addition of `added`, whose
    /// footers the writer holds, as one new version: the one way a
    /// version's files change, "remove these, add those". An empty
    /// `removed` is a commit that only adds files, an append, and the
    /// files removed need not share a partition. `data_change` is `false`
    /// for a rearrangement (compaction) that leaves the table's rows as
    /// they were. Nothing is committed when this version does not hold one
    /// of `removed` in its partition.
    fn commit(
        &self,
        removed: &[RemovedFile<'_>],
        added: &[TableFile],
        data_change: bool,
    ) -> Result<CommitOutcome>;
    /// How long a file this table no longer references is kept before it may
    /// be deleted.
    fn deleted_file_retention(&self) -> Result<Duration>;
    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the table's commit lock, and hand back the version that wrote with
    /// what it expired. `None` means nothing had expired, which is learned
    /// from the metadata the version holds, with nothing read. A race with
    /// another writer is retried inside. It lists nothing and deletes
    /// nothing. A table the format
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

/// A file a commit removes, and the partition it sits in. The partition is
/// named, so a format that holds its files a partition at a time finds the
/// file without looking through the others.
#[derive(Clone, Copy)]
pub(crate) struct RemovedFile<'a> {
    pub partition: &'a PartitionHandle,
    pub path: &'a ObjectPath,
}

/// A snapshot expiry a table committed.
pub(crate) struct SnapshotExpiry {
    /// The version the expiry wrote.
    pub table: Arc<dyn Table>,
    /// What the expiry dropped from the table's metadata, whose objects
    /// [`Table::delete_snapshot_objects`] of [`table`](Self::table) deletes.
    pub expired: ExpiredSnapshots,
}
