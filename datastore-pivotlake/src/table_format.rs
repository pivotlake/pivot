//! The table format a database keeps its tables in, and the seam each format
//! implements so the datastore above it is written once.
//!
//! A format owns only what differs between Delta Lake and Iceberg: how a
//! table's current version is read and advanced, how a set of file additions
//! and removals becomes one committed version, which footers it holds in
//! memory, and what its metadata lets maintenance delete. Everything else, the
//! database index, the transaction and DDL, the insert sink, the compaction
//! planner and actor, the vacuum driver and the query binding, lives above
//! this seam and never asks which format it is working on.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{RowGroupMetadata, ScanPlan};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Result;
use crate::file_entry::FileEntry;
use crate::file_layout::VisitFile;
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
    /// conditional write, readable by any Iceberg reader from the table's path.
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

/// What creates and loads tables in one format on one database's storage.
pub(crate) trait TableFactory: Send + Sync {
    /// Write the initial version of a new table at `directory` (its own
    /// directory under the database root) holding `adopted`, whose footers
    /// the caller already read, and return the table at that version.
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
}

/// One table at one committed version, as its format holds it. Immutable: a
/// refresh, a commit or a local file swap hands back the next version as a
/// new value, so a copy handed to a query never changes under it.
pub(crate) trait Table: Send + Sync {
    /// The committed version this value is at: monotonic per table, so a
    /// published copy is newer than the catalog's exactly when its version is
    /// higher.
    fn version(&self) -> u64;
    fn columns(&self) -> &[Column];
    fn partition_by(&self) -> &[String];
    fn sort_by(&self) -> &[String];

    /// Visit every live data file at this version, partition by partition: a
    /// partition's files come one after another, in an order the version
    /// fixes. Nothing is decoded for a file but what the visit reads. A
    /// format that holds its files a partition at a time has only a few
    /// partitions in hand at once.
    fn walk_files(&self, visit: &mut VisitFile<'_>) -> Result<()>;
    /// Every partition this version has a file in, each as the bytes a walk
    /// names it by.
    fn list_partitions(&self) -> Result<Vec<Vec<u8>>>;
    /// The partitions a routine compaction round looks at, each as the
    /// bytes a walk names it by: those a commit added a file to since a
    /// round last found nothing to do in them. A format that does not keep
    /// track lists every partition.
    fn list_dirty_partitions(&self) -> Result<Vec<Vec<u8>>>;
    /// Record that a compaction round looking at this version found nothing
    /// to do in `partition`, and say whether the partition is then no longer
    /// one a routine round looks at: it stays one when a later version
    /// added a file to it.
    fn settle_partition(&self, partition: &[u8]) -> bool;
    /// Visit the live data files of `partition`, in the order
    /// [`walk_files`](Self::walk_files) visits them.
    fn walk_partition_files(&self, partition: &[u8], visit: &mut VisitFile<'_>) -> Result<()>;
    /// Each of `paths`, in the order given, as the live file of `partition`
    /// at that path, with its partition tuple, its row count and its bounds
    /// on the columns the table is sorted by, as the metadata recorded
    /// them; `None` for a path this version has no file of the partition
    /// at.
    fn find_files(&self, partition: &[u8], paths: &[ObjectPath]) -> Result<Vec<Option<FileEntry>>>;
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
    fn count_rows(&self) -> Result<Option<u64>>;
    /// The live data files at this version, counted from the metadata the
    /// version holds, with nothing read; `None` when that metadata does not
    /// count them.
    fn count_files(&self) -> Option<u64>;
    /// Load what a scan of the whole table reads of the table's metadata
    /// and of its files' footers, so the scan reads none of it. A format
    /// whose versions hold all of it once loaded does nothing.
    fn preload_scan_metadata(&self) -> Result<()>;
    /// The field id each column (in `columns` order) is written under in new
    /// Parquet files, for a format that resolves columns by id; `None` for one
    /// that resolves them by name.
    fn field_ids(&self) -> Option<Vec<i32>>;
    /// The row groups of each of `files` (live files of this very version),
    /// in the order given: what a merge and a layout measurement read. A
    /// format that holds every footer in memory answers without I/O; one
    /// that keeps footers in a cache reads those the cache does not hold,
    /// and the caller keeps what it is handed only for as long as it works
    /// on it.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>>;
    /// Whether each column (in `columns` order) holds a NULL in no file of
    /// this version, as far as the metadata held proves it with nothing
    /// read; a format whose metadata proves it of no column lists none.
    fn list_never_null_columns(&self) -> Vec<bool>;
    /// What a scan over `filters`, the predicates pushed into it, reads:
    /// the row groups of the files this version's metadata cannot rule out,
    /// in an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization. A format may
    /// answer nullability for more files than the scan reads.
    fn plan_scan(&self, filters: &[Expression]) -> Result<ScanPlan>;
    /// The latest committed version, or `None` when this one is still current.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>>;
    /// Commit, on top of this version, the removal of `removed` (files the
    /// caller found live in this version) and the addition of `added`, whose
    /// footers the writer holds, as one new version. `data_change` is
    /// `false` for a rearrangement (compaction) that leaves the table's rows
    /// as they were.
    /// `None` means another writer committed a version first: the caller
    /// refreshes and retries.
    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        data_change: bool,
    ) -> Result<Option<Arc<dyn Table>>>;
    /// This version with `removed` dropped and `added` adopted, as a local view
    /// of a swap this process already committed through another copy; the
    /// next refresh reconciles it to the committed metadata.
    fn with_files_swapped(
        &self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
    ) -> Arc<dyn Table>;
    /// How long a file this table no longer references is kept before it may
    /// be deleted.
    fn deleted_file_retention(&self) -> Result<Duration>;
    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the table's commit lock, and hand back the deletion of what only they
    /// used. `None` when there was nothing to expire; a race with another
    /// writer is retried inside. It lists nothing.
    fn expire_snapshots(
        &self,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<Option<SnapshotExpiry>>;
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

/// The live files a commit removes or a swap drops, all of one partition:
/// a rewrite's inputs share one. The partition is named by the bytes a walk
/// names it by, so a format that holds its files a partition at a time
/// finds them without looking through the others.
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
    /// Deletes the objects the expired snapshots used and no retained one
    /// does, logging what it could not delete. The caller runs it once it
    /// has released the commit lock: it reads manifests and deletes objects,
    /// which no writer should wait for.
    pub delete_expired_objects: ExpiredObjectsDeletion,
}

/// The deferred half of a [`SnapshotExpiry`].
pub(crate) type ExpiredObjectsDeletion = Box<dyn FnOnce() + Send>;
