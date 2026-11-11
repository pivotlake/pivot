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

use object_storage::{FileRef, ObjectPath};
use parquet_engine::{FileStats, ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Result;
use crate::file_entry::FileEntry;
use crate::file_layout::VisitFile;
use crate::table::{TableFile, TableStorage};

/// The format every table of one database is kept in, recorded once in the
/// database's index when the database is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TableFormatKind {
    /// Delta Lake: a `_delta_log` per table, read and written through Delta
    /// Kernel.
    #[serde(alias = "deltalake")]
    Delta,
    /// Iceberg: numbered metadata files per table, each version committed with a
    /// conditional write, readable by any Iceberg reader from the table's path.
    Iceberg,
}

impl TableFormatKind {
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

impl fmt::Display for TableFormatKind {
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

impl FromStr for TableFormatKind {
    type Err = UnknownTableFormat;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        match name {
            "delta" | "deltalake" => Ok(Self::Delta),
            "iceberg" => Ok(Self::Iceberg),
            other => Err(UnknownTableFormat(other.to_string())),
        }
    }
}

/// One database's table format: what creates a table's metadata in storage
/// and what reads it back.
pub(crate) trait TableFormat: Send + Sync {
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
        id: Uuid,
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

    /// How many live data files this version has.
    fn count_files(&self) -> Result<usize>;
    /// Visit every live data file at this version, partition by partition: a
    /// partition's files come one after another, in an order the version
    /// fixes. Nothing is decoded for a file but what the visit reads.
    fn walk_files(&self, visit: &mut VisitFile<'_>) -> Result<()>;
    /// Whether this version holds a live file at `path`.
    fn contains_file(&self, path: &ObjectPath) -> Result<bool>;
    /// The live file at `path` with its partition tuple, its row count and
    /// its bounds on the columns the table is sorted by, as the metadata
    /// recorded them; `None` when this version has no such file.
    fn find_file(&self, path: &ObjectPath) -> Result<Option<FileEntry>>;
    /// Every statistic the metadata recorded for the live file at `path`;
    /// `None` for a file it recorded none for.
    fn file_stats(&self, path: &ObjectPath) -> Result<Option<Arc<FileStats>>>;
    /// Whether each column (in `columns` order) can hold a NULL in some
    /// file of this version, as the data shows it, not as the schema
    /// declares it. An empty table reports every column nullable.
    fn nullability(&self) -> Result<Vec<bool>>;
    /// The rows of every live data file at this version.
    fn total_rows(&self) -> Result<u64>;
    /// The rows a scan of the whole table returns, when the metadata this
    /// version holds counts them exactly; `None` when only a scan can.
    fn count_rows(&self) -> Result<Option<u64>>;
    /// The field id each column (in `columns` order) is written under in new
    /// Parquet files, for a format that resolves columns by id; `None` for one
    /// that resolves them by name.
    fn field_ids(&self) -> Option<Vec<i32>>;
    /// The row groups of each of `files` (live files of this very version),
    /// in the order given: what a merge and a layout measurement read. A
    /// format that holds every footer in memory answers without I/O; one
    /// that holds none reads them, and the caller keeps what it is handed
    /// only for as long as it works on it.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>>;
    /// The row groups a scan over `filters`, the predicates pushed into it,
    /// reads: those of the files this version's metadata cannot rule out, in
    /// an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization.
    fn plan_scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>>;
    /// The latest committed version, or `None` when this one is still current.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>>;
    /// Commit, on top of this version, the removal of `removed` (live files
    /// of this version) and the addition of `added`, whose footers the writer
    /// holds, as one new version dated `now_ms`. `data_change` is `false` for a
    /// rearrangement (compaction) that leaves the table's rows as they were.
    /// `None` means another writer committed a version first: the caller
    /// refreshes and retries.
    fn commit(
        &self,
        removed: &[ObjectPath],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> Result<Option<Arc<dyn Table>>>;
    /// This version with `removed` dropped and `added` adopted, as a local view
    /// of a swap this process already committed through another copy; the
    /// next refresh reconciles it to the committed metadata.
    fn with_files_swapped(&self, removed: &[ObjectPath], added: Vec<TableFile>) -> Arc<dyn Table>;
    /// How long a file this table no longer references is kept before it may
    /// be deleted.
    fn deleted_file_retention(&self) -> Result<Duration>;
    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the table's commit lock, and hand back the deletion of what only they
    /// used. `None` when there was nothing to expire; a race with another
    /// writer is retried inside. Routine maintenance: it lists nothing.
    fn expire_snapshots(
        &self,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<Option<SnapshotExpiry>>;
    /// Delete, through `storage`, what this version retired longer ago than
    /// the retention window as of `now_ms`, for a format whose retirements
    /// are not snapshots an expiry drops. Routine maintenance.
    fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()>;
    /// List the table's directory through `storage` and delete every object
    /// in it that this version does not use, as of `now_ms`. Never routine:
    /// only an explicit `VACUUM` asks for it.
    fn sweep_unused_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()>;
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
