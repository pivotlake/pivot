//! The table format a database keeps its tables in, and the seam each format
//! implements so the datastore above it is written once.
//!
//! A format owns only what differs between Delta Lake and Iceberg: how a
//! table's current version is read and advanced, how a set of file additions
//! and removals becomes one committed version, which footers it holds in
//! memory, and what its metadata lets vacuum delete. Everything else, the
//! database index, the transaction and DDL, the insert sink, the compaction
//! planner and actor, the vacuum driver and the query binding, lives above
//! this seam and never asks which format it is working on.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use object_storage::ObjectPath;
use parquet_engine::{ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Result;
use crate::file_entry::FileEntry;
use crate::table::{TableFile, TableStorage};

/// The format every table of one database is kept in, recorded once in the
/// database's index when the database is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TableFormat {
    /// Delta Lake: a `_delta_log` per table, read and written through Delta
    /// Kernel.
    #[serde(alias = "deltalake")]
    Delta,
    /// Iceberg: numbered metadata files per table, each version landed with a
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

/// One database's format backend: what creates a table's metadata in storage
/// and what reads it back.
pub(crate) trait Format: Send + Sync {
    /// Which table format this is, as the database index records it.
    fn name(&self) -> TableFormat;

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
    ) -> Result<Arc<dyn FormatTable>>;

    /// Read the table at `directory` at its latest committed version.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        id: Uuid,
    ) -> Result<Arc<dyn FormatTable>>;
}

/// One table at one committed version, as its format holds it. Immutable: a
/// refresh, a commit or a local file swap hands back the next version as a
/// new value, so a copy handed to a query never changes under it.
pub(crate) trait FormatTable: Send + Sync {
    /// The committed version this value is at: monotonic per table, so a
    /// published copy is newer than the catalog's exactly when its version is
    /// higher.
    fn version(&self) -> u64;
    fn columns(&self) -> &[Column];
    fn partition_by(&self) -> &[String];
    fn sort_by(&self) -> &[String];
    /// Every live data file at this version, with its partition tuple and
    /// stats as the metadata recorded them.
    fn entries(&self) -> &[FileEntry];
    /// The field id each column (in `columns` order) is written under in new
    /// Parquet files, for a format that resolves columns by id; `None` for one
    /// that resolves them by name.
    fn field_ids(&self) -> Option<Vec<i32>>;
    /// The row groups of each of `files` (entries of this very version), in
    /// the order given: what a scan, a merge and a layout measurement read.
    /// A format that holds every footer in memory answers without I/O; one
    /// that reads footers on demand reads the ones it has not yet.
    fn row_groups(&self, files: &[&FileEntry]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>>;
    /// The row groups a scan over `filters`, the predicates pushed into it,
    /// reads: those of the files this version's metadata cannot rule out, in
    /// an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization.
    fn scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>>;
    /// The latest committed version, or `None` when this one is still current.
    fn refresh(&self) -> Result<Option<Arc<dyn FormatTable>>>;
    /// Commit, on top of this version, the removal of `removed` (entries of
    /// this version) and the addition of `added`, whose footers the writer
    /// holds, as one new version dated `now_ms`. `data_change` is `false` for a
    /// rearrangement (compaction) that leaves the table's rows as they were.
    /// `None` means another writer landed a version first: the caller
    /// refreshes and retries.
    fn commit(
        &self,
        removed: &[FileEntry],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> Result<Option<Arc<dyn FormatTable>>>;
    /// This version with `removed` dropped and `added` adopted, as a local view
    /// of a swap this process already committed through another copy; the
    /// next refresh reconciles it to the committed metadata.
    fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Arc<dyn FormatTable>;
    /// How long a file this table no longer references is kept before vacuum
    /// may delete it.
    fn deleted_file_retention(&self) -> Result<Duration>;
    /// Advance the table's own metadata housekeeping (expiring what its
    /// retention allows) as of `now_ms`, under the table's commit lock. Returns
    /// the version it ended at when that wrote one, `None` when there was
    /// nothing to do; a race with another writer is retried inside.
    fn maintain(&self, now_ms: u64) -> Result<Option<Arc<dyn FormatTable>>>;
    /// Delete, through `storage`, every object under the table's directory that
    /// this version no longer needs and that has been unneeded for longer than
    /// the retention window as of `now_ms`: data files first, then whatever
    /// metadata objects the format keeps per version.
    fn sweep(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()>;
}
