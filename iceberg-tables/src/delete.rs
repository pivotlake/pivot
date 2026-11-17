//! Deleting rows by taking whole partitions out of a table.
//!
//! A DELETE writes no data file and no deletion vector: the files of every
//! partition its conditions cover whole leave the table in one snapshot,
//! and a DELETE whose conditions cut a partition, true of some of the rows
//! it may hold and not known to be true of them all, is refused before
//! anything is committed. How much of a partition the conditions cover is
//! read off the partition's tuple alone, so no file's bounds make a DELETE
//! succeed that the table's partitioning does not.
//!
//! The conditions are bound once, to the fields of the version the
//! statement read ([`Table::bind_delete_filter`]). Which files they take
//! out is selected from whichever version the removal is committed on
//! ([`Table::select_deleted_files`]), so a file another writer added to a
//! covered partition in between goes with the partition, and a rewrite of
//! its files in between changes nothing of what is deleted.

use iceberg::expr::BoundPredicate;
use iceberg::spec::FormatVersion;
use planner::expression::Expression;

use crate::describe::format_partition;
use crate::files::delete_coverage::{DeleteCoverage, PartitionCoverage};
use crate::files::deletes::count_live_rows;
use crate::files::file_index::{FileSelector, Partition};
use crate::files::pruning::bind_exact_condition;
use crate::table::{DecodedFiles, KeyedPartition, Table, WALKED_PARTITIONS_PER_LOAD};
use crate::types::partitioning::format_partition_fields;
use crate::{Error, Result};

/// The rows a DELETE removes: those every one of its conditions is true of,
/// each condition as a predicate over the table's fields.
#[derive(Debug, Clone)]
pub struct DeleteFilter {
    row_filters: Vec<BoundPredicate>,
}

/// The data files a DELETE takes out of one version of a table: every file
/// of the partitions it covers whole.
#[derive(Debug)]
pub struct DeletedFiles {
    /// Each file's path as the table's manifests record it.
    paths: Vec<String>,
    row_count: u64,
}

impl DeletedFiles {
    /// The rows the files hold, less those their deletion vectors had
    /// already deleted: the rows the DELETE removes.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Whether the DELETE takes no file out.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub(crate) fn paths(&self) -> &[String] {
        &self.paths
    }
}

impl Table {
    /// The rows a DELETE with `conditions` removes, each condition over
    /// this version's columns and a row removed when all are true of it;
    /// with no condition, every row. A condition that is not a comparison
    /// of a column with a constant, a test of a column for NULL, or an AND
    /// or OR of such, is refused: it could not be checked against a
    /// partition's values. So is a table of format version 1, which is not
    /// written.
    pub fn bind_delete_filter(&self, conditions: &[Expression]) -> Result<DeleteFilter> {
        let format_version = self.metadata().format_version();
        if format_version < FormatVersion::V2 {
            return Err(Error::UnsupportedFormatVersion {
                table: self.name().to_string(),
                version: format_version as u8,
            });
        }
        let schema = self.metadata().current_schema();
        let row_filters = conditions
            .iter()
            .map(|condition| {
                bind_exact_condition(schema, condition).ok_or_else(|| {
                    Error::UnsupportedDeleteCondition {
                        table: self.name().to_string(),
                        condition: condition.to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DeleteFilter { row_filters })
    }

    /// The files `filter` takes out of this version: every file of the
    /// partitions it covers whole. A partition it cuts that holds a row
    /// refuses the DELETE.
    ///
    /// Only the partitions the filter cannot rule out are looked at, found
    /// as a scan over the same conditions finds its own, and of those only
    /// the ones it covers or cuts are indexed, a few at a time.
    pub fn select_deleted_files(&self, filter: &DeleteFilter) -> Result<DeletedFiles> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name().to_string(),
            source: Box::new(source),
        };
        let coverage = DeleteCoverage::new(self.metadata(), &filter.row_filters)
            .map_err(invalid_partition_metadata)?;
        let selector = FileSelector::from_row_filters(self.metadata(), filter.row_filters.clone())
            .map_err(invalid_partition_metadata)?;
        let mut ahead = DecodedFiles::default();
        let mut partitions: Vec<KeyedPartition> = Vec::new();
        let mut coverages: Vec<PartitionCoverage> = Vec::new();
        for partition in self.select_partitions(&selector, &mut ahead)? {
            let partition_coverage = coverage
                .classify(&partition.partition)
                .map_err(invalid_partition_metadata)?;
            if partition_coverage != PartitionCoverage::Nothing {
                partitions.push(partition);
                coverages.push(partition_coverage);
            }
        }

        let mut files: Vec<(String, u64)> = Vec::new();
        let chunks = partitions
            .chunks(WALKED_PARTITIONS_PER_LOAD)
            .zip(coverages.chunks(WALKED_PARTITIONS_PER_LOAD));
        for (partitions, coverages) in chunks {
            let indexes = self.load_partitions(partitions, &mut ahead)?;
            for ((partition, partition_coverage), index) in
                partitions.iter().zip(coverages).zip(&indexes)
            {
                if *partition_coverage == PartitionCoverage::Whole {
                    files.extend(
                        index.files().map(|file| {
                            (file.data.file_path().to_string(), file.data.record_count())
                        }),
                    );
                    continue;
                }
                // A partition without a row is cut by nothing.
                if let Some(file) = index.files().find(|file| file.data.record_count() > 0) {
                    return Err(
                        self.build_cut_partition_error(&partition.partition, file.schema_id)
                    );
                }
            }
        }
        let row_count = self.count_live_rows_of(&files)?;
        Ok(DeletedFiles {
            paths: files.into_iter().map(|(path, _)| path).collect(),
            row_count,
        })
    }

    /// How many rows a scan of this version reads of `files`, each a data
    /// file's path and the rows it holds: the rows the files hold, less
    /// those this version's deletion vectors delete in them. Nothing is
    /// read for a table without a delete manifest.
    fn count_live_rows_of(&self, files: &[(String, u64)]) -> Result<u64> {
        let delete_manifests = match self.manifest_list().delete_manifests().next() {
            Some(_) if !files.is_empty() => {
                self.read_manifests(self.manifest_list().delete_manifests())?
            }
            _ => Vec::new(),
        };
        count_live_rows(delete_manifests, files)
    }

    /// Why a DELETE that cuts `partition`, which holds a file whose
    /// manifest was written under the schema `schema_id`, is refused.
    fn build_cut_partition_error(&self, partition: &Partition, schema_id: i32) -> Error {
        let table = self.name().to_string();
        let metadata = self.metadata();
        let Some(partition_spec) = metadata
            .partition_spec_by_id(partition.spec_id())
            .filter(|partition_spec| !partition_spec.fields().is_empty())
        else {
            return Error::DeleteCutsUnpartitionedFiles { table };
        };
        match format_partition(metadata, partition, schema_id) {
            Ok(partition) => Error::DeleteCutsPartition {
                table,
                partition,
                partition_by: format_partition_fields(metadata.current_schema(), partition_spec)
                    .join(", "),
            },
            Err(error) => error,
        }
    }
}
