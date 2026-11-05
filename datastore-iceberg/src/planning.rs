//! What a scan reads of a snapshot: the data files the iceberg crate's own
//! planning keeps for the scan's filters, with the deletion vector of each,
//! and the manifests a count or a listing reads for itself.
//!
//! The crate plans the scan: it reads the manifest list, keeps the manifests
//! whose partition summaries admit the filters, reads those, and keeps the
//! files whose partition values and column bounds do, pairing each with the
//! delete files that apply to it. Every manifest list and manifest it reads
//! comes from the cache the datastore shares between its tables, through the
//! table's file IO on a miss. What is read here directly, for a row count or
//! a listing, comes from the same cache.

use std::sync::Arc;

use futures::TryStreamExt;
use futures::future::try_join_all;
use iceberg::expr::Predicate;
use iceberg::scan::FileScanTask;
use iceberg::spec::{
    DataContentType, DataFileFormat, Manifest, ManifestContentType, ManifestEntryRef, ManifestFile,
    ManifestList,
};
use planner::expression::Expression;

use crate::deletes::DeletionVectorMetadata;
use crate::predicate::filters_to_predicate;
use crate::rest_catalog::block_on;
use crate::table::LoadedTable;
use crate::{Error, Result};

/// Where a data file is and how long, which is all a footer read needs.
pub(crate) struct FileDescriptor {
    pub path: String,
    pub length: u64,
}

impl FileDescriptor {
    pub fn new(path: &str, length: u64) -> Self {
        Self {
            path: path.to_string(),
            length,
        }
    }
}

/// A data file a scan reads, with its deletion vector when it has one.
pub(crate) struct PlannedFile {
    pub data_file: FileDescriptor,
    pub deletion_vector: Option<DeletionVectorMetadata>,
}

/// A data file the snapshot lists, with the manifest listing it.
pub(crate) struct ListedFile {
    pub manifest: Arc<Manifest>,
    pub entry: ManifestEntryRef,
}

impl LoadedTable {
    /// Plan the scan under `filters`: the non-empty data files that may hold
    /// a row satisfying every filter, in path order so that the same filters
    /// yield the same files in the same order. A file that is not Parquet,
    /// or a delete file that is not a deletion vector, is refused.
    pub(crate) fn plan_files(&self, filters: &[Expression]) -> Result<Vec<PlannedFile>> {
        let predicate = filters_to_predicate(self.table.metadata().current_schema(), filters);
        let mut scan = self.table.scan();
        if predicate != Predicate::AlwaysTrue {
            scan = scan.with_filter(predicate);
        }
        let scan = scan.build().map_err(|source| self.planning_error(source))?;
        let mut tasks: Vec<FileScanTask> =
            block_on(async { scan.plan_files().await?.try_collect().await })
                .map_err(|source| self.planning_error(source))?;
        tasks.sort_by(|left, right| left.data_file_path().cmp(right.data_file_path()));
        tasks
            .iter()
            .filter(|task| task.record_count() != Some(0))
            .map(|task| self.planned_file(task))
            .collect()
    }

    /// The data file of `task` with its deletion vector: the delete files
    /// the crate paired with it must be at most one, a deletion vector.
    fn planned_file(&self, task: &FileScanTask) -> Result<PlannedFile> {
        let path = task.data_file_path();
        if task.data_file_format() != DataFileFormat::Parquet {
            return Err(Error::NonParquetFile {
                table: self.name.to_string(),
                file: path.to_string(),
                format: task.data_file_format().to_string(),
            });
        }
        let mut deletion_vector = None;
        for delete in task.deletes() {
            let metadata = DeletionVectorMetadata::from_task_delete(delete, path)?;
            if deletion_vector.replace(metadata).is_some() {
                return Err(Error::DeletionVector {
                    file: path.to_string(),
                    message: "two live deletion vectors reference it".to_string(),
                });
            }
        }
        Ok(PlannedFile {
            data_file: FileDescriptor::new(path, task.file_size_in_bytes()),
            deletion_vector,
        })
    }

    /// Every live data file of the current snapshot, in manifest order. A
    /// delete file among the data manifests, or a file that is not Parquet,
    /// is refused.
    pub(crate) fn list_data_files(&self) -> Result<Vec<ListedFile>> {
        let Some(manifest_list) = self.load_manifest_list()? else {
            return Ok(Vec::new());
        };
        let data_manifests = manifest_list
            .entries()
            .iter()
            .filter(|manifest| manifest.content == ManifestContentType::Data);
        let manifests = self.load_manifests(data_manifests)?;
        let mut files = Vec::new();
        for manifest in manifests {
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                let file = entry.data_file();
                if file.content_type() != DataContentType::Data {
                    return Err(Error::DeleteFile {
                        file: file.file_path().to_string(),
                    });
                }
                if file.file_format() != DataFileFormat::Parquet {
                    return Err(Error::NonParquetFile {
                        table: self.name.to_string(),
                        file: file.file_path().to_string(),
                        format: file.file_format().to_string(),
                    });
                }
                files.push(ListedFile {
                    manifest: manifest.clone(),
                    entry: entry.clone(),
                });
            }
        }
        Ok(files)
    }

    /// The manifest list of the current snapshot, or `None` for a table
    /// without one.
    pub(crate) fn load_manifest_list(&self) -> Result<Option<Arc<ManifestList>>> {
        let Some(snapshot) = self.table.metadata().current_snapshot() else {
            return Ok(None);
        };
        let manifest_list = block_on(self.table.object_cache().get_manifest_list(
            snapshot,
            &self.table.metadata_ref(),
            self.table.file_io(),
            self.table.encryption_manager().cloned(),
        ))
        .map_err(|source| self.planning_error(source))?;
        Ok(Some(manifest_list))
    }

    /// `manifests`, read together, in the order given.
    pub(crate) fn load_manifests<'a>(
        &self,
        manifests: impl IntoIterator<Item = &'a ManifestFile>,
    ) -> Result<Vec<Arc<Manifest>>> {
        let cache = self.table.object_cache();
        let file_io = self.table.file_io();
        block_on(try_join_all(
            manifests
                .into_iter()
                .map(|manifest| cache.get_manifest(manifest, file_io)),
        ))
        .map_err(|source| self.planning_error(source))
    }

    fn planning_error(&self, source: iceberg::Error) -> Error {
        Error::Planning {
            table: self.name.to_string(),
            source: Box::new(source),
        }
    }
}

/// The delete manifests among `manifests`, a manifest list's entries, with a
/// file still live: one whose delete files were all removed carries nothing
/// to apply.
pub(crate) fn live_delete_manifests(
    manifests: &[ManifestFile],
) -> impl Iterator<Item = &ManifestFile> {
    manifests.iter().filter(|manifest| {
        manifest.content == ManifestContentType::Deletes
            && (manifest.has_added_files() || manifest.has_existing_files())
    })
}

/// The rows of the data files `manifests`, a manifest list's entries, name,
/// deleted ones included. Unknown or overflowing counts make planning fall
/// back to an ordinary scan.
pub(crate) fn count_rows(manifests: &[ManifestFile]) -> Option<i64> {
    sum_rows(
        manifests
            .iter()
            .filter(|manifest| manifest.content == ManifestContentType::Data),
    )
}

/// The rows of the data files less the rows the delete manifests' entries
/// count, from the manifest list alone: exact when every delete file is a
/// deletion vector, an estimate otherwise, never below zero.
pub(crate) fn estimate_rows(manifests: &[ManifestFile]) -> Option<i64> {
    let deleted = sum_rows(live_delete_manifests(manifests))?;
    Some(count_rows(manifests)?.saturating_sub(deleted).max(0))
}

fn sum_rows<'a>(manifests: impl Iterator<Item = &'a ManifestFile>) -> Option<i64> {
    manifests.into_iter().try_fold(0i64, |total, manifest| {
        let rows = manifest
            .added_rows_count?
            .checked_add(manifest.existing_rows_count?)?;
        total.checked_add(i64::try_from(rows).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use iceberg::spec::FieldSummary;

    use super::*;

    fn manifest(content: ManifestContentType, rows: Option<u64>, live_files: u32) -> ManifestFile {
        ManifestFile {
            manifest_path: format!("{content:?}.avro"),
            manifest_length: 100,
            partition_spec_id: 0,
            content,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 1,
            added_files_count: Some(live_files),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: rows,
            existing_rows_count: Some(2),
            deleted_rows_count: Some(0),
            partitions: Some(Vec::<FieldSummary>::new()),
            key_metadata: None,
            first_row_id: None,
        }
    }

    #[test]
    fn the_row_count_is_known_only_when_every_data_manifest_counts_its_rows() {
        let counted = [
            manifest(ManifestContentType::Data, Some(10), 1),
            manifest(ManifestContentType::Data, Some(20), 1),
            manifest(ManifestContentType::Deletes, Some(5), 1),
        ];
        let uncounted = [
            manifest(ManifestContentType::Data, Some(10), 1),
            manifest(ManifestContentType::Data, None, 1),
        ];

        assert_eq!(count_rows(&counted), Some(34));
        assert_eq!(count_rows(&uncounted), None);
    }

    #[test]
    fn the_estimate_discounts_the_live_delete_manifests_rows() {
        let manifests = [
            manifest(ManifestContentType::Data, Some(10), 1),
            manifest(ManifestContentType::Deletes, Some(5), 1),
            manifest(ManifestContentType::Deletes, Some(100), 0),
        ];

        assert_eq!(estimate_rows(&manifests), Some(5));
        assert_eq!(live_delete_manifests(&manifests).count(), 1);
    }
}
