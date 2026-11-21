//! A version's live files by the partition they sit in, read through the
//! parts of the table's data manifests the cache holds
//! ([`manifest_parts`](crate::files::manifest_parts)): the partitions a
//! version has a file in, and the files of one found by their paths.
//!
//! Nothing is kept of a lookup: it reads the files of the partition it
//! asks for, and no column's bounds with them, and a manifest is read and
//! parsed only for a part the cache does not hold.

use std::collections::HashMap;
use std::sync::Arc;

use object_storage::{FileRef, ObjectPath};
use parquet_engine::file_entry::FileEntry;

use super::Table;
use super::manifest_reader::{ManifestReader, PartitionSlice};
use crate::Result;
use crate::files::data_files::read_partition_values;
use crate::files::manifest_parts::{BoundSide, ColumnBounds, ManifestFiles};
use crate::files::manifests::ListedManifest;
use crate::files::partition::Partition;
use crate::files::sort_bounded_file::SortBoundedFile;
use crate::files::sort_bounded_file::SortColumns;

/// The files of one partition in one manifest, as the records maintenance
/// and a lookup take of them: each with its bounds on the columns its table
/// is sorted by.
pub(super) struct SliceRecords<'s> {
    files: &'s ManifestFiles,
    /// What the manifest records of each sort column in the files; `None`
    /// for a column it records nothing of, and for one its schema does not
    /// have as a primitive.
    sort_bounds: Vec<Option<&'s ColumnBounds>>,
    sort_columns: &'s Arc<SortColumns>,
}

impl<'s> SliceRecords<'s> {
    /// The records of the files of `slice`, which was read with the
    /// bounds of the columns of `sort_columns`.
    pub(super) fn new(slice: &'s PartitionSlice, sort_columns: &'s Arc<SortColumns>) -> Self {
        let sort_bounds = sort_columns
            .field_ids()
            .enumerate()
            .map(|(position, field_id)| {
                let bounds = slice.find_bounds(field_id);
                bounds.filter(|_| sort_columns.is_bounded_under(position, slice.files.schema_id()))
            })
            .collect();
        Self {
            files: &slice.files,
            sort_bounds,
            sort_columns,
        }
    }

    /// The record of the file at `position`.
    pub(super) fn build_record(&self, position: usize) -> SortBoundedFile {
        let file = FileRef {
            path: ObjectPath::new(self.files.path(position)),
            size: self.files.size(position),
        };
        let bounds = self.sort_bounds.iter().flat_map(|bounds| {
            [BoundSide::Lower, BoundSide::Upper]
                .map(|side| bounds.and_then(|bounds| bounds.read_literal(side, position)))
        });
        SortBoundedFile::new(
            file,
            self.files.record_count(position),
            self.files.commit_sequence(position),
            bounds.collect(),
            self.sort_columns,
        )
    }
}

/// A live file found by its path.
pub(crate) struct FoundFile {
    pub(crate) file: FileRef,
    /// The rows the file holds, deleted ones included.
    pub(crate) record_count: u64,
    /// The id of the table sort order the file's rows are in, as its
    /// manifest entry records it.
    sort_order_id: Option<i32>,
}

impl Table {
    /// Every partition this version has a file in, in key order. A
    /// manifest whose manifest-list entry names its one partition is not
    /// asked; the partitions any other holds are read off the part of it
    /// that lists them.
    pub(crate) fn list_partitions(&self) -> Result<Vec<Arc<Partition>>> {
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.listed_data_manifests().iter().collect();
        let mut reader = ManifestReader::new(self, Vec::new());
        reader.list_partitions(&manifests, &mut |_| Ok(false))
    }

    /// The live files of `partition` at `paths`, in the order given: `None`
    /// for a path this version has no file of the partition at. The
    /// partition's files are read through the parts of the manifests whose
    /// manifest-list entries do not rule it out, and no column's bounds
    /// with them.
    pub(crate) fn find_files(
        &self,
        partition: &Arc<Partition>,
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FoundFile>>> {
        let mut reader = ManifestReader::new(self, Vec::new());
        let mut loaded = reader.load_slices(std::slice::from_ref(partition))?;
        let slices = loaded
            .pop()
            .expect("the slices of each partition asked for are loaded");
        let asked: HashMap<&str, usize> = paths
            .iter()
            .enumerate()
            .map(|(position, path)| (path.as_str(), position))
            .collect();
        let mut found: Vec<Option<FoundFile>> = paths.iter().map(|_| None).collect();
        for slice in &slices {
            let files = &slice.files;
            for position in 0..files.count_files() {
                if let Some(asked_position) = asked.get(files.path(position)) {
                    found[*asked_position] = Some(FoundFile {
                        file: FileRef {
                            path: ObjectPath::new(files.path(position)),
                            size: files.size(position),
                        },
                        record_count: files.record_count(position),
                        sort_order_id: files.sort_order_id(position),
                    });
                }
            }
        }
        Ok(found)
    }

    /// The live files at `paths` in the partition with the key
    /// `partition`, in the order given, each as a merge of them reads it:
    /// its tuple, and the columns the table is sorted by as what its rows
    /// are sorted by when its manifest entry records the sort order Pivot's
    /// writer produces for them. No statistics are read. `None` for a path
    /// this version has no file of the partition at.
    pub fn find_datastore_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>> {
        // A lookup of no file names no partition.
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let partition = self.find_partition(partition)?;
        // The partition's tuple, the same for every file found, read for
        // the first.
        let mut tuple = None;
        let mut entries = Vec::with_capacity(paths.len());
        for found in self.find_files(&partition, paths)? {
            let Some(found) = found else {
                entries.push(None);
                continue;
            };
            let tuple = match &tuple {
                Some(tuple) => tuple,
                None => tuple.insert(read_partition_values(
                    self.metadata(),
                    self.columns(),
                    &partition,
                    &found.file.path,
                )?),
            };
            let is_in_sort_order =
                found.sort_order_id.is_some() && found.sort_order_id == self.writer_sort_order_id;
            let sorted_by: Arc<[String]> = if is_in_sort_order {
                Arc::from(self.sort_by.as_slice())
            } else {
                Arc::default()
            };
            entries.push(Some(FileEntry {
                file: found.file,
                partition: tuple.clone(),
                stats: None,
                sorted_by,
            }));
        }
        Ok(entries)
    }
}
