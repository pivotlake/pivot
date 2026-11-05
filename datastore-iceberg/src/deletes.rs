//! The snapshot's deletion vectors: reading the bitmaps of the data files a
//! scan reads, and counting the rows they delete from the delete manifests.
//! Only deletion vectors are applied: any other delete file fails the table
//! rather than being skipped.

use std::collections::HashMap;

use dispatch::io::FileRange;
use iceberg::delete_vector::DeleteVector;
use iceberg::scan::FileScanTaskDeleteFile;
use iceberg::spec::{DataContentType, DataFileFormat, Manifest};
use object_storage::{FileSlice, load_file_slices};
use roaring::RoaringTreemap;

use crate::planning::PlannedFile;
use crate::table::LoadedTable;
use crate::{Error, Result};

/// What a delete manifest entry records of a deletion vector: where its blob
/// lies in its Puffin file, and how many rows it deletes.
#[derive(Debug)]
pub(crate) struct DeletionVectorMetadata {
    puffin_path: String,
    puffin_size: u64,
    offset: u64,
    length: u64,
    /// The spec's "cardinality", which the entry records as `record_count`.
    deleted_row_count: u64,
}

/// What a delete file's manifest entry records, as the two sources of one
/// carry it: the entry itself, or the delete file of a planned scan task.
struct DeleteFileEntry<'a> {
    file_path: &'a str,
    content: DataContentType,
    format: DataFileFormat,
    file_size: u64,
    record_count: Option<u64>,
    referenced_data_file: Option<String>,
    content_offset: Option<i64>,
    content_size: Option<i64>,
}

impl DeletionVectorMetadata {
    /// The deletion vector `delete`, a delete file the crate paired with the
    /// data file `data_file`, must be. A delete file of any other kind is
    /// refused.
    pub(crate) fn from_task_delete(
        delete: &FileScanTaskDeleteFile,
        data_file: &str,
    ) -> Result<Self> {
        let (referenced, metadata) = Self::from_entry(DeleteFileEntry {
            file_path: &delete.file_path,
            content: delete.file_type,
            format: delete.file_format,
            file_size: delete.file_size_in_bytes,
            record_count: delete.record_count,
            referenced_data_file: delete.referenced_data_file.clone(),
            content_offset: delete.content_offset,
            content_size: delete.content_size_in_bytes,
        })?;
        if referenced != data_file {
            return Err(Error::DeletionVector {
                file: data_file.to_string(),
                message: format!("the deletion vector paired with it references `{referenced}`"),
            });
        }
        Ok(metadata)
    }

    /// The deletion vector `entry` must be, with the data file it applies to.
    fn from_entry(entry: DeleteFileEntry<'_>) -> Result<(String, Self)> {
        let is_deletion_vector = entry.content == DataContentType::PositionDeletes
            && entry.format == DataFileFormat::Puffin;
        if !is_deletion_vector {
            return Err(Error::DeleteFile {
                file: entry.file_path.to_string(),
            });
        }
        let invalid = |message: &str| Error::DeletionVector {
            file: entry.file_path.to_string(),
            message: message.to_string(),
        };
        let data_file = entry
            .referenced_data_file
            .ok_or_else(|| invalid("it names no referenced data file"))?;
        let offset = entry
            .content_offset
            .ok_or_else(|| invalid("it has no content offset"))?;
        let length = entry
            .content_size
            .ok_or_else(|| invalid("it has no content size"))?;
        let (Ok(offset), Ok(length)) = (u64::try_from(offset), u64::try_from(length)) else {
            return Err(invalid("its content offset or size is negative"));
        };
        let deleted_row_count = entry
            .record_count
            .ok_or_else(|| invalid("it has no record count"))?;
        Ok((
            data_file,
            Self {
                puffin_path: entry.file_path.to_string(),
                puffin_size: entry.file_size,
                offset,
                length,
                deleted_row_count,
            },
        ))
    }
}

impl LoadedTable {
    /// The deletion vector of each of `planned_files`, decoded to the positions it
    /// deletes: one per file, in order, `None` for a file without one. The
    /// blobs of the files a commit touched share one Puffin file; only the
    /// blobs of `planned_files` are read from it.
    pub(crate) fn read_deletion_vectors(
        &self,
        planned_files: &[PlannedFile],
    ) -> Result<Vec<Option<RoaringTreemap>>> {
        let mut blobs = Vec::new();
        for planned in planned_files {
            let Some(metadata) = &planned.deletion_vector else {
                continue;
            };
            let end = metadata.offset.checked_add(metadata.length);
            if end.is_none_or(|end| end > metadata.puffin_size) {
                return Err(Error::DeletionVector {
                    file: planned.data_file.path.clone(),
                    message: format!(
                        "its blob at offset {} of {} bytes is past the end of `{}`",
                        metadata.offset, metadata.length, metadata.puffin_path
                    ),
                });
            }
            blobs.push(FileSlice {
                file: self
                    .store
                    .data_file(&metadata.puffin_path, metadata.puffin_size)?,
                range: FileRange::new(metadata.offset as usize, metadata.length as usize),
            });
        }
        // One read per blob, in the order of `planned_files`.
        let mut loaded = load_file_slices(&self.dispatcher, &blobs)?.into_iter();
        planned_files
            .iter()
            .map(|planned| {
                if planned.deletion_vector.is_none() {
                    return Ok(None);
                }
                let blob = loaded.next().expect("one read per deletion vector").bytes;
                let deleted =
                    DeleteVector::deserialize(&blob).map_err(|source| Error::DeletionVector {
                        file: planned.data_file.path.clone(),
                        message: source.to_string(),
                    })?;
                Ok(Some(deleted.into_treemap()))
            })
            .collect()
    }
}

/// How many rows the deletion vectors the delete manifests `manifests`
/// declare delete in all, each one's count as its entry records it, with no
/// bitmap read.
pub(crate) fn count_deleted_rows<'a>(
    manifests: impl IntoIterator<Item = &'a Manifest>,
) -> Result<u64> {
    sum_deleted_rows(&locate_deletion_vectors(manifests)?)
}

fn sum_deleted_rows(by_data_file: &HashMap<String, DeletionVectorMetadata>) -> Result<u64> {
    by_data_file
        .iter()
        .try_fold(0u64, |total, (data_file, entry)| {
            total
                .checked_add(entry.deleted_row_count)
                .ok_or_else(|| Error::DeletionVector {
                    file: data_file.clone(),
                    message: "its deleted row count overflows the total".to_string(),
                })
        })
}

/// Where the deletion vector each live entry of `manifests` declares lies,
/// by the data file it applies to. A snapshot keeps one per data file: a
/// second one is corrupt, not a second set of deletes.
fn locate_deletion_vectors<'a>(
    manifests: impl IntoIterator<Item = &'a Manifest>,
) -> Result<HashMap<String, DeletionVectorMetadata>> {
    let mut by_data_file = HashMap::new();
    for manifest in manifests {
        for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
            let file = entry.data_file();
            let (data_file, metadata) = DeletionVectorMetadata::from_entry(DeleteFileEntry {
                file_path: file.file_path(),
                content: file.content_type(),
                format: file.file_format(),
                file_size: file.file_size_in_bytes(),
                record_count: Some(file.record_count()),
                referenced_data_file: file.referenced_data_file(),
                content_offset: file.content_offset(),
                content_size: file.content_size_in_bytes(),
            })?;
            if by_data_file.insert(data_file.clone(), metadata).is_some() {
                return Err(Error::DeletionVector {
                    file: data_file,
                    message: "two live deletion vectors reference it".to_string(),
                });
            }
        }
    }
    Ok(by_data_file)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use iceberg::spec::{
        DataFile, DataFileBuilder, FormatVersion, ManifestContentType, ManifestEntry,
        ManifestMetadata, ManifestStatus, NestedField, PartitionSpec, PrimitiveType, Schema, Type,
    };

    fn deletes_manifest(files: Vec<DataFile>) -> Manifest {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::optional(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .unwrap(),
        );
        Manifest::new(
            ManifestMetadata {
                schema: schema.clone(),
                schema_id: schema.schema_id(),
                partition_spec: PartitionSpec::builder(schema).build().unwrap(),
                format_version: FormatVersion::V3,
                content: ManifestContentType::Deletes,
            },
            files
                .into_iter()
                .map(|file| {
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(file)
                        .build()
                })
                .collect(),
        )
    }

    /// A deletion vector of `deleted_rows` rows of `data_file`, in `puffin`.
    fn deletion_vector(puffin: &str, data_file: &str, deleted_rows: u64) -> DataFileBuilder {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_path(puffin.into())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(deleted_rows)
            .referenced_data_file(Some(data_file.into()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40));
        builder
    }

    /// The delete file of a planned scan task, as the crate pairs one with
    /// a data file.
    fn task_delete(puffin: &str, data_file: &str) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: puffin.to_string(),
            file_size_in_bytes: 100,
            file_type: DataContentType::PositionDeletes,
            file_format: DataFileFormat::Puffin,
            partition_spec_id: 0,
            equality_ids: None,
            referenced_data_file: Some(data_file.to_string()),
            content_offset: Some(4),
            content_size_in_bytes: Some(40),
            record_count: Some(3),
            key_metadata: None,
        }
    }

    #[test]
    fn each_data_file_has_its_deletion_vector_and_the_counts_add_up() {
        let manifest = deletes_manifest(vec![
            deletion_vector("dv-1.puffin", "a.parquet", 3)
                .build()
                .unwrap(),
            deletion_vector("dv-1.puffin", "b.parquet", 4)
                .build()
                .unwrap(),
        ]);

        let metadata = locate_deletion_vectors([&manifest]).unwrap();

        assert_eq!(sum_deleted_rows(&metadata).unwrap(), 7);
        assert_eq!(metadata["b.parquet"].deleted_row_count, 4);
        assert!(!metadata.contains_key("c.parquet"));
    }

    #[test]
    fn a_second_deletion_vector_for_a_data_file_is_refused() {
        let manifest = deletes_manifest(vec![
            deletion_vector("dv-1.puffin", "a.parquet", 3)
                .build()
                .unwrap(),
            deletion_vector("dv-2.puffin", "a.parquet", 5)
                .build()
                .unwrap(),
        ]);

        let error = locate_deletion_vectors([&manifest]).unwrap_err();

        assert!(
            error.to_string().contains("two live deletion vectors"),
            "{error}"
        );
    }

    #[test]
    fn a_delete_file_that_is_not_a_deletion_vector_is_refused() {
        let position_deletes = deletion_vector("deletes.parquet", "a.parquet", 3)
            .file_format(DataFileFormat::Parquet)
            .build()
            .unwrap();
        let equality_deletes = deletion_vector("deletes.parquet", "a.parquet", 3)
            .content(DataContentType::EqualityDeletes)
            .file_format(DataFileFormat::Parquet)
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap();

        for file in [position_deletes, equality_deletes] {
            let manifest = deletes_manifest(vec![file]);
            let error = locate_deletion_vectors([&manifest]).unwrap_err();
            assert!(matches!(error, Error::DeleteFile { .. }), "{error}");
        }
    }

    #[test]
    fn a_deletion_vector_without_its_blob_location_is_refused() {
        let manifest = deletes_manifest(vec![
            deletion_vector("dv.puffin", "a.parquet", 3)
                .content_offset(None)
                .build()
                .unwrap(),
        ]);

        let error = locate_deletion_vectors([&manifest]).unwrap_err();

        assert!(error.to_string().contains("no content offset"), "{error}");
    }

    #[test]
    fn a_planned_tasks_deletion_vector_is_located_in_its_puffin_file() {
        let delete = task_delete("dv.puffin", "a.parquet");

        let metadata = DeletionVectorMetadata::from_task_delete(&delete, "a.parquet").unwrap();

        assert_eq!(metadata.puffin_path, "dv.puffin");
        assert_eq!((metadata.offset, metadata.length), (4, 40));
        assert_eq!(metadata.deleted_row_count, 3);
    }

    #[test]
    fn a_deletion_vector_paired_with_another_files_task_is_refused() {
        let delete = task_delete("dv.puffin", "a.parquet");

        let error = DeletionVectorMetadata::from_task_delete(&delete, "b.parquet").unwrap_err();

        assert!(
            error.to_string().contains("references `a.parquet`"),
            "{error}"
        );
    }
}
