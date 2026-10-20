//! The snapshot's delete files: reading the positions they delete from the
//! data files a scan reads, and counting the rows they delete, from the
//! delete manifests the pruning kept. Deletion vectors and Parquet
//! position-delete files are applied; any other delete file fails the table
//! rather than being skipped.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use dispatch::io::FileRange;
use dispatch::{DataFlowDispatcher, Projection};
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{DataContentType, DataFile, DataFileFormat, PrimitiveLiteral};
use object_storage::{FileSlice, load_file_slices};
use parquet_engine::{ParquetTable, TableColumns, load_file_row_groups, table_input};
use planner::catalog::Column;
use planner::types::Type;
use roaring::RoaringTreemap;

use crate::manifests::{FileDescriptor, LoadedManifest, describe_content};
use crate::store::TableStore;
use crate::{Error, Result};

/// The field ids the spec reserves for a position-delete file's columns: the
/// path of the data file a row deletes from, and the row's position in it.
const FILE_PATH_FIELD_ID: i32 = 2147483546;
const POSITION_FIELD_ID: i32 = 2147483545;

/// What a delete manifest entry records of a deletion vector: where its blob
/// lies in its Puffin file, and how many rows it deletes.
#[derive(Debug)]
struct DeletionVectorMetadata {
    puffin_path: String,
    puffin_size: u64,
    offset: u64,
    length: u64,
    /// The spec's "cardinality", which the entry records as `record_count`.
    deleted_row_count: u64,
}

/// What a delete manifest entry records of a Parquet position-delete file,
/// which lists (data file path, position) rows for any number of data files.
#[derive(Debug)]
struct PositionDeleteFile {
    path: String,
    size: u64,
    /// The file applies to data files of this sequence number or older.
    sequence_number: i64,
    /// The one data file every row names, when the entry records it.
    referenced_data_file: Option<String>,
    /// The bounds of the file-path column, when the entry records them.
    path_bounds: Option<(String, String)>,
}

impl PositionDeleteFile {
    /// Whether a row of this file may delete from `file`: the file is not
    /// newer than the deletes, and its path is one the file may name.
    fn may_delete_from(&self, file: &FileDescriptor) -> bool {
        self.sequence_number >= file.sequence_number
            && self
                .referenced_data_file
                .as_ref()
                .is_none_or(|referenced| *referenced == file.path)
            && self
                .path_bounds
                .as_ref()
                .is_none_or(|(lower, upper)| *lower <= file.path && file.path <= *upper)
    }
}

/// The delete files the delete manifests declare, by kind.
#[derive(Debug)]
struct DeleteFiles {
    /// A snapshot keeps one deletion vector per data file.
    deletion_vectors: HashMap<String, DeletionVectorMetadata>,
    position_deletes: Vec<PositionDeleteFile>,
}

/// The positions the delete manifests `manifests` delete from each of
/// `files`: one bitmap per file, in order, `None` for a file nothing deletes
/// from. The blobs of the files a commit touched share one Puffin file; only
/// the blobs of `files` are read from it. A position-delete file is read when
/// its sequence number and the paths it may name admit one of `files`, and
/// only its rows naming such a file count.
pub(crate) fn read_deleted_positions(
    manifests: Vec<LoadedManifest>,
    files: &[FileDescriptor],
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<Vec<Option<RoaringTreemap>>> {
    let delete_files = locate_delete_files(manifests)?;
    let mut deleted =
        read_deletion_vectors(&delete_files.deletion_vectors, files, store, dispatcher)?;
    read_position_deletes(
        &delete_files.position_deletes,
        files,
        store,
        dispatcher,
        &mut deleted,
    )?;
    Ok(deleted)
}

/// The deletion vector of each of `files`, decoded to the positions it
/// deletes, among `deletion_vectors`: one per file, in order, `None` for a
/// file without one.
fn read_deletion_vectors(
    deletion_vectors: &HashMap<String, DeletionVectorMetadata>,
    files: &[FileDescriptor],
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<Vec<Option<RoaringTreemap>>> {
    let metadata: Vec<Option<&DeletionVectorMetadata>> = files
        .iter()
        .map(|file| deletion_vectors.get(&file.path))
        .collect();
    let mut blobs = Vec::new();
    for (metadata, file) in metadata.iter().zip(files) {
        let Some(metadata) = metadata else {
            continue;
        };
        let end = metadata.offset.checked_add(metadata.length);
        if end.is_none_or(|end| end > metadata.puffin_size) {
            return Err(Error::DeletionVector {
                file: file.path.clone(),
                message: format!(
                    "its blob at offset {} of {} bytes is past the end of `{}`",
                    metadata.offset, metadata.length, metadata.puffin_path
                ),
            });
        }
        blobs.push(FileSlice {
            file: store.data_file(&metadata.puffin_path, metadata.puffin_size)?,
            range: FileRange::new(metadata.offset as usize, metadata.length as usize),
        });
    }
    // One read per blob, in the order of `files`.
    let mut loaded = load_file_slices(dispatcher, &blobs)?.into_iter();
    metadata
        .iter()
        .zip(files)
        .map(|(metadata, file)| {
            if metadata.is_none() {
                return Ok(None);
            }
            let blob = loaded.next().expect("one read per deletion vector").bytes;
            let deleted =
                DeleteVector::deserialize(&blob).map_err(|source| Error::DeletionVector {
                    file: file.path.clone(),
                    message: source.to_string(),
                })?;
            Ok(Some(deleted.into_treemap()))
        })
        .collect()
}

/// Add to `deleted`, one bitmap per file of `files`, the positions
/// `position_deletes` delete from the files. Each delete file that may name
/// one of `files` is decoded whole; a row naming a file it may not delete
/// from is passed over.
fn read_position_deletes(
    position_deletes: &[PositionDeleteFile],
    files: &[FileDescriptor],
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
    deleted: &mut [Option<RoaringTreemap>],
) -> Result<()> {
    let reads: Vec<(&PositionDeleteFile, HashMap<&str, usize>)> = position_deletes
        .iter()
        .map(|delete_file| {
            let targets: HashMap<&str, usize> = files
                .iter()
                .enumerate()
                .filter(|(_, file)| delete_file.may_delete_from(file))
                .map(|(index, file)| (file.path.as_str(), index))
                .collect();
            (delete_file, targets)
        })
        .filter(|(_, targets)| !targets.is_empty())
        .collect();
    let data_files = reads
        .iter()
        .map(|(delete_file, _)| store.data_file(&delete_file.path, delete_file.size))
        .collect::<Result<Vec<_>>>()?;
    let footers = load_file_row_groups(dispatcher, &data_files, position_delete_columns())?;
    for ((delete_file, targets), footer) in reads.into_iter().zip(footers) {
        let invalid = |message: &str| Error::PositionDeleteFile {
            file: delete_file.path.clone(),
            message: message.to_string(),
        };
        if footer.row_groups.is_empty() {
            continue;
        }
        let table = Arc::new(ParquetTable::new(footer.row_groups));
        let batches = table_input(dispatcher, &table, Projection::all(2), false).collect()?;
        for batch in batches {
            let paths = batch
                .column(0)
                .as_string_view_opt()
                .ok_or_else(|| invalid("its file-path column is not a string column"))?;
            let positions = batch
                .column(1)
                .as_primitive_opt::<Int64Type>()
                .ok_or_else(|| invalid("its position column is not a long column"))?;
            for (path, position) in paths.iter().zip(positions.iter()) {
                let (Some(path), Some(position)) = (path, position) else {
                    return Err(invalid("a row has no file path or no position"));
                };
                let Some(&index) = targets.get(path) else {
                    continue;
                };
                let position = u64::try_from(position)
                    .map_err(|_| invalid("a row has a negative position"))?;
                deleted[index]
                    .get_or_insert_with(RoaringTreemap::new)
                    .insert(position);
            }
        }
    }
    Ok(())
}

/// The columns a position-delete file is read by, matched by the field ids
/// the spec reserves for them; the deleted row's values, when kept, are not
/// read.
fn position_delete_columns() -> TableColumns {
    TableColumns::by_field_id(
        vec![
            Column {
                name: "file_path".to_string(),
                col_type: Type::Utf8,
            },
            Column {
                name: "pos".to_string(),
                col_type: Type::Int64,
            },
        ],
        vec![FILE_PATH_FIELD_ID, POSITION_FIELD_ID],
    )
}

/// How many rows the delete files the delete manifests `manifests` declare
/// delete in all, from their entries alone: the sum of the deletion vectors'
/// counts, each exact, or `None` when a position-delete file is among them,
/// since its rows may repeat another's or name a file no longer in the
/// snapshot.
pub(crate) fn count_deleted_rows(manifests: Vec<LoadedManifest>) -> Result<Option<u64>> {
    let delete_files = locate_delete_files(manifests)?;
    if !delete_files.position_deletes.is_empty() {
        return Ok(None);
    }
    sum_deleted_rows(&delete_files.deletion_vectors).map(Some)
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

/// The delete files each live entry of `manifests` declares: the deletion
/// vectors by the data file they apply to, and the position-delete files. A
/// snapshot keeps one vector per data file: a second one is corrupt, not a
/// second set of deletes.
fn locate_delete_files(manifests: Vec<LoadedManifest>) -> Result<DeleteFiles> {
    let mut deletion_vectors = HashMap::new();
    let mut position_deletes = Vec::new();
    for manifest in &manifests {
        for entry in manifest.live_entries() {
            let file = entry.data_file();
            match (file.content_type(), file.file_format()) {
                (DataContentType::PositionDeletes, DataFileFormat::Puffin) => {
                    let (data_file, metadata) = describe_deletion_vector(file)?;
                    if deletion_vectors
                        .insert(data_file.clone(), metadata)
                        .is_some()
                    {
                        return Err(Error::DeletionVector {
                            file: data_file,
                            message: "two live deletion vectors reference it".to_string(),
                        });
                    }
                }
                (DataContentType::PositionDeletes, DataFileFormat::Parquet) => {
                    position_deletes.push(describe_position_delete_file(
                        file,
                        manifest.sequence_number(entry),
                    )?);
                }
                (content, format) => {
                    return Err(Error::DeleteFile {
                        file: file.file_path().to_string(),
                        content: describe_content(content),
                        format: format.to_string(),
                    });
                }
            }
        }
    }
    Ok(DeleteFiles {
        deletion_vectors,
        position_deletes,
    })
}

/// The data file the deletion vector `file` applies to, and where its blob
/// lies.
fn describe_deletion_vector(file: &DataFile) -> Result<(String, DeletionVectorMetadata)> {
    let invalid = |message: &str| Error::DeletionVector {
        file: file.file_path().to_string(),
        message: message.to_string(),
    };
    let data_file = file
        .referenced_data_file()
        .ok_or_else(|| invalid("it names no referenced data file"))?;
    let offset = file
        .content_offset()
        .ok_or_else(|| invalid("it has no content offset"))?;
    let length = file
        .content_size_in_bytes()
        .ok_or_else(|| invalid("it has no content size"))?;
    let (Ok(offset), Ok(length)) = (u64::try_from(offset), u64::try_from(length)) else {
        return Err(invalid("its content offset or size is negative"));
    };
    let metadata = DeletionVectorMetadata {
        puffin_path: file.file_path().to_string(),
        puffin_size: file.file_size_in_bytes(),
        offset,
        length,
        deleted_row_count: file.record_count(),
    };
    Ok((data_file, metadata))
}

/// The position-delete file `file`, of data sequence number
/// `sequence_number`, with what its entry says about the files it names.
fn describe_position_delete_file(
    file: &DataFile,
    sequence_number: i64,
) -> Result<PositionDeleteFile> {
    let path_bound = |bounds: &HashMap<i32, iceberg::spec::Datum>| {
        bounds
            .get(&FILE_PATH_FIELD_ID)
            .map(|bound| match bound.literal() {
                PrimitiveLiteral::String(path) => Ok(path.clone()),
                _ => Err(Error::PositionDeleteFile {
                    file: file.file_path().to_string(),
                    message: "its file-path bound is not a string".to_string(),
                }),
            })
            .transpose()
    };
    let path_bounds = match (
        path_bound(file.lower_bounds())?,
        path_bound(file.upper_bounds())?,
    ) {
        (Some(lower), Some(upper)) => Some((lower, upper)),
        _ => None,
    };
    Ok(PositionDeleteFile {
        path: file.file_path().to_string(),
        size: file.file_size_in_bytes(),
        sequence_number,
        referenced_data_file: file.referenced_data_file(),
        path_bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::{
        DataFileBuilder, Datum, FormatVersion, Manifest, ManifestContentType, ManifestEntry,
        ManifestFile, ManifestMetadata, ManifestStatus, NestedField, PartitionSpec, PrimitiveType,
        Schema, Type,
    };

    /// A delete manifest of `files`, listed as committed at `sequence_number`.
    fn deletes_manifest(files: Vec<DataFile>, sequence_number: i64) -> LoadedManifest {
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
        let manifest = Manifest::new(
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
        );
        let listed = ManifestFile {
            manifest_path: "deletes.avro".to_string(),
            manifest_length: 100,
            partition_spec_id: 0,
            content: ManifestContentType::Deletes,
            sequence_number,
            min_sequence_number: sequence_number,
            added_snapshot_id: 1,
            added_files_count: None,
            existing_files_count: None,
            deleted_files_count: None,
            added_rows_count: None,
            existing_rows_count: None,
            deleted_rows_count: None,
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        LoadedManifest { listed, manifest }
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

    /// A Parquet position-delete file of `deleted_rows` rows.
    fn position_delete_file(path: &str, deleted_rows: u64) -> DataFileBuilder {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_path(path.into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(deleted_rows);
        builder
    }

    fn data_file(path: &str, sequence_number: i64) -> FileDescriptor {
        FileDescriptor {
            path: path.to_string(),
            length: 100,
            sequence_number,
        }
    }

    #[test]
    fn each_data_file_has_its_deletion_vector_and_the_counts_add_up() {
        let manifest = deletes_manifest(
            vec![
                deletion_vector("dv-1.puffin", "a.parquet", 3)
                    .build()
                    .unwrap(),
                deletion_vector("dv-1.puffin", "b.parquet", 4)
                    .build()
                    .unwrap(),
            ],
            2,
        );

        let located = locate_delete_files(vec![manifest]).unwrap();

        assert_eq!(sum_deleted_rows(&located.deletion_vectors).unwrap(), 7);
        assert_eq!(located.deletion_vectors["b.parquet"].deleted_row_count, 4);
        assert!(!located.deletion_vectors.contains_key("c.parquet"));
    }

    #[test]
    fn a_second_deletion_vector_for_a_data_file_is_refused() {
        let manifest = deletes_manifest(
            vec![
                deletion_vector("dv-1.puffin", "a.parquet", 3)
                    .build()
                    .unwrap(),
                deletion_vector("dv-2.puffin", "a.parquet", 5)
                    .build()
                    .unwrap(),
            ],
            2,
        );

        let error = locate_delete_files(vec![manifest]).unwrap_err();

        assert!(
            error.to_string().contains("two live deletion vectors"),
            "{error}"
        );
    }

    #[test]
    fn a_delete_file_that_is_neither_a_deletion_vector_nor_parquet_position_deletes_is_refused() {
        let equality_deletes = position_delete_file("deletes.parquet", 3)
            .content(DataContentType::EqualityDeletes)
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap();
        let avro_position_deletes = position_delete_file("deletes.avro", 3)
            .file_format(DataFileFormat::Avro)
            .build()
            .unwrap();

        for file in [equality_deletes, avro_position_deletes] {
            let manifest = deletes_manifest(vec![file], 2);
            let error = locate_delete_files(vec![manifest]).unwrap_err();
            assert!(matches!(error, Error::DeleteFile { .. }), "{error}");
        }
    }

    #[test]
    fn a_deletion_vector_without_its_blob_location_is_refused() {
        let manifest = deletes_manifest(
            vec![
                deletion_vector("dv.puffin", "a.parquet", 3)
                    .content_offset(None)
                    .build()
                    .unwrap(),
            ],
            2,
        );

        let error = locate_delete_files(vec![manifest]).unwrap_err();

        assert!(error.to_string().contains("no content offset"), "{error}");
    }

    #[test]
    fn a_position_delete_file_inherits_its_commits_sequence_number_and_applies_to_older_files() {
        let manifest = deletes_manifest(
            vec![position_delete_file("deletes.parquet", 3).build().unwrap()],
            5,
        );

        let located = locate_delete_files(vec![manifest]).unwrap();
        let delete_file = &located.position_deletes[0];

        assert_eq!(delete_file.sequence_number, 5);
        assert!(delete_file.may_delete_from(&data_file("a.parquet", 5)));
        assert!(delete_file.may_delete_from(&data_file("a.parquet", 1)));
        assert!(!delete_file.may_delete_from(&data_file("a.parquet", 6)));
    }

    #[test]
    fn a_position_delete_file_applies_only_to_the_files_its_entry_may_name() {
        let referencing = position_delete_file("deletes.parquet", 3)
            .referenced_data_file(Some("a.parquet".into()))
            .build()
            .unwrap();
        let bounded = position_delete_file("deletes.parquet", 3)
            .lower_bounds(HashMap::from([(
                FILE_PATH_FIELD_ID,
                Datum::string("s3://bucket/data/b"),
            )]))
            .upper_bounds(HashMap::from([(
                FILE_PATH_FIELD_ID,
                Datum::string("s3://bucket/data/d"),
            )]))
            .build()
            .unwrap();

        let located =
            locate_delete_files(vec![deletes_manifest(vec![referencing, bounded], 2)]).unwrap();
        let [referencing, bounded] = located.position_deletes.as_slice() else {
            panic!("two position-delete files");
        };

        assert!(referencing.may_delete_from(&data_file("a.parquet", 1)));
        assert!(!referencing.may_delete_from(&data_file("b.parquet", 1)));
        assert!(bounded.may_delete_from(&data_file("s3://bucket/data/c.parquet", 1)));
        assert!(!bounded.may_delete_from(&data_file("s3://bucket/data/a.parquet", 1)));
        assert!(!bounded.may_delete_from(&data_file("s3://bucket/data/e.parquet", 1)));
    }

    #[test]
    fn the_deleted_row_count_is_known_only_without_position_delete_files() {
        let vectors_only = deletes_manifest(
            vec![
                deletion_vector("dv.puffin", "a.parquet", 3)
                    .build()
                    .unwrap(),
            ],
            2,
        );
        let with_position_deletes = deletes_manifest(
            vec![
                deletion_vector("dv.puffin", "a.parquet", 3)
                    .build()
                    .unwrap(),
                position_delete_file("deletes.parquet", 2).build().unwrap(),
            ],
            2,
        );

        assert_eq!(count_deleted_rows(vec![vectors_only]).unwrap(), Some(3));
        assert_eq!(
            count_deleted_rows(vec![with_position_deletes]).unwrap(),
            None
        );
    }
}
