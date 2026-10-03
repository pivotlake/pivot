//! A pinned Iceberg snapshot with lazily loaded manifests and file metadata.
//! Pivot evaluates metadata bounds before reading surviving files' footers.

use ::pruning::Predicate;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{Field, Schema};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, Manifest, ManifestContentType, ManifestFile,
    ManifestList as IcebergManifestList, SchemaRef, TableMetadataRef, Transform,
};
use iceberg::table::Table;
use object_storage::{ExternalStoreFactory, load_objects};
use parquet_engine::{
    FileRowGroups, ParquetTable, TableColumns, load_file_row_groups, prune_row_groups,
};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::types::physical_arrow_type;

use crate::columns::table_columns;
use crate::metadata::{FileDescriptor, ManifestDescriptor, ManifestList, select_files};
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::{Error, Result};

pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    pub uuid: String,
    pub metadata_location: String,
    pub(super) metadata: TableMetadataRef,
    columns: TableColumns,
    pub(super) partition_by: Vec<String>,
    pub(super) sort_by: Vec<String>,
    manifest_list: ManifestList,
    store: TableStore,
    dispatcher: DataFlowDispatcher,
}

impl LoadedTable {
    /// Load the schema and manifest list, validating unsupported snapshot features
    /// before pruning can hide them. Manifests and data footers remain lazy.
    pub(crate) fn load(
        name: &SchemaQualifiedTableName,
        table: &Table,
        store: TableStore,
        manifest_list_size: Option<u64>,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let metadata = table.metadata_ref();
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: table_name.clone(),
            })?
            .to_string();
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable { table: table_name });
        }
        let schema = metadata.current_schema();
        let columns = table_columns(&table_name, schema)?;
        let partition_by = identity_columns(
            schema,
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );

        let manifest_list = match metadata.current_snapshot() {
            Some(snapshot) => {
                let size = manifest_list_size.expect(
                    "a table with a current snapshot is indexed with its manifest list's size",
                );
                load_manifest_list(
                    &table_name,
                    snapshot.manifest_list(),
                    size,
                    metadata.format_version(),
                    &store,
                    dispatcher,
                )?
            }
            None => Vec::new(),
        };

        let manifest_list = ManifestList::new(&metadata, &manifest_list).map_err(|source| {
            Error::InvalidPartitionMetadata {
                table: table_name,
                source: Box::new(source),
            }
        })?;
        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            metadata,
            columns,
            partition_by,
            sort_by,
            manifest_list,
            store,
            dispatcher: dispatcher.clone(),
        })
    }

    /// Select files using metadata bounds, then fetch and prune their row groups.
    /// Manifest order, then file-local row-group order, makes the same predicates
    /// yield the same row-group positions for scanning and materialization.
    pub(crate) fn load_scan_metadata(&self, predicates: &[Predicate]) -> Result<ParquetTable> {
        let selected = self.manifest_list.select(predicates);
        let manifests = self.load_manifests(selected.iter().copied())?;
        let files = select_files(self.metadata.current_schema(), &manifests, predicates).map_err(
            |source| Error::InvalidPartitionMetadata {
                table: self.name.to_string(),
                source: Box::new(source),
            },
        )?;
        tracing::debug!(table = %self.name, manifests = selected.len(), files = files.len(), "selected Iceberg files");
        let row_groups: Vec<_> = self
            .read_footers(&files)?
            .into_iter()
            .flat_map(|footer| footer.row_groups)
            .collect();
        let row_groups = prune_row_groups(&row_groups, predicates);
        Ok(if row_groups.is_empty() {
            ParquetTable::empty(self.declared_schema())
        } else {
            ParquetTable::new(row_groups)
        })
    }

    /// Read and parse the selected manifests, preserving request order despite
    /// parallel read completion so repeated preparations assign the same indexes.
    fn load_manifests<'a>(
        &self,
        entries: impl IntoIterator<Item = &'a ManifestDescriptor>,
    ) -> Result<Vec<Manifest>> {
        let files = entries
            .into_iter()
            .map(|manifest| self.store.data_file(&manifest.path, manifest.length))
            .collect::<Result<Vec<_>>>()?;
        let positions: HashMap<_, _> = files
            .iter()
            .enumerate()
            .map(|(row, file)| (file.file.path.as_str(), row))
            .collect();
        let mut manifests = Vec::with_capacity(files.len());
        for object in load_objects(&self.dispatcher, &files)? {
            let path = object.file.path.as_str().to_string();
            let manifest = Manifest::parse_avro(&object.bytes).map_err(|source| {
                Error::MalformedMetadataObject {
                    table: self.name.to_string(),
                    path: path.clone(),
                    source: Box::new(source),
                }
            })?;
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                let file = entry.data_file();
                if file.content_type() != DataContentType::Data {
                    return Err(Error::DeleteFiles {
                        table: self.name.to_string(),
                        manifest: path,
                    });
                }
                if file.file_format() != DataFileFormat::Parquet {
                    return Err(Error::NonParquetFile {
                        table: self.name.to_string(),
                        file: file.file_path().to_string(),
                        format: file.file_format().to_string(),
                    });
                }
            }
            manifests.push((positions[path.as_str()], manifest));
        }
        manifests.sort_unstable_by_key(|(index, _)| *index);
        Ok(manifests
            .into_iter()
            .map(|(_, manifest)| manifest)
            .collect())
    }

    pub(super) fn load_all_manifests(&self) -> Result<Vec<Manifest>> {
        self.load_manifests(self.manifest_list.manifests())
    }

    /// Read file metadata in parallel and return it in request order.
    pub(super) fn read_footers(&self, files: &[FileDescriptor]) -> Result<Vec<FileRowGroups>> {
        let data_files = files
            .iter()
            .map(|file| self.store.data_file(&file.path, file.length))
            .collect::<Result<Vec<_>>>()?;
        let mut loaded: HashMap<String, FileRowGroups> =
            load_file_row_groups(&self.dispatcher, &data_files, self.columns.clone())?
                .into_iter()
                .map(|footer| (footer.file.path.as_str().to_string(), footer))
                .collect();
        // A file the fetch returned nothing for would scan as empty, so the
        // scan fails rather than quietly losing rows.
        files
            .iter()
            .map(|file| {
                loaded
                    .remove(&file.path)
                    .ok_or_else(|| Error::FooterNotLoaded {
                        table: self.name.to_string(),
                        file: file.path.clone(),
                    })
            })
            .collect()
    }

    pub(crate) fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    pub(crate) fn nullability(&self) -> Vec<bool> {
        self.metadata
            .current_schema()
            .as_struct()
            .fields()
            .iter()
            .map(|field| !field.required)
            .collect()
    }

    fn declared_schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(
            self.columns()
                .iter()
                .zip(self.nullability())
                .map(|(column, nullable)| {
                    Field::new(
                        &column.name,
                        physical_arrow_type(&column.col_type),
                        nullable,
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub(crate) fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid.clone(),
            version: self.metadata_location.clone(),
        }
    }

    /// Exact for snapshots without delete files. Older writers may omit counts;
    /// in that case planning has no statistic and COUNT uses the ordinary scan.
    pub(crate) fn row_count(&self) -> Option<i64> {
        self.manifest_list.row_count()
    }
}

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source field id and transform) take by identity, in field order. A
/// transformed field (a bucket, a day) is not a column value, so it is left out.
fn identity_columns<'a>(
    schema: &SchemaRef,
    fields: impl Iterator<Item = (i32, &'a Transform)>,
) -> Vec<String> {
    fields
        .filter(|(_, transform)| **transform == Transform::Identity)
        .filter_map(|(source_id, _)| schema.name_by_field_id(source_id))
        .map(str::to_string)
        .collect()
}

/// The store `table`'s files are read through: opened with the credentials
/// the catalog vended for the table, or through `store_factory` when it
/// vended none. Opened when the table is indexed rather than when a query loads
/// it, since opening a store may ask the bucket for its region; and opened
/// again at every refresh, so vended credentials never outlive their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::StagedTable {
            table: table_name.clone(),
        })?;
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    TableStore::open(&table_name, metadata_location, store_factory, vended)
}

/// The size of `table`'s current manifest list, or `None` when the table has
/// no current snapshot. The snapshot names its manifest list without a
/// length, and a ring read needs one, so the store is asked once, when the
/// table is indexed, rather than on every query over it.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store: &TableStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let size = store
        .object_size(path)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
}

/// Read the snapshot's manifest list and return its data manifests in order.
/// Reject live delete files before pruning; empty delete manifests can be omitted.
fn load_manifest_list(
    table: &str,
    path: &str,
    size: u64,
    format_version: FormatVersion,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<Vec<ManifestFile>> {
    let bytes = load_objects(dispatcher, &[store.data_file(path, size)?])?
        .pop()
        .expect("load_objects returns one object per file")
        .bytes;
    let list =
        IcebergManifestList::parse_with_version(&bytes, format_version).map_err(|source| {
            Error::MalformedMetadataObject {
                table: table.to_string(),
                path: path.to_string(),
                source: Box::new(source),
            }
        })?;
    let mut manifests = Vec::new();
    for entry in list.consume_entries() {
        if entry.content == ManifestContentType::Deletes {
            if entry.has_added_files() || entry.has_existing_files() {
                return Err(Error::DeleteFiles {
                    table: table.to_string(),
                    manifest: entry.manifest_path,
                });
            }
            continue;
        }
        manifests.push(entry);
    }
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};

    use super::*;

    #[test]
    fn only_identity_fields_are_columns_in_field_order() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::required(3, "day", Type::Primitive(PrimitiveType::Date)).into(),
                ])
                .build()
                .unwrap(),
        );
        let fields = [
            (2, Transform::Identity),
            (1, Transform::Bucket(16)),
            (3, Transform::Day),
            (1, Transform::Identity),
        ];

        let columns = identity_columns(
            &schema,
            fields
                .iter()
                .map(|(source_id, transform)| (*source_id, transform)),
        );

        assert_eq!(columns, ["region", "id"]);
    }
}
