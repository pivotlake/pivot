//! One Iceberg table at one metadata location, loaded: its schema, and every
//! data file of its current snapshot with the row groups read from its footer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, Manifest, ManifestContentType, ManifestList,
    PrimitiveLiteral, SchemaRef, Transform,
};
use iceberg::table::Table;
use object_storage::{ExternalStoreFactory, load_objects};
use parquet_engine::{ParquetTable, RowGroupMetadata, TableColumns};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};

use crate::columns::table_columns;
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::{Error, Result};

/// A table as of one metadata location.
pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    /// The table's UUID: its identity across every metadata version.
    pub uuid: String,
    /// The metadata file this load was built from: the one the catalog
    /// pointed at as current when the table was indexed. A table keeps a
    /// metadata file per commit, so this location names the version read.
    pub metadata_location: String,
    columns: TableColumns,
    /// The identity-partitioned columns of the default partition spec, in
    /// spec order. A transformed partition (a bucket, a day) is not a column
    /// value, so it is not listed.
    partition_by: Vec<String>,
    /// The columns of the default sort order sorted on directly, in order.
    sort_by: Vec<String>,
    /// The current snapshot's data files: each as the catalog describes it
    /// (location, size, partition, the manifest's column bounds) with the row
    /// groups read from its footer. The one record of the table's content; a
    /// binding flattens the row groups into the view a scan reads.
    files: Vec<TableFile>,
    /// Whether each column can hold a NULL, judged from the footers rather
    /// than the schema: the planner routes a nullable column through its
    /// null-aware paths, so the flag must be true wherever the data can hold
    /// one and is best false wherever it provably cannot. A column absent from
    /// a file reads as NULL there, which its synthesized statistics record.
    pub nullability: Vec<bool>,
}

/// One data file of the current snapshot.
struct TableFile {
    /// The file's full location, as the manifest names it.
    location: String,
    size: u64,
    /// The partition the file belongs to, as `column=value` pairs in spec
    /// order, comma-separated. Empty for an unpartitioned table.
    partition: String,
    /// The file's column bounds from its manifest entry, as a JSON object
    /// keyed by column name whose values hold `min` and `max`.
    min_max_stats: String,
    row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    fn uncompressed_size(&self) -> u64 {
        self.row_groups
            .iter()
            .flat_map(|row_group| row_group.columns.iter())
            .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
            .sum()
    }
}

impl LoadedTable {
    /// Load `table` as the catalog last returned it: its current snapshot's
    /// manifest list (of `manifest_list_size` bytes, learned when the table
    /// was indexed), the manifests it names, and the footer of every live data
    /// file, each read through `store`, the table's store as opened when it
    /// was indexed.
    pub(crate) fn load(
        name: &SchemaQualifiedTableName,
        table: &Table,
        store: &TableStore,
        manifest_list_size: Option<u64>,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let metadata = table.metadata();
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: table_name.clone(),
            })?
            .to_string();
        if metadata.format_version() == FormatVersion::V3 {
            return Err(Error::UnsupportedFormatVersion {
                table: table_name,
                version: metadata.format_version() as u8,
            });
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

        let started = Instant::now();
        let manifests = match metadata.current_snapshot() {
            Some(snapshot) => {
                let size = manifest_list_size.expect(
                    "a table with a current snapshot is indexed with its manifest list's size",
                );
                let list = load_manifest_list(
                    &table_name,
                    snapshot.manifest_list(),
                    size,
                    metadata.format_version(),
                    store,
                    dispatcher,
                )?;
                load_manifests(&table_name, &list, store, dispatcher)?
            }
            None => HashMap::new(),
        };

        let manifests_loaded = Instant::now();
        let files = load_files(&table_name, &manifests, schema, &columns, store, dispatcher)?;
        tracing::debug!(
            table = %table_name,
            manifests = manifests.len(),
            files = files.len(),
            manifests_ms = (manifests_loaded - started).as_secs_f64() * 1e3,
            footers_ms = manifests_loaded.elapsed().as_secs_f64() * 1e3,
            "loaded table"
        );
        // Every row group is laid out in declared column order, so a column's
        // position addresses its chunk in each of them.
        let row_groups = || files.iter().flat_map(|file| &file.row_groups);
        let nullability = (0..columns.columns().len())
            .map(|column| {
                row_groups().next().is_none()
                    || row_groups().any(|row_group| row_group.column_may_hold_nulls(column))
            })
            .collect();

        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            columns,
            partition_by,
            sort_by,
            files,
            nullability,
        })
    }

    /// Every row group of every file, in file order.
    fn row_groups(&self) -> impl Iterator<Item = &Arc<RowGroupMetadata>> {
        self.files.iter().flat_map(|file| &file.row_groups)
    }

    /// The table as a scan reads it: every row group of every file, in file
    /// order, the view a row reference addresses a position in.
    pub(crate) fn parquet_table(&self) -> ParquetTable {
        ParquetTable::new(self.row_groups().cloned().collect())
    }

    pub(crate) fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    pub(crate) fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid.clone(),
            version: self.metadata_location.clone(),
        }
    }

    /// This table as the cross-datastore catalog describes it.
    pub(crate) fn describe(&self) -> DatastoreTableMetadata {
        let column_bytes = self.column_bytes();
        let columns = self
            .columns()
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let (bytes, bytes_uncompressed) = column_bytes[position];
                DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by.contains(&column.name),
                    is_sort_key: self.sort_by.contains(&column.name),
                }
            })
            .collect();
        let files = self
            .files
            .iter()
            .map(|file| DatastoreFileMetadata {
                path: file.location.clone(),
                bytes: file.size,
                bytes_uncompressed: file.uncompressed_size(),
                partition: file.partition.clone(),
                min_max_stats: file.min_max_stats.clone(),
            })
            .collect();
        DatastoreTableMetadata {
            name: self.name.clone(),
            id: self.uuid.clone(),
            columns,
            sort_by: self.sort_by.clone(),
            partition_by: self.partition_by.clone(),
            total_rows: self
                .row_groups()
                .map(|row_group| row_group.num_rows.max(0) as u64)
                .sum(),
            bytes: self.files.iter().map(|file| file.size).sum(),
            bytes_uncompressed: self.files.iter().map(|file| file.uncompressed_size()).sum(),
            files,
        }
    }

    /// What each column costs across the table's files: the bytes it occupies
    /// in storage and what they hold decoded. Every row group is laid out in
    /// declared column order, so a column's chunk is at its own position.
    fn column_bytes(&self) -> Vec<(u64, u64)> {
        let mut totals = vec![(0, 0); self.columns().len()];
        for row_group in self.row_groups() {
            for (position, chunk) in row_group.columns.iter().enumerate() {
                totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
            }
        }
        totals
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

/// The snapshot's manifest list at `path`, of `size` bytes, read through the
/// ring and parsed.
fn load_manifest_list(
    table: &str,
    path: &str,
    size: u64,
    format_version: FormatVersion,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<ManifestList> {
    let bytes = load_objects(dispatcher, &[store.data_file(path, size)?])?
        .pop()
        .expect("load_objects returns one object per file")
        .bytes;
    let list = ManifestList::parse_with_version(&bytes, format_version).map_err(|source| {
        Error::MalformedMetadataObject {
            table: table.to_string(),
            path: path.to_string(),
            source: Box::new(source),
        }
    })?;
    Ok(list)
}

/// The manifests `list` names, read through the ring in one dataflow, parsed
/// and keyed by path.
fn load_manifests(
    table: &str,
    list: &ManifestList,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<HashMap<String, Manifest>> {
    let mut manifests = HashMap::new();
    let mut to_fetch = Vec::new();
    for entry in list.entries() {
        // A delete manifest that still lists live delete files means rows
        // this reader would return were deleted. One that only records the
        // removal of its delete files carries nothing to apply.
        if entry.content == ManifestContentType::Deletes
            && (entry.has_added_files() || entry.has_existing_files())
        {
            return Err(Error::DeleteFiles {
                table: table.to_string(),
                manifest: entry.manifest_path.clone(),
            });
        }
        to_fetch.push(store.data_file(&entry.manifest_path, entry.manifest_length.max(0) as u64)?);
    }
    for object in load_objects(dispatcher, &to_fetch)? {
        let path = object.file.path.as_str().to_string();
        let manifest = Manifest::parse_avro(&object.bytes).map_err(|source| {
            Error::MalformedMetadataObject {
                table: table.to_string(),
                path: path.clone(),
                source: Box::new(source),
            }
        })?;
        manifests.insert(path, manifest);
    }
    Ok(manifests)
}

/// The live data files of `manifests`, each with its row groups. A file
/// `previous` already holds at this schema is reused whole; the rest have
/// their footers read in one dataflow.
fn load_files(
    table: &str,
    manifests: &HashMap<String, Manifest>,
    schema: &SchemaRef,
    columns: &TableColumns,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<Vec<TableFile>> {
    // Manifests iterate in path order so the file list is stable across loads.
    let mut manifest_paths: Vec<&String> = manifests.keys().collect();
    manifest_paths.sort();
    let mut files = Vec::new();
    let mut to_fetch = Vec::new();
    for path in manifest_paths {
        let manifest = &manifests[path];
        let spec = manifest.metadata().partition_spec();
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            // A data manifest only ever lists data files; anything else here
            // is a manifest this reader does not understand.
            if data_file.content_type() != DataContentType::Data {
                return Err(Error::DeleteFiles {
                    table: table.to_string(),
                    manifest: path.clone(),
                });
            }
            if data_file.file_format() != DataFileFormat::Parquet {
                return Err(Error::NonParquetFile {
                    table: table.to_string(),
                    file: data_file.file_path().to_string(),
                    format: data_file.file_format().to_string(),
                });
            }
            let location = data_file.file_path().to_string();
            to_fetch.push(store.data_file(&location, data_file.file_size_in_bytes())?);
            files.push(TableFile {
                partition: spec
                    .partition_to_path(data_file.partition(), schema.clone())
                    .replace('/', ","),
                min_max_stats: format_min_max_stats(schema, data_file),
                location,
                size: data_file.file_size_in_bytes(),
                row_groups: Vec::new(),
            });
        }
    }

    let mut fetched: HashMap<String, Vec<Arc<RowGroupMetadata>>> =
        parquet_engine::load_file_row_groups(dispatcher, &to_fetch, columns.clone())?
            .into_iter()
            .map(|loaded| (loaded.file.path.as_str().to_string(), loaded.row_groups))
            .collect();
    for file in &mut files {
        // A file the fetch returned nothing for would scan as empty, so the
        // table fails to load rather than quietly losing rows.
        file.row_groups = fetched
            .remove(&file.location)
            .ok_or_else(|| Error::FooterNotLoaded {
                table: table.to_string(),
                file: file.location.clone(),
            })?;
    }
    Ok(files)
}

/// A data file's column bounds as the catalog reports them: a JSON object
/// keyed by column name, each holding the manifest's `min` and `max` for the
/// column. Numbers and booleans are JSON scalars, everything else a string,
/// the shape the pivot datastore reports its file bounds in.
fn format_min_max_stats(schema: &SchemaRef, data_file: &iceberg::spec::DataFile) -> String {
    let mut field_ids: Vec<i32> = data_file
        .lower_bounds()
        .keys()
        .filter(|field_id| data_file.upper_bounds().contains_key(field_id))
        .copied()
        .collect();
    field_ids.sort_unstable();
    let fields: Vec<String> = field_ids
        .into_iter()
        .filter_map(|field_id| {
            let name = schema.name_by_field_id(field_id)?;
            let bound = |datum: &iceberg::spec::Datum| {
                if renders_unquoted_in_json(datum.literal()) {
                    datum.to_string()
                } else {
                    json_string(&datum.to_string())
                }
            };
            Some(format!(
                "{}:{{\"min\":{},\"max\":{}}}",
                json_string(name),
                bound(&data_file.lower_bounds()[&field_id]),
                bound(&data_file.upper_bounds()[&field_id])
            ))
        })
        .collect();
    format!("{{{}}}", fields.join(","))
}

/// Whether a literal is written bare in JSON, as a number or a boolean, rather
/// than quoted as a string.
fn renders_unquoted_in_json(literal: &PrimitiveLiteral) -> bool {
    matches!(
        literal,
        PrimitiveLiteral::Boolean(_)
            | PrimitiveLiteral::Int(_)
            | PrimitiveLiteral::Long(_)
            | PrimitiveLiteral::Float(_)
            | PrimitiveLiteral::Double(_)
            | PrimitiveLiteral::Int128(_)
    )
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("a Rust string is valid JSON")
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
