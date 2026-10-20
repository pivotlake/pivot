//! Loading a table from its metadata: the declared shape of the current
//! schema, and the live files of the current snapshot decoded from its
//! manifests.

use std::sync::Arc;

use iceberg::spec::{FormatVersion, SchemaRef, Transform};

use crate::catalog::TableCatalog;
use crate::columns::{read_logical_types, table_columns};
use crate::data_files::EntryContext;
use crate::layout::TableLocation;
use crate::manifest_writer::{ManifestCache, read_manifest, read_manifest_list};
use crate::state::{CurrentMetadata, TableState};
use crate::{Error, Result};

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source field id and transform) take by identity, in field order. A
/// transformed field (a bucket, a day) is not a column value, so it is left
/// out.
pub fn identity_columns<'a>(
    schema: &SchemaRef,
    fields: impl Iterator<Item = (i32, &'a Transform)>,
) -> Vec<String> {
    fields
        .filter(|(_, transform)| **transform == Transform::Identity)
        .filter_map(|(source_id, _)| schema.name_by_field_id(source_id))
        .map(str::to_string)
        .collect()
}

/// Load the table at `location` at the version `catalog` holds current: a
/// cold open.
pub fn load_table(
    catalog: &dyn TableCatalog,
    location: &TableLocation,
    table_name: &str,
) -> Result<TableState> {
    let current = catalog.load()?;
    load_state(location, table_name, Arc::new(current), None)
}

/// The table's state at a newer version than `state`, or `None` when
/// `state` is still current. Manifests the held state already decoded are
/// kept.
pub fn refresh_table(
    catalog: &dyn TableCatalog,
    state: &TableState,
    table_name: &str,
) -> Result<Option<TableState>> {
    let Some(current) = catalog.refresh(&state.current)? else {
        return Ok(None);
    };
    load_state(
        &state.location,
        table_name,
        Arc::new(current),
        Some(&state.manifests),
    )
    .map(Some)
}

/// The table's state at `current`: its declared shape and the files of its
/// current snapshot, decoding only the manifests `held` does not already.
/// A format version 1 table is refused: its manifests have no sequence
/// numbers, and this crate writes none.
pub fn load_state(
    location: &TableLocation,
    table_name: &str,
    current: Arc<CurrentMetadata>,
    held: Option<&ManifestCache>,
) -> Result<TableState> {
    let metadata = &current.metadata;
    let format_version = metadata.format_version();
    if format_version < FormatVersion::V2 {
        return Err(Error::UnsupportedFormatVersion {
            table: table_name.to_string(),
            version: format_version as u8,
        });
    }
    let schema = metadata.current_schema();
    let logical_types = read_logical_types(metadata.properties())?;
    let columns = table_columns(table_name, schema, &logical_types)?
        .columns()
        .to_vec();
    let partition_spec = metadata.default_partition_spec();
    let partition_by = identity_columns(
        schema,
        partition_spec
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
    let ctx = EntryContext {
        schema,
        partition_spec,
        columns: &columns,
        location,
        format_version,
    };

    let mut manifest_list = Vec::new();
    let mut manifest_list_size = None;
    let mut manifests = ManifestCache::new();
    let mut entries = Vec::new();
    if let Some(snapshot) = metadata.current_snapshot() {
        let list_bytes = read_metadata_object(location, table_name, snapshot.manifest_list())?;
        manifest_list_size = Some(list_bytes.len() as u64);
        manifest_list = read_manifest_list(
            table_name,
            snapshot.manifest_list(),
            &list_bytes,
            format_version,
        )?;
        for file in &manifest_list {
            let loaded = match held.and_then(|cache| cache.get(&file.manifest_path)) {
                Some(loaded) => Arc::clone(loaded),
                None => {
                    let bytes = read_metadata_object(location, table_name, &file.manifest_path)?;
                    Arc::new(read_manifest(table_name, file, &bytes, &ctx)?)
                }
            };
            entries.extend(loaded.entries.iter().cloned());
            manifests.insert(file.manifest_path.clone(), loaded);
        }
    }

    Ok(TableState {
        location: location.clone(),
        current,
        columns,
        partition_by,
        sort_by,
        manifest_list,
        manifest_list_size,
        manifests,
        entries,
    })
}

/// The bytes of the metadata object at `uri`, which must be in the table's
/// store.
pub fn read_metadata_object(
    location: &TableLocation,
    table_name: &str,
    uri: &str,
) -> Result<Vec<u8>> {
    let path = location.file_path_for_uri(uri)?;
    location
        .store()
        .get(&location.data_file_key(&path))?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: table_name.to_string(),
            path: uri.to_string(),
        })
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
