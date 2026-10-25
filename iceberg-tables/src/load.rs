//! Reading a table's metadata objects: the columns its partition spec and
//! sort order name, the objects under its metadata directory, and its
//! manifests decoded into this crate's data-file entries.

use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use iceberg::spec::{ManifestFile, SchemaRef, Transform};

use crate::data_files::EntryContext;
use crate::layout::TableLocation;
use crate::manifest_writer::decode_manifest;
use crate::manifests::{DecodedManifest, read_manifests};
use crate::store::TableStore;
use crate::{Error, Result};

/// The columns `fields` (a partition spec's or a sort order's, each a source
/// field id and its transform) name directly, in order. A transformed field
/// (bucket, truncate, a date part) is not a column of the table.
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

/// Read `files`, data manifests of the table, over the pool and decode each
/// into its live entries, in the order given.
pub(crate) fn decode_manifests(
    dispatcher: &DataFlowDispatcher,
    store: &TableStore,
    table_name: &str,
    files: &[ManifestFile],
    ctx: &EntryContext<'_>,
) -> Result<Vec<Arc<DecodedManifest>>> {
    read_manifests(table_name, files.iter(), store, dispatcher)?
        .into_iter()
        .zip(files)
        .map(|(manifest, file)| {
            let entries = decode_manifest(table_name, file, &manifest, ctx)?;
            Ok(Arc::new(DecodedManifest {
                file: file.clone(),
                manifest: Arc::new(manifest),
                entries,
            }))
        })
        .collect()
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
