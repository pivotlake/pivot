//! Reading a table's metadata objects: the columns its partition spec and
//! sort order name, the objects under its metadata directory, and its
//! manifests decoded into this crate's data-file entries.

use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    DataContentType, DataFileFormat, Manifest, ManifestFile, SchemaRef, Transform,
};

use crate::data_files::{DataFileEntry, EntryContext};
use crate::layout::TableLocation;
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

/// The live entries of the data manifest `file`, decoded from `manifest`,
/// with everything the manifest leaves implicit filled in. First row ids are
/// assigned the way the spec inherits them: an entry without one gets the
/// manifest's, advanced by the row counts of the entries assigned before it.
/// A delete file among a data manifest's entries, or a file that is not
/// Parquet, is refused.
fn decode_manifest(
    table: &str,
    file: &ManifestFile,
    manifest: &Manifest,
    ctx: &EntryContext<'_>,
) -> Result<Vec<DataFileEntry>> {
    let mut next_row_id = file.first_row_id.map(|first| first as i64);
    let mut entries = Vec::new();
    for entry in manifest.entries() {
        let data = entry.data_file();
        let first_row_id = match data.first_row_id() {
            Some(assigned) => Some(assigned),
            None if entry.is_alive() => {
                let assigned = next_row_id;
                if let Some(next) = next_row_id.as_mut() {
                    *next += data.record_count() as i64;
                }
                assigned
            }
            None => None,
        };
        if !entry.is_alive() {
            continue;
        }
        if data.content_type() != DataContentType::Data {
            return Err(Error::DeleteFile {
                file: data.file_path().to_string(),
            });
        }
        if data.file_format() != DataFileFormat::Parquet {
            return Err(Error::NonParquetFile {
                table: table.to_string(),
                file: data.file_path().to_string(),
                format: data.file_format().to_string(),
            });
        }
        entries.push(DataFileEntry::from_manifest_entry(
            entry,
            file,
            first_row_id,
            ctx,
        )?);
    }
    Ok(entries)
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
