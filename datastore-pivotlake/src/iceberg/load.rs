//! Loading a managed table from its store: finding the current metadata
//! version, reading it, and decoding the manifests of its current snapshot.

use std::sync::Arc;

use iceberg::spec::TableMetadata;

use super::data_files::EntryContext;
use super::layout::{
    TableLocation, VERSION_HINT_FILE, metadata_file_name, parse_hint, parse_metadata_file_version,
};
use super::manifest_writer::{ManifestCache, read_manifest, read_manifest_list};
use super::state::{CurrentMetadata, TableState};
use super::{Error, Result};
use datastore_iceberg::identity_columns;
use datastore_iceberg::read_logical_types;
use datastore_iceberg::table_columns;

/// The version the table's hint names, or `None` when there is no hint,
/// which only costs the reader a longer listing. A hint that does not parse
/// is an error.
pub(crate) fn read_version_hint(location: &TableLocation) -> Result<Option<u64>> {
    match location
        .store()
        .get(&location.metadata_key(VERSION_HINT_FILE))?
    {
        Some(bytes) => Ok(Some(parse_hint(&bytes)?)),
        None => {
            tracing::warn!(
                table = %location.table_uri(),
                "no version hint; listing the table's metadata from its first version"
            );
            Ok(None)
        }
    }
}

/// The table's current version: the highest metadata file listed from `from`
/// up, or `None` when the table has no metadata file at all. `from` names a
/// version known to have existed (the hint's, or the one a reader holds); a
/// listing that no longer shows it means it was deleted, which the caller
/// decides what to make of.
pub(crate) fn find_current_version(location: &TableLocation, from: u64) -> Result<Option<u64>> {
    // The padded names sort in version order, so the name of the version
    // before `from` is a cursor that admits `from` itself and everything newer.
    let after = match from {
        0 => String::new(),
        from => metadata_file_name(from - 1),
    };
    let listed = location
        .store()
        .list_from(&location.metadata_dir(), "v", &after)?;
    let mut versions: Vec<u64> = listed
        .objects
        .iter()
        .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
        .collect();
    versions.sort_unstable();
    if from > 0 && !versions.contains(&from) {
        return Err(Error::MetadataVersionMissing {
            table: location.table_uri(),
            version: from,
        });
    }
    Ok(versions.last().copied())
}

/// Read and parse metadata `version`.
pub(crate) fn read_metadata(location: &TableLocation, version: u64) -> Result<CurrentMetadata> {
    let name = metadata_file_name(version);
    let bytes = location
        .store()
        .get(&location.metadata_key(&name))?
        .ok_or_else(|| Error::MetadataVersionMissing {
            table: location.table_uri(),
            version,
        })?;
    let metadata: TableMetadata = serde_json::from_slice(&bytes)?;
    Ok(CurrentMetadata {
        version,
        metadata_location: location.metadata_uri(&name),
        metadata: Arc::new(metadata),
    })
}

/// Load the table at `location` from its current version: a cold open that
/// starts from the hint.
pub(crate) fn load_table(location: &TableLocation, table_name: &str) -> Result<TableState> {
    let hinted = read_version_hint(location)?.unwrap_or(0);
    let version = match find_current_version(location, hinted) {
        Ok(Some(version)) => version,
        Ok(None) => {
            return Err(Error::TableMetadataMissing {
                table: table_name.to_string(),
                location: location.table_uri(),
            });
        }
        // The hinted version was deleted from under a hint vacuum had not
        // caught up with yet: the whole listing is still exact.
        Err(Error::MetadataVersionMissing { .. }) if hinted > 0 => {
            tracing::warn!(
                table = table_name,
                hinted,
                "the hinted metadata version no longer exists; listing from the first version"
            );
            find_current_version(location, 0)?.ok_or_else(|| Error::TableMetadataMissing {
                table: table_name.to_string(),
                location: location.table_uri(),
            })?
        }
        Err(error) => return Err(error),
    };
    let current = read_metadata(location, version)?;
    load_state(location, table_name, Arc::new(current), None)
}

/// The table's state at a newer version than `state`, or `None` when
/// `state` is still current. Lists from the held version, so an unchanged
/// table costs one short listing; manifests the held state already decoded
/// are kept.
pub(crate) fn refresh_table(state: &TableState, table_name: &str) -> Result<Option<TableState>> {
    let version = match find_current_version(&state.location, state.current.version) {
        Ok(Some(version)) => version,
        Ok(None) => {
            return Err(Error::TableMetadataMissing {
                table: table_name.to_string(),
                location: state.location.table_uri(),
            });
        }
        // The held version was swept: this copy is older than the retention
        // window, and a cold open is the only exact reload.
        Err(Error::MetadataVersionMissing { .. }) => {
            return load_table(&state.location, table_name).map(Some);
        }
        Err(error) => return Err(error),
    };
    if version == state.current.version {
        return Ok(None);
    }
    let current = read_metadata(&state.location, version)?;
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
pub(crate) fn load_state(
    location: &TableLocation,
    table_name: &str,
    current: Arc<CurrentMetadata>,
    held: Option<&ManifestCache>,
) -> Result<TableState> {
    let metadata = &current.metadata;
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
    };

    let mut manifest_list = Vec::new();
    let mut manifest_list_size = None;
    let mut manifests = ManifestCache::new();
    let mut entries = Vec::new();
    if let Some(snapshot) = metadata.current_snapshot() {
        let list_bytes = read_metadata_object(location, table_name, snapshot.manifest_list())?;
        manifest_list_size = Some(list_bytes.len() as u64);
        manifest_list = read_manifest_list(table_name, snapshot.manifest_list(), &list_bytes)?;
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
fn read_metadata_object(location: &TableLocation, table_name: &str, uri: &str) -> Result<Vec<u8>> {
    let path = location.file_path_for_uri(uri)?;
    location
        .store()
        .get(&location.data_file_key(&path))?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: table_name.to_string(),
            path: uri.to_string(),
        })
}
