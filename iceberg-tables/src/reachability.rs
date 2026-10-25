//! What a table's retained snapshots and metadata log still reach: the files
//! some retained snapshot names, data files and deletion vectors alike, and
//! the metadata objects (manifests, manifest lists, metadata files) that are
//! still part of the table. What a sweep of the table's storage must keep.

use std::collections::HashSet;

use iceberg::spec::{ManifestFile, ManifestList as IcebergManifestList};
use object_storage::ObjectPath;

use crate::load::read_metadata_object;
use crate::manifests::read_manifests;
use crate::table::Table;
use crate::{Error, Result};

pub struct ReachableObjects {
    /// The files some retained snapshot names, as paths under the table.
    pub data_paths: HashSet<ObjectPath>,
    /// Metadata objects by file name, as they sit under the table's metadata
    /// directory.
    pub metadata_names: HashSet<String>,
}

/// Walk every retained snapshot of `table` from its manifest list down to
/// its files. The current snapshot's data manifests are the ones the table
/// holds decoded; every other manifest is read over the pool, each once
/// however many snapshots share it.
pub(crate) fn reachable_objects(table: &Table) -> Result<ReachableObjects> {
    let metadata = table.metadata();
    let location = table.location();
    let table_name = table.name();
    let mut data_paths = HashSet::new();
    let mut metadata_names = HashSet::new();
    metadata_names.insert(object_name(table.metadata_location()));
    for entry in metadata.metadata_log() {
        metadata_names.insert(object_name(&entry.metadata_file));
    }
    let mut manifests_seen: HashSet<String> = HashSet::new();
    let decoded = table.decoded_manifests()?;
    for snapshot in metadata.snapshots() {
        let list_uri = snapshot.manifest_list();
        metadata_names.insert(object_name(list_uri));
        let manifest_files: Vec<ManifestFile> =
            if Some(snapshot.snapshot_id()) == metadata.current_snapshot_id() {
                table.manifest_list().manifests().cloned().collect()
            } else {
                let bytes = read_metadata_object(location, table_name, list_uri)?;
                IcebergManifestList::parse_with_version(&bytes, metadata.format_version())
                    .map_err(|source| Error::MalformedMetadataObject {
                        table: table_name.to_string(),
                        path: list_uri.to_string(),
                        source: Box::new(source),
                    })?
                    .consume_entries()
                    .into_iter()
                    .collect()
            };
        let mut unread = Vec::new();
        for file in manifest_files {
            metadata_names.insert(object_name(&file.manifest_path));
            if !manifests_seen.insert(file.manifest_path.clone()) {
                continue;
            }
            match decoded
                .iter()
                .find(|manifest| manifest.file.manifest_path == file.manifest_path)
            {
                Some(manifest) => {
                    data_paths.extend(manifest.entries.iter().map(|entry| entry.file.path.clone()));
                }
                None => unread.push(file),
            }
        }
        for manifest in
            read_manifests(table_name, unread.iter(), table.store(), table.dispatcher())?
        {
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                data_paths.insert(location.file_path_for_uri(entry.data_file().file_path())?);
            }
        }
    }
    Ok(ReachableObjects {
        data_paths,
        metadata_names,
    })
}

/// The file name a metadata object's URI ends in.
pub fn object_name(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}
