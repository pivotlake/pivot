//! What a table's retained snapshots and metadata log still reach: the data
//! files some retained snapshot names, and the metadata objects (manifests,
//! manifest lists, metadata files) that are still part of the table. What a
//! sweep of the table's storage must keep.

use std::collections::HashSet;

use object_storage::ObjectPath;

use crate::Result;
use crate::data_files::EntryContext;
use crate::load::read_metadata_object;
use crate::manifest_writer::{read_manifest, read_manifest_list};
use crate::state::TableState;

pub struct ReachableObjects {
    pub data_paths: HashSet<ObjectPath>,
    /// Metadata objects by file name, as they sit under the table's metadata
    /// directory.
    pub metadata_names: HashSet<String>,
}

/// Walk every retained snapshot of the table in `state` from its manifest
/// list down to its data files. The current snapshot's manifests are already
/// decoded; older snapshots' are read, each once however many snapshots
/// share it.
pub fn reachable_objects(state: &TableState, table_name: &str) -> Result<ReachableObjects> {
    let metadata = &state.current.metadata;
    let location = &state.location;
    let ctx = EntryContext {
        schema: metadata.current_schema(),
        partition_spec: metadata.default_partition_spec(),
        columns: &state.columns,
        location,
        format_version: metadata.format_version(),
    };
    let mut data_paths = HashSet::new();
    let mut metadata_names = HashSet::new();
    metadata_names.insert(object_name(&state.current.metadata_location));
    for entry in metadata.metadata_log() {
        metadata_names.insert(object_name(&entry.metadata_file));
    }
    let mut manifests_seen: HashSet<String> = HashSet::new();
    for snapshot in metadata.snapshots() {
        let list_uri = snapshot.manifest_list();
        metadata_names.insert(object_name(list_uri));
        let manifest_files = if Some(snapshot.snapshot_id()) == metadata.current_snapshot_id() {
            state.manifest_list.clone()
        } else {
            let bytes = read_metadata_object(location, table_name, list_uri)?;
            read_manifest_list(table_name, list_uri, &bytes, metadata.format_version())?
        };
        for file in manifest_files {
            metadata_names.insert(object_name(&file.manifest_path));
            if !manifests_seen.insert(file.manifest_path.clone()) {
                continue;
            }
            let entries = match state.manifests.get(&file.manifest_path) {
                Some(loaded) => loaded.entries.clone(),
                None => {
                    let bytes = read_metadata_object(location, table_name, &file.manifest_path)?;
                    read_manifest(table_name, &file, &bytes, &ctx)?.entries
                }
            };
            data_paths.extend(entries.into_iter().map(|entry| entry.file.path));
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
