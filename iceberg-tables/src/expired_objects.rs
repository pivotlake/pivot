//! What an expiry leaves to delete: the objects the snapshots it dropped
//! used and no retained snapshot does (their data files and deletion
//! vectors, their manifests, their manifest lists and their statistics
//! files), and the previous metadata files it dropped from the metadata log.
//!
//! The rule is a set difference, never a deduction from a table's history:
//! everything reachable from a dropped snapshot, less everything the table
//! still uses after the expiry ([`Table::collect_objects_in_use`]). A file a
//! tag, a branch or any other retained snapshot names is therefore kept
//! whatever the ancestry of the snapshot that removed it, and so is a
//! metadata file the log still names.

use std::collections::{HashMap, HashSet};

use iceberg::spec::{ManifestFile, ManifestList as IcebergManifestList, TableMetadata};
use object_storage::ObjectPath;

use crate::objects_in_use::object_name;
use crate::table::Table;
use crate::{Error, Result};

/// The snapshots and previous metadata files one committed expiry dropped,
/// by what the metadata before it named for them and the metadata after it
/// no longer does. Only the commit that dropped them hands one out, so only
/// the process that committed an expiry deletes for it.
pub struct ExpiredHistory {
    /// The manifest lists of the dropped snapshots, by URI.
    manifest_lists: Vec<String>,
    /// The statistics and partition statistics files the expiry dropped
    /// from the metadata, by URI.
    statistics_files: Vec<String>,
    /// The previous metadata files the expiry dropped from the metadata
    /// log, by URI.
    metadata_files: Vec<String>,
}

impl ExpiredHistory {
    /// What the metadata `before` an expiry names that the metadata `after`
    /// it does not: the snapshots dropped, the statistics files dropped
    /// with them, and the previous metadata files dropped from the log.
    pub(crate) fn between(before: &TableMetadata, after: &TableMetadata) -> Self {
        let manifest_lists = before
            .snapshots()
            .filter(|snapshot| after.snapshot_by_id(snapshot.snapshot_id()).is_none())
            .map(|snapshot| snapshot.manifest_list().to_string())
            .collect();
        let retained_statistics: HashSet<&str> = list_statistics_files(after).collect();
        let statistics_files = list_statistics_files(before)
            .filter(|uri| !retained_statistics.contains(uri))
            .map(str::to_string)
            .collect();
        let retained_metadata_files: HashSet<&str> = list_logged_metadata_files(after).collect();
        let metadata_files = list_logged_metadata_files(before)
            .filter(|uri| !retained_metadata_files.contains(uri))
            .map(str::to_string)
            .collect();
        Self {
            manifest_lists,
            statistics_files,
            metadata_files,
        }
    }

    /// How many snapshots the expiry dropped.
    pub fn snapshot_count(&self) -> usize {
        self.manifest_lists.len()
    }

    /// How many previous metadata files the expiry dropped from the log.
    pub fn metadata_file_count(&self) -> usize {
        self.metadata_files.len()
    }
}

/// The previous metadata files the metadata log of `metadata` names, by URI.
fn list_logged_metadata_files(metadata: &TableMetadata) -> impl Iterator<Item = &str> {
    metadata
        .metadata_log()
        .iter()
        .map(|entry| entry.metadata_file.as_str())
}

/// The statistics and partition statistics files `metadata` names, by URI.
pub(crate) fn list_statistics_files(metadata: &TableMetadata) -> impl Iterator<Item = &str> {
    metadata
        .statistics_iter()
        .map(|file| file.statistics_path.as_str())
        .chain(
            metadata
                .partition_statistics_iter()
                .map(|file| file.statistics_path.as_str()),
        )
}

impl Table {
    /// The objects the snapshots `expired` used that this version, the one
    /// their expiry committed, does not use, and the metadata files `expired`
    /// that it no longer names: data files and deletion vectors first, then
    /// manifests, manifest lists, statistics files and metadata files. Each
    /// is named as the table names a file: relative to the table's directory
    /// when it lives under it, else by its store-root-absolute key.
    ///
    /// A manifest only the dropped snapshots list is read for the files live
    /// in it; a manifest a retained snapshot lists is in use, and so is
    /// every file live in it. Any read that fails fails the whole
    /// collection, so nothing is deleted on a partial view.
    pub fn collect_expired_objects(&self, expired: &ExpiredHistory) -> Result<Vec<ObjectPath>> {
        let in_use = self.collect_objects_in_use()?;
        let location = self.location();
        let table_name = self.name().to_string();
        let format_version = self.metadata().format_version();

        let mut unused_manifest_lists = Vec::new();
        let mut unused_manifests: HashMap<String, ManifestFile> = HashMap::new();
        for list_uri in &expired.manifest_lists {
            if in_use.metadata_names.contains(&object_name(list_uri)) {
                continue;
            }
            let bytes = location.read_metadata_object(&table_name, list_uri)?;
            let list = IcebergManifestList::parse_with_version(&bytes, format_version).map_err(
                |source| Error::MalformedMetadataObject {
                    table: table_name.clone(),
                    path: list_uri.clone(),
                    source: Box::new(source),
                },
            )?;
            for manifest in list.consume_entries() {
                if !in_use
                    .metadata_names
                    .contains(&object_name(&manifest.manifest_path))
                {
                    unused_manifests
                        .entry(manifest.manifest_path.clone())
                        .or_insert(manifest);
                }
            }
            unused_manifest_lists.push(list_uri);
        }

        let mut objects = Vec::new();
        let mut collected: HashSet<ObjectPath> = HashSet::new();
        let manifests = self.read_manifests(unused_manifests.values())?;
        for manifest in &manifests {
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                let path = location.file_path_for_uri(entry.data_file().file_path())?;
                if !in_use.file_paths.contains(&path) && collected.insert(path.clone()) {
                    objects.push(path);
                }
            }
        }
        let unused_statistics_files = expired
            .statistics_files
            .iter()
            .filter(|uri| !in_use.metadata_names.contains(&object_name(uri)));
        let unused_metadata_files = expired
            .metadata_files
            .iter()
            .filter(|uri| !in_use.metadata_names.contains(&object_name(uri)));
        for uri in unused_manifests
            .keys()
            .chain(unused_manifest_lists)
            .chain(unused_statistics_files)
            .chain(unused_metadata_files)
        {
            objects.push(location.file_path_for_uri(uri)?);
        }
        Ok(objects)
    }
}
