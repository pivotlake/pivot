//! What a table's retained snapshots and metadata log still use: the files
//! some retained snapshot names, data files and deletion vectors alike, and
//! the metadata objects (manifests, manifest lists, metadata files,
//! statistics files, the version hint) that are still part of the table.
//! What a sweep of the table's storage must keep, so it has to be complete:
//! a used object missing from it is one a sweep deletes.

use std::collections::HashSet;

use iceberg::spec::{ManifestFile, ManifestList as IcebergManifestList};
use object_storage::ObjectPath;

use crate::expired_objects::list_statistics_files;
use crate::layout::VERSION_HINT_FILE;
use crate::table::Table;
use crate::{Error, Result};

pub struct ObjectsInUse {
    /// The files some retained snapshot names, each as the table names a
    /// file: relative to the table's directory when it lives under it, else
    /// by its store-root-absolute key.
    pub file_paths: HashSet<ObjectPath>,
    /// Metadata objects by file name, wherever they sit: the names are
    /// unique to the object, so an object of the same name anywhere under
    /// the table is kept.
    pub metadata_names: HashSet<String>,
}

impl ObjectsInUse {
    /// Whether the table uses the object at `path`, named as the table
    /// names a file: it is a file some retained snapshot names, or it has
    /// the name of a metadata object the table uses.
    pub fn contains(&self, path: &ObjectPath) -> bool {
        self.file_paths.contains(path) || self.metadata_names.contains(path.name())
    }
}

impl Table {
    /// The data files and deletion vectors some retained snapshot names, and
    /// the metadata objects the table still uses: every retained snapshot's
    /// manifest list and manifests, the current metadata file and those the
    /// metadata log names, the statistics and partition statistics files
    /// the metadata names, and the version hint. What a sweep of the
    /// table's storage must keep. Every retained snapshot's manifest
    /// list is read for the manifests it names, the current one's included:
    /// the list this version holds leaves out a delete manifest with no file
    /// left to apply, which the snapshot still names. The files of the
    /// current snapshot's data manifests are the ones the table's index
    /// holds; every other manifest is read over the pool, each once however
    /// many snapshots share it.
    pub fn collect_objects_in_use(&self) -> Result<ObjectsInUse> {
        let metadata = self.metadata();
        let location = self.location();
        let table_name = &self.name().to_string();
        let mut file_paths = HashSet::new();
        let mut metadata_names = HashSet::new();
        metadata_names.insert(object_name(self.metadata_location()));
        for entry in metadata.metadata_log() {
            metadata_names.insert(object_name(&entry.metadata_file));
        }
        // The hint a path reader finds the current version by is named by no
        // metadata, and the statistics files only by the metadata document.
        metadata_names.insert(VERSION_HINT_FILE.to_string());
        metadata_names.extend(list_statistics_files(metadata).map(object_name));
        let mut manifests_seen: HashSet<String> = HashSet::new();
        let index = self.committed_index()?;
        for snapshot in metadata.snapshots() {
            let list_uri = snapshot.manifest_list();
            metadata_names.insert(object_name(list_uri));
            let bytes = location.read_metadata_object(table_name, list_uri)?;
            let manifest_files: Vec<ManifestFile> =
                IcebergManifestList::parse_with_version(&bytes, metadata.format_version())
                    .map_err(|source| Error::MalformedMetadataObject {
                        table: table_name.to_string(),
                        path: list_uri.to_string(),
                        source: Box::new(source),
                    })?
                    .consume_entries()
                    .into_iter()
                    .collect();
            let mut unread = Vec::new();
            for file in manifest_files {
                metadata_names.insert(object_name(&file.manifest_path));
                if !manifests_seen.insert(file.manifest_path.clone()) {
                    continue;
                }
                match index.manifest_files(&file.manifest_path) {
                    Some(files) => {
                        file_paths.extend(files.iter().map(|file| file.file.path.clone()));
                    }
                    None => unread.push(file),
                }
            }
            for manifest in self.read_manifests(unread.iter())? {
                for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                    file_paths.insert(location.file_path_for_uri(entry.data_file().file_path())?);
                }
            }
        }
        Ok(ObjectsInUse {
            file_paths,
            metadata_names,
        })
    }
}

/// The file name a metadata object's URI ends in.
pub fn object_name(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}
