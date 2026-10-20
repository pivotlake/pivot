//! One managed table at one committed metadata version, as loaded from its
//! metadata file: the base every commit from it is written on top of and the
//! view every read of it serves.

use std::sync::Arc;

use iceberg::spec::{ManifestFile, TableMetadata};
use planner::catalog::Column;

use super::data_files::DataFileEntry;
use super::layout::TableLocation;
use super::manifest_writer::ManifestCache;

/// A metadata file the table was loaded from: which numbered version it is,
/// where it is, and what it says.
pub(crate) struct CurrentMetadata {
    pub version: u64,
    /// The metadata file's URI, as the next version's metadata log records it.
    pub metadata_location: String,
    pub metadata: Arc<TableMetadata>,
}

/// The table at one version: its metadata, its declared shape, the files its
/// current snapshot holds, and the manifests they were read from.
pub(crate) struct TableState {
    pub location: TableLocation,
    pub current: Arc<CurrentMetadata>,
    pub columns: Vec<Column>,
    pub partition_by: Vec<String>,
    pub sort_by: Vec<String>,
    /// The current snapshot's data manifests, in manifest-list order.
    pub manifest_list: Vec<ManifestFile>,
    /// The size of the current snapshot's manifest list object, which a ring
    /// read of it needs; `None` for a table with no snapshot.
    pub manifest_list_size: Option<u64>,
    /// The manifests of `manifest_list`, decoded, by path.
    pub manifests: ManifestCache,
    /// Every live data file, in manifest-list then manifest order, each
    /// stamped with where its manifest records it.
    pub entries: Vec<DataFileEntry>,
}

impl TableState {
    /// The loaded manifests in manifest-list order.
    pub(crate) fn ordered_manifests(&self) -> Vec<Arc<super::manifest_writer::LoadedManifest>> {
        self.manifest_list
            .iter()
            .map(|file| {
                Arc::clone(
                    self.manifests
                        .get(&file.manifest_path)
                        .expect("every listed manifest is loaded with the state"),
                )
            })
            .collect()
    }
}
