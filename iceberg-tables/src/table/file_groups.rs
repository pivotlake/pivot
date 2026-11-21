//! A version's live files by the data manifest that lists them, as
//! maintenance takes them: a manifest never changes, so what it lists is
//! taken once, and a later version is caught up with by the manifests it
//! lists and an earlier one did not.
//!
//! The files are read through the parts of the manifests the cache holds
//! ([`manifest_parts`](crate::files::manifest_parts)), of which this reads
//! the files of every partition of a manifest and the bounds of the columns
//! the table is sorted by: a manifest is read and parsed only for a part
//! the cache does not hold, and a parse leaves every part of the manifest
//! there for the scans after it.

use std::collections::HashMap;
use std::sync::Arc;

use super::Table;
use super::live_files::SliceRecords;
use super::manifest_reader::ManifestReader;
use crate::files::manifests::ListedManifest;
use crate::files::sort_bounded_file::SortBoundedFile;
use crate::{Error, Result};

/// The live files one data manifest lists of one partition, in the
/// manifest's order.
pub struct ManifestPartitionFiles {
    /// The key of the partition the files sit in.
    pub partition_key: Arc<[u8]>,
    pub files: Vec<SortBoundedFile>,
}

impl Table {
    /// The path of every data manifest of this version, in manifest-list
    /// order. A manifest is never rewritten in place, so its path names
    /// the files it lists. Nothing is read.
    pub fn list_data_manifest_paths(&self) -> Vec<Arc<str>> {
        let manifests = self.manifest_list.listed_data_manifests();
        manifests
            .iter()
            .map(|manifest| Arc::clone(manifest.path()))
            .collect()
    }

    /// The live files each of the data manifests at `manifest_paths`
    /// lists, in the order given: for each manifest, the files of every
    /// partition it holds one of, in key order, each with its bounds on the
    /// columns this version is sorted by. A manifest is read and parsed
    /// only for what the cache does not hold of it; one whose manifest-list
    /// entry counts no live file is not read. A path that names no data
    /// manifest of this version is refused.
    pub fn load_manifest_files(
        &self,
        manifest_paths: &[Arc<str>],
    ) -> Result<Vec<Vec<ManifestPartitionFiles>>> {
        let listed: HashMap<&str, &ListedManifest> = self
            .manifest_list
            .listed_data_manifests()
            .iter()
            .map(|manifest| (&**manifest.path(), manifest))
            .collect();
        let manifests = manifest_paths
            .iter()
            .map(|path| {
                let manifest = listed.get(&**path).copied();
                manifest.ok_or_else(|| Error::UnlistedManifest {
                    table: self.name.to_string(),
                    path: path.to_string(),
                })
            })
            .collect::<Result<Vec<&ListedManifest>>>()?;

        let sort_columns = self.resolve_sort_columns();
        let mut reader = ManifestReader::new(self, sort_columns.field_ids().collect());
        let loaded = reader.load_manifests(&manifests)?;
        Ok(loaded
            .into_iter()
            .map(|partitions| {
                let partitions = partitions.into_iter();
                partitions
                    .map(|slice| {
                        let records = SliceRecords::new(&slice, &sort_columns);
                        let positions = 0..slice.files.count_files();
                        ManifestPartitionFiles {
                            partition_key: Arc::clone(slice.partition.key()),
                            files: positions
                                .map(|position| records.build_record(position))
                                .collect(),
                        }
                    })
                    .collect()
            })
            .collect())
    }
}
