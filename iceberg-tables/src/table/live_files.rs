//! A version's live files by the partition they sit in, read through the
//! parts of the table's data manifests the cache holds
//! ([`manifest_parts`](crate::files::manifest_parts)): the partitions a
//! version has a file in, and the files of one found by their paths.
//!
//! Nothing is kept of a lookup: it reads the files of the partition it
//! asks for, and no column's bounds with them, and a manifest is read and
//! parsed only for a part the cache does not hold.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::spec::{Literal, Transform, Type as IcebergType};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::PartitionValues;
use parquet_engine::file_entry::FileEntry;

use super::Table;
use super::manifest_reader::{ManifestReader, WantedPartitions};
use crate::files::data_files::read_partition_value_types;
use crate::files::manifest_list::ListedManifest;
use crate::files::partition::Partition;
use crate::types::columns::to_pivot_type;
use crate::types::datum::{literal_to_scalar, null_scalar};
use crate::{Error, Result};

/// A live file as the manifest that lists it records it, with that
/// manifest's path.
pub(crate) struct ListedFile {
    pub(crate) file: FileRef,
    /// The path the manifest list gives the manifest that holds the file.
    pub(crate) manifest_path: Arc<str>,
    /// The rows the file holds, deleted ones included.
    pub(crate) record_count: u64,
    /// The id of the table sort order the file's rows are in, as its
    /// manifest entry records it.
    sort_order_id: Option<i32>,
}

impl Table {
    /// The live files of `partition` at `paths`, in the order given: `None`
    /// for a path this version has no file of the partition at. The
    /// partition's files are read through the parts of the manifests that
    /// hold a file of it, and no column's bounds with them.
    ///
    /// A file is looked up by partition and path, and not in a manifest
    /// noted for it earlier, because a file outlives the manifest that
    /// lists it: a commit merges small manifests into one once enough have
    /// accumulated, and a commit that removes a file rewrites the manifest
    /// listing it, whose other files the new manifest then lists. A merge
    /// of files is selected at one version and committed on a later one, so
    /// a manifest noted at selection may be listed no longer while the file
    /// is still live, and a commit that looked only there would give the
    /// merge up. Merges in flight together make that the usual case: the
    /// first to commit rewrites the manifest the others' inputs are listed
    /// in. Asked of the version committed on, the lookup finds each file
    /// under whatever manifest lists it now.
    pub(crate) fn find_files(
        &self,
        partition: &Arc<Partition>,
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<ListedFile>>> {
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.list_data_manifests().iter().collect();
        let mut reader = ManifestReader::new(self, Vec::new());
        let wanted = WantedPartitions::Listed(std::slice::from_ref(partition));
        let matched = reader.match_partitions(&manifests, wanted)?;
        let slices = reader.load_slices(&matched)?;
        let asked: HashMap<&str, usize> = paths
            .iter()
            .enumerate()
            .map(|(position, path)| (path.as_str(), position))
            .collect();
        let mut listed_files: Vec<Option<ListedFile>> = paths.iter().map(|_| None).collect();
        for slice in slices.iter().flatten() {
            let files = &slice.files;
            for position in 0..files.count_files() {
                if let Some(asked_position) = asked.get(files.path(position)) {
                    listed_files[*asked_position] = Some(ListedFile {
                        file: FileRef {
                            path: ObjectPath::new(files.path(position)),
                            size: files.size(position),
                        },
                        manifest_path: Arc::clone(&slice.manifest_path),
                        record_count: files.record_count(position),
                        sort_order_id: files.sort_order_id(position),
                    });
                }
            }
        }
        Ok(listed_files)
    }

    /// The live files at `paths` in `partition`, in the order given, each
    /// as a merge of them reads it: its tuple, and the columns the table is
    /// sorted by as what its rows are sorted by when its manifest entry
    /// records the sort order Pivot's writer produces for them. No
    /// statistics are read. `None` for a path this version has no file of
    /// the partition at.
    pub fn find_file_entries(
        &self,
        partition: &Arc<Partition>,
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<FileEntry>>> {
        let listed_files = self.find_files(partition, paths)?;
        // The partition's values, the same for every file listed, are built
        // only when a file is: a partition that lists none may be of a spec
        // whose values have no type under the current schema.
        let values = if listed_files.iter().any(Option::is_some) {
            self.build_partition_values(partition)?
        } else {
            None
        };
        let build_entry = |listed: ListedFile| {
            let is_in_sort_order =
                listed.sort_order_id.is_some() && listed.sort_order_id == self.writer_sort_order_id;
            let sorted_by: Arc<[String]> = if is_in_sort_order {
                Arc::from(self.sort_by.as_slice())
            } else {
                Arc::default()
            };
            FileEntry {
                file: listed.file,
                partition: values.clone(),
                stats: None,
                sorted_by,
            }
        };
        let entries = listed_files
            .into_iter()
            .map(|listed| listed.map(build_entry));
        Ok(entries.collect())
    }

    /// The tuple of `partition`, a partition of a spec this table lists,
    /// as the datastore keys it: laid out by that spec, its field name to
    /// typed scalar of the declared columns' Pivot types, a NULL value as a
    /// null scalar. `None` for a spec without fields.
    pub(crate) fn build_partition_values(
        &self,
        partition: &Partition,
    ) -> Result<Option<PartitionValues>> {
        let metadata = self.metadata();
        let partition_spec = metadata
            .partition_spec_by_id(partition.spec_id())
            .expect("a partition is read under a spec its table lists");
        let fields = partition_spec.fields();
        if fields.is_empty() {
            return Ok(None);
        }
        let schema = metadata.current_schema();
        let value_types =
            read_partition_value_types(metadata, partition_spec).map_err(|message| {
                Error::UntypedPartitionSpec {
                    table: self.name().to_string(),
                    spec_id: partition.spec_id(),
                    message,
                }
            })?;
        let tuple = partition.tuple().fields();
        let mut values = PartitionValues::new();
        for (position, (field, value_type)) in fields.iter().zip(&value_types).enumerate() {
            // A field's values are kept as its source column's own Pivot
            // type for an identity field, else as the Pivot reading of the
            // transform's result type. A field whose source is not a
            // declared column is left out.
            let pivot_type = match field.transform {
                Transform::Identity => schema
                    .name_by_field_id(field.source_id)
                    .and_then(|name| self.columns().iter().find(|column| column.name == name))
                    .map(|column| column.col_type.clone()),
                _ => to_pivot_type(&IcebergType::Primitive(value_type.clone())),
            };
            let Some(pivot_type) = pivot_type else {
                continue;
            };
            let scalar = match tuple.get(position) {
                Some(Some(Literal::Primitive(literal))) => literal_to_scalar(literal, &pivot_type),
                _ => Some(null_scalar(&pivot_type)),
            };
            if let Some(scalar) = scalar {
                values.insert(field.name.clone(), scalar);
            }
        }
        Ok(Some(values))
    }
}
