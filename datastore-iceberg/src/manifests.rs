//! What a snapshot's metadata says about its manifests and data files, as the
//! statistics that prune them before anything further is read: a manifest by
//! the partition summaries of its manifest-list entry, a data file by its
//! partition values and column bounds.

use crate::Error;
use crate::columns::to_pivot_type;
use crate::pruning::{self, Bounds, Statistic, Statistics, Transform};
use crate::store::TableStore;
use dispatch::DataFlowDispatcher;
use iceberg::Result;
use iceberg::spec::{
    DataFile, Datum, Manifest, ManifestContentType, ManifestFile,
    ManifestList as IcebergManifestList, PartitionSpec, SchemaRef, TableMetadata,
};
use object_storage::load_objects;
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use std::collections::BTreeMap;

mod partition;
mod values;

use values::build_array;

pub(crate) struct ManifestDescriptor {
    pub path: String,
    pub length: u64,
    added_rows_count: Option<u64>,
    existing_rows_count: Option<u64>,
}

/// The snapshot's data manifests, in manifest-list order, with the bounds of
/// the partition values each one holds.
pub(crate) struct ManifestList {
    manifests: Vec<ManifestDescriptor>,
    statistics: Statistics,
}

impl ManifestList {
    /// Read the current snapshot's manifest list, of `size` bytes, and index
    /// its data manifests in order. A table without a snapshot has none. Live
    /// delete files are rejected before pruning can hide them; a delete
    /// manifest whose files were all removed carries nothing to apply.
    pub fn load(
        table: &str,
        metadata: &TableMetadata,
        size: Option<u64>,
        store: &TableStore,
        dispatcher: &DataFlowDispatcher,
    ) -> crate::Result<Self> {
        let mut manifests = Vec::new();
        if let Some(snapshot) = metadata.current_snapshot() {
            let path = snapshot.manifest_list();
            let size = size
                .expect("a table with a current snapshot is indexed with its manifest list's size");
            let bytes = load_objects(dispatcher, &[store.data_file(path, size)?])?
                .pop()
                .expect("load_objects returns one object per file")
                .bytes;
            let list = IcebergManifestList::parse_with_version(&bytes, metadata.format_version())
                .map_err(|source| Error::MalformedMetadataObject {
                table: table.to_string(),
                path: path.to_string(),
                source: Box::new(source),
            })?;
            for entry in list.consume_entries() {
                if entry.content == ManifestContentType::Deletes {
                    if entry.has_added_files() || entry.has_existing_files() {
                        return Err(Error::DeleteFiles {
                            table: table.to_string(),
                            manifest: entry.manifest_path,
                        });
                    }
                    continue;
                }
                manifests.push(entry);
            }
        }
        Self::new(metadata, &manifests).map_err(|source| Error::InvalidPartitionMetadata {
            table: table.to_string(),
            source: Box::new(source),
        })
    }

    /// Index the data manifests a manifest list names.
    fn new(metadata: &TableMetadata, manifests: &[ManifestFile]) -> Result<Self> {
        let mut statistics = Vec::new();
        for spec in metadata.partition_specs_iter() {
            // Each manifest is described by the fields of the spec it was
            // written under, and is unknown to the fields of every other spec.
            let spec_manifests: Vec<Option<&ManifestFile>> = manifests
                .iter()
                .map(|manifest| (manifest.partition_spec_id == spec.spec_id()).then_some(manifest))
                .collect();
            for field in partition::fields(metadata.current_schema(), spec) {
                statistics.push(field.manifest_statistic(&spec_manifests)?);
            }
        }
        Ok(Self {
            statistics: Statistics::new(manifests.len(), statistics),
            manifests: manifests
                .iter()
                .map(|manifest| ManifestDescriptor {
                    path: manifest.manifest_path.clone(),
                    length: manifest.manifest_length.max(0) as u64,
                    added_rows_count: manifest.added_rows_count,
                    existing_rows_count: manifest.existing_rows_count,
                })
                .collect(),
        })
    }

    /// The manifests that may list a file holding a row that satisfies every
    /// filter.
    pub fn select(&self, filters: &[Expression]) -> Vec<&ManifestDescriptor> {
        self.statistics.select(&self.manifests, filters).collect()
    }

    pub fn manifests(&self) -> &[ManifestDescriptor] {
        &self.manifests
    }

    /// Exact for snapshots without deletes. Unknown or overflowing counts make
    /// planning fall back to an ordinary scan.
    pub fn row_count(&self) -> Option<i64> {
        self.manifests.iter().try_fold(0i64, |total, manifest| {
            let rows = manifest
                .added_rows_count?
                .checked_add(manifest.existing_rows_count?)?;
            total.checked_add(i64::try_from(rows).ok()?)
        })
    }
}

/// Where a data file is and how long, which is all a footer read needs.
pub(crate) struct FileDescriptor {
    pub path: String,
    pub length: u64,
}

impl FileDescriptor {
    pub fn of(file: &DataFile) -> Self {
        Self {
            path: file.file_path().to_string(),
            length: file.file_size_in_bytes(),
        }
    }
}

/// The live, non-empty data files of `manifests` that may hold a row satisfying
/// every filter, in manifest order.
pub(crate) fn select_files(
    schema: &SchemaRef,
    manifests: &[Manifest],
    filters: &[Expression],
) -> Result<Vec<FileDescriptor>> {
    let files: Vec<LiveFile> = manifests.iter().flat_map(live_files).collect();
    let statistics = file_statistics(schema, manifests, &files, filters)?;
    Ok(statistics
        .select(&files, filters)
        .filter(|file| file.data.record_count() > 0)
        .map(|file| FileDescriptor::of(file.data))
        .collect())
}

/// A data file a manifest lists as part of the snapshot.
pub(crate) struct LiveFile<'a> {
    pub manifest: &'a Manifest,
    pub data: &'a DataFile,
}

pub(crate) fn live_files(manifest: &Manifest) -> impl Iterator<Item = LiveFile<'_>> {
    manifest
        .entries()
        .iter()
        .filter(|entry| entry.is_alive())
        .map(move |entry| LiveFile {
            manifest,
            data: entry.data_file(),
        })
}

/// The statistics of `files` for the columns `filters` compare, one slot per
/// file: each column's bounds and null count, and the file's value of every
/// partition field derived from the column.
fn file_statistics(
    schema: &SchemaRef,
    manifests: &[Manifest],
    files: &[LiveFile],
    filters: &[Expression],
) -> Result<Statistics> {
    // A manifest bounds whole primitive columns. The fields of a VARIANT
    // column are pruned per row group.
    let columns: Vec<usize> = pruning::columns(&pruning::comparisons(filters))
        .into_iter()
        .filter(|column| column.path.is_empty())
        .map(|column| column.column_idx)
        .collect();
    let mut statistics: Vec<Statistic> = columns
        .iter()
        .filter_map(|&column_idx| {
            Some(Statistic {
                column: column_idx.into(),
                transform: Transform::Identity,
                bounds: column_bounds(schema, column_idx, files)?,
            })
        })
        .collect();
    // Each manifest records the spec its files were written under. A file is
    // described by the fields of that spec, and is unknown to every other's.
    let specs: BTreeMap<i32, &PartitionSpec> = manifests
        .iter()
        .map(|manifest| &manifest.metadata().partition_spec)
        .map(|spec| (spec.spec_id(), spec))
        .collect();
    for (spec_id, spec) in specs {
        let spec_files: Vec<Option<&LiveFile>> = files
            .iter()
            .map(|file| {
                let written_under = file.manifest.metadata().partition_spec.spec_id();
                (written_under == spec_id).then_some(file)
            })
            .collect();
        for field in partition::fields(schema, spec) {
            if columns.contains(&field.column_idx) {
                statistics.push(field.file_statistic(&spec_files)?);
            }
        }
    }
    Ok(Statistics::new(files.len(), statistics))
}

/// The bounds a manifest records for a column of the current schema, or `None`
/// for a column whose type has no bounds to compare.
fn column_bounds(schema: &SchemaRef, column_idx: usize, files: &[LiveFile]) -> Option<Bounds> {
    let field = &schema.as_struct().fields()[column_idx];
    field.field_type.as_primitive_type()?;
    let data_type = physical_arrow_type(&to_pivot_type(&field.field_type)?);
    let id = field.id;
    let lower = files
        .iter()
        .map(|file| file.data.lower_bounds().get(&id).map(Datum::literal));
    let upper = files
        .iter()
        .map(|file| file.data.upper_bounds().get(&id).map(Datum::literal));
    Some(Bounds {
        lower: build_array(&data_type, lower),
        upper: build_array(&data_type, upper),
        // Initial defaults are rejected on load: a column the file's schema
        // lacks is NULL in every row.
        all_null: files
            .iter()
            .map(|file| {
                let null_count = file.data.null_value_counts().get(&id);
                Some(
                    file.manifest.metadata().schema.field_by_id(id).is_none()
                        || null_count == Some(&file.data.record_count()),
                )
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests;
