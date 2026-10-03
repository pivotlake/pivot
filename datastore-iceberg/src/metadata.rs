//! What a snapshot's metadata says about its manifests and data files, as the
//! statistics that prune them before anything further is read: a manifest by
//! the partition summaries of its manifest-list entry, a data file by its
//! partition values and column bounds.

use crate::columns::to_pivot_type;
use crate::values::build_array;
use ::pruning::{Bounds, Predicate, Statistic, Statistics, Transform};
use iceberg::spec::{DataFile, Datum, Manifest, ManifestFile, SchemaRef, TableMetadata};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;

mod partition;

use partition::PartitionSpecs;

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
    pub fn new(metadata: &TableMetadata, manifests: &[ManifestFile]) -> Result<Self> {
        let mut specs = PartitionSpecs::default();
        for manifest in manifests {
            let spec_id = manifest.partition_spec_id;
            let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("unknown partition spec {spec_id}"),
                )
            })?;
            specs.insert(metadata.current_schema(), spec);
        }
        let statistics = specs
            .distinct_fields()
            .into_iter()
            .map(|field| field.manifest_statistic(&specs, manifests))
            .collect::<Result<Vec<_>>>()?;
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
    /// predicate.
    pub fn select(&self, predicates: &[Predicate]) -> Vec<&ManifestDescriptor> {
        self.statistics
            .select(&self.manifests, predicates)
            .collect()
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
/// every predicate, in manifest order.
pub(crate) fn select_files(
    schema: &SchemaRef,
    manifests: &[Manifest],
    predicates: &[Predicate],
) -> Result<Vec<FileDescriptor>> {
    let files: Vec<LiveFile> = manifests.iter().flat_map(live_files).collect();
    let statistics = file_statistics(schema, &files, predicates)?;
    Ok(statistics
        .select(&files, predicates)
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

/// The statistics of `files` for the columns `predicates` read, one slot per
/// file: each column's bounds and null count, and the file's value of every
/// partition field derived from the column.
fn file_statistics(
    schema: &SchemaRef,
    files: &[LiveFile],
    predicates: &[Predicate],
) -> Result<Statistics> {
    // A manifest bounds whole primitive columns. The fields of a VARIANT
    // column are pruned per row group.
    let columns: Vec<usize> = Predicate::columns(predicates)
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
    let mut specs = PartitionSpecs::default();
    for file in files {
        specs.insert(schema, &file.manifest.metadata().partition_spec);
    }
    for field in specs.distinct_fields() {
        if columns.contains(&field.column_idx) {
            statistics.push(field.file_statistic(&specs, files)?);
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
