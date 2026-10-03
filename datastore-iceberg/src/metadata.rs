//! What a snapshot's metadata says about its manifests and data files, as the
//! statistics that prune them before anything further is read: a manifest by
//! the partition summaries of its manifest-list entry, a data file by its
//! partition values and column bounds.

use crate::columns::to_pivot_type;
use crate::values::build_array;
use ::pruning::{Bounds, Predicate, Statistic, Statistics, Transform};
use iceberg::spec::{
    DataFile, Datum, Manifest, ManifestFile, PrimitiveType, SchemaRef, TableMetadata,
};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

mod partition;

use partition::PartitionField;

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
        let mut fields_by_spec = BTreeMap::new();
        for manifest in manifests {
            let spec_id = manifest.partition_spec_id;
            if fields_by_spec.contains_key(&spec_id) {
                continue;
            }
            let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("unknown partition spec {spec_id}"),
                )
            })?;
            fields_by_spec.insert(spec_id, partition::fields(metadata.current_schema(), spec));
        }
        let statistics = partition::statistics(
            manifests,
            |manifest| &fields_by_spec[&manifest.partition_spec_id],
            PartitionField::manifest_bounds,
        )?;
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

/// A data file to read: where it is and how long, which is all a footer read
/// needs.
pub(crate) struct FileDescriptor {
    pub path: String,
    pub length: u64,
    /// The filtered columns that may hold a NaN in this file.
    nan_columns: Vec<usize>,
}

impl FileDescriptor {
    pub fn of(file: &DataFile) -> Self {
        Self {
            path: file.file_path().to_string(),
            length: file.file_size_in_bytes(),
            nan_columns: Vec::new(),
        }
    }

    /// The predicates this file's Parquet statistics can answer. Those
    /// statistics skip NaNs, which Pivot orders outside the finite range, so
    /// they say nothing about a column that may hold one.
    pub fn row_group_predicates<'a>(&self, predicates: &'a [Predicate]) -> Cow<'a, [Predicate]> {
        if self.nan_columns.is_empty() {
            return Cow::Borrowed(predicates);
        }
        predicates
            .iter()
            .filter(|predicate| !self.nan_columns.contains(&predicate.column.column_idx))
            .cloned()
            .collect()
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
    let columns = Predicate::columns(predicates);
    Ok(statistics
        .select(&files, predicates)
        .filter(|file| file.data.record_count() > 0)
        .map(|file| FileDescriptor {
            nan_columns: columns
                .iter()
                .map(|column| column.column_idx)
                .filter(|&column_idx| may_hold_nan(schema, column_idx, file.data))
                .collect(),
            ..FileDescriptor::of(file.data)
        })
        .collect())
}

/// Whether `file` may hold a NaN in the column: only a floating-point column
/// can, and not in a file the manifest counts none in.
fn may_hold_nan(schema: &SchemaRef, column_idx: usize, file: &DataFile) -> bool {
    let field = &schema.as_struct().fields()[column_idx];
    let is_floating = matches!(
        field.field_type.as_primitive_type(),
        Some(PrimitiveType::Float | PrimitiveType::Double)
    );
    is_floating && file.nan_value_counts().get(&field.id) != Some(&0)
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
    let mut fields_by_spec = BTreeMap::new();
    for file in files {
        let spec = &file.manifest.metadata().partition_spec;
        fields_by_spec.entry(spec.spec_id()).or_insert_with(|| {
            let mut fields = partition::fields(schema, spec);
            fields.retain(|field| columns.contains(&field.column_idx));
            fields
        });
    }
    statistics.extend(partition::statistics(
        files,
        |file| &fields_by_spec[&file.manifest.metadata().partition_spec.spec_id()],
        PartitionField::file_bounds,
    )?);
    Ok(Statistics::new(files.len(), statistics))
}

/// The bounds a manifest records for a column of the current schema, or `None`
/// for a column whose type has no bounds to compare.
fn column_bounds(schema: &SchemaRef, column_idx: usize, files: &[LiveFile]) -> Option<Bounds> {
    let field = &schema.as_struct().fields()[column_idx];
    field.field_type.as_primitive_type()?;
    let data_type = physical_arrow_type(&to_pivot_type(&field.field_type)?);
    let id = field.id;
    // Iceberg bounds skip NaNs, which Pivot orders outside the finite range,
    // so they do not bound a column that may hold one.
    let bound = |bounds: fn(&DataFile) -> &HashMap<i32, Datum>| {
        build_array(
            &data_type,
            files.iter().map(|file| {
                bounds(file.data)
                    .get(&id)
                    .filter(|_| !may_hold_nan(schema, column_idx, file.data))
                    .map(Datum::literal)
            }),
        )
    };
    Some(Bounds {
        lower: bound(DataFile::lower_bounds),
        upper: bound(DataFile::upper_bounds),
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
