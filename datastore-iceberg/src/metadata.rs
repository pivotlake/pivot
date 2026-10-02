//! Manifest and data-file descriptors paired with Arrow statistics in the same
//! order. This adapter resolves field identity; the shared evaluator owns
//! comparisons and partition projection.

use crate::columns::to_pivot_type;
use crate::values::build_array;
use ::pruning::{
    ColumnBounds, ColumnPredicate, ColumnStatistics, PartitionStatistics, StatisticsBatch,
};
use arrow_array::{Array, BooleanArray};
use arrow_schema::ArrowError;
use iceberg::spec::{
    DataFile, Datum, Manifest, ManifestFile, ManifestMetadata, SchemaRef, TableMetadata,
};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;
use std::collections::{BTreeMap, BTreeSet};

mod partition;

pub(crate) struct ManifestDescriptor {
    pub path: String,
    pub length: u64,
    added_rows_count: Option<u64>,
    existing_rows_count: Option<u64>,
}

/// Descriptors and partition bounds are built together in manifest-list order
/// and never reordered. One statistics batch covers every historical spec;
/// expressions absent from a spec are inapplicable for its rows.
pub(crate) struct ManifestList {
    manifests: Vec<ManifestDescriptor>,
    statistics: StatisticsBatch,
}

impl ManifestList {
    pub fn new(metadata: &TableMetadata, manifests: &[ManifestFile]) -> Result<Self> {
        let mut by_spec = BTreeMap::new();
        let spec_ids: BTreeSet<_> = manifests
            .iter()
            .map(|manifest| manifest.partition_spec_id)
            .collect();
        for spec_id in spec_ids {
            let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("unknown partition spec {spec_id}"),
                )
            })?;
            let schema = metadata.current_schema();
            by_spec.insert(spec_id, partition::columns(schema, schema, spec)?);
        }
        let columns = partition::distinct(by_spec.values())
            .into_iter()
            .map(|column| {
                let rows = manifests.iter().map(|manifest| {
                    by_spec[&manifest.partition_spec_id]
                        .iter()
                        .find(|binding| binding.expression == column.expression)
                        .map(|binding| (binding, manifest))
                });
                Ok(PartitionStatistics {
                    expression: column.expression.clone(),
                    bounds: column.manifest_bounds(rows)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let statistics =
            StatisticsBatch::new(manifests.len(), BTreeMap::new(), columns).map_err(arrow_error)?;
        let manifests = manifests
            .iter()
            .map(|manifest| ManifestDescriptor {
                path: manifest.manifest_path.clone(),
                length: manifest.manifest_length.max(0) as u64,
                added_rows_count: manifest.added_rows_count,
                existing_rows_count: manifest.existing_rows_count,
            })
            .collect();
        Ok(Self {
            manifests,
            statistics,
        })
    }

    pub fn select(&self, predicates: &[ColumnPredicate]) -> Result<Vec<&ManifestDescriptor>> {
        let keep = self.statistics.prune(predicates).map_err(arrow_error)?;
        Ok(self
            .manifests
            .iter()
            .zip(keep.values())
            .filter_map(|(manifest, keep)| keep.then_some(manifest))
            .collect())
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

#[derive(Clone)]
pub(crate) struct FileDescriptor {
    pub path: String,
    pub length: u64,
    record_count: u64,
}

/// Live files and their bounds in manifest order. The private fields are built,
/// selected and sliced together, so each descriptor always matches its stats row.
/// SDK objects and their encoded bounds are temporary decoding inputs.
pub(crate) struct DataFiles {
    files: Vec<FileDescriptor>,
    statistics: StatisticsBatch,
}

impl DataFiles {
    pub fn new(schema: &SchemaRef, manifests: &[Manifest]) -> Result<Self> {
        let files: Vec<_> = manifests
            .iter()
            .flat_map(|manifest| {
                manifest
                    .entries()
                    .iter()
                    .filter(|entry| entry.is_alive())
                    .map(|entry| (manifest.metadata(), entry.data_file()))
            })
            .collect();
        let statistics = file_statistics(schema, &files)?;
        let files = files
            .into_iter()
            .map(|(_, file)| FileDescriptor {
                path: file.file_path().to_string(),
                length: file.file_size_in_bytes(),
                record_count: file.record_count(),
            })
            .collect();
        Ok(Self { files, statistics })
    }

    pub fn select(self, predicates: &[ColumnPredicate]) -> Result<Self> {
        let keep = self.statistics.prune(predicates).map_err(arrow_error)?;
        let keep: BooleanArray = self
            .files
            .iter()
            .zip(keep.values())
            .map(|(file, keep)| keep && file.record_count > 0)
            .collect();
        let statistics = self.statistics.filter(&keep).map_err(arrow_error)?;
        let files = self
            .files
            .into_iter()
            .zip(keep.values())
            .filter_map(|(file, keep)| keep.then_some(file))
            .collect();
        Ok(Self { files, statistics })
    }

    pub fn files(&self) -> &[FileDescriptor] {
        &self.files
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            files: self.files[offset..offset + len].to_vec(),
            statistics: self.statistics.slice(offset, len),
        }
    }

    pub fn nan_free_columns(&self, row: usize) -> Vec<usize> {
        self.statistics
            .column_stats()
            .iter()
            .filter_map(|(&column, statistics)| {
                let proof = statistics.primitive()?.nan_free.as_ref()?;
                (proof.is_valid(row) && proof.value(row)).then_some(column)
            })
            .collect()
    }
}

fn file_statistics(
    schema: &SchemaRef,
    files: &[(&ManifestMetadata, &DataFile)],
) -> Result<StatisticsBatch> {
    let mut columns = BTreeMap::new();
    for (index, field) in schema.as_struct().fields().iter().enumerate() {
        // File bounds cover primitive table columns only. VARIANT path bounds
        // belong to the Parquet row-group adapter.
        if field.field_type.as_primitive_type().is_none() {
            continue;
        }
        let Some(pivot_type) = to_pivot_type(&field.field_type) else {
            continue;
        };
        let data_type = physical_arrow_type(&pivot_type);
        let id = field.id;
        let bounds = ColumnBounds {
            validity: None,
            lower: Some(build_array(
                &data_type,
                files
                    .iter()
                    .map(|(_, file)| file.lower_bounds().get(&id).map(Datum::literal)),
            )),
            upper: Some(build_array(
                &data_type,
                files
                    .iter()
                    .map(|(_, file)| file.upper_bounds().get(&id).map(Datum::literal)),
            )),
            // Initial defaults are rejected on load: an absent field is NULL.
            all_null: Some(
                files
                    .iter()
                    .map(|(metadata, file)| {
                        metadata.schema.field_by_id(id).is_none()
                            || file.null_value_counts().get(&id) == Some(&file.record_count())
                    })
                    .collect(),
            ),
            nan_free: Some(
                files
                    .iter()
                    .map(|(_, file)| file.nan_value_counts().get(&id) == Some(&0))
                    .collect(),
            ),
        };
        columns.insert(index, ColumnStatistics::from(bounds));
    }
    let mut by_schema_spec = BTreeMap::new();
    for (metadata, _) in files {
        let key = (
            metadata.schema.schema_id(),
            metadata.partition_spec.spec_id(),
        );
        if let std::collections::btree_map::Entry::Vacant(entry) = by_schema_spec.entry(key) {
            entry.insert(partition::columns(
                schema,
                &metadata.schema,
                &metadata.partition_spec,
            )?);
        }
    }
    let mut partitions = Vec::new();
    for column in partition::distinct(by_schema_spec.values()) {
        let rows = files.iter().map(|(metadata, file)| {
            by_schema_spec[&(
                metadata.schema.schema_id(),
                metadata.partition_spec.spec_id(),
            )]
                .iter()
                .find(|binding| binding.expression == column.expression)
                .map(|binding| (binding, *file))
        });
        let bounds = column.file_bounds(rows)?;
        partitions.push(PartitionStatistics {
            expression: column.expression.clone(),
            bounds,
        });
    }
    StatisticsBatch::new(files.len(), columns, partitions).map_err(arrow_error)
}

fn arrow_error(source: ArrowError) -> Error {
    Error::new(ErrorKind::DataInvalid, "could not evaluate metadata bounds").with_source(source)
}

#[cfg(test)]
mod tests;
