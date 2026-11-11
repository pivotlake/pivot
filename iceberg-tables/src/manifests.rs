//! A snapshot's manifests, and which of them a scan reads: a manifest is
//! pruned by the partition summaries of its manifest-list entry. A delete
//! manifest is summarized by the partitions of the data files its deletes
//! apply to, so the pruning that keeps a data file keeps its deletes.

use crate::Error;
use crate::pruning::{self, FieldRange};
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    DataFile, Datum, ManifestContentType, ManifestFile, ManifestList as IcebergManifestList,
    PartitionSpecRef, SchemaRef, TableMetadata,
};
use iceberg::{ErrorKind, Result};
use planner::expression::Expression;
use std::collections::HashMap;

/// The snapshot's manifests, in manifest-list order: its data manifests, and
/// the delete manifests with a file still live.
#[derive(Clone)]
pub struct ManifestList {
    data_manifests: Vec<ListedManifest>,
    delete_manifests: Vec<ListedManifest>,
    schema: SchemaRef,
    partition_specs: Vec<PartitionSpecRef>,
}

/// A manifest the list names, with what its entry says about the partition
/// values of the manifest's files.
#[derive(Clone)]
struct ListedManifest {
    file: ManifestFile,
    partition_ranges: Vec<FieldRange>,
}

impl ManifestList {
    /// The manifest list of a table without a snapshot: no manifests.
    pub fn empty(metadata: &TableMetadata) -> Self {
        Self::new(metadata, Vec::new()).expect("no manifest has a summary to misread")
    }

    /// The current snapshot's manifest list, decoded from `bytes`, its
    /// manifests kept in order. A delete manifest whose files were all
    /// removed carries nothing to apply and is left out.
    pub fn parse(table: &str, metadata: &TableMetadata, bytes: &[u8]) -> crate::Result<Self> {
        let path = metadata
            .current_snapshot()
            .expect("a manifest list is parsed for a table with a current snapshot")
            .manifest_list();
        let list = IcebergManifestList::parse_with_version(bytes, metadata.format_version())
            .map_err(|source| Error::MalformedMetadataObject {
                table: table.to_string(),
                path: path.to_string(),
                source: Box::new(source),
            })?;
        let mut manifests = Vec::new();
        for entry in list.consume_entries() {
            // A delete manifest whose delete files were all removed has
            // nothing left to apply; skipping it saves reading it.
            if entry.content == ManifestContentType::Deletes
                && !entry.has_added_files()
                && !entry.has_existing_files()
            {
                continue;
            }
            manifests.push(entry);
        }
        Self::listing(table, metadata, manifests)
    }

    /// `manifests`, the current snapshot's, kept in order with their
    /// partition summaries read: what a commit that wrote them hands back.
    pub(crate) fn listing(
        table: &str,
        metadata: &TableMetadata,
        manifests: Vec<ManifestFile>,
    ) -> crate::Result<Self> {
        Self::new(metadata, manifests).map_err(|source| Error::InvalidPartitionMetadata {
            table: table.to_string(),
            source: Box::new(source),
        })
    }

    /// Keep `manifests`, data and delete ones apart, each with its partition
    /// summaries read once.
    fn new(metadata: &TableMetadata, manifests: Vec<ManifestFile>) -> Result<Self> {
        let mut listed = Vec::new();
        let mut listed_deletes = Vec::new();
        for file in manifests {
            let partition_ranges = read_partition_summaries(metadata, &file)?;
            let manifest = ListedManifest {
                file,
                partition_ranges,
            };
            match manifest.file.content {
                ManifestContentType::Data => listed.push(manifest),
                ManifestContentType::Deletes => listed_deletes.push(manifest),
            }
        }
        Ok(Self {
            data_manifests: listed,
            delete_manifests: listed_deletes,
            schema: metadata.current_schema().clone(),
            partition_specs: metadata.partition_specs_iter().cloned().collect(),
        })
    }

    /// The delete manifests whose deletes may apply to a file holding a row
    /// that satisfies every filter.
    pub fn select_delete_manifests(&self, filters: &[Expression]) -> Result<Vec<&ManifestFile>> {
        let row_filters = pruning::bind_filters(&self.schema, filters);
        let partition_filters =
            pruning::project_filters_per_spec(&self.schema, &self.partition_specs, &row_filters)?;
        Ok(select_manifests(&self.delete_manifests, &partition_filters))
    }

    /// The data manifests.
    pub fn data_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.data_manifests.iter().map(|manifest| &manifest.file)
    }

    /// Every manifest: the data manifests, then the delete manifests with a
    /// file still live.
    pub fn manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.data_manifests().chain(self.delete_manifests())
    }

    /// The delete manifests with a file still live.
    pub fn delete_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.delete_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The rows of the data files, deleted ones included. Unknown or
    /// overflowing counts make planning fall back to an ordinary scan.
    pub fn row_count(&self) -> Option<i64> {
        self.data_manifests().try_fold(0i64, |total, manifest| {
            let rows = manifest
                .added_rows_count?
                .checked_add(manifest.existing_rows_count?)?;
            total.checked_add(i64::try_from(rows).ok()?)
        })
    }

    /// The rows of the data files less the rows the delete manifests' entries
    /// count, from the manifest list alone: exact when every delete file is a
    /// deletion vector, an estimate otherwise, never below zero.
    pub fn estimated_row_count(&self) -> Option<i64> {
        let deleted = self.delete_manifests().try_fold(0i64, |total, manifest| {
            let rows = manifest
                .added_rows_count?
                .checked_add(manifest.existing_rows_count?)?;
            total.checked_add(i64::try_from(rows).ok()?)
        })?;
        Some(self.row_count()?.saturating_sub(deleted).max(0))
    }
}

/// The manifests among `manifests` whose partition summaries may match the
/// filter projected onto their spec, in `partition_filters`; a manifest of a
/// spec with no projected filter is kept.
fn select_manifests<'a>(
    manifests: &'a [ListedManifest],
    partition_filters: &HashMap<i32, BoundPredicate>,
) -> Vec<&'a ManifestFile> {
    manifests
        .iter()
        .filter(|manifest| {
            partition_filters
                .get(&manifest.file.partition_spec_id)
                .is_none_or(|partition_filter| {
                    pruning::may_match(partition_filter, &manifest.partition_ranges)
                })
        })
        .map(|manifest| &manifest.file)
        .collect()
}

/// What the manifest-list entry `manifest` says about the partition values of
/// the manifest's files: a range for each field of the spec it was written
/// under whose values can prune. Nothing when the entry has no summaries, or
/// the table does not list the spec.
fn read_partition_summaries(
    metadata: &TableMetadata,
    manifest: &ManifestFile,
) -> Result<Vec<FieldRange>> {
    let (Some(partition_spec), Some(summaries)) = (
        metadata.partition_spec_by_id(manifest.partition_spec_id),
        &manifest.partitions,
    ) else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for (position, field) in partition_spec.fields().iter().enumerate() {
        let Some(value_type) = pruning::find_partition_value_type(metadata.current_schema(), field)
        else {
            continue;
        };
        let summary = summaries.get(position).ok_or_else(|| {
            iceberg::Error::new(
                ErrorKind::DataInvalid,
                "manifest summary does not match its partition spec",
            )
        })?;
        let decode = |bound: &Option<iceberg::spec::ByteBuf>| {
            bound
                .as_ref()
                .map(|bytes| Datum::try_from_bytes(bytes, value_type.clone()))
                .transpose()
        };
        ranges.push(FieldRange {
            field_id: field.field_id,
            lower: decode(&summary.lower_bound)?,
            upper: decode(&summary.upper_bound)?,
            // A summary without bounds has no non-null value to bound.
            all_null: summary.contains_null
                && summary.lower_bound.is_none()
                && summary.upper_bound.is_none(),
        });
    }
    Ok(ranges)
}

/// Where a data file is and how long, which is all a footer read needs.
pub struct FileDescriptor {
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

#[cfg(test)]
mod tests;
