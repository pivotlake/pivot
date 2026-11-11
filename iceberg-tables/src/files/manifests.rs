//! A snapshot's manifests, and which of them a scan reads: a manifest is
//! pruned by the partition summaries of its manifest-list entry. A delete
//! manifest is summarized by the partitions of the data files its deletes
//! apply to, so the pruning that keeps a data file keeps its deletes.
//!
//! The list is all a table version holds of its data files until something
//! asks for a partition: an entry's summaries bound each partition field's
//! values across the manifest, so they say which manifests may hold a
//! partition, never which partitions a manifest holds. That is learned when
//! the manifest is first decoded and kept beside its entry as a
//! [`ManifestSummary`], shared by every version that lists the manifest.

use crate::Error;
use crate::files::file_index::{ManifestSummary, Partition};
use crate::files::pruning::{self, FieldRange};
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    Datum, Literal, ManifestContentType, ManifestFile, ManifestList as IcebergManifestList,
    PartitionSpecRef, PrimitiveLiteral, SchemaRef, TableMetadata,
};
use iceberg::{ErrorKind, Result};
use planner::expression::Expression;
use std::collections::HashMap;
use std::mem::discriminant;
use std::sync::{Arc, OnceLock};

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
pub(crate) struct ListedManifest {
    file: ManifestFile,
    /// The manifest's path, which names its content: a manifest is never
    /// rewritten in place.
    path: Arc<str>,
    partition_ranges: Vec<FieldRange>,
    /// The bounds of `partition_ranges` on the tuple positions they bound,
    /// as a partition's tuple is checked against them.
    tuple_bounds: Vec<TupleBound>,
    /// What decoding the manifest learned of it, once something decoded it.
    summary: Arc<OnceLock<ManifestSummary>>,
}

/// What a manifest-list entry says of one position of its manifest's
/// partition tuples.
#[derive(Clone)]
struct TupleBound {
    position: usize,
    /// The lowest and the highest non-null value; `None` for an entry that
    /// records none, and for a float, whose NaNs the bounds leave out.
    range: Option<(PrimitiveLiteral, PrimitiveLiteral)>,
    contains_null: bool,
}

impl ListedManifest {
    pub(crate) fn file(&self) -> &ManifestFile {
        &self.file
    }

    /// The manifest's path, which names its content.
    pub(crate) fn path(&self) -> &Arc<str> {
        &self.path
    }

    /// What decoding the manifest learned of it; `None` until something
    /// decoded it.
    pub(crate) fn summary(&self) -> Option<&ManifestSummary> {
        self.summary.get()
    }

    /// Keep `summary`, what a decode of the manifest learned of it.
    pub(crate) fn learn(&self, summary: ManifestSummary) {
        // A manifest never changes, so a summary set meanwhile is this one.
        let _ = self.summary.set(summary);
    }

    /// Whether the manifest may hold a file of `partition`, by its entry's
    /// summaries: it was written under the partition's spec, and no summary
    /// rules the tuple's value at its position out.
    pub(crate) fn may_hold_partition(&self, partition: &Partition) -> bool {
        if self.file.partition_spec_id != partition.spec_id() {
            return false;
        }
        let tuple = partition.tuple().fields();
        self.tuple_bounds.iter().all(|bound| {
            match tuple.get(bound.position) {
                // A tuple shorter than its spec says nothing of the field.
                None => true,
                Some(None) => bound.contains_null,
                Some(Some(Literal::Primitive(value))) => {
                    let Some((lowest, highest)) = &bound.range else {
                        return true;
                    };
                    // Values of one kind order as the field's values do;
                    // a value of another kind than its bounds is not
                    // compared with them.
                    let is_comparable = [lowest, highest]
                        .iter()
                        .all(|limit| discriminant(*limit) == discriminant(value));
                    !is_comparable || (lowest <= value && value <= highest)
                }
                Some(Some(_)) => true,
            }
        })
    }
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
            let (partition_ranges, tuple_bounds) = read_partition_summaries(metadata, &file)?;
            let manifest = ListedManifest {
                path: Arc::from(file.manifest_path.as_str()),
                file,
                partition_ranges,
                tuple_bounds,
                summary: Arc::default(),
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

    /// The data manifests whose files may hold a row that satisfies every
    /// one of `row_filters`, filters bound to the table's current schema.
    pub(crate) fn select_data_manifests(
        &self,
        row_filters: &[BoundPredicate],
    ) -> Result<Vec<&ListedManifest>> {
        let partition_filters =
            pruning::project_filters_per_spec(&self.schema, &self.partition_specs, row_filters)?;
        Ok(self
            .data_manifests
            .iter()
            .filter(|manifest| may_manifest_match(manifest, &partition_filters))
            .collect())
    }

    /// The data manifests, each with what its entry and its decodes say of
    /// it.
    pub(crate) fn listed_data_manifests(&self) -> &[ListedManifest] {
        &self.data_manifests
    }

    /// Take over what `previous`, another version's list, learned of the
    /// data manifests both lists name.
    pub(crate) fn share_summaries_with(&mut self, previous: &ManifestList) {
        let known: HashMap<&str, &ListedManifest> = previous
            .data_manifests
            .iter()
            .map(|manifest| (manifest.path.as_ref(), manifest))
            .collect();
        for manifest in &mut self.data_manifests {
            if let Some(earlier) = known.get(manifest.path.as_ref()) {
                manifest.path = Arc::clone(&earlier.path);
                manifest.summary = Arc::clone(&earlier.summary);
            }
        }
    }

    /// The data manifests.
    pub fn data_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.data_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The delete manifests with a file still live.
    pub fn delete_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.delete_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The live data files, as the entries of the data manifests count
    /// them; `None` when an entry counts none, which a manifest list of
    /// format version 1 may leave out.
    pub fn count_data_files(&self) -> Option<u64> {
        self.data_manifests().try_fold(0u64, |total, manifest| {
            let files = manifest
                .added_files_count?
                .checked_add(manifest.existing_files_count?)?;
            total.checked_add(u64::from(files))
        })
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
        .filter(|manifest| may_manifest_match(manifest, partition_filters))
        .map(|manifest| &manifest.file)
        .collect()
}

/// Whether the partition summaries of `manifest` may match the filter
/// projected onto its spec, in `partition_filters`; a manifest of a spec
/// with no projected filter may.
fn may_manifest_match(
    manifest: &ListedManifest,
    partition_filters: &HashMap<i32, BoundPredicate>,
) -> bool {
    partition_filters
        .get(&manifest.file.partition_spec_id)
        .is_none_or(|partition_filter| {
            pruning::may_match(partition_filter, &manifest.partition_ranges)
        })
}

/// What the manifest-list entry `manifest` says about the partition values of
/// the manifest's files: a range for each field of the spec it was written
/// under whose values can prune, as a filter reads them and as a partition's
/// tuple is checked against. Nothing when the entry has no summaries, or the
/// table does not list the spec.
fn read_partition_summaries(
    metadata: &TableMetadata,
    manifest: &ManifestFile,
) -> Result<(Vec<FieldRange>, Vec<TupleBound>)> {
    let (Some(partition_spec), Some(summaries)) = (
        metadata.partition_spec_by_id(manifest.partition_spec_id),
        &manifest.partitions,
    ) else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut ranges = Vec::new();
    let mut tuple_bounds = Vec::new();
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
        let (lower, upper) = (decode(&summary.lower_bound)?, decode(&summary.upper_bound)?);
        let is_float = matches!(
            value_type,
            iceberg::spec::PrimitiveType::Float | iceberg::spec::PrimitiveType::Double
        );
        tuple_bounds.push(TupleBound {
            position,
            range: match (&lower, &upper) {
                (Some(lower), Some(upper)) if !is_float => {
                    Some((lower.literal().clone(), upper.literal().clone()))
                }
                _ => None,
            },
            contains_null: summary.contains_null,
        });
        ranges.push(FieldRange {
            field_id: field.field_id,
            lower,
            upper,
            // A summary without bounds has no non-null value to bound.
            all_null: summary.contains_null
                && summary.lower_bound.is_none()
                && summary.upper_bound.is_none(),
        });
    }
    Ok((ranges, tuple_bounds))
}

#[cfg(test)]
mod tests;
