//! A snapshot's manifests, and which of them a scan reads: a manifest is
//! pruned by the partition summaries of its manifest-list entry. A delete
//! manifest is summarized by the partitions of the data files its deletes
//! apply to, so the pruning that keeps a data file keeps its deletes.
//!
//! The list is all a table version holds of its data files: an entry's
//! summaries bound each partition field's values across the manifest, so
//! they say which manifests may hold a partition. Which partitions a
//! manifest holds they say only for one they leave a single partition, of a
//! spec without fields or with every field bounded to one value
//! (`EntryPartitions`); for any other it is learned by decoding the
//! manifest, and kept nowhere.

use crate::Error;
use crate::files::file_index::Partition;
use crate::files::pruning::{self, FieldRange, ValueRange};
use iceberg::expr::{BoundPredicate, PredicateOperator};
use iceberg::spec::{
    Datum, Literal, ManifestContentType, ManifestFile, ManifestList as IcebergManifestList,
    PartitionSpecRef, PrimitiveLiteral, PrimitiveType, SchemaRef, Struct, TableMetadata,
};
use iceberg::{ErrorKind, Result};
use planner::expression::Expression;
use std::collections::HashMap;
use std::mem::discriminant;
use std::sync::Arc;

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
    /// What the entry says of each partition field of the spec the manifest
    /// was written under, in the spec's field order, as a filter reads it
    /// and as a partition's tuple is checked against it. A field whose
    /// values cannot prune is left out, which is why each summary carries
    /// its field's position in the tuple.
    field_summaries: Vec<PartitionFieldSummary>,
}

/// What a manifest-list entry alone says of the partitions its manifest
/// holds a live file of.
pub(crate) enum EntryPartitions<'m> {
    /// The manifest holds no live file.
    None,
    /// Every live file of the manifest sits in this partition.
    One(OnlyPartition<'m>),
    /// The entry does not tell: the manifest's own entries do.
    Unknown,
}

/// The one partition a manifest-list entry leaves its manifest's files, as
/// the entry's summaries tell it: of a spec without fields, or with every
/// field bounded to a single value. It is read off the summaries each time
/// it is asked for, and nothing is built until the partition itself is.
#[derive(Clone, Copy)]
pub(crate) struct OnlyPartition<'m> {
    spec_id: i32,
    /// A summary for every field of the spec, each leaving its field one
    /// value.
    field_summaries: &'m [PartitionFieldSummary],
}

impl OnlyPartition<'_> {
    /// Whether `other` is this partition: of the same spec, with the same
    /// value in every field.
    pub(crate) fn is_same_as(&self, other: &OnlyPartition<'_>) -> bool {
        self.spec_id == other.spec_id
            && self
                .field_summaries
                .iter()
                .zip(other.field_summaries)
                .all(|(summary, other)| summary.find_only_value() == other.find_only_value())
    }

    /// The partition itself: its spec, and the tuple of the one value each
    /// field is bounded to.
    pub(crate) fn build_partition(&self) -> Partition {
        let tuple = self.field_summaries.iter().map(|summary| {
            summary
                .find_only_value()
                .expect("every summary of an only partition leaves its field one value")
                .map(|value| Literal::Primitive(value.clone()))
        });
        Partition::new(self.spec_id, Struct::from_iter(tuple))
    }
}

/// What a manifest-list entry says of one partition field of its
/// manifest's files.
#[derive(Clone)]
struct PartitionFieldSummary {
    /// The field's place in a partition's tuple.
    position: usize,
    /// The field's id, its lowest and highest non-null values, and whether
    /// every value is NULL.
    range: FieldRange,
    /// Whether any file's value is NULL.
    contains_null: bool,
}

impl ValueRange for PartitionFieldSummary {
    fn field_id(&self) -> i32 {
        self.range.field_id()
    }

    fn may_satisfy(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        self.range.may_satisfy(operator, constant)
    }
}

impl PartitionFieldSummary {
    /// The lowest and the highest non-null value, as a tuple's value is
    /// checked against them; `None` for an entry that records either one
    /// missing, and for a float or a double, whose NaNs the bounds leave
    /// out.
    fn read_comparable_bounds(&self) -> Option<(&PrimitiveLiteral, &PrimitiveLiteral)> {
        let (lowest, highest) = (self.range.lower.as_ref()?, self.range.upper.as_ref()?);
        let is_float = matches!(
            lowest.data_type(),
            PrimitiveType::Float | PrimitiveType::Double
        );
        (!is_float).then(|| (lowest.literal(), highest.literal()))
    }

    /// Whether a file of the manifest may hold `value` in the field, NULL
    /// as `None`.
    fn may_hold(&self, value: Option<&Literal>) -> bool {
        match value {
            None => self.contains_null,
            Some(Literal::Primitive(value)) => {
                let Some((lowest, highest)) = self.read_comparable_bounds() else {
                    return true;
                };
                // Values of one kind order as the field's values do; a
                // value of another kind than its bounds is not compared
                // with them.
                let is_comparable = [lowest, highest]
                    .iter()
                    .all(|limit| discriminant(*limit) == discriminant(value));
                !is_comparable || (lowest <= value && value <= highest)
            }
            Some(_) => true,
        }
    }

    /// The one value the summary leaves the field, NULL as `None`; `None`
    /// when it leaves several, or does not say.
    fn find_only_value(&self) -> Option<Option<&PrimitiveLiteral>> {
        match self.read_comparable_bounds() {
            Some((lowest, highest)) if lowest == highest && !self.contains_null => {
                Some(Some(lowest))
            }
            None if self.range.all_null => Some(None),
            _ => None,
        }
    }
}

impl ListedManifest {
    pub(crate) fn file(&self) -> &ManifestFile {
        &self.file
    }

    /// The manifest's path, which names its content.
    pub(crate) fn path(&self) -> &Arc<str> {
        &self.path
    }

    /// What the entry alone says of the partitions the manifest holds a
    /// live file of, in the table `metadata` describes. An entry that does
    /// not count its manifest's files does not tell whether it holds one.
    pub(crate) fn read_partitions(&self, metadata: &TableMetadata) -> EntryPartitions<'_> {
        let live_files = self
            .file
            .added_files_count
            .zip(self.file.existing_files_count)
            .map(|(added, existing)| u64::from(added) + u64::from(existing));
        match live_files {
            Some(0) => EntryPartitions::None,
            Some(_) => match self.find_only_partition(metadata) {
                Some(partition) => EntryPartitions::One(partition),
                None => EntryPartitions::Unknown,
            },
            None => EntryPartitions::Unknown,
        }
    }

    /// The one partition the entry leaves the manifest's files: that of
    /// the empty tuple for a spec without fields, or of the one value each
    /// field is bounded to. `None` when a field is left more than one value
    /// or none is recorded for it, and for a spec `metadata` does not list.
    fn find_only_partition(&self, metadata: &TableMetadata) -> Option<OnlyPartition<'_>> {
        let partition_spec = metadata.partition_spec_by_id(self.file.partition_spec_id)?;
        if self.field_summaries.len() != partition_spec.fields().len() {
            return None;
        }
        self.field_summaries
            .iter()
            .all(|summary| summary.find_only_value().is_some())
            .then_some(OnlyPartition {
                spec_id: self.file.partition_spec_id,
                field_summaries: &self.field_summaries,
            })
    }

    /// Whether the manifest may hold a file of `partition`, by its entry's
    /// summaries: it was written under the partition's spec, and no summary
    /// rules the tuple's value at its position out.
    pub(crate) fn may_hold_partition(&self, partition: &Partition) -> bool {
        if self.file.partition_spec_id != partition.spec_id() {
            return false;
        }
        let tuple = partition.tuple().fields();
        self.field_summaries.iter().all(|summary| {
            match tuple.get(summary.position) {
                // A tuple shorter than its spec says nothing of the field.
                None => true,
                Some(value) => summary.may_hold(value.as_ref()),
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
    /// partition summaries read.
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
            let field_summaries = read_partition_summaries(metadata, &file)?;
            let manifest = ListedManifest {
                path: Arc::from(file.manifest_path.as_str()),
                file,
                field_summaries,
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

    /// The data manifests, each with what its entry says of it.
    pub(crate) fn listed_data_manifests(&self) -> &[ListedManifest] {
        &self.data_manifests
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
            pruning::may_match(partition_filter, &manifest.field_summaries)
        })
}

/// What the manifest-list entry `manifest` says about the partition values of
/// the manifest's files: a summary for each field of the spec it was written
/// under whose values can prune, in the spec's field order. Nothing when the
/// entry has no summaries, or the table does not list the spec.
fn read_partition_summaries(
    metadata: &TableMetadata,
    manifest: &ManifestFile,
) -> Result<Vec<PartitionFieldSummary>> {
    let (Some(partition_spec), Some(summaries)) = (
        metadata.partition_spec_by_id(manifest.partition_spec_id),
        &manifest.partitions,
    ) else {
        return Ok(Vec::new());
    };
    let mut field_summaries = Vec::new();
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
        // A summary without bounds has no non-null value to bound.
        let is_all_null =
            summary.contains_null && summary.lower_bound.is_none() && summary.upper_bound.is_none();
        field_summaries.push(PartitionFieldSummary {
            position,
            range: FieldRange {
                field_id: field.field_id,
                lower,
                upper,
                all_null: is_all_null,
            },
            contains_null: summary.contains_null,
        });
    }
    Ok(field_summaries)
}

#[cfg(test)]
mod tests;
