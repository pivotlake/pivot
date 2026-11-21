//! A snapshot's manifests, and which of them a scan reads: a manifest is
//! pruned by the partition summaries of its manifest-list entry. A delete
//! manifest is summarized by the partitions of the data files its deletes
//! apply to, so the pruning that keeps a data file keeps its deletes.
//!
//! The list is all a table version holds of its data files: an entry's
//! summaries bound each partition field's values across the manifest, so
//! they say which manifests may hold a partition. Which partitions a
//! manifest holds a live file of they say only for one whose entry counts no
//! live file, and for one they leave a single partition, of a spec without
//! fields or with every field bounded to one value: those are derived from
//! the entry when asked ([`ListedManifest::find_listed_partitions`]). For
//! any other they are learned by decoding the manifest, and the list keeps
//! none.
//!
//! An entry's bounds are read in the type a field's values have under the
//! table's current schema, and a manifest's own entries hold them in the
//! type they had when the manifest was written: a partition derived from an
//! entry is the one the manifest holds all the same, since a partition is
//! told by its values whatever width they are stored in.

use crate::Error;
use crate::files::manifest_parts::ManifestPartitions;
use crate::files::partition::Partition;
use crate::files::pruning::{self, FieldRange, ValueRange};
use iceberg::expr::{BoundPredicate, PredicateOperator};
use iceberg::spec::{
    Datum, Literal, ManifestContentType, ManifestFile, ManifestList as IcebergManifestList,
    PrimitiveLiteral, PrimitiveType, Struct, TableMetadata,
};
use iceberg::{ErrorKind, Result};
use std::mem::discriminant;
use std::sync::Arc;

/// The snapshot's manifests, in manifest-list order: its data manifests, and
/// the delete manifests with a file still live.
#[derive(Clone)]
pub struct ManifestList {
    data_manifests: Vec<ListedManifest>,
    delete_manifests: Vec<ListedManifest>,
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

    fn may_match_comparison(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        self.range.may_match_comparison(operator, constant)
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
    fn find_only_value(&self) -> Option<Option<Literal>> {
        match self.read_comparable_bounds() {
            Some((lowest, highest)) if lowest == highest && !self.contains_null => {
                Some(Some(Literal::Primitive(lowest.clone())))
            }
            None if self.range.all_null => Some(None),
            _ => None,
        }
    }
}

impl ListedManifest {
    /// The manifest `file` names in the manifest list of the table
    /// `metadata` describes, with what the entry says about the partition
    /// values of the manifest's files: a summary for each field of the spec
    /// it was written under whose values can prune, in the spec's field
    /// order. None when the entry has no summaries, or the table does not
    /// list the spec.
    fn new(metadata: &TableMetadata, file: ManifestFile) -> Result<Self> {
        let mut field_summaries = Vec::new();
        if let (Some(partition_spec), Some(summaries)) = (
            metadata.partition_spec_by_id(file.partition_spec_id),
            &file.partitions,
        ) {
            for (position, field) in partition_spec.fields().iter().enumerate() {
                let Some(value_type) =
                    pruning::find_partition_value_type(metadata.current_schema(), field)
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
                // No bound covers a NaN, which only a float or a double
                // holds: a summary of one that does not say it holds none
                // is read as holding one.
                let may_hold_nan =
                    matches!(value_type, PrimitiveType::Float | PrimitiveType::Double)
                        && summary.contains_nan != Some(false);
                // A summary without bounds has no non-null value to bound,
                // besides a NaN.
                let is_all_null = summary.contains_null
                    && summary.lower_bound.is_none()
                    && summary.upper_bound.is_none()
                    && !may_hold_nan;
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
        }
        Ok(Self {
            path: Arc::from(file.manifest_path.as_str()),
            file,
            field_summaries,
        })
    }

    pub(crate) fn file(&self) -> &ManifestFile {
        &self.file
    }

    /// The manifest's path, which names its content.
    pub(crate) fn path(&self) -> &Arc<str> {
        &self.path
    }

    /// The live files the entry counts in the manifest; `None` for an entry
    /// that does not count them.
    fn count_live_files(&self) -> Option<u64> {
        let file = &self.file;
        Some(u64::from(file.added_files_count?) + u64::from(file.existing_files_count?))
    }

    /// The partitions the manifest holds a live file of, in the table
    /// `metadata` describes, when the entry alone tells them: none for an
    /// entry that counts no live file, else the one partition the entry
    /// leaves the live files, that of the empty tuple for a spec without
    /// fields, or of the one value each field is bounded to. `None` when
    /// the entry does not count its manifest's files, when a field is left
    /// more than one value or none is recorded for it, and for a spec
    /// `metadata` does not list: the manifest's own entries tell then.
    pub(crate) fn find_listed_partitions(
        &self,
        metadata: &TableMetadata,
    ) -> Option<ManifestPartitions> {
        if self.count_live_files()? == 0 {
            return Some(ManifestPartitions::new(Vec::new()));
        }
        let spec_id = self.file.partition_spec_id;
        let partition_spec = metadata.partition_spec_by_id(spec_id)?;
        if self.field_summaries.len() != partition_spec.fields().len() {
            return None;
        }
        let tuple = self
            .field_summaries
            .iter()
            .map(PartitionFieldSummary::find_only_value)
            .collect::<Option<Vec<_>>>()?;
        let only_partition = Arc::new(Partition::new(spec_id, Struct::from_iter(tuple)));
        Some(ManifestPartitions::new(vec![only_partition]))
    }

    /// Whether the entry's partition summaries may match
    /// `partition_filter`, what a scan's filters require of a tuple of the
    /// spec the manifest was written under.
    pub(crate) fn may_match_predicate(&self, partition_filter: &BoundPredicate) -> bool {
        pruning::may_match_predicate(partition_filter, &self.field_summaries)
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
        // A delete manifest whose delete files were all removed has nothing
        // left to apply; skipping it saves reading it.
        let has_nothing_to_apply = |entry: &ManifestFile| {
            entry.content == ManifestContentType::Deletes
                && !entry.has_added_files()
                && !entry.has_existing_files()
        };
        let entries = list.consume_entries().into_iter();
        let manifests = entries
            .filter(|entry| !has_nothing_to_apply(entry))
            .collect();
        Self::new(metadata, manifests).map_err(|source| Error::InvalidPartitionMetadata {
            table: table.to_string(),
            source: Box::new(source),
        })
    }

    /// Keep `manifests`, data and delete ones apart, each with its partition
    /// summaries read once.
    fn new(metadata: &TableMetadata, manifests: Vec<ManifestFile>) -> Result<Self> {
        let mut data_manifests = Vec::new();
        let mut delete_manifests = Vec::new();
        for file in manifests {
            let manifest = ListedManifest::new(metadata, file)?;
            match manifest.file.content {
                ManifestContentType::Data => data_manifests.push(manifest),
                ManifestContentType::Deletes => delete_manifests.push(manifest),
            }
        }
        Ok(Self {
            data_manifests,
            delete_manifests,
        })
    }

    /// The data manifests, each with what the list derives from its entry
    /// for selecting through it ([`ListedManifest`]): its shared path and
    /// its partition summaries. [`data_manifests`](Self::data_manifests)
    /// hands the same manifests over as the bare entries.
    pub(crate) fn list_data_manifests(&self) -> &[ListedManifest] {
        &self.data_manifests
    }

    /// The delete manifests with a file still live, each with what the
    /// list derives from its entry for selecting through it
    /// ([`ListedManifest`]). [`delete_manifests`](Self::delete_manifests)
    /// hands the same manifests over as the bare entries.
    pub(crate) fn list_delete_manifests(&self) -> &[ListedManifest] {
        &self.delete_manifests
    }

    /// The entries of the data manifests as the manifest list wrote them,
    /// for whoever reads a manifest or names its file, outside the crate
    /// too. [`list_data_manifests`](Self::list_data_manifests) hands the
    /// same manifests over with what selecting through them needs.
    pub fn data_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.data_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The entries of the delete manifests with a file still live, as the
    /// manifest list wrote them.
    /// [`list_delete_manifests`](Self::list_delete_manifests) hands the
    /// same manifests over with what selecting through them needs.
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
    pub fn count_rows(&self) -> Option<i64> {
        sum_live_rows(self.data_manifests())
    }

    /// The rows of the data files less the rows the delete manifests' entries
    /// count, from the manifest list alone: exact when every delete file is a
    /// deletion vector, an estimate otherwise, never below zero.
    pub fn estimate_row_count(&self) -> Option<i64> {
        let deleted = sum_live_rows(self.delete_manifests())?;
        Some(self.count_rows()?.saturating_sub(deleted).max(0))
    }
}

/// The rows the entries `manifests` count in their manifests' live files;
/// `None` when one counts none, or the total overflows.
fn sum_live_rows<'m>(mut manifests: impl Iterator<Item = &'m ManifestFile>) -> Option<i64> {
    manifests.try_fold(0i64, |total, manifest| {
        let rows = manifest
            .added_rows_count?
            .checked_add(manifest.existing_rows_count?)?;
        total.checked_add(i64::try_from(rows).ok()?)
    })
}

#[cfg(test)]
mod tests;
