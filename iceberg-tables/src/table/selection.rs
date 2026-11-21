//! Which files a scan reads, selected through the parts of the table's data
//! manifests the cache holds ([`manifest_reader`](super::manifest_reader)),
//! of which a scan reads the columns its filters compare.
//!
//! A scan's files come in the order its version fixes and nothing sorts:
//! partition by partition in key order, a partition's manifests in
//! manifest-list order, and a manifest's files as it lists them.
//!
//! What a scan's filters require is resolved once (`FileSelector`): of a
//! manifest by its manifest-list entry, of a partition by its tuple, and of
//! a file by the bounds its manifest records for the columns the filters
//! compare. The filters are bound to the table's schema once, and
//! projected once onto each partition spec the table lists. Filters that
//! leave every partition field only a few values name the scan's
//! partitions themselves, with no manifest read.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use iceberg::expr::{BoundPredicate, PredicateOperator};
use iceberg::spec::{Datum, Schema, TableMetadata};
use object_storage::ObjectPath;
use planner::expression::Expression;

use super::manifest_reader::{ManifestReader, PartitionSlice};
use super::{SelectedFiles, Table};
use crate::files::manifest_parts::ManifestFiles;
use crate::files::manifests::ListedManifest;
use crate::files::partition::Partition;
use crate::files::partitions_from_filter::{
    MAX_PARTITIONS_FROM_FILTER, enumerate_tuples_from_filter,
};
use crate::files::pruning::{self, FieldRange, PartitionPruner, StoredColumn};
use crate::{Error, Result};

/// Which declared columns may hold a NULL in the files a scan selected.
struct SelectedNulls {
    /// The field id of each declared column; `None` for a column the
    /// table's schema gives none.
    field_ids: Vec<Option<i32>>,
    /// Whether the table's schema lets a file hold a NULL in each declared
    /// column.
    declared_nullability: Vec<bool>,
    /// Whether some selected file may hold a NULL in each declared column.
    may_hold_nulls: Vec<bool>,
    is_any_file_selected: bool,
}

impl SelectedNulls {
    fn new(table: &Table) -> Self {
        let schema = table.metadata().current_schema();
        Self {
            field_ids: table
                .columns()
                .iter()
                .map(|column| schema.field_id_by_name(&column.name))
                .collect(),
            declared_nullability: table.declared_nullability(),
            may_hold_nulls: vec![false; table.columns().len()],
            is_any_file_selected: false,
        }
    }

    /// Count the files of `files` whose bit `is_selected` has set among
    /// the selected: a column may hold a NULL in one unless its manifest
    /// entry counts none, which an entry of a file written without the
    /// column does not. A column the table's schema lets no file hold a
    /// NULL in is not looked at.
    fn observe(&mut self, files: &ManifestFiles, is_selected: &BooleanBuffer) {
        if is_selected.count_set_bits() == 0 {
            return;
        }
        self.is_any_file_selected = true;
        let columns = self.field_ids.iter().zip(&self.declared_nullability);
        for ((field_id, is_declared_nullable), may_hold_nulls) in
            columns.zip(&mut self.may_hold_nulls)
        {
            if *may_hold_nulls || !is_declared_nullable {
                continue;
            }
            *may_hold_nulls = match field_id.and_then(|field_id| files.find_stats_field(field_id)) {
                None => true,
                Some(field_position) => files.may_hold_nulls(is_selected, field_position),
            };
        }
    }

    /// Whether each declared column may hold a NULL in a selected file.
    fn into_nullability(self) -> Vec<bool> {
        self.declared_nullability
            .iter()
            .zip(self.may_hold_nulls)
            .map(|(is_declared_nullable, may_hold_nulls)| {
                *is_declared_nullable && (may_hold_nulls || !self.is_any_file_selected)
            })
            .collect()
    }
}

impl Table {
    /// What a scan over the filters of `selector` selects: the files it reads, each located
    /// for reading, those that the files' partition values and column
    /// bounds cannot rule out, by partition in key order, a partition's
    /// manifests in manifest-list order and a manifest's files as it lists
    /// them; the parts of the table's manifests they were selected
    /// through; and whether each declared column may hold a NULL in one of
    /// them.
    ///
    /// Only the partitions the filters keep are read. Filters that leave
    /// every partition field only a few values name those partitions
    /// themselves. Otherwise they are those held by the manifests whose
    /// manifest-list entries the filters cannot rule out. A manifest is
    /// read and parsed only for a part of it the cache does not hold.
    pub(crate) fn select_scan_files(&self, selector: &FileSelector) -> Result<SelectedFiles> {
        let mut reader = ManifestReader::new(self, selector.compared_field_ids().collect());
        let partitions = self.select_partitions(selector, &mut reader)?;
        let partition_slices = reader.load_slices(&partitions)?;

        let mut files = Vec::new();
        let mut nulls = SelectedNulls::new(self);
        for slice in partition_slices.iter().flatten() {
            let is_selected = selector.select_files(slice);
            nulls.observe(&slice.files, &is_selected);
            for position in is_selected.set_indices() {
                files.push(self.location.data_file_at(
                    &ObjectPath::new(slice.files.path(position)),
                    slice.files.size(position),
                )?);
            }
        }
        tracing::debug!(
            table = %self.name,
            partitions = partitions.len(),
            files = files.len(),
            "selected Iceberg files"
        );
        Ok(SelectedFiles {
            files,
            read_parts: reader.read_parts,
            nullability: nulls.into_nullability(),
        })
    }

    /// The partitions whose tuple the filters of `selector` keep, in key
    /// order: those computed from the filters alone
    /// ([`compute_partitions_from_filter`](Self::compute_partitions_from_filter)),
    /// or else those held by the data manifests whose manifest-list entries
    /// the filters cannot rule out. A manifest parsed to learn its
    /// partitions leaves the files of the kept ones with `reader`.
    fn select_partitions(
        &self,
        selector: &FileSelector,
        reader: &mut ManifestReader<'_>,
    ) -> Result<Vec<Arc<Partition>>> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let manifests = selector.select_manifests(self.manifest_list.listed_data_manifests());
        let mut tuple_values = Vec::new();
        let mut keeps = |partition: &Partition| {
            selector
                .may_partition_match(partition, &mut tuple_values)
                .map_err(invalid_partition_metadata)
        };
        if let Some(from_filter) =
            self.compute_partitions_from_filter(selector, &manifests, &mut keeps)?
        {
            return Ok(from_filter);
        }
        let partitions = reader.list_partitions(&manifests, &mut keeps)?;
        let mut kept = Vec::with_capacity(partitions.len());
        for partition in partitions {
            if keeps(&partition)? {
                kept.push(partition);
            }
        }
        Ok(kept)
    }

    /// The partitions a scan over the filters of `selector` may read a
    /// file of, in key order, computed from the filters
    /// with no manifest read: of the combinations of the values the filters
    /// leave the partition fields, under each spec `manifests` were written
    /// under, those whose tuple `keeps` accepts and that the manifest-list
    /// entry of one of `manifests`, the data manifests the filters cannot
    /// rule out, admits.
    /// One of them may hold no file. `None` when the filters leave a field
    /// of one of the specs unrestricted, or leave more than
    /// [`MAX_PARTITIONS_FROM_FILTER`] combinations: the partitions are then
    /// listed from the manifests.
    fn compute_partitions_from_filter(
        &self,
        selector: &FileSelector,
        manifests: &[&ListedManifest],
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
    ) -> Result<Option<Vec<Arc<Partition>>>> {
        let mut spec_ids: Vec<i32> = manifests
            .iter()
            .map(|manifest| manifest.file().partition_spec_id)
            .collect();
        spec_ids.sort_unstable();
        spec_ids.dedup();
        let Some(from_filter) = selector.enumerate_partitions_from_filter(
            self.metadata(),
            &spec_ids,
            MAX_PARTITIONS_FROM_FILTER,
        ) else {
            return Ok(None);
        };
        let mut partitions = BTreeMap::new();
        for partition in from_filter {
            let is_admitted = manifests
                .iter()
                .any(|manifest| manifest.may_hold_partition(&partition));
            if is_admitted && keeps(&partition)? {
                partitions.insert(Arc::clone(partition.key()), Arc::new(partition));
            }
        }
        Ok(Some(partitions.into_values().collect()))
    }
}

/// What a scan's filters require of a table's partitions and files,
/// resolved once for all of them: a file is described by the bounds its
/// manifest records for its columns, and by the partition values of the
/// spec it was written under when the table lists that spec.
pub(crate) struct FileSelector {
    row_filters: Vec<BoundPredicate>,
    /// What the filters require of a tuple of each partition spec the table
    /// lists, in spec id order.
    partition_pruners: Vec<(i32, PartitionPruner)>,
    /// How the files of each schema the table lists record the columns the
    /// filters compare, in schema id order.
    columns_per_schema: Vec<(i32, Vec<StoredColumn>)>,
    columns_of_unlisted_schema: Vec<StoredColumn>,
}

impl FileSelector {
    /// What `filters` require of the files of the table `metadata`
    /// describes.
    pub(crate) fn new(metadata: &TableMetadata, filters: &[Expression]) -> iceberg::Result<Self> {
        let schema = metadata.current_schema();
        let row_filters = pruning::bind_filters(schema, filters);
        let mut partition_pruners = Vec::new();
        for partition_spec in metadata.partition_specs_iter() {
            let pruner = PartitionPruner::new(schema, partition_spec, &row_filters)?;
            partition_pruners.push((partition_spec.spec_id(), pruner));
        }
        partition_pruners.sort_unstable_by_key(|(spec_id, _)| *spec_id);
        let compared_columns = pruning::list_compared_fields(&row_filters);
        let resolve_columns = |file_schema: Option<&Schema>| -> Vec<StoredColumn> {
            compared_columns
                .iter()
                .map(|column| StoredColumn::resolve(column, file_schema))
                .collect()
        };
        let mut columns_per_schema: Vec<(i32, Vec<StoredColumn>)> = metadata
            .schemas_iter()
            .map(|file_schema| (file_schema.schema_id(), resolve_columns(Some(file_schema))))
            .collect();
        columns_per_schema.sort_unstable_by_key(|(schema_id, _)| *schema_id);
        let columns_of_unlisted_schema = resolve_columns(None);
        Ok(Self {
            row_filters,
            partition_pruners,
            columns_per_schema,
            columns_of_unlisted_schema,
        })
    }

    /// Those of `manifests` whose files may hold a row satisfying every
    /// filter, by the partition summaries of their manifest-list entries
    /// against what the filters require of a tuple of each one's spec. A
    /// manifest of a spec the table does not list is kept.
    pub(crate) fn select_manifests<'m>(
        &self,
        manifests: &'m [ListedManifest],
    ) -> Vec<&'m ListedManifest> {
        let may_match = |manifest: &&ListedManifest| {
            let pruner = find_by_id(&self.partition_pruners, manifest.file().partition_spec_id);
            pruner.is_none_or(|pruner| manifest.may_match(pruner.partition_filter()))
        };
        manifests.iter().filter(may_match).collect()
    }

    /// Whether a file of `partition` may hold a row satisfying every filter,
    /// by the partition's tuple alone. `values` is scratch space the caller
    /// keeps from one partition to the next.
    pub(crate) fn may_partition_match(
        &self,
        partition: &Partition,
        values: &mut Vec<FieldRange>,
    ) -> iceberg::Result<bool> {
        match find_by_id(&self.partition_pruners, partition.spec_id()) {
            Some(pruner) => pruner.may_match(partition.tuple(), values),
            None => Ok(true),
        }
    }

    /// The partitions of the specs `spec_ids` a row satisfying every filter
    /// may sit in, computed from the filters alone when they leave every
    /// field of each of those specs only a few values: every combination
    /// of the values, whether or not the table has a file in it. `None`
    /// when a spec is not one `metadata` lists, the filters leave a field
    /// of one unrestricted, or the partitions number more than `limit`.
    pub(crate) fn enumerate_partitions_from_filter(
        &self,
        metadata: &TableMetadata,
        spec_ids: &[i32],
        limit: usize,
    ) -> Option<Vec<Partition>> {
        let mut partitions = Vec::new();
        for spec_id in spec_ids {
            let pruner = find_by_id(&self.partition_pruners, *spec_id)?;
            let partition_spec = metadata.partition_spec_by_id(*spec_id)?;
            let tuples = enumerate_tuples_from_filter(
                metadata,
                partition_spec,
                pruner.partition_filter(),
                limit - partitions.len(),
            )?;
            partitions.extend(
                tuples
                    .into_iter()
                    .map(|tuple| Partition::new(*spec_id, tuple)),
            );
        }
        Some(partitions)
    }

    /// The ids of the fields the filters compare with constants, each once:
    /// the columns whose bounds [`select_files`](Self::select_files) reads.
    pub(crate) fn compared_field_ids(&self) -> impl Iterator<Item = i32> {
        self.columns_of_unlisted_schema
            .iter()
            .map(StoredColumn::field_id)
    }

    /// The files of `slice`, those of one partition whose tuple the
    /// filters keep in one manifest, read with the bounds of the compared
    /// columns, that are not empty and whose column bounds may hold a row
    /// satisfying every filter: a bit for each file, set for one that is.
    pub(crate) fn select_files(&self, slice: &PartitionSlice) -> BooleanBuffer {
        let files = &slice.files;
        let columns = match find_by_id(&self.columns_per_schema, files.schema_id()) {
            Some(columns) => columns,
            None => &self.columns_of_unlisted_schema,
        };
        let selection = BoundsSelection {
            columns,
            file_count: files.count_files(),
            slice,
        };
        let mut selected = files.select_files_with_rows();
        for row_filter in &self.row_filters {
            selected &= &selection.may_match(row_filter);
        }
        selected
    }
}

/// What one manifest records of the columns a scan's filters compare, in
/// the files of one partition: which of those files may hold something a
/// predicate is true for.
struct BoundsSelection<'s> {
    /// How the files of the manifest record the compared columns.
    columns: &'s [StoredColumn],
    file_count: usize,
    /// The files, with what the manifest records of the compared columns
    /// in them.
    slice: &'s PartitionSlice,
}

impl BoundsSelection<'_> {
    /// A bit for each file: whether it may hold something `predicate` is
    /// true for, what [`pruning::may_match`] answers of the file's manifest
    /// entry. The bits of a predicate's two sides are combined a word at a
    /// time.
    fn may_match(&self, predicate: &BoundPredicate) -> BooleanBuffer {
        match predicate {
            BoundPredicate::AlwaysFalse => BooleanBuffer::new_unset(self.file_count),
            BoundPredicate::And(children) => {
                let [left, right] = children.inputs();
                &self.may_match(left) & &self.may_match(right)
            }
            BoundPredicate::Or(children) => {
                let [left, right] = children.inputs();
                &self.may_match(left) | &self.may_match(right)
            }
            BoundPredicate::Binary(comparison) => self.may_satisfy(
                comparison.term().field().id,
                comparison.op(),
                std::iter::once(comparison.literal()),
            ),
            // A field is in the set when it equals one of its values.
            BoundPredicate::Set(set) if set.op() == PredicateOperator::In => self.may_satisfy(
                set.term().field().id,
                PredicateOperator::Eq,
                set.literals().iter(),
            ),
            _ => BooleanBuffer::new_set(self.file_count),
        }
    }

    /// A bit for each file: whether it may hold a value of the field
    /// `field_id` that satisfies `<operator> constant` for one of
    /// `constants`. A field that is none of the compared columns is
    /// unknown, and may.
    fn may_satisfy<'c>(
        &self,
        field_id: i32,
        operator: PredicateOperator,
        constants: impl ExactSizeIterator<Item = &'c Datum>,
    ) -> BooleanBuffer {
        let Some(column) = self
            .columns
            .iter()
            .find(|column| column.field_id() == field_id)
        else {
            return BooleanBuffer::new_set(self.file_count);
        };
        let Some(bounds) = self.slice.find_bounds(field_id) else {
            // The manifest records nothing of the column: its values are
            // unknown in a file that has the column, and all NULL in one
            // that does not.
            return if column.is_written() && constants.len() > 0 {
                BooleanBuffer::new_set(self.file_count)
            } else {
                BooleanBuffer::new_unset(self.file_count)
            };
        };
        constants.fold(
            BooleanBuffer::new_unset(self.file_count),
            |satisfied, constant| &satisfied | &bounds.may_satisfy(column, operator, constant),
        )
    }
}

/// The entry under `id` of `entries`, which are in id order.
fn find_by_id<Entry>(entries: &[(i32, Entry)], id: i32) -> Option<&Entry> {
    let position = entries
        .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
        .ok()?;
    Some(&entries[position].1)
}
