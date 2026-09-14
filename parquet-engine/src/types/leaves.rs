//! Converts between nested Arrow columns and Parquet's depth-first leaves.

use std::ops::Range;
use std::sync::Arc;

use ahash::HashMap;
use arrow_array::{Array, ArrayRef, BinaryViewArray, Datum, StructArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use dispatch::{Projection, VariantExtract};

use crate::RowGroupMetadata;
use crate::types::metadata::QueryRowGroupMetadata;

/// Returns the primitive descendants of `fields` in depth-first order.
///
/// This is the same order that Parquet uses for column chunks.
pub fn leaf_fields(fields: &Fields) -> Vec<FieldRef> {
    let mut leaves = Vec::new();
    for field in fields {
        push_leaf_fields(field, &mut leaves);
    }
    leaves
}

fn push_leaf_fields(field: &FieldRef, leaves: &mut Vec<FieldRef>) {
    match field.data_type() {
        DataType::Struct(children) => children
            .iter()
            .for_each(|child| push_leaf_fields(child, leaves)),
        _ => leaves.push(field.clone()),
    }
}

/// Returns the number of Parquet leaves spanned by `field`.
///
/// A primitive spans one leaf, while a struct spans all its descendant leaves.
pub fn leaf_count(field: &FieldRef) -> usize {
    match field.data_type() {
        DataType::Struct(children) => children.iter().map(leaf_count).sum(),
        _ => 1,
    }
}

/// Returns the index of the first leaf belonging to `column`.
pub fn first_leaf(fields: &Fields, column: usize) -> usize {
    fields.iter().take(column).map(leaf_count).sum()
}

/// Returns the file leaf range spanned by top-level `column`.
pub(crate) fn leaf_range(fields: &Fields, column: usize) -> Range<usize> {
    let start = first_leaf(fields, column);
    start..start + leaf_count(&fields[column])
}

/// How one projected output column resolves against this row group's layout.
///
/// The fetch side and the decoder both derive their work from the same
/// resolution, which keeps fetched chunks aligned with their decoders when
/// files have different shredding layouts.
pub(crate) enum OutputRead {
    /// Every leaf of a top-level column: a plain column read, and the fallback
    /// for an extract this file's layout cannot satisfy any smaller way.
    WholeColumn(Range<usize>),
    /// The one shredded typed leaf holding every value for a scalar extract's
    /// path.
    TypedLeaf(usize),
    /// The pruned path subtree an extract needs when one complete typed leaf
    /// cannot answer it, folded into
    /// `reconstructed_variant_field`.
    PrunedVariant {
        /// One source per primitive leaf in the reconstructed field.
        leaf_sources: Vec<usize>,
        reconstructed_variant_field: FieldRef,
    },
    /// An extract whose path row-group statistics prove holds SQL NULL in
    /// every row: nothing on the path can produce a value, so the output is
    /// all NULL and only the metadata leaf is read.
    AbsentPath {
        /// The variant's metadata leaf. It supplies each batch's row count,
        /// and the metadata blob when the output is itself a variant.
        metadata_leaf: usize,
    },
}

impl OutputRead {
    fn create(
        fields: &Fields,
        metadata: &QueryRowGroupMetadata,
        column: usize,
        extract: Option<&VariantExtract>,
    ) -> Self {
        let Some(extract) = extract else {
            return Self::WholeColumn(leaf_range(fields, column));
        };
        let value_leaves =
            variant_shredded_leaves(fields, column, &extract.path).zip(extract.as_type.as_ref());
        if let Some((leaves, target)) = &value_leaves
            && let Some(read) = Self::try_create_typed_leaf(metadata, leaves, target)
        {
            return read;
        }
        Self::try_create_pruned_variant(
            fields,
            column,
            &extract.path,
            metadata,
            value_leaves
                .as_ref()
                .map(|(leaves, target)| (leaves, *target)),
        )
        .unwrap_or_else(|| Self::WholeColumn(leaf_range(fields, column)))
    }

    /// The read that satisfies a scalar extract of `path` from one shredded
    /// typed leaf, or `None` when this row group cannot answer it that way.
    ///
    /// The path must resolve to a typed scalar leaf under `column`, and every
    /// untyped `value` leaf along the path must contribute only SQL NULL under
    /// this cast. Otherwise some values live outside the typed leaf and the
    /// caller has to reconstruct the on-path variant.
    pub(crate) fn try_create_typed_leaf(
        metadata: &QueryRowGroupMetadata,
        leaves: &ShreddedScalarPath,
        target: &DataType,
    ) -> Option<Self> {
        let terminal = leaves.value_leaves.len().saturating_sub(1);
        for (level, &value_leaf) in leaves.value_leaves.iter().enumerate() {
            if !variant_value_leaf_is_semantically_null(
                metadata.get_metadata(),
                value_leaf,
                target,
                level == terminal,
            ) {
                return None;
            }
        }
        Some(OutputRead::TypedLeaf(leaves.typed_leaf))
    }

    /// The read that satisfies an extract of `path` from its on-path subtree
    /// alone.
    ///
    /// Only the metadata, the untyped `value` at each level along the path, and
    /// the whole subtree at the path are read; sibling fields are skipped.
    /// Folding those leaves into `reconstructed_variant_field` yields the input
    /// variant with every off-path field dropped, so `variant_get` over it
    /// reconstructs the sub-variant exactly as it would over the full column,
    /// untyped fallbacks and all, while touching far fewer column chunks.
    pub(crate) fn try_create_pruned_variant(
        fields: &Fields,
        column: usize,
        path: &[String],
        metadata: &QueryRowGroupMetadata,
        value_leaves: Option<(&ShreddedScalarPath, &DataType)>,
    ) -> Option<Self> {
        let mut leaf_sources = Vec::new();
        let reconstructed_variant_field = prune_variant_node(
            &fields[column],
            path,
            first_leaf(fields, column),
            &mut leaf_sources,
            metadata,
            value_leaves,
        )?;
        // Metadata is always retained, and every other retained leaf lies on
        // the path and can produce values there. A tree that kept nothing
        // else proves the path all SQL NULL in this row group.
        let (metadata_offset, _) = find_child_leaf_offset(&fields[column], "metadata")?;
        let metadata_leaf = first_leaf(fields, column) + metadata_offset;
        if leaf_sources.iter().all(|&leaf| leaf == metadata_leaf) {
            return Some(OutputRead::AbsentPath { metadata_leaf });
        }
        Some(OutputRead::PrunedVariant {
            leaf_sources,
            reconstructed_variant_field,
        })
    }

    /// The file source for each leaf this output reconstructs, in depth-first
    /// order.
    fn leaf_sources(&self) -> Vec<usize> {
        match self {
            OutputRead::WholeColumn(leaves) => leaves.clone().collect(),
            OutputRead::TypedLeaf(leaf) => vec![*leaf],
            OutputRead::PrunedVariant { leaf_sources, .. } => leaf_sources.clone(),
            OutputRead::AbsentPath { metadata_leaf } => vec![*metadata_leaf],
        }
    }
}

/// Resolves each of `projection`'s output columns against this row group.
pub(crate) fn resolve_output_reads(
    fields: &Fields,
    metadata: &QueryRowGroupMetadata,
    projection: &Projection,
) -> Vec<OutputRead> {
    projection
        .column_indices
        .iter()
        .enumerate()
        .map(|(output_idx, &column)| {
            OutputRead::create(fields, metadata, column, projection.extract_at(output_idx))
        })
        .collect()
}

/// The leaves a row group reads, and which of them each output column folds.
pub(crate) struct LeafPlan {
    /// The distinct file leaves, in the order they are fetched and decoded.
    pub file_leaves: Vec<usize>,
    /// Per output column, the positions within `file_leaves` it folds, in
    /// depth-first order.
    pub output_positions: Vec<Vec<usize>>,
}

/// Collects `reads` into the distinct leaves to fetch and each output's view of
/// them.
///
/// Outputs routinely want the same leaf: a query extracting four paths from one
/// variant column this file does not shred resolves all four to the whole
/// column. Reading that leaf once and letting each output fold its own view
/// costs one decode instead of four.
pub(crate) fn plan_leaves(reads: &[OutputRead]) -> LeafPlan {
    let mut file_leaves: Vec<usize> = Vec::new();
    let mut position_of: HashMap<usize, usize> = HashMap::default();
    let output_positions = reads
        .iter()
        .map(|read| {
            read.leaf_sources()
                .into_iter()
                .map(|leaf| {
                    *position_of.entry(leaf).or_insert_with(|| {
                        file_leaves.push(leaf);
                        file_leaves.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    LeafPlan {
        file_leaves,
        output_positions,
    }
}

/// Returns the distinct file leaf indices needed for `projection`, in fetch
/// order.
pub(crate) fn projected_leaves(
    fields: &Fields,
    metadata: &QueryRowGroupMetadata,
    projection: &Projection,
) -> Vec<usize> {
    plan_leaves(&resolve_output_reads(fields, metadata, projection)).file_leaves
}

/// Whether a binary variant fallback can only contribute SQL NULL to this cast.
///
/// Physical nulls are absent values. A present canonical `[0x00]` value is JSON
/// null. At an ancestor of the requested path it means that the path is absent;
/// at the terminal value it is interchangeable with absence for every scalar
/// cast except text. Any missing or broader statistics are not proof and keep
/// the fallback read.
pub fn variant_value_leaf_is_semantically_null(
    row_group: &RowGroupMetadata,
    leaf: usize,
    target: &DataType,
    terminal: bool,
) -> bool {
    let Some(stats) = row_group.leaf_statistics(leaf) else {
        return false;
    };
    if stats.null_count == Some(row_group.num_rows) {
        return true;
    }
    (!terminal || !target.is_string())
        && statistic_is_json_null(stats.min().as_ref())
        && statistic_is_json_null(stats.max().as_ref())
}

fn statistic_is_json_null(statistic: Option<&arrow_array::Scalar<ArrayRef>>) -> bool {
    let Some(statistic) = statistic else {
        return false;
    };
    let (array, _) = statistic.get();
    let Some(value) = array.as_any().downcast_ref::<BinaryViewArray>() else {
        return false;
    };
    value.len() == 1 && value.value(0) == [0_u8]
}

/// Describes where a shredded scalar path lives in one file's column chunks.
///
/// Shredding stores each field in a typed `typed_value` column and an untyped
/// binary `value` column. Values that match the shredded type use the typed
/// column, while other values use an untyped `value` column. Recursive
/// shredding adds an untyped column at every level of the path. A cast can read
/// only the typed column when every one of those untyped columns is either
/// physically null or proven equivalent to SQL NULL for that cast.
pub struct ShreddedScalarPath {
    /// This leaf holds the shredded values and their predicate statistics.
    pub typed_leaf: usize,
    /// These untyped leaves can hold values missing from the typed leaf.
    pub value_leaves: Vec<usize>,
}

/// Resolves a shredded scalar `path` under variant `column` to its file leaves.
///
/// This returns `None` when the file does not shred the path or when the path
/// ends at a shredded object instead of a scalar.
///
/// For example, `user.id` maps to
/// `column.typed_value.user.typed_value.id.typed_value`.
pub fn variant_shredded_leaves(
    fields: &Fields,
    column: usize,
    path: &[String],
) -> Option<ShreddedScalarPath> {
    let mut field_first_leaf = first_leaf(fields, column);
    let mut current_field = &fields[column];
    let mut value_leaves = Vec::new();
    // Each shredding level may hold an untyped `value` next to the
    // `typed_value` branch.
    for segment in path {
        if let Some((value_offset, _)) = find_child_leaf_offset(current_field, "value") {
            value_leaves.push(field_first_leaf + value_offset);
        }
        let (typed_offset, typed_field) = find_child_leaf_offset(current_field, "typed_value")?;
        let (segment_offset, segment_field) = find_child_leaf_offset(typed_field, segment)?;
        field_first_leaf += typed_offset + segment_offset;
        current_field = segment_field;
    }
    if let Some((value_offset, _)) = find_child_leaf_offset(current_field, "value") {
        value_leaves.push(field_first_leaf + value_offset);
    }
    let (typed_offset, typed_field) = find_child_leaf_offset(current_field, "typed_value")?;
    if matches!(typed_field.data_type(), DataType::Struct(_)) {
        // The path names an object shredded further, not a typed leaf.
        return None;
    }
    Some(ShreddedScalarPath {
        typed_leaf: field_first_leaf + typed_offset,
        value_leaves,
    })
}

/// Prunes a variant node to the single branch that reaches `path`.
///
/// The function retains metadata and untyped `value` children. It narrows
/// `typed_value` to the on-path child until the path ends, then retains the
/// complete remaining subtree. Leaves proven to be all SQL NULL are omitted
/// from both the reconstructed field and `leaf_sources`, and a branch left
/// without any leaves is dropped entirely. This returns `None` when `field`
/// is not a variant struct at all.
fn prune_variant_node(
    field: &FieldRef,
    path: &[String],
    first_leaf: usize,
    leaf_sources: &mut Vec<usize>,
    metadata: &QueryRowGroupMetadata,
    value_leaves: Option<(&ShreddedScalarPath, &DataType)>,
) -> Option<FieldRef> {
    let DataType::Struct(children) = field.data_type() else {
        return None;
    };
    let mut pruned_children = Vec::with_capacity(children.len());
    let mut child_first_leaf = first_leaf;
    for child in children {
        let child_leaf_count = leaf_count(child);
        if child.name() == "typed_value" {
            if path.is_empty() {
                // The path ends here, so the entire subtree is retained.
                if let Some(child) = prune_null_leaves(
                    child,
                    child_first_leaf,
                    leaf_sources,
                    metadata,
                    value_leaves,
                ) {
                    pruned_children.push(child);
                }
            } else {
                // Only the on-path child of the shredded object is retained.
                let DataType::Struct(object_fields) = child.data_type() else {
                    // The path continues but this level is a typed leaf. The
                    // current node's `value` is still sufficient to resolve a
                    // binary fallback, so simply omit this unusable branch.
                    child_first_leaf += child_leaf_count;
                    continue;
                };
                let mut object_child_first_leaf = child_first_leaf;
                let mut pruned_path_field = None;
                for object_child in object_fields {
                    if object_child.name() == path[0].as_str() {
                        pruned_path_field = prune_variant_node(
                            object_child,
                            &path[1..],
                            object_child_first_leaf,
                            leaf_sources,
                            metadata,
                            value_leaves,
                        );
                        break;
                    }
                    object_child_first_leaf += leaf_count(object_child);
                }
                // A subtree that retained no leaves cannot produce values at
                // the path (and its bare structs could not even be
                // reconstructed), so the whole branch is dropped.
                if let Some(pruned_path_field) = pruned_path_field
                    && leaf_count(&pruned_path_field) > 0
                {
                    pruned_children.push(Arc::new(Field::new(
                        child.name(),
                        DataType::Struct(Fields::from(vec![pruned_path_field.as_ref().clone()])),
                        child.is_nullable(),
                    )));
                }
            }
        } else {
            // Metadata and untyped `value` children stay on the path unless
            // row-group statistics prove they can contribute only SQL NULL.
            if let Some(child) = prune_null_leaves(
                child,
                child_first_leaf,
                leaf_sources,
                metadata,
                value_leaves,
            ) {
                pruned_children.push(child);
            }
        }
        child_first_leaf += child_leaf_count;
    }
    Some(Arc::new(Field::new(
        field.name(),
        DataType::Struct(Fields::from(pruned_children)),
        field.is_nullable(),
    )))
}

/// Removes all-null primitive descendants from a retained field, appending the
/// file source for every descendant that remains.
fn prune_null_leaves(
    field: &FieldRef,
    first_leaf: usize,
    leaf_sources: &mut Vec<usize>,
    metadata: &QueryRowGroupMetadata,
    value_leaves: Option<(&ShreddedScalarPath, &DataType)>,
) -> Option<FieldRef> {
    let DataType::Struct(children) = field.data_type() else {
        if leaf_is_semantically_null(field, first_leaf, metadata, value_leaves) {
            return None;
        }
        leaf_sources.push(first_leaf);
        return Some(field.clone());
    };

    let mut pruned_children = Vec::with_capacity(children.len());
    let mut child_first_leaf = first_leaf;
    for child in children {
        if let Some(child) = prune_null_leaves(
            child,
            child_first_leaf,
            leaf_sources,
            metadata,
            value_leaves,
        ) {
            pruned_children.push(child);
        }
        child_first_leaf += leaf_count(child);
    }
    (!pruned_children.is_empty()).then(|| {
        Arc::new(Field::new(
            field.name(),
            DataType::Struct(Fields::from(pruned_children)),
            field.is_nullable(),
        ))
    })
}

fn leaf_is_semantically_null(
    field: &FieldRef,
    leaf: usize,
    metadata: &QueryRowGroupMetadata,
    value_leaves: Option<(&ShreddedScalarPath, &DataType)>,
) -> bool {
    // Every reconstructed Variant needs its metadata blob.
    if field.name() == "metadata" {
        return false;
    }
    if metadata
        .get_metadata()
        .leaf_statistics(leaf)
        .and_then(|statistics| statistics.null_count)
        == Some(metadata.num_rows())
    {
        return true;
    }
    value_leaves.is_some_and(|(value_leaves, target)| {
        value_leaves
            .value_leaves
            .iter()
            .position(|&value_leaf| value_leaf == leaf)
            .is_some_and(|level| {
                variant_value_leaf_is_semantically_null(
                    metadata.get_metadata(),
                    leaf,
                    target,
                    level + 1 == value_leaves.value_leaves.len(),
                )
            })
    })
}

/// Finds a direct child and its leaf offset within `field`.
///
/// The offset counts the leaves of preceding siblings. This returns `None`
/// when `field` is not a struct or does not contain the named child.
/// Per file leaf, whether it belongs to a shredded variant and is not its
/// `metadata`: the leaves a reassembly of the variant reads once and drops.
/// Their arrays are short-lived while every other leaf's lives as long as
/// the batch, so a decoder keeps them apart (see
/// [`WorkerAllocator`](crate::reading::decoding::WorkerAllocator)).
pub(crate) fn short_lived_variant_leaves(fields: &Fields) -> Vec<bool> {
    let mut short_lived = vec![false; fields.iter().map(leaf_count).sum()];
    for (column, field) in fields.iter().enumerate() {
        if !crate::is_variant_field(field) || find_child_leaf_offset(field, "typed_value").is_none()
        {
            continue;
        }
        let range = leaf_range(fields, column);
        let metadata =
            find_child_leaf_offset(field, "metadata").map(|(offset, _)| range.start + offset);
        for leaf in range {
            short_lived[leaf] = Some(leaf) != metadata;
        }
    }
    short_lived
}

fn find_child_leaf_offset<'a>(field: &'a FieldRef, name: &str) -> Option<(usize, &'a FieldRef)> {
    let DataType::Struct(children) = field.data_type() else {
        return None;
    };
    let mut offset = 0;
    for child in children {
        if child.name() == name {
            return Some((offset, child));
        }
        offset += leaf_count(child);
    }
    None
}

/// Reconstructs one top-level Arrow column from decoded Parquet leaf arrays.
///
/// Parquet stores a nested column as depth-first leaf column chunks, while
/// Arrow represents it as one array. This function walks `field` in order,
/// taking the next decoded leaf array for each primitive descendant and
/// assembling each struct from the arrays of its children (which are themselves
/// assembled the same way). A primitive field is a plain pass-through. It is
/// the inverse of [`leaf_fields`].
pub(crate) fn reconstruct_column_from_leaves(
    field: &FieldRef,
    leaf_arrays: &mut impl Iterator<Item = ArrayRef>,
) -> ArrayRef {
    /// Tracks the fields and arrays of a struct that is being reconstructed.
    ///
    /// The next field to fill is always `fields[arrays.len()]`.
    struct PartialStruct {
        fields: Fields,
        arrays: Vec<ArrayRef>,
        /// This records whether the field containing this struct is nullable.
        ///
        /// A required struct cannot have a mask because its parent interprets
        /// child nulls as evidence that the parent is null.
        nullable: bool,
    }
    impl PartialStruct {
        fn start(fields: &Fields, nullable: bool) -> Self {
            Self {
                fields: fields.clone(),
                arrays: Vec::with_capacity(fields.len()),
                nullable,
            }
        }
        fn get_next_unfilled_field(&self) -> Option<&FieldRef> {
            self.fields.get(self.arrays.len())
        }
        /// Finishes this partial struct and reconstructs its null buffer.
        ///
        /// Parquet stores only leaf arrays, so a struct's own null buffer is
        /// not available directly. A null in a non-nullable child must come
        /// from the struct or one of its ancestors, so combining those child
        /// null buffers recovers the mask that makes their nulls valid.
        ///
        /// If no non-nullable child provides a mask and this struct is
        /// nullable, its nulls are inferred from rows where every child is
        /// absent. Shredded variant `typed_value` structs rely on this fallback
        /// to distinguish rows that have a value from rows that do not. The
        /// fallback is not applied to a non-nullable struct, since absent
        /// nullable children do not make a required struct null.
        fn finish_struct_array(self) -> ArrayRef {
            let mut nulls: Option<NullBuffer> = None;
            for (field, array) in self.fields.iter().zip(&self.arrays) {
                if !field.is_nullable() {
                    nulls = NullBuffer::union(nulls.as_ref(), array.nulls());
                }
            }
            if self.nullable && nulls.is_none() {
                nulls = infer_nulls_from_children(&self.arrays);
            }
            Arc::new(StructArray::new(self.fields, self.arrays, nulls))
        }
    }

    let DataType::Struct(children) = field.data_type() else {
        // A primitive column is exactly the next decoded leaf array.
        return leaf_arrays
            .next()
            .expect("one decoded leaf array per leaf field");
    };
    // A stack of open structs supports arbitrarily deep schemas. The bottom
    // entry is the column itself and closes by returning.
    let mut open_structs = vec![PartialStruct::start(children, field.is_nullable())];
    loop {
        let innermost = open_structs
            .last()
            .expect("the column's own struct only closes by returning");
        match innermost.get_next_unfilled_field().cloned() {
            Some(field) => match field.data_type() {
                // A nested struct must be filled before it can be added here.
                DataType::Struct(children) => {
                    open_structs.push(PartialStruct::start(children, field.is_nullable()))
                }
                // A primitive field owns exactly the next decoded leaf array.
                _ => open_structs
                    .last_mut()
                    .expect("just inspected")
                    .arrays
                    .push(
                        leaf_arrays
                            .next()
                            .expect("one decoded leaf array per leaf field"),
                    ),
            },
            // A complete struct is added to its parent as one array. Completing
            // the column's own struct produces the reconstructed column.
            None => {
                let finished = open_structs.pop().expect("just inspected");
                match open_structs.last_mut() {
                    Some(enclosing) => enclosing.arrays.push(finished.finish_struct_array()),
                    None => return finished.finish_struct_array(),
                }
            }
        }
    }
}

/// Infers a struct's null buffer from the presence of its child `arrays`.
///
/// A row is present when at least one child is present, so the mask is null
/// exactly where every child is absent, and `None` when no such row exists. A
/// child that is present in every row makes the struct present in every row,
/// which ends the search. A struct child that carries no mask of its own has
/// its presence inferred from its descendants by [`find_present_rows`].
fn infer_nulls_from_children(arrays: &[ArrayRef]) -> Option<NullBuffer> {
    let mut valid: Option<BooleanBuffer> = None;
    for array in arrays {
        let child = find_present_rows(array)?;
        valid = Some(match valid {
            None => child,
            Some(valid) => &valid | &child,
        });
    }
    valid.map(NullBuffer::new)
}

/// Finds the rows where `array` holds a value.
///
/// A struct carrying a mask of its own is answered by it. One without has to be
/// answered by its descendants, because the struct that says whether a variant
/// field is present sits above children that are themselves non-nullable, and a
/// non-nullable child never carries a mask. What it has instead is its own
/// children, one level further down, being null. This returns `None` when every
/// row is present.
fn find_present_rows(array: &ArrayRef) -> Option<BooleanBuffer> {
    if let Some(nulls) = array.nulls() {
        return Some(nulls.inner().clone());
    }
    let structure = array.as_any().downcast_ref::<StructArray>()?;
    let mut valid: Option<BooleanBuffer> = None;
    for child in structure.columns() {
        let child = find_present_rows(child)?;
        valid = Some(match valid {
            None => child,
            Some(valid) => &valid | &child,
        });
    }
    valid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::dummy_metadata;
    use crate::types::metadata::RowSelection;
    use arrow_array::{BinaryViewArray, Int64Array};
    use arrow_schema::Field;

    /// Builds a shredded variant struct for the supplied paths.
    ///
    /// The result has the shape
    /// `{metadata, value, typed_value{<path>{value, typed_value}}}`.
    fn build_variant_field(paths: &[(&str, DataType)]) -> Field {
        let shredded: Vec<Field> = paths
            .iter()
            .map(|(name, ty)| {
                let leaves = vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", ty.clone(), true),
                ];
                Field::new(*name, DataType::Struct(leaves.into()), true)
            })
            .collect();
        let children = vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
            Field::new("typed_value", DataType::Struct(shredded.into()), true),
        ];
        Field::new("v", DataType::Struct(children.into()), true)
    }

    /// Builds a shredded variant field with the layout used in a written file.
    ///
    /// Each path's `{value, typed_value}` pair is non-nullable. Only the
    /// enclosing `typed_value` records whether a row contains any path.
    fn build_file_variant_field(paths: &[(&str, DataType)]) -> Field {
        let shredded: Vec<Field> = paths
            .iter()
            .map(|(name, ty)| {
                let pair = vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", ty.clone(), true),
                ];
                Field::new(*name, DataType::Struct(pair.into()), false)
            })
            .collect();
        let children = vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
            Field::new("typed_value", DataType::Struct(shredded.into()), true),
        ];
        Field::new("v", DataType::Struct(children.into()), true)
    }

    fn build_path(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| s.to_string()).collect()
    }

    /// A row that has none of the shredded paths must come back null at
    /// `typed_value`, which is how a reader tells "this row had no value for
    /// the field" from "it had one". Nothing else carries that: the pairs
    /// beneath are non-nullable and never hold a mask of their own.
    #[test]
    fn verifies_shredded_variant_nulls_for_rows_without_paths() {
        let field = Arc::new(build_file_variant_field(&[
            ("a", DataType::Int64),
            ("b", DataType::Int64),
        ]));
        // Row 0 has only `a`, row 1 only `b`, row 2 neither.
        let mut leaves = vec![
            Arc::new(BinaryViewArray::from(vec![Some(&b"m"[..]); 3])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 3])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1), None, None])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![None, Some(2), None])) as ArrayRef,
        ]
        .into_iter();

        let column = reconstruct_column_from_leaves(&field, &mut leaves);

        let variant = column.as_any().downcast_ref::<StructArray>().unwrap();
        let typed_value = variant.column(2);
        assert!(!typed_value.is_null(0), "row 0 has `a`");
        assert!(!typed_value.is_null(1), "row 1 has `b`");
        assert!(typed_value.is_null(2), "row 2 has neither");
    }

    /// The pairs under `typed_value` are non-nullable, so a row missing one of
    /// them says nothing about the row as a whole. Masking them anyway reads,
    /// to the rule that a null in a non-nullable child means the parent is
    /// null, as the whole variant being absent wherever one path is.
    #[test]
    fn verifies_missing_shredded_path_does_not_null_variant() {
        let field = Arc::new(build_file_variant_field(&[
            ("a", DataType::Int64),
            ("b", DataType::Int64),
        ]));
        let mut leaves = vec![
            Arc::new(BinaryViewArray::from(vec![Some(&b"m"[..]); 2])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1), Some(2)])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![None, None])) as ArrayRef,
        ]
        .into_iter();

        let column = reconstruct_column_from_leaves(&field, &mut leaves);

        // Every row has `a`; none has `b`. Both rows are still present.
        assert_eq!(column.null_count(), 0);
        let variant = column.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(variant.column(2).null_count(), 0);
    }

    /// Outputs that resolve to the same leaves read them once. The third
    /// output's typed leaf is already covered by the whole-column reads, so it
    /// reuses that position instead of adding another chunk to fetch.
    #[test]
    fn verifies_outputs_share_the_leaves_they_both_read() {
        let reads = vec![
            OutputRead::WholeColumn(0..3),
            OutputRead::WholeColumn(0..3),
            OutputRead::TypedLeaf(1),
        ];

        let plan = plan_leaves(&reads);

        assert_eq!(plan.file_leaves, vec![0, 1, 2]);
        assert_eq!(
            plan.output_positions,
            vec![vec![0, 1, 2], vec![0, 1, 2], vec![1]]
        );
    }

    /// Leaves no output shares are still fetched once each, in first-appearance
    /// order, because that order is what the fetcher and the decoder agree on.
    #[test]
    fn verifies_distinct_leaves_keep_first_appearance_order() {
        let reads = vec![
            OutputRead::TypedLeaf(4),
            OutputRead::TypedLeaf(1),
            OutputRead::TypedLeaf(4),
        ];

        let plan = plan_leaves(&reads);

        assert_eq!(plan.file_leaves, vec![4, 1]);
        assert_eq!(plan.output_positions, vec![vec![0], vec![1], vec![0]]);
    }

    #[test]
    fn verifies_typed_leaf_resolution_for_each_shredded_path() {
        let fields = Fields::from(vec![build_variant_field(&[
            ("age", DataType::Int64),
            ("name", DataType::Utf8),
        ])]);

        let age = variant_shredded_leaves(&fields, 0, &build_path(&["age"])).unwrap();
        let name = variant_shredded_leaves(&fields, 0, &build_path(&["name"])).unwrap();

        // The layout is metadata=0, value=1, age.value=2,
        // age.typed_value=3, name.value=4, and name.typed_value=5.
        assert_eq!(age.typed_leaf, 3);
        assert_eq!(age.value_leaves, vec![1, 2]);
        assert_eq!(name.typed_leaf, 5);
        assert_eq!(name.value_leaves, vec![1, 4]);
    }

    #[test]
    fn verifies_unshredded_path_has_no_typed_leaf() {
        let fields = Fields::from(vec![build_variant_field(&[("age", DataType::Int64)])]);

        assert!(variant_shredded_leaves(&fields, 0, &build_path(&["missing"])).is_none());
    }

    #[test]
    fn verifies_pruned_variant_reads_only_the_path_subtree() {
        let fields = Fields::from(vec![build_variant_field(&[
            ("age", DataType::Int64),
            ("name", DataType::Utf8),
        ])]);

        let read = OutputRead::try_create_pruned_variant(
            &fields,
            0,
            &build_path(&["age"]),
            &dummy_metadata(RowSelection::All),
            None,
        )
        .unwrap();

        // The read takes metadata=0, value=1, age.value=2, and age.typed_value=3.
        // It skips both leaves for name.
        let OutputRead::PrunedVariant {
            leaf_sources,
            reconstructed_variant_field,
        } = read
        else {
            panic!("a shredded path prunes to its own subtree");
        };
        assert_eq!(leaf_sources, vec![0, 1, 2, 3]);
        // The pruned field folds exactly the read leaves and nothing else.
        assert_eq!(leaf_count(&reconstructed_variant_field), leaf_sources.len());
    }

    #[test]
    fn verifies_pruned_variant_keeps_nested_path_value_leaves() {
        let id = Field::new(
            "id",
            DataType::Struct(
                vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Int64, true),
                ]
                .into(),
            ),
            true,
        );
        let user = Field::new(
            "user",
            DataType::Struct(
                vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Struct(vec![id].into()), true),
                ]
                .into(),
            ),
            true,
        );
        let doc = Field::new(
            "v",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Struct(vec![user].into()), true),
                ]
                .into(),
            ),
            true,
        );
        let fields = Fields::from(vec![doc]);

        let read = OutputRead::try_create_pruned_variant(
            &fields,
            0,
            &build_path(&["user", "id"]),
            &dummy_metadata(RowSelection::All),
            None,
        )
        .unwrap();

        // The read takes metadata=0, value=1, user.value=2, user.id.value=3, and
        // user.id.typed_value=4. Reading every untyped value keeps the extract
        // sound.
        let OutputRead::PrunedVariant {
            leaf_sources,
            reconstructed_variant_field,
        } = read
        else {
            panic!("a shredded path prunes to its own subtree");
        };
        assert_eq!(leaf_sources, vec![0, 1, 2, 3, 4]);
        assert_eq!(leaf_count(&reconstructed_variant_field), leaf_sources.len());
    }

    #[test]
    fn verifies_unshredded_path_prunes_to_the_binary_fallback() {
        let fields = Fields::from(vec![build_variant_field(&[("age", DataType::Int64)])]);

        let read = OutputRead::try_create_pruned_variant(
            &fields,
            0,
            &build_path(&["missing"]),
            &dummy_metadata(RowSelection::All),
            None,
        )
        .unwrap();

        let OutputRead::PrunedVariant { leaf_sources, .. } = read else {
            panic!("the root binary value can contain the unshredded path");
        };
        assert_eq!(leaf_sources, vec![0, 1]);
    }

    #[test]
    fn verifies_shredded_path_is_scoped_to_its_column() {
        let fields = Fields::from(vec![
            Field::new("hello", DataType::Int64, false),
            build_variant_field(&[("hello", DataType::Int64)]),
        ]);

        let hello = variant_shredded_leaves(&fields, 1, &build_path(&["hello"])).unwrap();

        // The top-level hello=0 is outside the variant's range. The variant's
        // layout is metadata=1, value=2, hello.value=3, hello.typed_value=4.
        assert_eq!(hello.typed_leaf, 4);
        assert_eq!(hello.value_leaves, vec![2, 3]);
    }

    #[test]
    fn verifies_typed_leaf_resolution_for_nested_path() {
        // The `user` object has a `typed_value` that shreds `id`.
        let id = Field::new(
            "id",
            DataType::Struct(
                vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Int64, true),
                ]
                .into(),
            ),
            true,
        );
        let user = Field::new(
            "user",
            DataType::Struct(
                vec![
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Struct(vec![id].into()), true),
                ]
                .into(),
            ),
            true,
        );
        let doc = Field::new(
            "v",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("typed_value", DataType::Struct(vec![user].into()), true),
                ]
                .into(),
            ),
            true,
        );
        let fields = Fields::from(vec![doc]);

        let leaves = variant_shredded_leaves(&fields, 0, &build_path(&["user", "id"])).unwrap();

        // The layout is metadata=0, value=1, user.value=2, user.id.value=3,
        // and user.id.typed_value=4.
        assert_eq!(leaves.typed_leaf, 4);
        assert_eq!(leaves.value_leaves, vec![1, 2, 3]);
    }
}
