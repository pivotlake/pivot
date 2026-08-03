//! Conversion between nested Arrow columns and Parquet's depth-first leaves.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use dispatch::Projection;

use crate::parquet::types::metadata::QueryRowGroupMetadata;

/// The depth-first leaf fields of `fields` (struct fields expanded to their
/// primitive descendants), in the same order as a Parquet file's column chunks.
pub(crate) fn leaf_fields(fields: &Fields) -> Vec<FieldRef> {
    let mut leaves = Vec::new();
    for field in fields {
        push_leaf_fields(field, &mut leaves);
    }
    leaves
}

fn push_leaf_fields(field: &FieldRef, out: &mut Vec<FieldRef>) {
    match field.data_type() {
        DataType::Struct(children) => children.iter().for_each(|c| push_leaf_fields(c, out)),
        _ => out.push(field.clone()),
    }
}

/// Number of leaves (column chunks) the field spans: `1` for a primitive, the
/// descendant-leaf count for a struct.
pub(crate) fn leaf_count(field: &FieldRef) -> usize {
    match field.data_type() {
        DataType::Struct(children) => children.iter().map(leaf_count).sum(),
        _ => 1,
    }
}

/// The index of `column`'s first leaf: the leaves of every field before it.
pub(crate) fn first_leaf(fields: &Fields, column: usize) -> usize {
    fields.iter().take(column).map(leaf_count).sum()
}

/// The typed leaf a pushed variant extract can be read directly from, when
/// `path` is shredded to a typed (non-object) leaf under `column` AND every
/// row's value for the path lives in that leaf in this row group (no residual
/// `value` fallback holds anything), so reading the leaf alone is exact.
/// `None` means read the whole variant and extract (the path isn't shredded
/// here, or some rows fell to the residual, invisible to the typed leaf).
pub(crate) fn direct_extract_typed_leaf(
    fields: &Fields,
    metadata: &QueryRowGroupMetadata,
    column: usize,
    path: &[String],
) -> Option<usize> {
    let leaves = variant_shredded_leaves(fields, column, path)?;
    let num_rows = metadata.num_rows();
    for &fallback in &leaves.value_fallbacks {
        if metadata
            .get_metadata()
            .leaf_statistics(fallback)?
            .null_count?
            != num_rows
        {
            return None;
        }
    }
    Some(leaves.typed_value)
}

/// The leaf (column-chunk) indices a `projection` reads against this file's
/// `fields` and row-group stats. Each output is one of: a plain column (its
/// run of leaves), a pushed extract read directly from its shredded typed leaf
/// (one leaf), or a pushed extract that falls back to the whole variant (its
/// run of leaves) so `variant_get` can reconstruct it. The fetcher and the
/// decoder both resolve outputs this way, so a fetched chunk's position always
/// lines up with its decoder, even when another file shreds a variant column
/// into a different number of leaves.
pub(crate) fn projected_leaves(
    fields: &Fields,
    metadata: &QueryRowGroupMetadata,
    projection: &Projection,
) -> Vec<usize> {
    let mut leaves = Vec::with_capacity(projection.column_indices.len());
    for (output_idx, &column) in projection.column_indices.iter().enumerate() {
        if let Some(extract) = projection.extract_at(output_idx) {
            match &extract.as_type {
                // A scalar extract reads only the shredded typed leaf when the
                // path is fully shredded here.
                Some(_) => {
                    if let Some(typed_leaf) =
                        direct_extract_typed_leaf(fields, metadata, column, &extract.path)
                    {
                        leaves.push(typed_leaf);
                        continue;
                    }
                }
                // A bare extract reads only the path's subtree (plus residuals)
                // when the path is shredded here.
                None => {
                    if let Some(plan) = plan_variant_extract(fields, column, &extract.path) {
                        leaves.extend(plan.file_leaves);
                        continue;
                    }
                }
            }
        }
        let start = first_leaf(fields, column);
        leaves.extend(start..start + leaf_count(&fields[column]));
    }
    leaves
}

/// The column-chunk indices that decide whether a shredded variant path can
/// be pruned by statistics.
pub(crate) struct ShreddedPathLeaves {
    /// The typed leaf holding the path's shredded values, whose min/max
    /// statistics a predicate compares against.
    pub typed_value: usize,
    /// The binary `value` leaves along the path. A row's value for the path
    /// may live in any of them instead of the typed leaf (an unshredded row,
    /// or a value of another variant type), invisible to the typed leaf's
    /// statistics; pruning is sound only when each is all-null in the row
    /// group.
    pub value_fallbacks: Vec<usize>,
}

/// Resolve `path`, shredded under the variant column `column`, to its leaves
/// in THIS file's schema; `None` when the file doesn't shred that path (no
/// pruning, always sound).
///
/// For example, `user.id` maps to
/// `column.typed_value.user.typed_value.id.typed_value`.
pub(crate) fn variant_shredded_leaves(
    fields: &Fields,
    column: usize,
    path: &[String],
) -> Option<ShreddedPathLeaves> {
    let mut offset = first_leaf(fields, column);
    let mut field = &fields[column];
    let mut value_fallbacks = Vec::new();
    // Each shredding level holds a binary `value` fallback next to the
    // `typed_value` we descend. A level without a `value` child has no
    // fallback to guard against.
    for segment in path {
        if let Some((value_offset, _)) = child_leaf_offset(field, "value") {
            value_fallbacks.push(offset + value_offset);
        }
        let (typed_offset, typed) = child_leaf_offset(field, "typed_value")?;
        let (segment_offset, segment_field) = child_leaf_offset(typed, segment)?;
        offset += typed_offset + segment_offset;
        field = segment_field;
    }
    if let Some((value_offset, _)) = child_leaf_offset(field, "value") {
        value_fallbacks.push(offset + value_offset);
    }
    let (typed_offset, typed) = child_leaf_offset(field, "typed_value")?;
    if matches!(typed.data_type(), DataType::Struct(_)) {
        // The path names an object shredded further, not a typed leaf.
        return None;
    }
    Some(ShreddedPathLeaves {
        typed_value: offset + typed_offset,
        value_fallbacks,
    })
}

/// A plan for reading a bare (uncast) variant sub-extraction at a path: the
/// file leaf indices to read and the pruned variant field to fold them into.
///
/// Only the metadata, the residual `value` at each level along the path, and
/// the whole subtree at the path are read; sibling fields are skipped. Folding
/// the read leaves into `nest_field` yields a variant that is the input with
/// every off-path field dropped, so `variant_get(folded, path)` reconstructs
/// the sub-variant exactly as it would over the full column, residual fallbacks
/// and all, while touching far fewer column chunks.
pub(crate) struct VariantExtractPlan {
    /// File leaf (column-chunk) indices to read, in depth-first order.
    pub file_leaves: Vec<usize>,
    /// The pruned variant struct field the read leaves fold back into.
    pub nest_field: FieldRef,
}

/// Plan a bare variant extraction of `path` under variant `column` against this
/// file's `fields`. `None` when the path isn't shredded here (the caller reads
/// the whole variant instead).
pub(crate) fn plan_variant_extract(
    fields: &Fields,
    column: usize,
    path: &[String],
) -> Option<VariantExtractPlan> {
    let base = first_leaf(fields, column);
    let mut file_leaves = Vec::new();
    let nest_field = prune_variant_node(&fields[column], path, base, &mut file_leaves)?;
    Some(VariantExtractPlan {
        file_leaves,
        nest_field,
    })
}

/// Prune variant node `field` (a `{[metadata], [value], typed_value}` struct,
/// with `base` the file index of its first leaf) to the single branch that
/// reaches `path`: metadata and residual `value` children are kept, and
/// `typed_value` is narrowed to just the on-path child until `path` empties, at
/// which point the whole subtree is kept. Appends the kept leaves (absolute
/// file indices, depth-first) to `leaves`. `None` when `path` isn't shredded.
fn prune_variant_node(
    field: &FieldRef,
    path: &[String],
    base: usize,
    leaves: &mut Vec<usize>,
) -> Option<FieldRef> {
    let DataType::Struct(children) = field.data_type() else {
        return None;
    };
    let mut pruned = Vec::with_capacity(children.len());
    let mut cursor = base;
    for child in children {
        let count = leaf_count(child);
        if child.name() == "typed_value" {
            if path.is_empty() {
                // The path ends here: keep the whole subtree.
                leaves.extend(cursor..cursor + count);
                pruned.push(child.clone());
            } else {
                // Descend the shredded object to only its on-path child.
                let DataType::Struct(object) = child.data_type() else {
                    // The path continues but this level is a typed leaf.
                    return None;
                };
                let mut sub = cursor;
                let mut kept = None;
                for object_child in object {
                    if object_child.name() == path[0].as_str() {
                        kept = Some(prune_variant_node(object_child, &path[1..], sub, leaves)?);
                        break;
                    }
                    sub += leaf_count(object_child);
                }
                let kept = kept?;
                pruned.push(Arc::new(Field::new(
                    child.name(),
                    DataType::Struct(Fields::from(vec![kept.as_ref().clone()])),
                    child.is_nullable(),
                )));
            }
        } else {
            // A metadata or residual `value` leaf: keep and read it.
            leaves.extend(cursor..cursor + count);
            pruned.push(child.clone());
        }
        cursor += count;
    }
    Some(Arc::new(Field::new(
        field.name(),
        DataType::Struct(Fields::from(pruned)),
        field.is_nullable(),
    )))
}

/// The leaf offset (within `field`) and field of `field`'s direct child named
/// `name`, counting the leaves of earlier siblings; `None` when `field` isn't
/// a struct or has no such child.
fn child_leaf_offset<'a>(field: &'a FieldRef, name: &str) -> Option<(usize, &'a FieldRef)> {
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

/// Rebuilds top-level columns from decoded leaf arrays.
///
/// The decoder hands us one flat array per leaf, in the same depth-first
/// order the file stores its column chunks. The output batch wants one array
/// per top-level field, where a struct field is a [`StructArray`] holding its
/// children. So we walk the schema's fields in order: a primitive field
/// simply takes the next decoded array, and a struct field first collects an
/// array for each of its children (its own structs collect theirs, and so
/// on), then wraps them in a `StructArray`. For a flat schema this is a plain
/// pass-through. The inverse of [`leaf_fields`].
///
/// A struct wrapper's null buffer is derived from its non-nullable children:
/// a null in such a child can only mean the struct (or an ancestor) is null
/// at that row, since the child itself never is. Without this mask,
/// `StructArray::new` rejects the child's nulls as unmasked. A nullable struct
/// left without a mask that way takes instead the one that is null where every
/// child is: a variant's `typed_value` needs it, since its reader tells a row
/// that has a value for the field from one that does not by exactly that mask.
pub(crate) fn nest_leaves_into_columns(
    fields: &Fields,
    leaf_arrays: &mut impl Iterator<Item = ArrayRef>,
) -> Vec<ArrayRef> {
    /// A struct column mid-rebuild (or the top level itself): the fields it
    /// wants, and the arrays collected for them so far. The next field to
    /// fill is always `fields[arrays.len()]`.
    struct PartialStruct {
        fields: Fields,
        arrays: Vec<ArrayRef>,
        /// Whether the field this struct fills is itself nullable. A struct that
        /// cannot be null must not be given a mask, both because it would say
        /// something untrue and because its parent reads a null in it as proof
        /// that the parent is null.
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
        fn next_unfilled_field(&self) -> Option<&FieldRef> {
            self.fields.get(self.arrays.len())
        }
        fn into_struct_array(self) -> ArrayRef {
            // Null wherever any non-nullable child is null: the child itself
            // can't be, so the null came from this struct or an ancestor.
            let mut nulls: Option<NullBuffer> = None;
            for (field, array) in self.fields.iter().zip(&self.arrays) {
                if !field.is_nullable() {
                    nulls = NullBuffer::union(nulls.as_ref(), array.nulls());
                }
            }
            // No non-nullable child said anything, so fall back to the only
            // thing the children still say between them: the struct is null
            // exactly where all of them are. A variant's `typed_value` needs
            // this, since its reader tells a row that has a value for the field
            // from one that does not by that mask. Only for a struct that can be
            // null in the first place: giving a non-nullable one a mask reads,
            // to the rule above, as its parent being null, which would drop the
            // whole enclosing struct wherever one field happened to be absent.
            if self.nullable && nulls.is_none() {
                nulls = null_where_every_child_is_null(&self.arrays);
            }
            Arc::new(StructArray::new(self.fields, self.arrays, nulls))
        }
    }

    // Keep the open structs in a stack to support deeply nested schemas.
    // The top level is the batch itself and is never null.
    let mut open = vec![PartialStruct::start(fields, false)];
    loop {
        let innermost = open.last().expect("the top level only closes by returning");
        match innermost.next_unfilled_field().cloned() {
            Some(field) => match field.data_type() {
                // The next field is itself a struct: open it and fill it first.
                DataType::Struct(children) => {
                    open.push(PartialStruct::start(children, field.is_nullable()))
                }
                // A primitive field owns exactly the next decoded leaf array.
                _ => open.last_mut().expect("just inspected").arrays.push(
                    leaf_arrays
                        .next()
                        .expect("one decoded leaf array per leaf field"),
                ),
            },
            // Every field is filled: close this struct and hand it to the one
            // it lives in as a single array. Closing the top level means every
            // column is rebuilt, and those are the output.
            None => {
                let finished = open.pop().expect("just inspected");
                match open.last_mut() {
                    Some(enclosing) => enclosing.arrays.push(finished.into_struct_array()),
                    None => return finished.arrays,
                }
            }
        }
    }
}

/// The mask that is null exactly where every one of `arrays` is null, or `None`
/// when there is no such row. A child that is null nowhere makes the struct
/// present everywhere, which is why one of those ends the search.
fn null_where_every_child_is_null(arrays: &[ArrayRef]) -> Option<NullBuffer> {
    let mut valid: Option<BooleanBuffer> = None;
    for array in arrays {
        let child = rows_present_in(array)?;
        valid = Some(match valid {
            None => child,
            Some(valid) => &valid | &child,
        });
    }
    valid.map(NullBuffer::new)
}

/// The rows where `array` holds something, or `None` when that is every row.
///
/// A struct carrying a mask of its own is answered by it. One without has to be
/// answered by its descendants, because the struct that says whether a variant
/// field is present sits above children that are themselves non-nullable, and a
/// non-nullable child never carries a mask. What it has instead is its own
/// children, one level further down, being null.
fn rows_present_in(array: &ArrayRef) -> Option<BooleanBuffer> {
    if let Some(nulls) = array.nulls() {
        return Some(nulls.inner().clone());
    }
    let structure = array.as_any().downcast_ref::<StructArray>()?;
    let mut valid: Option<BooleanBuffer> = None;
    for child in structure.columns() {
        let child = rows_present_in(child)?;
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
    use arrow_array::{BinaryViewArray, Int64Array};
    use arrow_schema::Field;

    /// A shredded variant struct:
    /// `{metadata, value, typed_value{<path>{value, typed_value}}}`.
    fn variant(paths: &[(&str, DataType)]) -> Field {
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

    /// A variant shredded on `paths`, laid out the way a written file has it:
    /// each path's `{value, typed_value}` pair is non-nullable, and only the
    /// `typed_value` holding them says whether a row has any of them at all.
    fn file_shredded_variant(paths: &[(&str, DataType)]) -> Field {
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

    fn path(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| s.to_string()).collect()
    }

    /// A row that has none of the shredded paths must come back null at
    /// `typed_value`, which is how a reader tells "this row had no value for
    /// the field" from "it had one". Nothing else carries that: the pairs
    /// beneath are non-nullable and never hold a mask of their own.
    #[test]
    fn shredded_variant_is_null_where_a_row_has_no_shredded_path() {
        let fields = Fields::from(vec![file_shredded_variant(&[
            ("a", DataType::Int64),
            ("b", DataType::Int64),
        ])]);
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

        let columns = nest_leaves_into_columns(&fields, &mut leaves);

        let variant = columns[0].as_any().downcast_ref::<StructArray>().unwrap();
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
    fn a_shredded_path_missing_from_a_row_does_not_null_the_variant() {
        let fields = Fields::from(vec![file_shredded_variant(&[
            ("a", DataType::Int64),
            ("b", DataType::Int64),
        ])]);
        let mut leaves = vec![
            Arc::new(BinaryViewArray::from(vec![Some(&b"m"[..]); 2])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1), Some(2)])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![None, None])) as ArrayRef,
        ]
        .into_iter();

        let columns = nest_leaves_into_columns(&fields, &mut leaves);

        // Every row has `a`; none has `b`. Both rows are still present.
        assert_eq!(columns[0].null_count(), 0);
        let variant = columns[0].as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(variant.column(2).null_count(), 0);
    }

    #[test]
    fn variant_typed_leaf_resolves_each_shredded_path() {
        let fields = Fields::from(vec![variant(&[
            ("age", DataType::Int64),
            ("name", DataType::Utf8),
        ])]);

        let age = variant_shredded_leaves(&fields, 0, &path(&["age"])).unwrap();
        let name = variant_shredded_leaves(&fields, 0, &path(&["name"])).unwrap();

        // metadata=0 value=1 | age.value=2 age.typed_value=3
        //                    | name.value=4 name.typed_value=5
        assert_eq!(age.typed_value, 3);
        assert_eq!(age.value_fallbacks, vec![1, 2]);
        assert_eq!(name.typed_value, 5);
        assert_eq!(name.value_fallbacks, vec![1, 4]);
    }

    #[test]
    fn variant_typed_leaf_is_none_for_an_unshredded_path() {
        let fields = Fields::from(vec![variant(&[("age", DataType::Int64)])]);

        assert!(variant_shredded_leaves(&fields, 0, &path(&["missing"])).is_none());
    }

    #[test]
    fn plan_variant_extract_reads_only_the_path_subtree() {
        let fields = Fields::from(vec![variant(&[
            ("age", DataType::Int64),
            ("name", DataType::Utf8),
        ])]);

        let plan = plan_variant_extract(&fields, 0, &path(&["age"])).unwrap();

        // metadata=0 value=1 | age.value=2 age.typed_value=3; name's 4,5 skipped.
        assert_eq!(plan.file_leaves, vec![0, 1, 2, 3]);
        // The pruned field folds exactly the read leaves and nothing else.
        assert_eq!(leaf_count(&plan.nest_field), plan.file_leaves.len());
    }

    #[test]
    fn plan_variant_extract_keeps_residuals_along_a_nested_path() {
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

        let plan = plan_variant_extract(&fields, 0, &path(&["user", "id"])).unwrap();

        // metadata=0 value=1 user.value=2 user.id.value=3 user.id.typed_value=4: every
        // residual `value` along the path is read so `variant_get` stays sound.
        assert_eq!(plan.file_leaves, vec![0, 1, 2, 3, 4]);
        assert_eq!(leaf_count(&plan.nest_field), plan.file_leaves.len());
    }

    #[test]
    fn plan_variant_extract_is_none_for_an_unshredded_path() {
        let fields = Fields::from(vec![variant(&[("age", DataType::Int64)])]);

        assert!(plan_variant_extract(&fields, 0, &path(&["missing"])).is_none());
    }

    #[test]
    fn variant_typed_leaf_counts_preceding_columns() {
        let fields = Fields::from(vec![
            Field::new("id", DataType::Int64, false),
            variant(&[("age", DataType::Int64)]),
        ]);

        let age = variant_shredded_leaves(&fields, 1, &path(&["age"])).unwrap();

        // id=0 | metadata=1 value=2 age.value=3 age.typed_value=4
        assert_eq!(age.typed_value, 4);
        assert_eq!(age.value_fallbacks, vec![2, 3]);
    }

    #[test]
    fn variant_typed_leaf_resolves_a_nested_path() {
        // `user` shredded as an object whose own typed_value shreds `id`.
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

        let leaves = variant_shredded_leaves(&fields, 0, &path(&["user", "id"])).unwrap();

        // metadata=0 value=1 user.value=2 user.id.value=3 user.id.typed_value=4
        assert_eq!(leaves.typed_value, 4);
        assert_eq!(leaves.value_fallbacks, vec![1, 2, 3]);
    }
}
