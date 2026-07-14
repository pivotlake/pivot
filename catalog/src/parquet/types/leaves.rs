//! Flattening a nested Arrow schema into leaf columns, and folding decoded leaf
//! arrays back into it.
//!
//! Parquet stores (and pivot decodes) one column chunk per *leaf* of the
//! schema tree, depth-first; a flat schema is the degenerate case where every
//! top-level field is its own leaf. [`leaf_fields`] is the per-leaf primitive
//! field list in chunk order, shared by the footer reader (per-leaf statistics)
//! and the row-group decoder (per-leaf decoder types).
//! [`nest_leaves_into_columns`] is the inverse: it folds a decoded depth-first
//! stream of leaf arrays back under their struct parents to match the nested
//! output schema.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, FieldRef, Fields};

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

/// The leaf (column-chunk) indices a projection of top-level `columns` reads,
/// resolved against this file's `fields`: each column expands to its run of
/// leaves. A flat column is one leaf, so for a flat schema this is the
/// identity. The fetcher and the decoder both iterate this list, so a fetched
/// chunk's position always lines up with its decoder, even when another file
/// shreds a variant column into a different number of leaves.
pub(crate) fn projected_leaves(fields: &Fields, columns: &[usize]) -> Vec<usize> {
    let mut leaves = Vec::with_capacity(columns.len());
    for &column in columns {
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
/// Shredding nests one level per path segment: the leaf for `user.id` is
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

/// Turn decoded leaf arrays back into the schema's columns.
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
/// `StructArray::new` rejects the child's nulls as unmasked. Structs whose
/// children are all nullable get no mask; their null rows decode as
/// all-null children, which readers treat the same way.
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
    }
    impl PartialStruct {
        fn start(fields: &Fields) -> Self {
            Self {
                fields: fields.clone(),
                arrays: Vec::with_capacity(fields.len()),
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
            Arc::new(StructArray::new(self.fields, self.arrays, nulls))
        }
    }

    // `open` is the chain of structs we are currently inside, outermost (the
    // top level) first. Iterating with this explicit chain instead of
    // recursing means a deeply nested document schema can't overflow the call
    // stack.
    let mut open = vec![PartialStruct::start(fields)];
    loop {
        let innermost = open.last().expect("the top level only closes by returning");
        match innermost.next_unfilled_field().cloned() {
            Some(field) => match field.data_type() {
                // The next field is itself a struct: open it and fill it first.
                DataType::Struct(children) => open.push(PartialStruct::start(children)),
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

#[cfg(test)]
mod tests {
    use super::*;
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

    fn path(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| s.to_string()).collect()
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
