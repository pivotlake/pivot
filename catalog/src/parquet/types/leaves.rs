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

use arrow_array::{ArrayRef, StructArray};
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

/// The column-chunk index of the typed leaf that `path` is shredded into under
/// the variant column `column`, resolved in THIS file's schema; `None` when the
/// file doesn't shred that path. Used to prune row groups by a shredded path's
/// statistics, and `None` (no pruning) is always sound.
///
/// Shredding nests one level per path segment: the leaf for `user.id` is
/// `column.typed_value.user.typed_value.id.typed_value`.
pub(crate) fn variant_typed_leaf(fields: &Fields, column: usize, path: &[String]) -> Option<usize> {
    let mut names = Vec::with_capacity(path.len() * 2 + 1);
    for segment in path {
        names.push("typed_value");
        names.push(segment.as_str());
    }
    names.push("typed_value");
    let offset = field_leaf_offset(&fields[column], &names)?;
    Some(first_leaf(fields, column) + offset)
}

/// Depth-first leaf offset, within `field`, of the descendant reached by the
/// `names` path (each a child field name), or `None` if the path doesn't lead
/// to a leaf. Counts the leaves of earlier siblings skipped along the way.
fn field_leaf_offset(field: &FieldRef, names: &[&str]) -> Option<usize> {
    let Some((head, tail)) = names.split_first() else {
        // Path fully consumed: the target is reached iff it's a leaf.
        return match field.data_type() {
            DataType::Struct(_) => None,
            _ => Some(0),
        };
    };
    let DataType::Struct(children) = field.data_type() else {
        return None; // the path continues, but we're already at a leaf
    };
    let mut offset = 0;
    for child in children {
        if child.name() == head {
            return field_leaf_offset(child, tail).map(|inner| offset + inner);
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
/// The struct wrappers get no null buffer of their own: when a row's struct
/// value is null, the file's definition levels already made every leaf of
/// that row null, and readers of a struct column take a row's nullness from
/// the leaves.
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
            Arc::new(StructArray::new(self.fields, self.arrays, None))
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

        // metadata=0 value=1 | age.value=2 age.typed_value=3
        //                    | name.value=4 name.typed_value=5
        assert_eq!(variant_typed_leaf(&fields, 0, &path(&["age"])), Some(3));
        assert_eq!(variant_typed_leaf(&fields, 0, &path(&["name"])), Some(5));
    }

    #[test]
    fn variant_typed_leaf_is_none_for_an_unshredded_path() {
        let fields = Fields::from(vec![variant(&[("age", DataType::Int64)])]);

        assert_eq!(variant_typed_leaf(&fields, 0, &path(&["missing"])), None);
    }

    #[test]
    fn variant_typed_leaf_counts_preceding_columns() {
        let fields = Fields::from(vec![
            Field::new("id", DataType::Int64, false),
            variant(&[("age", DataType::Int64)]),
        ]);

        // id=0 | metadata=1 value=2 age.value=3 age.typed_value=4
        assert_eq!(variant_typed_leaf(&fields, 1, &path(&["age"])), Some(4));
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

        // metadata=0 value=1 user.value=2 user.id.value=3 user.id.typed_value=4
        assert_eq!(
            variant_typed_leaf(&fields, 0, &path(&["user", "id"])),
            Some(4)
        );
    }
}
