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
