//! Flattens a column into the primitive leaves Parquet actually stores.
//!
//! Parquet has no nested storage: a column chunk holds one *leaf*, and a struct
//! exists only as a group in the footer schema. A flat column is its own single
//! leaf, so this is a no-op for every column the writer handled before variants;
//! a variant column shreds into `{metadata, value, typed_value{..}}` and yields
//! one leaf per primitive under it.
//!
//! Leaves come out depth-first, left to right — the order Parquet numbers column
//! chunks in, and the same order the reader's `leaf_fields` expands a schema
//! into. The two must agree, or every chunk after the first variant would line
//! up against the wrong decoder.
//!
//! ## Definition levels
//!
//! A nested leaf can be absent because it is null itself or because an ancestor
//! is, and Parquet stores only the values that are present. A definition level
//! per row says how far down the path the row got: `max_def_level` means the
//! leaf itself holds a value, anything less means it went missing at that depth.
//! Levels are what let a shredded `typed_value` be absent on exactly the rows
//! that fell back to the binary `value`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray};
use arrow_schema::{DataType, Field};

use super::super::error::WriteResult;

/// One primitive leaf of a column, ready to encode.
pub(super) struct Leaf {
    /// The leaf's path from the top-level column down, e.g. `["attrs",
    /// "typed_value", "user", "typed_value"]` — the column chunk's
    /// `path_in_schema` in the footer.
    pub(super) path: Vec<String>,
    /// The present rows' values, in order and with the absent rows dropped —
    /// exactly the values Parquet stores.
    pub(super) values: ArrayRef,
    /// One level per row of the column (not per stored value). `None` when
    /// nothing on the path is nullable, which is the flat, required case: every
    /// row is present, so the pages carry no level section at all.
    pub(super) def_levels: Option<Arc<[i16]>>,
    /// The level meaning "this leaf holds a value" — the number of nullable
    /// fields on the path. Zero exactly when `def_levels` is `None`.
    pub(super) max_def_level: i16,
}

impl Leaf {
    /// Rows this leaf spans, absent ones included. Parquet counts a page's
    /// values in rows rather than in stored values, because the definition
    /// levels cover every row.
    pub(super) fn rows(&self) -> usize {
        match &self.def_levels {
            Some(levels) => levels.len(),
            None => self.values.len(),
        }
    }

    /// Whether row `row` holds a stored value.
    pub(super) fn is_present(&self, row: usize) -> bool {
        match &self.def_levels {
            Some(levels) => levels[row] == self.max_def_level,
            None => true,
        }
    }
}

/// Flatten `field` into its primitive leaves, depth-first.
pub(super) fn flatten(field: &Field, values: &ArrayRef) -> WriteResult<Vec<Leaf>> {
    let mut leaves = Vec::new();
    // A top-level column starts defined at level 0 on every row: it has no
    // ancestor that could be absent.
    append_leaves(
        field,
        values,
        &mut Vec::new(),
        &Defined::root(),
        &mut leaves,
    )?;
    Ok(leaves)
}

/// How far down each row is defined at the node being walked, and the level that
/// means "defined all the way to here". A row below `max` went missing at an
/// ancestor and stays missing for every leaf beneath it.
///
/// The levels are shared rather than copied: descending through a required field
/// leaves them untouched, and a variant shreds into many such fields.
struct Defined {
    levels: Option<Arc<[i16]>>,
    max: i16,
}

impl Defined {
    /// The root: every row defined, at level 0.
    fn root() -> Self {
        Self {
            levels: None,
            max: 0,
        }
    }

    /// Descend into `field`, whose values are `values`. A nullable field adds a
    /// level: its non-null rows reach the deeper one, its null rows stop at this
    /// one. A required field cannot be absent, so it passes the levels straight
    /// through.
    fn descend(&self, field: &Field, values: &ArrayRef) -> Self {
        if !field.is_nullable() {
            return Self {
                levels: self.levels.clone(),
                max: self.max,
            };
        }
        let max = self.max + 1;
        let levels = (0..values.len())
            .map(|row| match &self.levels {
                // Already absent higher up: that level is this row's final one.
                Some(levels) if levels[row] < self.max => levels[row],
                _ if values.is_null(row) => self.max,
                _ => max,
            })
            .collect();
        Self {
            levels: Some(levels),
            max,
        }
    }
}

/// Append the leaves at or under `field` to `leaves`, depth-first. `path` names
/// the fields from the top-level column down to and including `field`.
fn append_leaves(
    field: &Field,
    values: &ArrayRef,
    path: &mut Vec<String>,
    parent: &Defined,
    leaves: &mut Vec<Leaf>,
) -> WriteResult<()> {
    path.push(field.name().clone());
    let defined = parent.descend(field, values);

    match field.data_type() {
        DataType::Struct(fields) => {
            // A struct stores nothing itself — it is only a footer group. Its
            // children carry its absence in their own levels.
            for (child, child_values) in fields.iter().zip(values.as_struct().columns()) {
                append_leaves(child.as_ref(), child_values, path, &defined, leaves)?;
            }
        }
        _ => leaves.push(build_leaf(path.clone(), values, defined)?),
    }

    path.pop();
    Ok(())
}

/// Build a leaf from the rows defined down to it: keep the present values, and
/// hand on the levels that say which rows those were.
fn build_leaf(path: Vec<String>, values: &ArrayRef, defined: Defined) -> WriteResult<Leaf> {
    let values = match &defined.levels {
        // Nothing on the path is nullable: the values are already what Parquet
        // stores, and filtering would only copy them.
        None => values.clone(),
        Some(levels) => {
            let present: BooleanArray = levels.iter().map(|&l| l == defined.max).collect();
            if present.false_count() == 0 {
                values.clone()
            } else {
                arrow_select::filter::filter(values, &present)?
            }
        }
    };
    Ok(Leaf {
        path,
        values,
        def_levels: defined.levels,
        max_def_level: defined.max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int64Type;
    use arrow_array::{Int64Array, StringArray, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields};

    fn field(name: &str, data_type: DataType, nullable: bool) -> Field {
        Field::new(name, data_type, nullable)
    }

    /// A flat required column is its own leaf with no levels at all — the path
    /// every non-variant column still takes.
    #[test]
    fn a_flat_required_column_is_one_leaf_without_levels() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));

        let leaves = flatten(&field("n", DataType::Int64, false), &values).unwrap();

        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].path, vec!["n"]);
        assert_eq!(leaves[0].max_def_level, 0);
        assert!(leaves[0].def_levels.is_none());
        assert_eq!(leaves[0].values.len(), 3);
    }

    /// A nullable leaf stores only its non-null values; the levels carry which
    /// rows those were.
    #[test]
    fn a_nullable_leaf_drops_its_null_values() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));

        let leaves = flatten(&field("n", DataType::Int64, true), &values).unwrap();

        assert_eq!(leaves[0].max_def_level, 1);
        assert_eq!(leaves[0].def_levels.as_deref(), Some([1, 0, 1].as_slice()));
        assert_eq!(
            leaves[0].values.as_primitive::<Int64Type>(),
            &Int64Array::from(vec![1, 3])
        );
        assert_eq!(leaves[0].rows(), 3);
    }

    /// A struct contributes no leaf of its own; its children come out
    /// depth-first, with the struct's name leading their paths.
    #[test]
    fn a_struct_yields_its_children_depth_first() {
        let inner = Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        let values: ArrayRef = Arc::new(StructArray::new(
            inner.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["x", "y"])),
            ],
            None,
        ));

        let leaves = flatten(&field("s", DataType::Struct(inner), false), &values).unwrap();

        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].path, vec!["s", "a"]);
        assert_eq!(leaves[1].path, vec!["s", "b"]);
    }

    /// A row absent at a nullable struct stays absent for the leaf below it, and
    /// that leaf's level records the depth it went missing at rather than its
    /// own max — the shape a shredded variant's `typed_value` takes.
    #[test]
    fn an_absent_struct_row_stops_its_childs_level_short() {
        let inner = Fields::from(vec![Field::new("a", DataType::Int64, true)]);
        // Row 1's struct is null, so its child's value there is never stored.
        let values: ArrayRef = Arc::new(StructArray::new(
            inner.clone(),
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))],
            Some(NullBuffer::from(vec![true, false, true])),
        ));

        let leaves = flatten(&field("s", DataType::Struct(inner), true), &values).unwrap();

        // max_def 2: 2 = value present, 1 = struct present but field null,
        // 0 = the struct itself absent.
        assert_eq!(leaves[0].max_def_level, 2);
        assert_eq!(leaves[0].def_levels.as_deref(), Some([2, 0, 2].as_slice()));
        assert_eq!(
            leaves[0].values.as_primitive::<Int64Type>(),
            &Int64Array::from(vec![1, 3])
        );
    }

    /// A leaf under a nullable path that happens to be null nowhere still stores
    /// every row, so the encoder keeps its no-copy path.
    #[test]
    fn a_fully_present_nullable_leaf_stores_every_row() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2)]));

        let leaves = flatten(&field("n", DataType::Int64, true), &values).unwrap();

        assert_eq!(leaves[0].def_levels.as_deref(), Some([1, 1].as_slice()));
        assert_eq!(leaves[0].values.len(), 2);
        assert!(leaves[0].is_present(0));
    }
}
