//! Generic column projection, with no dependency on any particular file format.
//!
//! A [`Projection`] (which columns to read) is the one scan concept the engine
//! itself passes around. Row-group statistics and pruning predicates are a
//! storage concern and live in `catalog`, not here.

use arrow_schema::Schema;

/// An ordered set of column indices that a scan should read.
///
/// Can be built from explicit indices, from an Arrow [`Schema`], or by
/// resolving field names against a schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Column indices to read.
    pub column_indices: Vec<usize>,
}

impl Projection {
    /// Create a projection for specific column indices.
    pub fn columns(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            column_indices: indices.into_iter().collect(),
        }
    }

    /// Create a projection that includes all `num_columns` columns (`0..num_columns`).
    pub fn all(num_columns: usize) -> Self {
        Self {
            column_indices: (0..num_columns).collect(),
        }
    }

    /// Create a projection that includes every field in `schema`.
    pub fn all_from_schema(schema: &Schema) -> Self {
        Self::all(schema.fields().len())
    }

    /// Resolve field names to column indices via `schema`.
    ///
    /// Panics if any name is not found in the schema.
    pub fn from_field_names<'a>(schema: &Schema, names: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            column_indices: names
                .into_iter()
                .map(|name| {
                    schema
                        .index_of(name)
                        .unwrap_or_else(|_| panic!("field '{name}' not found in schema"))
                })
                .collect(),
        }
    }

    /// Returns the projected column indices as a slice.
    pub fn indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// Returns `true` if `column_idx` is part of this projection.
    pub fn includes(&self, column_idx: &usize) -> bool {
        self.column_indices.contains(column_idx)
    }
}
