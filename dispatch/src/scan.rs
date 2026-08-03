//! Generic column projection, with no dependency on any particular file format.
//!
//! A [`Projection`] (which columns to read) is the one scan concept the engine
//! itself passes around. Row-group statistics and pruning predicates are a
//! storage concern and live in `catalog`, not here.

use arrow_schema::Schema;

/// Field name of the global row-group column a scan appends when emitting
/// row-group metadata for late materialization (a run-end-encoded group id).
pub const ROW_GROUP_IDX_FIELD: &str = "row_group_idx";
/// Field name of the per-row index column appended alongside [`ROW_GROUP_IDX_FIELD`].
pub const ROW_IDX_FIELD: &str = "row_idx";

/// Number of trailing row-group-metadata columns on a batch with this schema
/// (`2` if it ends with the [`ROW_GROUP_IDX_FIELD`]/[`ROW_IDX_FIELD`] pair a
/// metadata-emitting scan appends, else `0`).
///
/// Operators that drop or reorder columns between such a scan and the parquet
/// materializer (notably projections) consult this so they can carry the
/// metadata pair through untouched — the materializer reads it positionally
/// from the end of the batch.
pub fn trailing_metadata_columns(schema: &Schema) -> usize {
    let fields = schema.fields();
    let n = fields.len();
    if n >= 2
        && fields[n - 2].name() == ROW_GROUP_IDX_FIELD
        && fields[n - 1].name() == ROW_IDX_FIELD
    {
        2
    } else {
        0
    }
}

/// A pushed-down variant field extract on a projected column: read only the
/// leaves at `path` and emit them, instead of materializing the whole variant.
/// Produced when DuckDB's projection-pushdown pushes a `variant_extract` into
/// the scan; the reader resolves the path against each file's shredding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantExtract {
    /// Object-field path from the variant root to the referenced field.
    pub path: Vec<String>,
    /// The physical arrow type to emit the extracted value as, when the path
    /// is cast to a scalar. `None` for a bare extraction, which yields a
    /// sub-variant (the object at the path) rather than a typed leaf.
    pub as_type: Option<arrow_schema::DataType>,
}

/// An ordered set of column indices that a scan should read.
///
/// Can be built from explicit indices, from an Arrow [`Schema`], or by
/// resolving field names against a schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Column indices to read.
    pub column_indices: Vec<usize>,
    /// Per output column (parallel to `column_indices`), a pushed field extract
    /// to apply instead of reading the whole column. Empty means no extracts:
    /// every output is a whole-column read (the common case).
    pub extracts: Vec<Option<VariantExtract>>,
}

impl Projection {
    /// The pushed extract for output column `output_idx`, if any. Safe whether
    /// or not `extracts` was populated.
    pub fn extract_at(&self, output_idx: usize) -> Option<&VariantExtract> {
        self.extracts.get(output_idx).and_then(|e| e.as_ref())
    }

    /// A projection of columns, some of which carry a pushed field extract
    /// (parallel to `indices`).
    pub fn columns_with_extracts(
        indices: impl IntoIterator<Item = usize>,
        extracts: Vec<Option<VariantExtract>>,
    ) -> Self {
        Self {
            column_indices: indices.into_iter().collect(),
            extracts,
        }
    }

    /// Create a projection for specific column indices.
    pub fn columns(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            column_indices: indices.into_iter().collect(),
            extracts: Vec::new(),
        }
    }

    /// Create a projection that includes all `num_columns` columns (`0..num_columns`).
    pub fn all(num_columns: usize) -> Self {
        Self {
            extracts: Vec::new(),
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
            extracts: Vec::new(),
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
