//! Metadata types that describe Parquet row groups and their column chunks.
//!
//! These structs sit between the raw Parquet file metadata and the rest of the
//! query engine. [`RowGroupMetadata`] captures the static, file-level
//! information for a single row group (offsets, sizes, schema), while
//! [`QueryRowGroupMetadata`] augments it with per-query state such as the set
//! of row indices that survived predicate filtering. [`ColumnChunkMeta`] holds
//! the byte-level layout of an individual column chunk needed by the
//! decompressor to locate pages on disk.

use crate::operations::unary::parquet::types::table::ParquetTable;
use arrow_schema::SchemaRef;
use std::fs::File;
use std::sync::Arc;

/// Byte-level layout of a single column chunk within a row group.
///
/// Used by the decompressor to seek directly to the dictionary and data pages
/// on disk without re-parsing Parquet footer metadata at read time.
#[derive(Clone)]
pub struct ColumnChunkMeta {
    /// Offset of the dictionary page, if the column uses dictionary encoding.
    pub dictionary_page_offset: Option<i64>,
    /// Offset of the first data page.
    pub data_page_offset: i64,
    /// Total size of all compressed pages in this column chunk (bytes).
    pub total_compressed_size: i64,
    /// Maximum definition level for this column (indicates nesting / nullability depth).
    pub max_def_level: i16,
}

/// Static, file-level metadata for a single Parquet row group.
///
/// One instance exists per row group across all files that make up a
/// [`ParquetTable`]. It is cheaply shared (`Arc`) because multiple concurrent
/// queries may reference the same row group.
#[derive(Clone)]
pub struct RowGroupMetadata {
    /// Open file handle for the Parquet file that contains this row group.
    pub file: Arc<File>,
    /// Arrow schema describing the columns in this row group.
    pub schema: SchemaRef,
    /// Per-column-chunk byte layout (offsets and sizes).
    pub columns: Vec<ColumnChunkMeta>,
    /// Total number of rows in this row group.
    pub num_rows: i64,
    /// Index of this row group within its Parquet file.
    pub file_row_group_idx: usize,
    /// The global index of the row group within the table's Vec<RowGroupMetadata>. Note that this
    /// should not be conflated with the row group number within a particular parquet file.
    pub global_row_group_idx: usize,
}

/// Row-group metadata augmented with per-query filtering state.
///
/// Wraps a shared [`RowGroupMetadata`] and optionally carries the sorted row
/// indices that survived predicate evaluation. When `filtered_indices` is
/// `None`, the entire row group is read; when `Some`, only those rows are
/// materialized.
#[derive(Clone)]
pub struct QueryRowGroupMetadata {
    /// The underlying static row-group metadata.
    pub row_group_metadata: Arc<RowGroupMetadata>,
    /// Sorted row indices to read, or `None` to read the full row group.
    pub filtered_indices: Option<Vec<u32>>,
    /// Global row-group index (same as [`RowGroupMetadata::global_row_group_idx`]).
    pub row_group_index: usize,
}

impl QueryRowGroupMetadata {
    pub fn new(table: &ParquetTable, index: usize, filtered_indices: Option<Vec<u32>>) -> Self {
        Self {
            row_group_metadata: table.row_groups[index].clone(),
            filtered_indices,
            row_group_index: index,
        }
    }

    /// Get the corresponding RowGroupMetadata from the table
    pub fn get_metadata(&self) -> &RowGroupMetadata {
        &self.row_group_metadata
    }

    pub fn num_rows(&self) -> i64 {
        self.row_group_metadata.num_rows
    }

    pub fn columns(&self) -> &[ColumnChunkMeta] {
        &self.row_group_metadata.columns
    }

    pub fn index(&self) -> usize {
        self.row_group_index
    }

    pub fn filtered_indices(&self) -> &Option<Vec<u32>> {
        &self.filtered_indices
    }
}
