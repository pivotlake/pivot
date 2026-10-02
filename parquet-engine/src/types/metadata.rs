//! Metadata types that describe Parquet row groups and their column chunks.
//!
//! These structs sit between the raw Parquet file metadata and the rest of the
//! query engine. [`RowGroupMetadata`] captures the static, file-level
//! information for a single row group (offsets, sizes, schema), while
//! [`QueryRowGroupMetadata`] augments it with per-query state such as the set
//! of row indices that survived predicate filtering. [`ColumnChunkMeta`] holds
//! the byte-level layout of an individual column chunk needed by the
//! decompressor to locate pages on disk.

use crate::thrift::general::CompressionCodec;
use crate::types::table::ParquetTable;
use arrow_array::{Array, ArrayRef, Scalar};
use arrow_schema::{Fields, SchemaRef};
use dispatch::io::OpenFile;
use pruning::StatisticsBatch;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Physical leaf statistics for readers and logical column statistics for
/// pruning. Both views share the same Arrow min/max buffers; physical leaf
/// positions and counts remain a Parquet concern.
#[derive(Clone)]
pub(crate) struct FileStatistics {
    pub(crate) bounds: StatisticsBatch,
    leaves: Vec<Option<DecodedLeafStatistics>>,
}

impl Default for FileStatistics {
    fn default() -> Self {
        Self::new(Vec::new(), &[], &Fields::empty())
    }
}

impl FileStatistics {
    pub(crate) fn new(
        leaves: Vec<Option<DecodedLeafStatistics>>,
        row_counts: &[i64],
        fields: &Fields,
    ) -> Self {
        let columns = crate::pushdown::statistics_columns(fields, &leaves, row_counts);
        let bounds = StatisticsBatch::new(row_counts.len(), columns, Vec::new())
            .expect("footer columns have one slot per row group");
        Self { bounds, leaves }
    }
}

/// Decoded footer statistics retained in physical leaf order. Logical pruning
/// bounds share these arrays, including for safe typed VARIANT paths.
#[derive(Clone)]
pub(crate) struct DecodedLeafStatistics {
    pub(crate) min: Option<ArrayRef>,
    pub(crate) max: Option<ArrayRef>,
    pub(crate) null_counts: Vec<Option<i64>>,
    pub(crate) distinct_counts: Vec<Option<i64>>,
}

impl DecodedLeafStatistics {
    pub(crate) fn row(&self, row: usize) -> ColumnStatistics<'_> {
        ColumnStatistics {
            row_group: row,
            min: self.min.as_ref(),
            max: self.max.as_ref(),
            null_count: self.null_counts[row],
            distinct_count: self.distinct_counts[row],
        }
    }
}

/// One row group's view of a leaf's statistics. Scalar consumers such as
/// aggregate statistics and compaction borrow the canonical bounds arrays;
/// [`Self::min`] and [`Self::max`] slice them only when a scalar is requested.
pub struct ColumnStatistics<'a> {
    row_group: usize,
    min: Option<&'a ArrayRef>,
    max: Option<&'a ArrayRef>,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
}

impl<'a> ColumnStatistics<'a> {
    /// This row group's lower bound, or `None` where the writer recorded none
    /// (an all-null column, or statistics omitted). A present bound is valid but
    /// may be conservative rather than the true extremum.
    pub fn min(&self) -> Option<Scalar<ArrayRef>> {
        self.bound(self.min)
    }

    /// This row group's upper bound, on the same terms as [`Self::min`].
    pub fn max(&self) -> Option<Scalar<ArrayRef>> {
        self.bound(self.max)
    }

    /// Slice this row group's own value out of a leaf's column of bounds.
    fn bound(&self, column: Option<&'a ArrayRef>) -> Option<Scalar<ArrayRef>> {
        let column = column?;
        (!column.is_null(self.row_group)).then(|| Scalar::new(column.slice(self.row_group, 1)))
    }
}

/// Byte-level layout of a single column chunk within a row group.
///
/// Used by the decompressor to seek directly to the dictionary and data pages
/// on disk without re-parsing Parquet footer metadata at read time.
#[derive(Clone)]
pub struct ColumnChunkMeta {
    /// Compression codec of every page in this chunk, from the footer. Whether
    /// the engine can decompress it is checked when a page is actually read,
    /// so a chunk in an unsupported codec is fine as long as row-group pruning
    /// or the projection keeps its pages from being decoded.
    pub codec: CompressionCodec,
    /// Offset of the dictionary page, if the column uses dictionary encoding.
    pub dictionary_page_offset: Option<i64>,
    /// Offset of the first data page.
    pub data_page_offset: i64,
    /// Total size of all compressed pages in this column chunk (bytes).
    pub total_compressed_size: i64,
    /// What those pages hold decoded (bytes), before compression.
    pub total_uncompressed_size: i64,
    /// Maximum definition level for this column (indicates nesting / nullability depth).
    pub max_def_level: i16,
    /// The chunk's Parquet physical type id, disambiguating storages that
    /// share an arrow type (e.g. a decimal chunk may be INT32, INT64, or
    /// FIXED_LEN_BYTE_ARRAY).
    pub physical_type: i32,
    /// The schema's `type_length` for a FIXED_LEN_BYTE_ARRAY leaf: the byte
    /// width of each value.
    pub fixed_len_byte_width: Option<i32>,
    /// True when this chunk has a dictionary page and every one of its data
    /// pages is dictionary-encoded (per the footer's `encoding_stats`). Only
    /// then is it sound to prune the whole row group when the dictionary does
    /// not contain a pushed-down equality constant — otherwise a non-dictionary
    /// data page could hold a matching value absent from the dictionary.
    pub data_pages_all_dictionary: bool,
    /// True for a leaf the file has no chunk for: a declared column the file
    /// predates (see [`ColumnResolution::ByFieldId`]). Nothing is read for it;
    /// every row decodes as NULL.
    ///
    /// [`ColumnResolution::ByFieldId`]: crate::ColumnResolution::ByFieldId
    pub absent: bool,
}

impl ColumnChunkMeta {
    /// The chunk of a leaf the file has no chunk for. Its layout fields are
    /// placeholders: no reader dereferences them, because the leaf is never
    /// fetched. `max_def_level` is one, the level of an optional leaf, since an
    /// absent column is nullable by construction.
    pub fn absent() -> Self {
        Self {
            codec: CompressionCodec::UNCOMPRESSED,
            dictionary_page_offset: None,
            data_page_offset: 0,
            total_compressed_size: 0,
            total_uncompressed_size: 0,
            max_def_level: 1,
            physical_type: 0,
            fixed_len_byte_width: None,
            data_pages_all_dictionary: false,
            absent: true,
        }
    }
}

/// Static, file-level metadata for a single Parquet row group.
///
/// One instance exists per row group across all files that make up a
/// [`ParquetTable`]. It is cheaply shared (`Arc`) because multiple concurrent
/// queries may reference the same row group.
#[derive(Clone)]
pub struct RowGroupMetadata {
    /// The open file holding this row group's bytes (local file or remote
    /// object) — what the fetcher reads from and the compressed cache keys on.
    pub open_file: OpenFile,
    /// Arrow schema describing the columns in this row group.
    pub schema: SchemaRef,
    /// Per-column-chunk byte layout (offsets and sizes).
    pub columns: Vec<ColumnChunkMeta>,
    /// The whole file's statistics, shared by every row group in it and read at
    /// this row group's `file_row_group_idx`.
    pub(crate) statistics: Arc<FileStatistics>,
    /// Total number of rows in this row group.
    pub num_rows: i64,
    /// Index of this row group within its Parquet file. Note that this should
    /// not be conflated with the row group's *global* index, which is simply
    /// its position in the table's flat `row_groups` list.
    pub file_row_group_idx: usize,
    /// How many of this row group's pages are currently resident in the
    /// decompressed cache. The cache itself maintains it (a clone rides on every
    /// cached page, incremented on insert and decremented when the page leaves);
    /// the scan feed reads it to hand out cache-covered row groups first, so a
    /// repeated scan consumes what's already decompressed before its own churn
    /// can evict it.
    pub live_decompressed_pages: Arc<AtomicUsize>,
}

impl RowGroupMetadata {
    /// This row group's statistics for the top-level column `column`, if the
    /// file recorded any for it. Chunks are per leaf, so the column is resolved
    /// to its first leaf. Callers that need a nested field's statistics resolve
    /// that field to its own leaf instead.
    pub fn column_statistics(&self, column: usize) -> Option<ColumnStatistics<'_>> {
        self.leaf_statistics(super::leaves::first_leaf(self.schema.fields(), column))
    }

    /// Whether the top-level column `column` can hold a NULL in this row group.
    /// A `REQUIRED` column cannot; an `OPTIONAL` one (writers like DuckDB mark
    /// every column `OPTIONAL` even when no value is ever NULL) is refined by
    /// the chunk's `null_count` statistic when the schema is flat enough to map
    /// fields to leaves. Nested fields (a shredded variant) span several leaves,
    /// so the field-to-chunk mapping does not hold and the answer stays
    /// conservative.
    pub fn column_may_hold_nulls(&self, column: usize) -> bool {
        if !self.schema.field(column).is_nullable() {
            return false;
        }
        if self.columns.len() != self.schema.fields().len() {
            return true;
        }
        self.leaf_statistics(column)
            .and_then(|stats| stats.null_count)
            .is_none_or(|null_count| null_count > 0)
    }

    /// This row group's statistics for the chunk at `leaf`, a raw column-chunk
    /// index. For pruning by a leaf that isn't a column's first (a shredded
    /// variant path's typed leaf); resolve the index against this row group's
    /// own schema, since leaf positions differ per file.
    ///
    /// `None` means the file recorded nothing for that leaf in any of its row
    /// groups. A leaf the file records elsewhere but not here comes back with
    /// every field empty, which reads the same to every caller.
    pub fn leaf_statistics(&self, leaf: usize) -> Option<ColumnStatistics<'_>> {
        Some(
            self.statistics
                .leaves
                .get(leaf)?
                .as_ref()?
                .row(self.file_row_group_idx),
        )
    }
}

/// Which rows of a row group are read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowSelection {
    /// Every row of the row group.
    All,
    /// The sorted row indices that survived predicate evaluation, shared
    /// by every page and range of the row group.
    Indices(Arc<[u32]>),
}

/// Row-group metadata augmented with per-query filtering state.
///
/// Wraps a shared [`RowGroupMetadata`] and the [`RowSelection`] describing
/// which of the row group's rows are read through it.
#[derive(Clone)]
pub struct QueryRowGroupMetadata {
    /// The underlying static row-group metadata.
    pub row_group_metadata: Arc<RowGroupMetadata>,
    /// The rows read through this metadata.
    selection: RowSelection,
    /// The row group's global index — its position in the table's flat
    /// `row_groups` list, which is how the materializer addresses it back.
    pub row_group_index: usize,
    /// Shared across every page of this row group: flipped once the row group
    /// is pruned (a dictionary excludes a pushed-down equality constant), so
    /// the decompressor skips the remaining, not-yet-touched pages instead of
    /// decompressing them only to be discarded.
    pruned: Arc<AtomicBool>,
}

impl QueryRowGroupMetadata {
    /// Metadata for reading the `selection` of row group `index`.
    pub fn new(table: &ParquetTable, index: usize, selection: RowSelection) -> Self {
        Self {
            row_group_metadata: table.row_groups[index].clone(),
            selection,
            row_group_index: index,
            pruned: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The rows read through this metadata.
    pub fn selection(&self) -> &RowSelection {
        &self.selection
    }

    /// Whether this row group has been pruned (no row can match a pushed-down
    /// predicate), so its remaining pages need not be decompressed or decoded.
    pub fn is_pruned(&self) -> bool {
        self.pruned.load(Ordering::Relaxed)
    }

    /// Mark this row group pruned. Visible (best-effort) to every page sharing
    /// this metadata — in particular to the decompressor handling later pages.
    pub fn mark_pruned(&self) {
        self.pruned.store(true, Ordering::Relaxed);
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
}
