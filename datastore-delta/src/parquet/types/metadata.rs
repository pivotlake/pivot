//! Metadata types that describe Parquet row groups and their column chunks.
//!
//! These structs sit between the raw Parquet file metadata and the rest of the
//! query engine. [`RowGroupMetadata`] captures the static, file-level
//! information for a single row group (offsets, sizes, schema), while
//! [`QueryRowGroupMetadata`] augments it with per-query state such as the set
//! of row indices that survived predicate filtering. [`ColumnChunkMeta`] holds
//! the byte-level layout of an individual column chunk needed by the
//! decompressor to locate pages on disk.

use crate::parquet::filter_cache::FilterResultCache;
use crate::parquet::types::table::ParquetTable;
use arrow_array::{ArrayRef, Scalar};
use arrow_schema::SchemaRef;
use dispatch::io::OpenFile;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Decoded min/max (and counts) for one column of a row group. Each bound is an
/// arrow [`Scalar<ArrayRef>`] — the same shape the planner uses for SQL
/// constants — so a pushdown predicate can compare a query constant directly
/// against `min` / `max` without further conversion.
///
/// `None` for either side means the writer didn't record that bound (e.g.
/// all-null column, or stats omitted); a present `min`/`max` is a valid bound
/// but may be conservative rather than the true extremum.
#[derive(Clone, Debug)]
pub struct ColumnStatistics {
    pub min: Option<Scalar<ArrayRef>>,
    pub max: Option<Scalar<ArrayRef>>,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
}

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
    /// Decoded min/max for this chunk, when the writer recorded statistics
    /// and the column's Arrow type is one we know how to decode.
    pub statistics: Option<ColumnStatistics>,
    /// True when this chunk has a dictionary page and every one of its data
    /// pages is dictionary-encoded (per the footer's `encoding_stats`). Only
    /// then is it sound to prune the whole row group when the dictionary does
    /// not contain a pushed-down equality constant — otherwise a non-dictionary
    /// data page could hold a matching value absent from the dictionary.
    pub data_pages_all_dictionary: bool,
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
    /// Remembered outcomes of pushed-down equality predicates a scan evaluated
    /// against this row group's data. The next scan consults it at claim time
    /// and skips the row group when a remembered predicate (or conjunction of
    /// predicates) matched no row. Shared (`Arc`) like the row group itself, so
    /// every query sees one cache; dropped with the metadata when the file is
    /// rewritten.
    pub filter_cache: Arc<FilterResultCache>,
}

impl RowGroupMetadata {
    /// Decoded min/max (and counts) for the top-level column `column`, if the
    /// writer recorded statistics for it. Chunks are per leaf, so the column is
    /// resolved to its first leaf. Callers that need a nested field's statistics
    /// resolve that field to its own leaf instead.
    pub fn column_statistics(&self, column: usize) -> Option<&ColumnStatistics> {
        self.leaf_statistics(super::leaves::first_leaf(self.schema.fields(), column))
    }

    /// Decoded min/max (and counts) for the chunk at `leaf`, a raw column-chunk
    /// index. For pruning by a leaf that isn't a column's first (a shredded
    /// variant path's typed leaf); resolve the index against this row group's
    /// own schema, since leaf positions differ per file.
    pub fn leaf_statistics(&self, leaf: usize) -> Option<&ColumnStatistics> {
        self.columns.get(leaf).and_then(|c| c.statistics.as_ref())
    }
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
    /// The row group's global index — its position in the table's flat
    /// `row_groups` list, which is how the materializer addresses it back.
    pub row_group_index: usize,
    /// Shared across every page of this row group: the decoder flips it once the
    /// row group is pruned (e.g. a dictionary excludes a pushed-down equality
    /// constant), letting the decompressor skip the remaining, not-yet-touched
    /// pages instead of decompressing them only for the decoder to discard.
    pruned: Arc<AtomicBool>,
}

impl QueryRowGroupMetadata {
    pub fn new(table: &ParquetTable, index: usize, filtered_indices: Option<Vec<u32>>) -> Self {
        Self {
            row_group_metadata: table.row_groups[index].clone(),
            filtered_indices,
            row_group_index: index,
            pruned: Arc::new(AtomicBool::new(false)),
        }
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

    /// A handle to the shared pruned flag, for a holder (the row-group decoder)
    /// that needs to flip it later. Cloning is cheap — an `Arc` bump.
    pub fn pruned_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pruned)
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
