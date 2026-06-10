//! Metadata types that describe Parquet row groups and their column chunks.
//!
//! These structs sit between the raw Parquet file metadata and the rest of the
//! query engine. [`RowGroupMetadata`] captures the static, file-level
//! information for a single row group (offsets, sizes, schema), while
//! [`QueryRowGroupMetadata`] augments it with per-query state such as the set
//! of row indices that survived predicate filtering. [`ColumnChunkMeta`] holds
//! the byte-level layout of an individual column chunk needed by the
//! decompressor to locate pages on disk.

use crate::parquet::types::table::ParquetTable;
use arrow_array::{ArrayRef, Scalar};
use arrow_schema::SchemaRef;
use dispatch::io::{FileLocation, RemoteFile};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Where a row group's bytes live, and how the file cache addresses them: a
/// local file (read via the io_uring file path) or a remote object (read via
/// HTTP range requests on the same ring). A [`FileLocation`] is derived from
/// this to key the cache and to tell the fetcher which kind of IO to issue.
#[derive(Clone)]
pub enum FileSource {
    /// An open local file. The `Arc<File>` keeps the fd alive for the cache.
    Local(Arc<File>),
    /// A remote object addressed by an (already DNS-resolved, possibly
    /// presigned) URL.
    Remote(Arc<RemoteFile>),
}

impl FileSource {
    /// The [`FileLocation`] used to key the file cache and select the IO path.
    pub fn location(&self) -> FileLocation {
        match self {
            FileSource::Local(file) => FileLocation::Local(file.as_raw_fd()),
            FileSource::Remote(remote) => FileLocation::Remote(remote.clone()),
        }
    }
}

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
    /// Maximum definition level for this column (indicates nesting / nullability depth).
    pub max_def_level: i16,
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
    /// Where the Parquet file's bytes live (local fd or remote object).
    pub source: FileSource,
    /// Arrow schema describing the columns in this row group.
    pub schema: SchemaRef,
    /// Per-column-chunk byte layout (offsets and sizes).
    pub columns: Vec<ColumnChunkMeta>,
    /// Total number of rows in this row group.
    pub num_rows: i64,
    /// Index of this row group within its Parquet file.
    pub file_row_group_idx: usize,
    /// The global index of the row group within the table's `Vec<RowGroupMetadata>`. Note that this
    /// should not be conflated with the row group number within a particular parquet file.
    pub global_row_group_idx: usize,
}

impl RowGroupMetadata {
    /// Decoded min/max (and counts) for column `idx`, if the writer recorded
    /// statistics for it.
    pub fn column_statistics(&self, idx: usize) -> Option<&ColumnStatistics> {
        self.columns.get(idx).and_then(|c| c.statistics.as_ref())
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
    /// Global row-group index (same as [`RowGroupMetadata::global_row_group_idx`]).
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

    /// A fresh per-query view of the same row group, reading only
    /// `filtered_indices` (or everything when `None`), with its own pruned
    /// flag. Used by the staged scan to build the phase-B (remaining columns)
    /// request from the phase-A (filter columns) metadata: the new view shares
    /// the static [`RowGroupMetadata`] but none of the old read's state.
    pub fn with_filtered_indices(&self, filtered_indices: Option<Vec<u32>>) -> Self {
        Self {
            row_group_metadata: self.row_group_metadata.clone(),
            filtered_indices,
            row_group_index: self.row_group_index,
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
