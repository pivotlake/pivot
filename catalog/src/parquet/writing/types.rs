//! Messages that flow between the write pipeline's stages.
//!
//! The unit of encode work is one **column chunk** (a single column's values for
//! one row group): the [`partition`](super::partition) stage emits a
//! [`ColumnChunkJob`], the [`encoder`](super::encoder) turns it into an
//! [`EncodedColumnChunk`] (PLAIN, or dictionary-encoded), and the
//! [`assembler`](super::assembler) lays the chunks out into a file. Each carries
//! the row group's shared [`RowGroupHeader`] for routing and provenance.

use std::collections::HashMap;
use std::sync::Arc;

use crate::SortBounds;
use arrow_array::{ArrayRef, Scalar};
use arrow_schema::SchemaRef;
use dispatch::{Identifier, WorkerIdOutput};

/// Identifies a row group across the pipeline so its column chunks reassemble
/// together.
pub(crate) type RowGroupId = u64;

/// Identifies an output file across the pipeline so its row groups assemble
/// together (and route to one worker).
pub(crate) type FileId = u64;

/// One sort column's min/max (+ null count) over a single row group, as
/// single-element Arrow arrays — written into the row group's footer
/// `Statistics` and aggregated into the file's manifest `sort_bounds`.
pub(crate) struct SortColStat {
    /// Column index within the schema.
    pub(crate) column: usize,
    pub(crate) min: ArrayRef,
    pub(crate) max: ArrayRef,
    pub(crate) null_count: i64,
}

/// The sort-column statistics for one row group (one [`SortColStat`] per
/// `sort_by` column; empty when the table has no sort key).
pub(crate) struct RowGroupSortStats {
    pub(crate) cols: Vec<SortColStat>,
}

/// Per-row-group provenance threaded from the [`partition`](super::partition) stage to
/// the [`assembler`](super::assembler). The file-level fields (`file_id`,
/// `n_row_groups`, `partition`) are shared by every row group of a file;
/// `sort_stats` is this row group's own. Always present (an unpartitioned,
/// unsorted write carries one with `partition`/`sort_bounds` `None` and empty
/// `sort_stats`). `Arc` so the chunk stages clone it cheaply.
pub(crate) struct PartitionTag {
    /// The output file this row group belongs to; the assembler packs all row
    /// groups of one `file_id` into a single Parquet file.
    pub(crate) file_id: FileId,
    /// Total row groups in this file, so the assembler knows when it's complete.
    pub(crate) n_row_groups: usize,
    pub(crate) partition: Option<HashMap<String, Scalar<ArrayRef>>>,
    /// The file's sort-key bounds (min/max per sort column over the whole file),
    /// recorded in the manifest. File-level: the same on every row group of the
    /// file. `None` when the table has no sort key.
    pub(crate) sort_bounds: Option<SortBounds>,
    /// This row group's own sort-column min/max, written into its footer
    /// `Statistics`.
    pub(crate) sort_stats: RowGroupSortStats,
}

/// A finished Parquet file from the write pipeline, with the manifest metadata to
/// record for it (`partition`/`sort_bounds` are `None` for an unpartitioned,
/// unsorted write).
pub struct EncodedFile {
    pub bytes: Vec<u8>,
    /// The footer metadata written into `bytes`. Kept so a consumer that records
    /// the file's row groups can build them straight from here instead of parsing
    /// the footer back out of a file it just produced.
    pub metadata: thriftparquet::footer::FileMetaData,
    pub partition: Option<HashMap<String, Scalar<ArrayRef>>>,
    pub sort_bounds: Option<SortBounds>,
}

/// The per-row-group metadata every column chunk of a row group shares: its
/// identity and owner worker (for routing), schema, and [`PartitionTag`]. Built
/// once by the [`partition`](super::partition) stage and shared by `Arc`, so each
/// column job clones a single pointer instead of re-copying all of this. The row
/// group is complete once the assembler has one chunk per `schema` column.
pub(crate) struct RowGroupHeader {
    pub(crate) row_group_id: RowGroupId,
    /// Worker that owns this row group; its chunks all route there for assembly
    /// (`file_id % worker_count`, so a file's row groups co-locate).
    pub(crate) dest_worker: usize,
    pub(crate) schema: SchemaRef,
    pub(crate) tag: Arc<PartitionTag>,
}

/// One column's values for one row group, to encode into a column chunk.
pub(crate) struct ColumnChunkJob {
    pub(crate) header: Arc<RowGroupHeader>,
    /// Index of this column in the schema.
    pub(crate) column: usize,
    pub(crate) values: ArrayRef,
}

/// An encoded page (data or dictionary): its snappy-compressed body behind a
/// thrift header. The sizes feed the column-chunk footer metadata.
pub(crate) struct EncodedPage {
    pub(crate) num_rows: i64,
    /// Uncompressed size of the page body (before snappy).
    pub(crate) uncompressed_size: usize,
    /// Size of the page's thrift header (precedes the compressed body).
    pub(crate) header_len: usize,
    /// The page on the wire: header followed by the compressed body.
    pub(crate) bytes: Vec<u8>,
}

/// A fully-encoded column chunk, routed back to its row group's owner worker for
/// assembly. Either PLAIN (`dictionary_page` is `None`) or dictionary-encoded
/// (`dictionary_page` holds the distinct values; `data_pages` hold RLE-encoded
/// indices).
pub(crate) struct EncodedColumnChunk {
    pub(crate) header: Arc<RowGroupHeader>,
    pub(crate) column: usize,
    pub(crate) dictionary_page: Option<EncodedPage>,
    pub(crate) data_pages: Vec<EncodedPage>,
}

impl WorkerIdOutput for EncodedColumnChunk {
    fn worker_id(&self) -> Identifier {
        self.header.dest_worker
    }
}
