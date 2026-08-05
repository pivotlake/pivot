//! Messages that flow between the write pipeline's stages.
//!
//! The unit of encode work is one **column chunk** (a single column's values for
//! one row group): the [`partition`](super::partition) stage emits a
//! [`ColumnChunkJob`], the [`encoder`](super::encoder) turns it into an
//! [`EncodedColumnChunk`] holding one [`EncodedLeaf`] per leaf of that column
//! (PLAIN, or dictionary-encoded), and the [`assembler`](super::assembler) lays
//! the leaves out into a file. Each carries the row group's shared
//! [`RowGroupHeader`] for routing and provenance.

use std::sync::Arc;

use crate::SortBounds;
use arrow_array::ArrayRef;
use arrow_schema::{DataType, SchemaRef};
use dispatch::memory::{FileBytes, Slab};
use dispatch::{Identifier, WorkerIdOutput};
use thriftparquet::footer::Statistics;
use thriftparquet::general::Encoding;

/// Identifies a row group across the pipeline so its column chunks reassemble
/// together.
pub(crate) type RowGroupId = u64;

/// Identifies an output file across the pipeline so its row groups assemble
/// together (and route to one worker).
pub(crate) type FileId = u64;

/// Per-row-group provenance threaded from the [`partition`](super::partition)
/// stage to the [`assembler`](super::assembler). Every field is file-level, so a
/// file's row groups all carry the same one. Always present (an unpartitioned,
/// unsorted write carries one with `partition`/`sort_bounds` `None`). `Arc` so
/// the chunk stages clone it cheaply.
pub(crate) struct PartitionTag {
    /// The output file this row group belongs to; the assembler packs all row
    /// groups of one `file_id` into a single Parquet file.
    pub(crate) file_id: FileId,
    /// Total row groups in this file, so the assembler knows when it's complete.
    pub(crate) n_row_groups: usize,
    pub(crate) partition: Option<crate::PartitionValues>,
    /// The file's sort-key bounds (min/max per sort column over the whole file),
    /// recorded in the manifest. `None` when the table has no sort key. The
    /// per-row-group, per-column statistics the footer carries are a separate
    /// thing, computed by the encoder for every leaf.
    pub(crate) sort_bounds: Option<SortBounds>,
}

/// A finished Parquet file from the write pipeline, with the manifest metadata to
/// record for it (`partition`/`sort_bounds` are `None` for an unpartitioned,
/// unsorted write).
///
/// The bytes are the slabs its pages were encoded into, in file order, so the
/// file is never assembled into one buffer. They are ring memory, so this must
/// be dropped on the dispatch worker that assembled it, which is where the
/// upload operators consume it.
pub(crate) struct AssembledFile {
    pub bytes: FileBytes,
    /// The footer metadata written into `bytes`. Kept so a consumer that records
    /// the file's row groups can build them straight from here instead of parsing
    /// the footer back out of a file it just produced.
    pub metadata: thriftparquet::footer::FileMetaData,
    pub partition: Option<crate::PartitionValues>,
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
    /// The type each variant column shreds into before encoding, index-aligned
    /// with `schema`'s columns and `None` for every other column. Decided per
    /// file by the partition stage; applied per chunk by the encoder, which is
    /// what keeps the rewrite on the parallel side of the pipeline.
    pub(crate) shredding_types: Arc<[Option<DataType>]>,
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
    /// The page on the wire, header followed by the compressed body, in one
    /// slab. Every stage after this moves the slab rather than its bytes, so
    /// this is the only place a page's bytes are written.
    pub(crate) bytes: Slab,
}

/// One leaf's encoded pages. Parquet stores a chunk per *leaf*, not per column:
/// a flat column has exactly one, and a shredded variant one per primitive under
/// its `{metadata, value, typed_value{..}}` struct. A dictionary-encoded leaf
/// holds its distinct values in `dictionary_page` and RLE-encoded indices in
/// `data_pages`; otherwise the data pages carry the values themselves, in
/// whichever encoding `data_page_encoding` names.
pub(crate) struct EncodedLeaf {
    /// The leaf's path from its top-level column down, e.g. `["attrs",
    /// "typed_value", "user", "typed_value"]` — the footer's `path_in_schema`.
    pub(crate) path: Vec<String>,
    pub(crate) physical_type: i32,
    /// This leaf's min/max and null count over the row group, for the footer.
    /// Every leaf carries them, so a reader can prune by any column.
    pub(crate) statistics: Statistics,
    pub(crate) dictionary_page: Option<EncodedPage>,
    /// How the data pages encode their values, which the footer reports so a
    /// reader knows which decoder to use.
    pub(crate) data_page_encoding: Encoding,
    pub(crate) data_pages: Vec<EncodedPage>,
}

/// A fully-encoded column chunk, routed back to its row group's owner worker for
/// assembly: one column's leaves, in the depth-first order Parquet numbers them.
pub(crate) struct EncodedColumnChunk {
    pub(crate) header: Arc<RowGroupHeader>,
    pub(crate) column: usize,
    pub(crate) leaves: Vec<EncodedLeaf>,
}

impl WorkerIdOutput for EncodedColumnChunk {
    fn worker_id(&self) -> Identifier {
        self.header.dest_worker
    }
}
