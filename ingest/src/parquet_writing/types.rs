//! Messages that flow between the write pipeline's stages.
//!
//! They mirror the read pipeline's page types: each carries the routing
//! metadata (row-group id, owner worker) a downstream stage needs to keep a row
//! group's data together, plus the payload for that stage. The unwrapped
//! [`PageJob`] / [`EncodedPage`] payloads are produced and consumed by adjacent
//! stages ([`planner`](super::planner) → [`encoder`](super::encoder) →
//! [`assembler`](super::assembler)), so they live here rather than in any one
//! stage.

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use dispatch::{Identifier, WorkerIdOutput};

/// Identifies a row group across the pipeline so its pages reassemble together.
pub(crate) type RowGroupId = u64;

/// One row group's worth of rows (possibly several batches), produced by
/// [`builder`](super::builder) and cut into pages by [`planner`](super::planner).
pub(crate) struct RowGroupBatch {
    pub(crate) rg_id: RowGroupId,
    /// Worker that owns this row group (`rg_id % worker_count`); its pages all
    /// route there for assembly.
    pub(crate) dest_worker: usize,
    pub(crate) schema: SchemaRef,
    pub(crate) batches: Vec<RecordBatch>,
}

/// One page to encode: the column slices (in row order) that make up the page.
/// More than one slice when the page spans a batch boundary.
pub(crate) struct PageJob {
    pub(crate) column: usize,
    /// Order of this page within its column chunk.
    pub(crate) page_index: usize,
    pub(crate) num_rows: i64,
    pub(crate) pieces: Vec<ArrayRef>,
}

/// A [`PageJob`] tagged with its row group's identity and total page count.
pub(crate) struct PipePageJob {
    pub(crate) rg_id: RowGroupId,
    pub(crate) dest_worker: usize,
    /// Total pages in this row group (all columns) — lets the assembler know
    /// when the row group is complete.
    pub(crate) n_pages: usize,
    pub(crate) schema: SchemaRef,
    pub(crate) job: PageJob,
}

/// An encoded data page: its snappy-compressed body behind a thrift header,
/// tagged with its position. The sizes feed the column-chunk footer metadata.
pub(crate) struct EncodedPage {
    pub(crate) column: usize,
    pub(crate) page_index: usize,
    pub(crate) num_rows: i64,
    /// Uncompressed size of the page's values (PLAIN bytes).
    pub(crate) uncompressed_size: usize,
    /// Size of the page's thrift header (precedes the compressed values).
    pub(crate) header_len: usize,
    /// The page on the wire: header followed by the compressed values.
    pub(crate) bytes: Vec<u8>,
}

/// An [`EncodedPage`] routed back to its row group's owner worker for assembly.
pub(crate) struct PipeEncodedPage {
    pub(crate) rg_id: RowGroupId,
    pub(crate) dest_worker: usize,
    pub(crate) n_pages: usize,
    pub(crate) schema: SchemaRef,
    pub(crate) page: EncodedPage,
}

impl WorkerIdOutput for PipeEncodedPage {
    fn worker_id(&self) -> Identifier {
        self.dest_worker
    }
}
