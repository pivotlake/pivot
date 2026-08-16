use crate::parquet::types::leaves::projected_leaves;
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::projection::Projection;
use crate::parquet::types::thrift::headers::PageHeader;
use crate::parquet::types::thrift::parquet_thrift::ThriftReadInputProtocol;
use bytes::Bytes;
use dispatch::io::{CacheTiers, FileRange, PendingIoRequest, RangePart};
use dispatch::memory::{MultiBufferReader, ReaderPosition};

/// A claim on one row group: which row group to read and which of its leaf
/// columns the projection needs. Purely descriptive - no cache lookups or IO
/// happen until the fetcher turns it into a [`PendingIoRequest`] (see
/// [`into_pending_io`](Self::into_pending_io)) and the worker registers that.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    /// The file leaf columns to read, in fetch order.
    leaves: Vec<usize>,
}

impl RowGroupRequest {
    /// Build a claim for all projected columns in the given row group.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        // The projection names top-level columns; resolve each one's run of leaf
        // chunks in this file's layout. The decoder resolves leaves the same
        // way, so a fetched chunk's position lines up with its decoder.
        let fields = metadata_handle.get_metadata().schema.fields();
        let leaves = projected_leaves(fields, &metadata_handle, projection);
        Self {
            metadata: metadata_handle,
            leaves,
        }
    }

    /// The IO request reading this row group: one byte range per projected leaf
    /// column chunk, servable from the decompressed cache (whole pages) and the
    /// compressed cache alike. The row group's metadata rides along as the
    /// request state and comes back with the response, where
    /// [`RowGroupBuffer::from_response`] picks it up.
    pub fn into_pending_io(self) -> PendingIoRequest {
        let columns = self.metadata.columns();
        let ranges = self
            .leaves
            .iter()
            .map(|&leaf| {
                let column = &columns[leaf];
                FileRange {
                    offset: column
                        .dictionary_page_offset
                        .unwrap_or(column.data_page_offset) as usize,
                    len: column.total_compressed_size as usize,
                }
            })
            .collect();
        PendingIoRequest {
            file: self.metadata.get_metadata().open_file.clone(),
            ranges,
            tiers: CacheTiers::DecompressedAndCompressed,
            state: Box::new(self.metadata),
        }
    }
}

/// One resolved piece of a column chunk, in file order, named by the form its
/// bytes are in:
///
/// - [`Compressed`](Self::Compressed): raw compressed bytes for the indexer to
///   Thrift-parse into pages.
/// - [`Decompressed`](Self::Decompressed): a page served straight from the
///   decompressed cache before any IO was issued: its identity (`offset`,
///   `span`), its header (parsed back out of the hit - see
///   [`parse_page_header`]), and its decompressed bytes, pinned. The indexer
///   builds a `CompressedPage` from it directly, no parsing or decompression
///   needed.
pub enum ColumnPart {
    Compressed {
        offset: usize,
        bytes: Vec<Bytes>,
    },
    Decompressed {
        offset: usize,
        span: usize,
        header: Box<PageHeader>,
        data: Vec<Bytes>,
    },
}

impl ColumnPart {
    /// Give a resolved range part its parquet meaning: raw bytes stay raw, and
    /// a decompressed-cache hit gets its page header parsed back out of the
    /// bytes the cache stored with it.
    fn from_range_part(part: RangePart) -> Self {
        match part {
            RangePart::Compressed { offset, bytes } => ColumnPart::Compressed { offset, bytes },
            RangePart::Decompressed {
                offset,
                span,
                header,
                data,
            } => ColumnPart::Decompressed {
                offset,
                span,
                header: Box::new(parse_page_header(&header)),
                data,
            },
        }
    }
}

/// Parse a page's Thrift header back out of the bytes a decompressed-cache hit
/// returned (the `Decompressor` serialized them at insert time) - cheap, a few
/// bytes, entirely in memory; no disk access and nothing to do with the
/// (possibly much larger) page payload.
fn parse_page_header(bytes: &[Bytes]) -> PageHeader {
    let mut position = ReaderPosition::default();
    let mut reader = MultiBufferReader::new(bytes, &mut position);
    let mut prot = ThriftReadInputProtocol::new(&mut reader);
    PageHeader::read_thrift_without_stats(&mut prot)
        .expect("a decompressed-cache hit's header round-trips: this cache wrote it")
}

pub struct RowGroupBuffer {
    pub metadata: QueryRowGroupMetadata,
    /// Each projected column's resolved parts, in file order.
    pub columns: Vec<Vec<ColumnPart>>,
    /// The worker every page of this row group returns to for decode. It must
    /// be the worker that claimed the row group: that worker holds the row
    /// group's decoder state and accounts for the claim (see
    /// `RowGroupFetcher`). The middle stages may run on stealing siblings, so
    /// the id is stamped at claim time, not by whoever runs a stage.
    pub worker_id: usize,
}

impl RowGroupBuffer {
    /// Assemble the buffer from a completed IO response: the row group's
    /// metadata (round-tripped through the request state) and each column's
    /// resolved parts. Runs on the worker that claimed the row group - a
    /// staged request is registered and delivered on the worker whose operator
    /// staged it - so that worker is recorded as the decode owner.
    pub fn from_response(metadata: QueryRowGroupMetadata, columns: Vec<Vec<RangePart>>) -> Self {
        Self {
            metadata,
            columns: columns
                .into_iter()
                .map(|parts| parts.into_iter().map(ColumnPart::from_range_part).collect())
                .collect(),
            worker_id: dispatch::worker::WORKER_IDX.get(),
        }
    }
}
