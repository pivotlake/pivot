use crate::types::leaves::projected_leaves;
use crate::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::types::projection::Projection;
use crate::types::thrift::headers::PageHeader;
use crate::types::thrift::parquet_thrift::ThriftReadInputProtocol;
use bytes::Bytes;
use dispatch::io::{FileRange, OpenFile, ReadData, ReadResponse};
use dispatch::memory::{MultiBufferReader, ReaderPosition};

/// One resolved piece of a column chunk, in file order, named by the form its
/// bytes are in:
///
/// - [`Compressed`](Self::Compressed): raw compressed bytes for the indexer to
///   Thrift-parse into pages. Dispatch has already resolved any compressed-cache
///   misses before constructing this value, so every byte run is pinned and
///   ready to read.
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

/// Parse a page's Thrift header back out of the bytes a decompressed-cache hit
/// returned (see [`dispatch::memory::Segment::Cached`], which
/// stores whatever the `Decompressor`
/// serialized at insert time) - cheap, a few bytes, entirely in memory; no disk
/// access and nothing to do with the (possibly much larger) page payload.
fn parse_page_header(bytes: &[Bytes]) -> PageHeader {
    let mut position = ReaderPosition::default();
    let mut reader = MultiBufferReader::new(bytes, &mut position);
    let mut prot = ThriftReadInputProtocol::new(&mut reader);
    PageHeader::read_thrift_without_stats(&mut prot)
        .expect("a decompressed-cache hit's header round-trips: this cache wrote it")
}

/// Tracks the IO state for an entire row group read.
///
/// It contains only durable file locations. Cache lookup, extent allocation,
/// physical reads happen later in dispatch's worker-local requester and its
/// private tracker.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    open_file: OpenFile,
    locations: Vec<FileRange>,
}

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let open_file = metadata_handle.get_metadata().open_file.clone();
        let columns = metadata_handle.columns();
        // The projection names top-level columns; fetch each one's run of leaf
        // chunks in this file's layout. The decoder resolves leaves the same
        // way, so a fetched chunk's position lines up with its decoder.
        let fields = metadata_handle.get_metadata().schema.fields();
        let leaves = projected_leaves(fields, &metadata_handle, projection);

        let locations = leaves
            .iter()
            .map(|&leaf| {
                let meta: &ColumnChunkMeta = &columns[leaf];
                FileRange::new(
                    meta.dictionary_page_offset.unwrap_or(meta.data_page_offset) as usize,
                    meta.total_compressed_size as usize,
                )
            })
            .collect();

        Self {
            metadata: metadata_handle,
            open_file,
            locations,
        }
    }

    pub fn open_file(&self) -> &OpenFile {
        &self.open_file
    }

    pub fn locations(&self) -> &[FileRange] {
        &self.locations
    }

    /// Pair this row group's domain metadata with dispatch's resolved response.
    /// Runs on the claiming worker (the fetcher emits from the worker that
    /// admitted the request), so the buffer records it as the decode owner.
    pub fn into_row_group_buffer(self, response: ReadResponse) -> RowGroupBuffer {
        let locations = response.into_locations();
        assert_eq!(locations.len(), self.locations.len());
        RowGroupBuffer {
            metadata: self.metadata,
            columns: locations
                .into_iter()
                .map(|parts| {
                    parts
                        .into_iter()
                        .map(|part| match part {
                            ReadData::Compressed { offset, bytes } => {
                                ColumnPart::Compressed { offset, bytes }
                            }
                            ReadData::Decompressed {
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
                        })
                        .collect()
                })
                .collect(),
            worker_id: dispatch::worker::WORKER_IDX.get(),
        }
    }
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
