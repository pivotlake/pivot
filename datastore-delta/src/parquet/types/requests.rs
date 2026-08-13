use crate::parquet::request_tracker::PendingRequest;
use crate::parquet::types::leaves::projected_leaves;
use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::page_directory::leaf_entry;
use crate::parquet::types::projection::Projection;
use crate::parquet::types::thrift::headers::PageHeader;
use crate::parquet::types::thrift::parquet_thrift::ThriftReadInputProtocol;
use bytes::Bytes;
use dispatch::io::{FsReadRequest, FsRequest, HttpGetRequest, OpenFile};
use dispatch::memory::{CacheLookup, MultiBufferReader, ReaderPosition, Segment, memory_ctx};
use std::ops::Range;

/// One resolved piece of a column chunk, in file order, named by the form its
/// bytes are in:
///
/// - [`Compressed`](Self::Compressed): raw compressed bytes for the indexer to
///   Thrift-parse into pages. They may be fully resident in the compressed
///   cache or partly read from disk/HTTP; `Payload` tracks that IO state -
///   [`CacheLookup`]s while this row group's reads may still be in flight,
///   plain [`Bytes`] once every read has landed (see
///   [`into_ready`](Self::into_ready)).
/// - [`Decompressed`](Self::Decompressed): a page served straight from the
///   decompressed cache before any IO was issued: its identity (`offset`,
///   `span`), its header (parsed back out of the hit - see
///   [`parse_page_header`]), and its decompressed bytes, pinned. The indexer
///   builds a `CompressedPage` from it directly, no parsing or decompression
///   needed.
pub enum ColumnPart<Payload = Vec<Bytes>> {
    Compressed {
        offset: usize,
        bytes: Payload,
    },
    Decompressed {
        offset: usize,
        span: usize,
        header: Box<PageHeader>,
        data: Vec<Bytes>,
    },
}

impl ColumnPart<Vec<CacheLookup>> {
    /// Resolve the IO state away, one landed [`Bytes`] run per lookup. Only
    /// valid once every queued read has landed.
    fn into_ready(self) -> ColumnPart {
        match self {
            Self::Compressed { offset, bytes } => ColumnPart::Compressed {
                offset,
                bytes: bytes.into_iter().map(CacheLookup::into_data).collect(),
            },
            Self::Decompressed {
                offset,
                span,
                header,
                data,
            } => ColumnPart::Decompressed {
                offset,
                span,
                header,
                data,
            },
        }
    }
}

/// Tracks the IO state for a single column chunk within a row group.
///
/// The chunk's byte range is resolved against the decompressed cache first
/// ([`get_range`](dispatch::memory::DecompressedCache::get_range)): any page
/// already decompressed there is pinned on the spot and needs no IO at all. What's
/// left is looked up in the compressed cache exactly as before, one
/// [`CacheLookup`] per contiguous cached/missing run, and queued for IO.
struct ColumnRequest {
    parts: Vec<ColumnPart<Vec<CacheLookup>>>,
}

impl ColumnRequest {
    /// Build a `ColumnRequest` from column chunk metadata, queueing the reads
    /// for any missing sub-blocks onto the filesystem or HTTP request list
    /// according to where the file lives.
    ///
    /// `spans` are the byte ranges of the chunk to read, in file order: the
    /// whole chunk for an ordinary claim, and just the dictionary page plus
    /// this claim's own data pages for a split one.
    fn from(
        spans: &[Range<usize>],
        open_file: &OpenFile,
        fs_requests: &mut Vec<FsRequest>,
        http_requests: &mut Vec<HttpGetRequest>,
    ) -> Self {
        let parts = spans
            .iter()
            .flat_map(|span| {
                memory_ctx()
                    .decompressed_cache()
                    .get_range(open_file, span.start, span.len())
            })
            .map(|segment| match segment {
                Segment::Gap { offset, len } => ColumnPart::Compressed {
                    offset,
                    bytes: compressed_lookup(open_file, offset, len, fs_requests, http_requests),
                },
                Segment::Cached {
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
            .collect();
        Self { parts }
    }

    /// Consume this request into the column's resolved parts, in file order. Only
    /// valid once every queued read has landed.
    fn into_source(self) -> Vec<ColumnPart> {
        self.parts.into_iter().map(ColumnPart::into_ready).collect()
    }
}

/// Look up `[offset, offset+len)` of `open_file` in the compressed cache, queueing
/// any missing blocks onto the filesystem or HTTP request list according to where
/// the file lives.
fn compressed_lookup(
    open_file: &OpenFile,
    offset: usize,
    len: usize,
    fs_requests: &mut Vec<FsRequest>,
    http_requests: &mut Vec<HttpGetRequest>,
) -> Vec<CacheLookup> {
    let parts = memory_ctx().compressed_cache().get(open_file, offset, len);
    for lookup in &parts {
        if let Some(block) = lookup.missing() {
            match open_file {
                OpenFile::Local(file) => fs_requests.push(FsRequest::Read(FsReadRequest {
                    file: file.clone(),
                    block: block.clone(),
                })),
                OpenFile::Remote(remote) => http_requests.push(HttpGetRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                }),
            }
        }
    }
    parts
}

/// Parse a page's Thrift header back out of the bytes a decompressed-cache hit
/// returned (see [`Segment::Cached`], which
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

/// The chunk's whole byte range: its dictionary page, if any, through its last
/// data page.
fn whole_chunk(meta: &ColumnChunkMeta) -> Range<usize> {
    let start = meta.dictionary_page_offset.unwrap_or(meta.data_page_offset) as usize;
    start..start + meta.total_compressed_size as usize
}

/// Tracks the IO state for an entire row group read.
///
/// Created by `from()`, which resolves every projected column against the
/// decompressed cache and then the compressed cache, queuing read requests for any
/// still-missing sub-blocks. As completions arrive the requester fills the blocks
/// directly into their cache slots and the fetcher counts down via
/// [`complete_one`](Self::complete_one) until [`complete`](Self::complete) holds.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    column_requests: Vec<ColumnRequest>,
    /// Each fetched column's leaf index within the row group, which is how the
    /// page directory is keyed.
    leaves: Vec<usize>,
    /// The row each fetched column's first data page starts on.
    first_rows: Vec<usize>,
    /// Local filesystem reads not yet handed to the fetcher (drained when the
    /// row group is admitted).
    pending_fs: Vec<FsRequest>,
    /// Remote HTTP reads not yet handed to the fetcher.
    pending_http: Vec<HttpGetRequest>,
    /// Outstanding read count (decremented as each of this row group's reads
    /// lands).
    remaining: usize,
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

        let mut pending_fs = vec![];
        let mut pending_http = vec![];
        let rows = metadata_handle.row_range();
        let directories = &metadata_handle.get_metadata().page_directories;
        let checkpoints = &metadata_handle.get_metadata().decode_checkpoints;
        let mut first_rows = Vec::with_capacity(leaves.len());
        let column_requests = leaves
            .iter()
            .map(|&leaf| {
                let (spans, first_row) = match directories.get(leaf) {
                    // A split reads only the pages it decodes, plus the
                    // dictionary every reader of the chunk needs. Its first page
                    // is wherever it enters the chunk, which is the page holding
                    // the recorded position it resumes from - earlier than the
                    // page holding its first row whenever the checkpoint sits in
                    // the page before it. Without a directory there is nothing to
                    // seek by, so the whole chunk is read.
                    Some(directory) if metadata_handle.is_split() => {
                        let entry = leaf_entry(directory, checkpoints.get(leaf), rows.start);
                        let pages = entry.page_idx..directory.pages_covering(rows.clone()).end;
                        let mut spans: Vec<Range<usize>> =
                            directory.dictionary_span().into_iter().collect();
                        if let Some(data) = directory.span_of(pages.clone()) {
                            // Splitting at page zero leaves the dictionary
                            // directly abutting the data, so the two reads are
                            // merged rather than issued as neighbours.
                            match spans.last_mut() {
                                Some(dict) if dict.end == data.start => dict.end = data.end,
                                _ => spans.push(data),
                            }
                        }
                        (spans, directory.first_row_of(entry.page_idx))
                    }
                    _ => (vec![whole_chunk(&columns[leaf])], 0),
                };
                first_rows.push(first_row);
                ColumnRequest::from(&spans, &open_file, &mut pending_fs, &mut pending_http)
            })
            .collect();

        Self {
            column_requests,
            remaining: pending_fs.len() + pending_http.len(),
            pending_fs,
            pending_http,
            metadata: metadata_handle,
            leaves,
            first_rows,
        }
    }

    /// Record that one queued read has landed (and been committed to its slot).
    pub fn complete_one(&mut self) {
        self.remaining -= 1;
    }

    /// True when every read has completed and the row group is ready to consume.
    pub fn complete(&self) -> bool {
        self.remaining == 0
    }

    /// Consume this request into a `RowGroupBuffer`.
    /// Runs on the claiming worker (the fetcher emits from the worker that
    /// admitted the request), so the buffer records it as the decode owner.
    pub fn into_row_group_buffer(self) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: self.metadata,
            columns: self
                .column_requests
                .into_iter()
                .map(ColumnRequest::into_source)
                .collect(),
            worker_id: dispatch::worker::WORKER_IDX.get(),
            leaves: self.leaves,
            first_rows: self.first_rows,
        }
    }
}

impl PendingRequest for RowGroupRequest {
    fn pending_fs(&mut self) -> &mut Vec<FsRequest> {
        &mut self.pending_fs
    }

    fn pending_http(&mut self) -> &mut Vec<HttpGetRequest> {
        &mut self.pending_http
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
    /// Each fetched column's leaf index within the row group, parallel to
    /// `columns`. The indexer records each chunk's page directory under it.
    pub leaves: Vec<usize>,
    /// The row each fetched column's first data page starts on, parallel to
    /// `columns`. Zero for a whole-chunk read; for a split it is where that
    /// column's own first page begins, which is at or before the first row the
    /// split emits.
    pub first_rows: Vec<usize>,
}
