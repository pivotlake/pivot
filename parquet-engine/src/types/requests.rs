use crate::thrift::general::CompressionCodec;
use crate::thrift::headers::PageHeader;
use crate::thrift::parquet_thrift::ThriftReadInputProtocol;
use crate::types::leaves::projected_leaves;
use crate::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata, RowSelection};
use crate::types::page_index::DataPageLocation;
use crate::types::projection::Projection;
use bytes::Bytes;
use dispatch::BUFFER_SIZE;
use dispatch::io::{FileRange, OpenFile, ReadAheadCharge, ReadData, ReadResponse, SlabRead};
use dispatch::memory::{MultiBufferReader, ReaderPosition, Segment, SlabAllocator, memory_ctx};
use std::ops::Range;

/// The most bytes of pages holding no selected row that a read of selected
/// pages takes in to stay one stretch. Two reads a short hop apart cost more
/// than the bytes between them, most of all over the network, where each
/// stretch is a request of its own.
const MAX_UNSELECTED_BYTES_READ_BETWEEN_PAGES: usize = 64 * 1024;

/// The block size a local file opened for direct IO is read in: a direct
/// read of one starts and ends on a multiple of it, in memory aligned to it.
const DIRECT_IO_BLOCK_BYTES: usize = 4096;

/// A row group's read as the fetcher issues it: for each projected leaf, the
/// pages the decompressed cache already held when the read was planned and the
/// byte ranges still to read. Built on the fetching worker right before its IO
/// (see [`from`](Self::from)), so the cached pages it pins are pinned only
/// while the read is in flight. Compressed-cache lookup, extent allocation and
/// physical reads happen in dispatch's worker-local requester.
///
/// A read of every row takes each leaf's whole chunk. A read of selected rows
/// takes, for a leaf whose pages the file's offset index lists, only the
/// dictionary page and the data pages holding a selected row.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    open_file: OpenFile,
    /// Each projected leaf's chunk, in leaf order.
    columns: Vec<ColumnReference>,
}

enum ColumnReference {
    /// The chunk's parts in file order, tagged with the page codec its footer
    /// entry recorded so the buffer tags the column's parts with it.
    InFile {
        codec: CompressionCodec,
        parts: Vec<ColumnChunkReference>,
    },
    /// The chunk's dictionary page and the data pages holding a selected
    /// row, in file order, of the leaf at `leaf` of the row group, whose
    /// pages the offset index lists.
    SelectedPages {
        codec: CompressionCodec,
        leaf: usize,
        parts: Vec<ColumnChunkReference>,
    },
    /// The file has no chunk for the leaf (see [`ColumnChunkMeta::absent`]):
    /// nothing is read for it, and its column of the buffer comes back empty.
    Absent,
}

enum ColumnChunkReference {
    /// A page the decompressed cache held: resolved, nothing to read.
    Cached(CachedColumnChunk),
    /// A stretch of whole pages the cache did not hold, read from the file.
    ToRead(FileRange),
}

impl ColumnChunkReference {
    /// Split `bytes`, a stretch of whole pages of `open_file`, into the pages
    /// the decompressed cache holds, pinned, and the stretches between them
    /// left to read. A read that bypasses the caches takes nothing from the
    /// cache: the whole stretch is left to read.
    fn resolve_cached_pages(
        open_file: &OpenFile,
        bytes: Range<usize>,
        bypasses_caches: bool,
    ) -> Vec<Self> {
        if bypasses_caches {
            return vec![ColumnChunkReference::ToRead(FileRange::new(
                bytes.start,
                bytes.len(),
            ))];
        }
        memory_ctx()
            .page_caches()
            .decompressed
            .get_range(open_file, bytes.start, bytes.len())
            .into_iter()
            .map(|segment| match segment {
                Segment::Cached {
                    offset,
                    span,
                    value: page,
                } => ColumnChunkReference::Cached(CachedColumnChunk::Decompressed {
                    offset,
                    span,
                    header: Box::new(Self::parse_page_header(&page.header)),
                    data: page.data,
                }),
                Segment::Gap { offset, len } => {
                    ColumnChunkReference::ToRead(FileRange::new(offset, len))
                }
            })
            .collect()
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
}

/// One resolved piece of a column chunk, in file order, named by the form its
/// bytes are in:
///
/// - [`Compressed`](Self::Compressed): raw compressed bytes for the indexer to
///   Thrift-parse into pages. Dispatch has already resolved any compressed-cache
///   misses before constructing this value, so every byte run is pinned and
///   ready to read.
/// - [`Decompressed`](Self::Decompressed): a page served straight from the
///   decompressed cache before any IO was issued: its identity (`offset`,
///   `span`), its header, and its decompressed bytes, pinned. The indexer
///   builds a `CompressedPage` from it directly, no parsing or decompression
///   needed.
pub enum CachedColumnChunk {
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

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group: each
    /// chunk, or the pages of it that hold selected rows, is looked up in the
    /// decompressed cache, the pages it holds are taken, pinned, and the
    /// stretches between them become ranges to read.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let row_group = metadata_handle.get_metadata();
        let open_file = row_group.open_file.clone();
        let columns = metadata_handle.columns();
        // The projection names top-level columns; fetch each one's run of leaf
        // chunks in this file's layout. The decoder resolves leaves the same
        // way, so a fetched chunk's position lines up with its decoder.
        let fields = metadata_handle.get_metadata().schema.fields();
        let leaves = projected_leaves(fields, row_group, projection);
        let bypasses_caches = metadata_handle.bypasses_caches();

        let planned = leaves
            .iter()
            .map(|&leaf| {
                let meta: &ColumnChunkMeta = &columns[leaf];
                if meta.absent {
                    return ColumnReference::Absent;
                }
                let chunk_bytes = meta.byte_range();
                if let RowSelection::Indices(selected_rows) = metadata_handle.selection()
                    && let Some(pages) = row_group.locate_leaf_pages(leaf)
                {
                    return ColumnReference::SelectedPages {
                        codec: meta.codec,
                        leaf,
                        parts: plan_selected_page_reads(chunk_bytes.start, pages, selected_rows)
                            .into_iter()
                            .flat_map(|read| {
                                ColumnChunkReference::resolve_cached_pages(
                                    &open_file,
                                    read,
                                    bypasses_caches,
                                )
                            })
                            .collect(),
                    };
                }
                ColumnReference::InFile {
                    codec: meta.codec,
                    parts: ColumnChunkReference::resolve_cached_pages(
                        &open_file,
                        chunk_bytes,
                        bypasses_caches,
                    ),
                }
            })
            .collect();

        Self {
            metadata: metadata_handle,
            open_file,
            columns: planned,
        }
    }

    pub fn open_file(&self) -> &OpenFile {
        &self.open_file
    }

    /// The byte ranges left to read, in file order within each leaf.
    pub fn file_ranges(&self) -> impl Iterator<Item = FileRange> + '_ {
        self.columns
            .iter()
            .flat_map(|column| match column {
                ColumnReference::InFile { parts, .. }
                | ColumnReference::SelectedPages { parts, .. } => parts.as_slice(),
                ColumnReference::Absent => &[],
            })
            .filter_map(|part| match part {
                ColumnChunkReference::ToRead(range) => Some(*range),
                ColumnChunkReference::Cached(_) => None,
            })
    }

    /// Whether the read goes around the caches (see
    /// [`QueryRowGroupMetadata::bypassing_caches`]). Such a read is issued
    /// with [`plan_direct_reads`](Self::plan_direct_reads).
    pub fn bypasses_caches(&self) -> bool {
        self.metadata.bypasses_caches()
    }

    /// The stretches of the file that cover the ranges left to read, in file
    /// order: ranges that touch, or lie a short hop apart, are one stretch. A
    /// read of every column of a row group is one stretch, since a row
    /// group's chunks follow each other in the file.
    fn list_stretches_to_read(&self) -> Vec<Range<usize>> {
        let mut ranges: Vec<FileRange> = self.file_ranges().collect();
        ranges.sort_unstable_by_key(|range| range.offset);
        let mut stretches: Vec<Range<usize>> = Vec::new();
        for range in ranges {
            let range_end = range.offset + range.len;
            match stretches.last_mut() {
                Some(last)
                    if range.offset <= last.end + MAX_UNSELECTED_BYTES_READ_BETWEEN_PAGES =>
                {
                    last.end = last.end.max(range_end)
                }
                _ => stretches.push(range.offset..range_end),
            }
        }
        stretches
    }

    /// The reads of a request that bypasses the caches: every stretch of the
    /// file it needs, cut into pieces of at most one buffer, each read into a
    /// slab of `allocator`'s. A slab holds no more than a buffer, and a row
    /// group is far larger, so its bytes arrive as several pieces.
    ///
    /// A local file is opened for direct IO, which reads whole blocks into
    /// block-aligned memory: its stretches are widened to block boundaries,
    /// and the last piece of a stretch may then reach past the end of the
    /// file, so it asks only for the bytes up to the end of the stretch.
    ///
    /// The response has one location per piece, which
    /// [`into_row_group_buffer`](Self::into_row_group_buffer) cuts the
    /// requested ranges out of.
    pub fn plan_direct_reads(&self, allocator: &mut SlabAllocator) -> Vec<SlabRead> {
        let block_bytes = match self.open_file {
            OpenFile::Local(_) => DIRECT_IO_BLOCK_BYTES,
            OpenFile::Remote(_) => 1,
        };
        let mut reads = Vec::new();
        for stretch in self.list_stretches_to_read() {
            let mut piece_start = stretch.start / block_bytes * block_bytes;
            while piece_start < stretch.end {
                let bytes_wanted = (stretch.end - piece_start).min(BUFFER_SIZE);
                let slab_bytes = bytes_wanted.next_multiple_of(block_bytes);
                reads.push(SlabRead {
                    offset: piece_start,
                    slab: allocator.get_aligned_slab(slab_bytes, block_bytes, false),
                    len: bytes_wanted,
                });
                piece_start += bytes_wanted;
            }
        }
        reads
    }

    /// The bytes of each range left to read, in the order of
    /// [`file_ranges`](Self::file_ranges), cut out of `pieces`: the locations
    /// of the response to [`plan_direct_reads`](Self::plan_direct_reads), in
    /// file order. A range that crosses from one piece into the next comes
    /// back as a run of each, the way a range the compressed cache holds in
    /// several blocks does, so a page that straddles two pieces is parsed
    /// like one that straddles two cache blocks.
    fn cut_ranges_from_pieces(&self, pieces: Vec<ReadData>) -> Vec<ReadData> {
        self.file_ranges()
            .map(|range| {
                let range_end = range.offset + range.len;
                let first_piece = pieces.partition_point(|piece| piece.offset <= range.offset) - 1;
                let mut bytes = Vec::new();
                let mut position = range.offset;
                for piece in &pieces[first_piece..] {
                    if position == range_end {
                        break;
                    }
                    assert_eq!(piece.bytes.len(), 1, "a piece is one slab");
                    let slab = &piece.bytes[0];
                    let start_in_slab = position - piece.offset;
                    let end_in_slab = (range_end - piece.offset).min(slab.len());
                    bytes.push(slab.slice(start_in_slab..end_in_slab));
                    position = piece.offset + end_in_slab;
                }
                assert_eq!(position, range_end, "the pieces cover every range");
                ReadData {
                    offset: range.offset,
                    bytes,
                }
            })
            .collect()
    }

    /// Give the row group back, dropping the cached pages taken for it. For a
    /// request that will be built again later, by whichever worker asks for
    /// its read.
    pub fn into_metadata(self) -> QueryRowGroupMetadata {
        self.metadata
    }

    /// Whether nothing is left to read: every projected leaf is absent from
    /// the file or entirely held by the decompressed cache, so the buffer is
    /// built with no response
    /// ([`into_row_group_buffer`](Self::into_row_group_buffer) with `None`).
    pub fn reads_nothing(&self) -> bool {
        self.file_ranges().next().is_none()
    }

    /// Pair this row group's domain metadata with dispatch's resolved
    /// `response`, `None` for a request that [reads nothing](Self::reads_nothing).
    /// Runs on the claiming worker (the fetcher emits from the worker that
    /// admitted the request), so the buffer records it as the decode owner.
    ///
    /// `read_ahead_charge` is what the claim counts against its worker's
    /// read-ahead budget, `None` for a row group that was expected to be in
    /// memory and was. The buffer carries it to whichever worker starts on the
    /// row group.
    pub fn into_row_group_buffer(
        self,
        response: Option<ReadResponse>,
        read_ahead_charge: Option<ReadAheadCharge>,
    ) -> RowGroupBuffer {
        let locations = response
            .map(|response| response.into_locations())
            .unwrap_or_default();
        // A read that bypasses the caches comes back as the pieces it was
        // read in, not as the ranges it asked for.
        let locations = if self.metadata.bypasses_caches() {
            self.cut_ranges_from_pieces(locations)
        } else {
            locations
        };
        let mut resolved = locations.into_iter();
        let mut resolve_parts = |parts: Vec<ColumnChunkReference>| -> Vec<CachedColumnChunk> {
            parts
                .into_iter()
                .map(|part| match part {
                    ColumnChunkReference::Cached(part) => part,
                    ColumnChunkReference::ToRead(_) => {
                        let read = resolved
                            .next()
                            .expect("the response resolves every requested range");
                        CachedColumnChunk::Compressed {
                            offset: read.offset,
                            bytes: read.bytes,
                        }
                    }
                })
                .collect()
        };
        let columns = self
            .columns
            .into_iter()
            .map(|column| match column {
                ColumnReference::InFile { codec, parts } => ColumnBuffer::Resolved {
                    codec,
                    parts: resolve_parts(parts),
                },
                ColumnReference::SelectedPages { codec, leaf, parts } => {
                    ColumnBuffer::SelectedPages {
                        codec,
                        leaf,
                        parts: resolve_parts(parts),
                    }
                }
                ColumnReference::Absent => ColumnBuffer::Absent,
            })
            .collect();
        assert!(
            resolved.next().is_none(),
            "the response resolves only the requested ranges"
        );
        RowGroupBuffer {
            metadata: self.metadata,
            columns,
            worker_id: dispatch::worker::WORKER_IDX.get(),
            read_ahead_charge,
        }
    }
}

/// The stretches of a chunk to read for the rows `selected_rows`, sorted row
/// group rows: the dictionary page, which is whatever the chunk holds before
/// its first data page, and each of `pages`, the chunk's data pages in file
/// order, that holds a selected row. Pages back to back are read as one
/// stretch, and so are pages a short hop apart.
fn plan_selected_page_reads(
    chunk_start: usize,
    pages: impl Iterator<Item = DataPageLocation>,
    selected_rows: &[u32],
) -> Vec<Range<usize>> {
    let mut reads: Vec<Range<usize>> = Vec::new();
    let mut rows_left = selected_rows;
    for page in pages {
        if page.page_idx == 0 && page.file_offset > chunk_start {
            reads.push(chunk_start..page.file_offset);
        }
        let rows_in_page = rows_left.partition_point(|&row| row < page.rows.end);
        rows_left = &rows_left[rows_in_page..];
        if rows_in_page == 0 {
            continue;
        }
        let page_end = page.file_offset + page.compressed_size;
        match reads.last_mut() {
            Some(last)
                if page.file_offset - last.end <= MAX_UNSELECTED_BYTES_READ_BETWEEN_PAGES =>
            {
                last.end = page_end
            }
            _ => reads.push(page.file_offset..page_end),
        }
    }
    reads
}

pub struct RowGroupBuffer {
    pub metadata: QueryRowGroupMetadata,
    /// Each projected column's resolved bytes, in file order.
    pub columns: Vec<ColumnBuffer>,
    /// The worker every page of this row group returns to for decode. It must
    /// be the worker that claimed the row group: that worker holds the row
    /// group's decoder state and accounts for the claim (see
    /// `RowGroupFetcher`). The middle stages may run on stealing siblings, so
    /// the id is stamped at claim time, not by whoever runs a stage.
    pub worker_id: usize,
    /// What this row group counts against its claiming worker's read-ahead
    /// budget, `None` for one that was expected to be in memory and was. It is
    /// given back when the buffer is dropped, which is when a worker has
    /// started on the row group.
    pub read_ahead_charge: Option<ReadAheadCharge>,
}

/// One projected column of a row group as the ring resolved it.
pub enum ColumnBuffer {
    /// The column's parts, in file order, tagged with the codec its compressed
    /// pages were written with (from the column chunk's footer metadata).
    Resolved {
        codec: CompressionCodec,
        parts: Vec<CachedColumnChunk>,
    },
    /// The dictionary page and the data pages holding a selected row, in
    /// file order, of the leaf at `leaf` of the row group, read by the file's
    /// offset index. The indexer stands in a page without bytes for each
    /// data page that was not read.
    SelectedPages {
        codec: CompressionCodec,
        leaf: usize,
        parts: Vec<CachedColumnChunk>,
    },
    /// A leaf the file has no chunk for (see [`ColumnChunkMeta::absent`]):
    /// nothing was read, and the indexer stands in one page spanning every row
    /// so the column decodes as NULL.
    Absent,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Data pages of `page_size` bytes and ten rows each, back to back from
    /// file offset `first_page_offset`.
    fn pages_of_ten_rows(
        first_page_offset: usize,
        page_size: usize,
        page_count: usize,
    ) -> impl Iterator<Item = DataPageLocation> {
        (0..page_count).map(move |page_idx| DataPageLocation {
            page_idx,
            file_offset: first_page_offset + page_idx * page_size,
            compressed_size: page_size,
            rows: page_idx as u32 * 10..(page_idx as u32 + 1) * 10,
        })
    }

    const MIB: usize = 1024 * 1024;

    #[test]
    fn only_the_pages_holding_selected_rows_are_read() {
        let pages = pages_of_ten_rows(0, MIB, 5);

        let reads = plan_selected_page_reads(0, pages, &[10, 19, 40]);

        assert_eq!(reads, vec![MIB..2 * MIB, 4 * MIB..5 * MIB]);
    }

    #[test]
    fn rows_at_the_edges_of_a_page_select_that_page_alone() {
        let last_row_of_second = plan_selected_page_reads(0, pages_of_ten_rows(0, MIB, 5), &[19]);
        let first_row_of_third = plan_selected_page_reads(0, pages_of_ten_rows(0, MIB, 5), &[20]);

        assert_eq!(last_row_of_second, vec![MIB..2 * MIB]);
        assert_eq!(first_row_of_third, vec![2 * MIB..3 * MIB]);
    }

    #[test]
    fn selected_pages_back_to_back_are_one_read() {
        let pages = pages_of_ten_rows(0, MIB, 5);

        let reads = plan_selected_page_reads(0, pages, &[15, 25, 35]);

        assert_eq!(reads, vec![MIB..4 * MIB]);
    }

    #[test]
    fn the_dictionary_page_is_read_with_the_selected_pages() {
        let pages = pages_of_ten_rows(1000, MIB, 5);

        let reads = plan_selected_page_reads(200, pages, &[45]);

        assert_eq!(reads, vec![200..1000, 1000 + 4 * MIB..1000 + 5 * MIB]);
    }

    #[test]
    fn a_short_hop_between_selected_pages_is_read_through() {
        let pages = pages_of_ten_rows(0, 1024, 5);

        let reads = plan_selected_page_reads(0, pages, &[5, 45]);

        assert_eq!(reads, vec![0..5 * 1024]);
    }
}
