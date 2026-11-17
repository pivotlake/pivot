//! Splits raw column buffers into individual compressed pages and sends them in an order
//! optimised for downstream decoding.
//!
//! The Indexer sits between IO (which delivers whole column chunks, or pages already
//! resolved straight from the decompressed cache - see
//! [`CachedColumnChunk`]) and the Decompressor
//! (which expects individual pages). For each column it walks its parts in file
//! order: a `Compressed` part's raw bytes are Thrift-parsed into pages same as
//! always; a `Decompressed` part is already one whole decompressed page, so it
//! becomes a `CompressedPage` directly, no parsing needed. When filtered indices are
//! present it attaches a [`FilterMask`] to each data page so the decoder knows which
//! rows to keep.
//!
//! A column read by the file's offset index holds only the pages with selected
//! rows (see [`ColumnBuffer::SelectedPages`]). Each page read is placed by the
//! index, and every data page that was not read gets a stand-in without bytes
//! whose mask keeps no row, so the stages downstream see every page of the
//! chunk, as they do for a chunk read whole, and skip the stand-ins as they
//! skip any page without a selected row.
//!
//! ## Emission order
//!
//! Pages are expected to be emitted onto a **LIFO** channel, so the last page sent is the first one
//! received.
//! We exploit this by sending in reverse-priority order:
//!
//! 1. **Data pages** — interleaved across columns so the column with the fewest emitted rows
//!    goes next. This lets the decoder produce `RecordBatch`es as early as possible, keeping
//!    recently-decompressed pages hot in cache.
//! 2. **Dictionary pages** — sent last so they arrive first, ensuring the decoder has the
//!    dictionary before any RLE-dictionary-encoded data page.

use crate::general_err;
use crate::thrift::general::{CompressionCodec, Encoding, PageType};
use crate::thrift::headers::{DataPageHeader, PageHeader};
use crate::thrift::parquet_thrift::{ParquetError, ThriftReadInputProtocol};
use crate::types::filter_mask::FilterMask;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::page::CompressedPage;
use crate::types::requests::{CachedColumnChunk, ColumnBuffer, RowGroupBuffer};
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{MultiBufferReader, ReaderPosition};
use std::collections::VecDeque;

pub type IndexerFactory = DefaultUnaryFactory<Indexer>;

/// Splits a [`RowGroupBuffer`] into individual [`CompressedPage`]s.
#[derive(Default)]
pub struct Indexer {}

/// Running position while emitting one column's pages: how many data pages (and
/// their rows) have been emitted so far. Threaded across all of a column's parts
/// so `page_idx` and filter-mask row offsets stay continuous regardless of which
/// parts came pre-resolved from the decompressed cache.
#[derive(Default)]
struct PageCursor {
    data_page_idx: usize,
    row_offset: u32,
}

impl PageCursor {
    /// This page's `(page_idx, row_offset)`, then advance past it if it's a data
    /// page (dictionary pages don't count).
    fn advance(&mut self, header: &PageHeader) -> (usize, u32) {
        let at = (self.data_page_idx, self.row_offset);
        if header.r#type == PageType::DATA_PAGE {
            self.data_page_idx += 1;
            self.row_offset += header.data_page_num_values() as u32;
        }
        at
    }
}

/// A page's body, either still compressed (needs the decompressor) or already
/// decompressed (a decompressed-cache hit, resolved before this row group's IO was
/// even issued).
enum PagePayload {
    Compressed(Vec<Bytes>),
    Decompressed(Vec<Bytes>),
}

/// Carries the context for emitting one column's pages: the column's
/// identity and codec, the query's view of the row group, the worker that
/// claimed it, and the running [`PageCursor`] threaded across the column's
/// parts.
struct ColumnPageBuilder {
    col_idx: usize,
    codec: CompressionCodec,
    query_row_group_metadata: QueryRowGroupMetadata,
    worker_id: usize,
    cursor: PageCursor,
}

/// A page of a column as read, before it is placed among the column's pages.
struct ReadPage {
    file_offset: usize,
    span: usize,
    header: PageHeader,
    payload: PagePayload,
}

impl ColumnPageBuilder {
    /// Build the next `CompressedPage` of a column read whole, attaching the
    /// current query's filter mask and advancing the cursor.
    fn build_page(&mut self, page: ReadPage) -> CompressedPage {
        let (page_idx, row_offset) = self.cursor.advance(&page.header);
        self.build_page_at(page_idx, row_offset, page)
    }

    /// Build one `CompressedPage`, the column's data page number `page_idx`
    /// starting at row `row_offset` (both zero for a dictionary page),
    /// attaching the current query's filter mask.
    fn build_page_at(&self, page_idx: usize, row_offset: u32, page: ReadPage) -> CompressedPage {
        let ReadPage {
            file_offset,
            span,
            header,
            payload,
        } = page;
        // A whole row group carries no mask; the materializer's row groups
        // mask down to their surviving rows.
        let filter_mask = (header.r#type == PageType::DATA_PAGE)
            .then(|| {
                let page_end = row_offset + header.data_page_num_values() as u32;
                match self.query_row_group_metadata.selection() {
                    RowSelection::All => None,
                    RowSelection::Indices(indices) => {
                        Some(FilterMask::new(row_offset, page_end, indices))
                    }
                }
            })
            .flatten();
        let (data, decompressed) = match payload {
            PagePayload::Compressed(bytes) => (bytes, None),
            PagePayload::Decompressed(bytes) => (Vec::new(), Some(bytes)),
        };

        CompressedPage {
            // This is the claimer, not necessarily this worker: the indexer
            // often runs on a stealing sibling, but the decode must return to
            // the worker that claimed the row group (it owns the decoder
            // state and the claim accounting).
            worker_id: self.worker_id,
            codec: self.codec,
            row_group: self.query_row_group_metadata.clone(),
            column_idx: self.col_idx,
            file_offset,
            span,
            page_idx,
            first_row: row_offset,
            data,
            decompressed,
            filter_mask,
            header,
        }
    }
}

/// Walk a `Compressed` part's raw byte stream and Thrift-parse it into pages,
/// handing each to `take_page`. Each iteration reads a page header, then slices out
/// the compressed payload via [`MultiBufferReader::copy_out_buffers`] (zero-copy).
fn parse_compressed_pages(
    part_offset: usize,
    buffers: &[Bytes],
    mut take_page: impl FnMut(ReadPage),
) -> Result<(), ParquetError> {
    let mut position = ReaderPosition::default();
    let buffers_length = buffers.len();
    let mut reader = MultiBufferReader::new(buffers, &mut position);
    loop {
        // Bytes consumed so far = this page's offset within the part, which the
        // part's own file offset turns into the page's absolute file offset.
        let page_start = reader.consumed();
        let header = {
            let mut prot = ThriftReadInputProtocol::new(&mut reader);
            PageHeader::read_thrift_without_stats(&mut prot)?
        };
        let data = reader.copy_out_buffers(header.compressed_page_size as usize);
        let span = reader.consumed() - page_start;

        take_page(ReadPage {
            file_offset: part_offset + page_start,
            span,
            header,
            payload: PagePayload::Compressed(data),
        });

        if reader.remaining_in_cur() == 0 && reader.position().buffer_index == buffers_length - 1 {
            break;
        }
    }
    Ok(())
}

/// Hand each page of a column's resolved `parts` to `take_page`, in file
/// order.
fn read_pages(
    parts: Vec<CachedColumnChunk>,
    mut take_page: impl FnMut(ReadPage),
) -> Result<(), ParquetError> {
    for part in parts {
        match part {
            CachedColumnChunk::Compressed { offset, bytes } => {
                parse_compressed_pages(offset, &bytes, &mut take_page)?
            }
            CachedColumnChunk::Decompressed {
                offset,
                span,
                header,
                data,
            } => take_page(ReadPage {
                file_offset: offset,
                span,
                header: *header,
                payload: PagePayload::Decompressed(data),
            }),
        }
    }
    Ok(())
}

/// A data page header for a page of `rows` rows that carries no bytes.
fn header_of_page_without_bytes(rows: i32) -> PageHeader {
    PageHeader {
        r#type: PageType::DATA_PAGE,
        uncompressed_page_size: 0,
        compressed_page_size: 0,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: rows,
            encoding: Encoding::PLAIN,
            definition_level_encoding: Encoding::RLE,
            repetition_level_encoding: Encoding::RLE,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    }
}

/// Build the pages of a column read by the offset index of the row group's
/// leaf `leaf`: the pages in `parts`, placed by the index, and a stand-in
/// for every data page of the chunk that was not read.
///
/// A page read from where the index lists none, or holding another number of
/// rows than the index gives it, is an error: the index does not describe
/// the chunk, and rows placed by it would be the wrong rows.
fn build_selected_column_pages(
    builder: ColumnPageBuilder,
    leaf: usize,
    parts: Vec<CachedColumnChunk>,
) -> Result<Vec<CompressedPage>, ParquetError> {
    let mut pages_read = VecDeque::new();
    read_pages(parts, |page| pages_read.push_back(page))?;
    let row_group = builder.query_row_group_metadata.row_group_metadata.clone();
    let locations = row_group
        .locate_leaf_pages(leaf)
        .expect("a column is read by the offset index only when the index lists its pages");

    let mut pages = Vec::with_capacity(128);
    while let Some(page) = pages_read.pop_front_if(|page| page.header.r#type != PageType::DATA_PAGE)
    {
        pages.push(builder.build_page_at(0, 0, page));
    }
    for location in locations {
        let rows = location.rows.len() as i32;
        let page = match pages_read.pop_front_if(|page| page.file_offset == location.file_offset) {
            Some(page) if page.header.r#type != PageType::DATA_PAGE => {
                return Err(general_err!(
                    "the offset index lists a data page at offset {} but the page there is a {}",
                    location.file_offset,
                    page.header.r#type
                ));
            }
            Some(page) if page.header.data_page_num_values() != rows => {
                return Err(general_err!(
                    "the offset index gives the page at offset {} {} rows but it holds {}",
                    location.file_offset,
                    rows,
                    page.header.data_page_num_values()
                ));
            }
            Some(page) => page,
            None => ReadPage {
                file_offset: location.file_offset,
                span: location.compressed_size,
                header: header_of_page_without_bytes(rows),
                payload: PagePayload::Compressed(Vec::new()),
            },
        };
        pages.push(builder.build_page_at(location.page_idx, location.rows.start, page));
    }
    if let Some(page) = pages_read.front() {
        return Err(general_err!(
            "the offset index lists no page at offset {}, where one was read",
            page.file_offset
        ));
    }
    Ok(pages)
}

/// Build one column's pages, in file order, from its resolved parts.
///
/// A column the file has no chunk for gets one stand-in data page spanning
/// every row of the row group, carrying no bytes: it moves through the
/// pipeline like any page, so the row group's ranges become ready and the
/// column's decoder emits a NULL per row.
fn build_column_pages(
    col_idx: usize,
    query_row_group_metadata: QueryRowGroupMetadata,
    column: ColumnBuffer,
    worker_id: usize,
) -> Result<Vec<CompressedPage>, ParquetError> {
    let mut builder = ColumnPageBuilder {
        col_idx,
        codec: CompressionCodec::UNCOMPRESSED,
        query_row_group_metadata,
        worker_id,
        cursor: PageCursor::default(),
    };
    match column {
        ColumnBuffer::Resolved { codec, parts } => {
            builder.codec = codec;
            let mut pages = Vec::with_capacity(128);
            read_pages(parts, |page| pages.push(builder.build_page(page)))?;
            Ok(pages)
        }
        ColumnBuffer::SelectedPages { codec, leaf, parts } => {
            builder.codec = codec;
            build_selected_column_pages(builder, leaf, parts)
        }
        ColumnBuffer::Absent => {
            let rows = builder.query_row_group_metadata.num_rows().max(0) as i32;
            Ok(vec![builder.build_page(ReadPage {
                file_offset: 0,
                span: 0,
                header: header_of_page_without_bytes(rows),
                payload: PagePayload::Decompressed(Vec::new()),
            })])
        }
    }
}

impl Unary<RowGroupBuffer, CompressedPage> for Indexer {
    fn consume(
        &mut self,
        buffer: RowGroupBuffer,
        sender: &mut dyn Sender<CompressedPage>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let mut pages_per_column: Vec<_> = buffer
            .columns
            .into_iter()
            .enumerate()
            .map(|(col_idx, column)| {
                build_column_pages(col_idx, buffer.metadata.clone(), column, buffer.worker_id)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::op_err)?;

        // We want to send the pages out in an order that will be best for decoding. If we were to,
        // for example, send all of column A pages and then only column B pages, we would be unable to
        // send out record batches until AFTER we finished ALL of column A. This is horrible from a
        // cache standpoint, as it means pages we touched for decompressing won't be in cache for
        // the decoder.
        // Therefore, we try to send out pages in "intelligently"- first we send out dict pages,
        // and then we try sending out pages in an order where we will be able to send out record
        // batches as soon as possible. We do this by always selecting pages from the column that
        // has sent out the minimum amount of records so far
        let dict_pages: Vec<_> = pages_per_column
            .iter_mut()
            .flat_map(|pages| {
                pages
                    .extract_if(..pages.len(), |p| {
                        p.header.r#type == PageType::DICTIONARY_PAGE
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut rows_emitted = vec![0u64; pages_per_column.len()];

        while let Some(col) = (0..pages_per_column.len())
            .filter(|i| !pages_per_column[*i].is_empty())
            .min_by_key(|i| rows_emitted[*i])
        {
            let page = pages_per_column[col].pop().unwrap();
            rows_emitted[col] += page.header.data_page_num_values() as u64;
            sender.send(page)?;
        }

        // We send the dict pages at the end, so they'll be received first (remember, we're LIFO!)
        for page in dict_pages {
            sender.send(page)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{dummy_metadata, dummy_row_group};
    use crate::thrift::footer::{OffsetIndex, PageLocation};
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::{
        DataPageHeader as ThriftDataPageHeader, DictionaryPageHeader as ThriftDictionaryPageHeader,
    };
    use crate::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
    use crate::types::metadata::{ColumnChunkMeta, RowGroupMetadata};
    use crate::types::page_index::RowGroupPageIndex;
    use crate::types::table::ParquetTable;
    use bytes::Bytes;
    use dispatch::test_utils::{CollectSender, run_unary};
    use std::sync::Arc;

    /// Serialize a dispatch `PageHeader` to Thrift compact bytes.
    fn serialize_header(header: &PageHeader) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut prot = ThriftCompactOutputProtocol::new(&mut buf);
            header.write_thrift(&mut prot).unwrap();
        }
        buf
    }

    /// Build a data page header for serialization.
    fn data_page_header(num_values: i32, compressed_size: i32) -> PageHeader {
        PageHeader {
            r#type: PageType::DATA_PAGE,
            uncompressed_page_size: compressed_size,
            compressed_page_size: compressed_size,
            crc: None,
            data_page_header: Some(ThriftDataPageHeader {
                num_values,
                encoding: Encoding::PLAIN,
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: None,
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        }
    }

    /// Build a dictionary page header for serialization.
    fn dict_page_header(num_values: i32, compressed_size: i32) -> PageHeader {
        PageHeader {
            r#type: PageType::DICTIONARY_PAGE,
            uncompressed_page_size: compressed_size,
            compressed_page_size: compressed_size,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(ThriftDictionaryPageHeader {
                num_values,
                encoding: Encoding::PLAIN_DICTIONARY,
                is_sorted: None,
            }),
            data_page_header_v2: None,
        }
    }

    /// Concatenate serialized page headers + payloads into a single `Compressed` part.
    fn make_column_buffer(pages: &[(PageHeader, Vec<u8>)]) -> Vec<CachedColumnChunk> {
        let mut buf = Vec::new();
        for (header, payload) in pages {
            buf.extend_from_slice(&serialize_header(header));
            buf.extend_from_slice(payload);
        }
        vec![CachedColumnChunk::Compressed {
            offset: 0,
            bytes: vec![Bytes::from(buf)],
        }]
    }

    fn make_row_group_buffer(
        columns: Vec<Vec<CachedColumnChunk>>,
        selection: RowSelection,
    ) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: dummy_metadata(selection),
            columns: columns
                .into_iter()
                .map(|parts| ColumnBuffer::Resolved {
                    codec: CompressionCodec::SNAPPY,
                    parts,
                })
                .collect(),
            worker_id: 0,
            read_ahead_charge: None,
        }
    }

    /// Single data page → 1 CompressedPage with correct column_idx, page_idx, and payload.
    #[test]
    fn test_single_data_page() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let header = data_page_header(10, payload.len() as i32);
        let col = make_column_buffer(&[(header, payload.clone())]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        let page = &pages[0];
        assert_eq!(page.column_idx, 0);
        assert_eq!(page.page_idx, 0);
        assert_eq!(page.header.r#type, PageType::DATA_PAGE);
        assert_eq!(page.header.data_page_num_values(), 10);
        let data: Vec<u8> = page.data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(data, payload);
        assert!(page.filter_mask.is_none());
    }

    /// Three data pages → all 3 emitted with correct page_idx and payload.
    /// Note: .pop() reverses order within a column, so pages come out last-first.
    #[test]
    fn test_multiple_data_pages() {
        let payloads: Vec<Vec<u8>> = vec![vec![10, 11], vec![20, 21, 22], vec![30]];
        let pages: Vec<_> = payloads
            .iter()
            .map(|p| (data_page_header(5, p.len() as i32), p.clone()))
            .collect();
        let col = make_column_buffer(&pages);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        let mut page_indices: Vec<usize> = pages.iter().map(|p| p.page_idx).collect();
        page_indices.sort();
        assert_eq!(page_indices, vec![0, 1, 2]);
        for page in &pages {
            assert_eq!(page.header.r#type, PageType::DATA_PAGE);
            let data: Vec<u8> = page.data.iter().flat_map(|b| b.to_vec()).collect();
            assert_eq!(data, payloads[page.page_idx]);
        }
    }

    /// Dict page + 2 data pages → dict page sent last (LIFO: received first).
    #[test]
    fn test_dict_pages_sent_last() {
        let dict_payload = vec![0xDD];
        let data_payload_0 = vec![0xA0];
        let data_payload_1 = vec![0xA1];
        let col = make_column_buffer(&[
            (dict_page_header(3, dict_payload.len() as i32), dict_payload),
            (
                data_page_header(10, data_payload_0.len() as i32),
                data_payload_0,
            ),
            (
                data_page_header(10, data_payload_1.len() as i32),
                data_payload_1,
            ),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].header.r#type, PageType::DATA_PAGE);
        assert_eq!(pages[1].header.r#type, PageType::DATA_PAGE);
        assert_eq!(pages[2].header.r#type, PageType::DICTIONARY_PAGE);
    }

    /// Two columns, each with 1 data page → pages interleaved by column.
    #[test]
    fn test_two_columns_interleaved() {
        let col_a = make_column_buffer(&[(data_page_header(100, 2), vec![0xAA, 0xAA])]);
        let col_b = make_column_buffer(&[(data_page_header(100, 2), vec![0xBB, 0xBB])]);
        let buffer = make_row_group_buffer(vec![col_a, col_b], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let col_ids: Vec<usize> = pages.iter().map(|p| p.column_idx).collect();
        assert!(col_ids.contains(&0));
        assert!(col_ids.contains(&1));
    }

    /// Two columns with unequal pages: col A has 2 pages (50 rows each), col B has 1 page
    /// (100 rows). Min-rows-emitted interleaving: A(50), B(100), A(50).
    #[test]
    fn test_interleave_unequal_pages() {
        let col_a = make_column_buffer(&[
            (data_page_header(50, 1), vec![0xA0]),
            (data_page_header(50, 1), vec![0xA1]),
        ]);
        let col_b = make_column_buffer(&[(data_page_header(100, 1), vec![0xB0])]);
        let buffer = make_row_group_buffer(vec![col_a, col_b], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        let col_ids: Vec<usize> = pages.iter().map(|p| p.column_idx).collect();
        assert_eq!(col_ids, vec![0, 1, 0]);
    }

    /// With an index selection, data pages get a FilterMask; dict pages don't.
    #[test]
    fn test_filter_mask_on_data_pages_only() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(10, 1), vec![0xAA]),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::Indices(vec![2, 5].into()));

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let data_page = pages
            .iter()
            .find(|p| p.header.r#type == PageType::DATA_PAGE)
            .unwrap();
        let dict_page = pages
            .iter()
            .find(|p| p.header.r#type == PageType::DICTIONARY_PAGE)
            .unwrap();
        assert!(data_page.filter_mask.is_some());
        assert_eq!(data_page.filter_mask.as_ref().unwrap().rows(), 2);
        assert!(dict_page.filter_mask.is_none());
    }

    #[test]
    fn test_no_filter_mask_without_indices() {
        let col = make_column_buffer(&[(data_page_header(10, 2), vec![0xAA, 0xBB])]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        assert!(pages[0].filter_mask.is_none());
    }

    #[test]
    fn test_dict_does_not_increment_page_idx() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(5, 1), vec![0xA0]),
            (data_page_header(5, 1), vec![0xA1]),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        let mut data_page_indices: Vec<usize> = pages
            .iter()
            .filter(|p| p.header.r#type == PageType::DATA_PAGE)
            .map(|p| p.page_idx)
            .collect();
        data_page_indices.sort();
        assert_eq!(data_page_indices, vec![0, 1]);
    }

    /// A `Decompressed` part (a decompressed-cache hit) becomes a `CompressedPage`
    /// with no parsing: its bytes flow through as `decompressed`, not `data`.
    #[test]
    fn test_decompressed_part_skips_parsing() {
        let header = data_page_header(7, 0);
        let data = vec![Bytes::from(vec![9u8, 9, 9])];
        let col = vec![CachedColumnChunk::Decompressed {
            offset: 42,
            span: 20,
            header: Box::new(header),
            data: data.clone(),
        }];
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].file_offset, 42);
        assert_eq!(pages[0].span, 20);
        assert!(pages[0].data.is_empty());
        assert_eq!(pages[0].decompressed, Some(data));
    }

    /// `page_idx` and filter-mask row offsets stay continuous across a
    /// `Compressed` part followed by a `Decompressed` one in the same column.
    #[test]
    fn test_cursor_continuous_across_mixed_parts() {
        let payload = vec![0xA0];
        let mut parts =
            make_column_buffer(&[(data_page_header(10, payload.len() as i32), payload)]);
        parts.push(CachedColumnChunk::Decompressed {
            offset: 1000,
            span: 5,
            header: Box::new(data_page_header(10, 0)),
            data: vec![Bytes::from(vec![1u8])],
        });
        let buffer = make_row_group_buffer(vec![parts], RowSelection::Indices(vec![15].into()));

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let mut by_idx: Vec<_> = pages.iter().collect();
        by_idx.sort_by_key(|p| p.page_idx);
        assert_eq!(by_idx[0].page_idx, 0);
        assert_eq!(by_idx[1].page_idx, 1);
        // Row 15 falls in the second page's [10, 20) range, not the first's [0, 10).
        assert!(by_idx[0].filter_mask.as_ref().unwrap().all_false());
        assert_eq!(by_idx[1].filter_mask.as_ref().unwrap().rows(), 1);
    }

    /// Where the data pages of [`three_indexed_pages`] start in their file.
    const INDEXED_PAGES_OFFSET: usize = 100;

    /// A row group of thirty rows whose one leaf spans three data pages of
    /// ten rows that the file's offset index lists, read for `selection`:
    /// its metadata and the bytes of each page.
    fn three_indexed_pages(selection: RowSelection) -> (QueryRowGroupMetadata, Vec<Vec<u8>>) {
        let pages: Vec<Vec<u8>> = (0..3u8)
            .map(|page| {
                let mut bytes = serialize_header(&data_page_header(10, 1));
                bytes.push(page);
                bytes
            })
            .collect();
        let mut page_locations = Vec::new();
        let mut offset = INDEXED_PAGES_OFFSET;
        for (page, bytes) in pages.iter().enumerate() {
            page_locations.push(PageLocation {
                offset: offset as i64,
                compressed_page_size: bytes.len() as i32,
                first_row_index: page as i64 * 10,
            });
            offset += bytes.len();
        }
        let chunk = ColumnChunkMeta {
            data_page_offset: INDEXED_PAGES_OFFSET as i64,
            total_compressed_size: (offset - INDEXED_PAGES_OFFSET) as i64,
            absent: false,
            ..ColumnChunkMeta::absent()
        };
        let offset_index = OffsetIndex { page_locations };
        let page_index =
            RowGroupPageIndex::from_offset_indexes([Some((&chunk, &offset_index))].into_iter(), 30)
                .unwrap();
        let row_group = RowGroupMetadata {
            columns: vec![chunk],
            num_rows: 30,
            page_index: page_index.map(Arc::new),
            ..(*dummy_row_group()).clone()
        };
        let table = ParquetTable::new(vec![Arc::new(row_group)]);
        (QueryRowGroupMetadata::new(&table, 0, selection), pages)
    }

    /// The buffer of a read that took `parts` of the leaf of
    /// [`three_indexed_pages`] from the file.
    fn buffer_of_selected_pages(
        metadata: QueryRowGroupMetadata,
        parts: Vec<CachedColumnChunk>,
    ) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata,
            columns: vec![ColumnBuffer::SelectedPages {
                codec: CompressionCodec::SNAPPY,
                leaf: 0,
                parts,
            }],
            worker_id: 0,
            read_ahead_charge: None,
        }
    }

    #[test]
    fn a_page_read_by_the_offset_index_is_placed_among_stand_ins_for_the_others() {
        let (metadata, pages) = three_indexed_pages(RowSelection::Indices(vec![15].into()));
        let second_page = CachedColumnChunk::Compressed {
            offset: INDEXED_PAGES_OFFSET + pages[0].len(),
            bytes: vec![Bytes::from(pages[1].clone())],
        };
        let buffer = buffer_of_selected_pages(metadata, vec![second_page]);

        let mut pages = run_unary(Indexer {}, vec![buffer]);

        pages.sort_by_key(|page| page.page_idx);
        let first_rows: Vec<u32> = pages.iter().map(|page| page.first_row).collect();
        let kept_rows: Vec<usize> = pages
            .iter()
            .map(|page| page.filter_mask.as_ref().unwrap().rows())
            .collect();
        let bytes: Vec<usize> = pages
            .iter()
            .map(|page| page.data.iter().map(Bytes::len).sum())
            .collect();
        assert_eq!(first_rows, vec![0, 10, 20]);
        assert_eq!(kept_rows, vec![0, 1, 0]);
        assert_eq!(bytes, vec![0, 1, 0]);
    }

    #[test]
    fn a_page_holding_other_rows_than_the_offset_index_gives_it_is_an_error() {
        let (metadata, _) = three_indexed_pages(RowSelection::Indices(vec![5].into()));
        let mut first_page = serialize_header(&data_page_header(7, 1));
        first_page.push(0);
        let buffer = buffer_of_selected_pages(
            metadata,
            vec![CachedColumnChunk::Compressed {
                offset: INDEXED_PAGES_OFFSET,
                bytes: vec![Bytes::from(first_page)],
            }],
        );
        let mut test_io = dispatch::TestOperatorIO::default();

        let result = Indexer {}.consume(buffer, &mut CollectSender::new(), &mut test_io.io());

        assert!(result.is_err());
    }

    #[test]
    fn a_page_read_from_where_the_offset_index_lists_none_is_an_error() {
        let (metadata, pages) = three_indexed_pages(RowSelection::Indices(vec![5].into()));
        let buffer = buffer_of_selected_pages(
            metadata,
            vec![CachedColumnChunk::Compressed {
                offset: INDEXED_PAGES_OFFSET + 1,
                bytes: vec![Bytes::from(pages[0].clone())],
            }],
        );
        let mut test_io = dispatch::TestOperatorIO::default();

        let result = Indexer {}.consume(buffer, &mut CollectSender::new(), &mut test_io.io());

        assert!(result.is_err());
    }
}
