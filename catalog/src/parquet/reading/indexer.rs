//! Splits raw column buffers into individual compressed pages and sends them in an order
//! optimised for downstream decoding.
//!
//! The Indexer sits between IO (which delivers whole column chunks, or pages already
//! resolved straight from the decompressed cache - see
//! [`ColumnPart`]) and the Decompressor
//! (which expects individual pages). For each column it walks its parts in file
//! order: a `Compressed` part's raw bytes are Thrift-parsed into pages same as
//! always; a `Decompressed` part is already one whole decompressed page, so it
//! becomes a `CompressedPage` directly, no parsing needed. When filtered indices are
//! present it attaches a [`FilterMask`] to each data page so the decoder knows which
//! rows to keep.
//!
//! A row group claimed with a [`DecodeFanOut`] plan is instead split into
//! row-range decode slices, each routed to a different worker as a dense mini
//! row group (see [`fan_out_row_group`]), so a scan with fewer row groups than
//! workers still decodes on the whole pool.
//!
//! [`DecodeFanOut`]: crate::parquet::types::metadata::DecodeFanOut
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

use crate::parquet::types::filter_mask::FilterMask;
use crate::parquet::types::metadata::{DecodeSlice, QueryRowGroupMetadata, SliceRelease};
use crate::parquet::types::page::CompressedPage;
use crate::parquet::types::requests::{ColumnPart, RowGroupBuffer};
use crate::parquet::types::thrift::general::PageType;
use crate::parquet::types::thrift::headers::PageHeader;
use crate::parquet::types::thrift::parquet_thrift::{ParquetError, ThriftReadInputProtocol};
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{MultiBufferReader, ReaderPosition};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

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
/// identity, the query's view of the row group, the worker that claimed it,
/// and the running [`PageCursor`] threaded across the column's parts.
struct ColumnPageBuilder {
    col_idx: usize,
    query_row_group_metadata: QueryRowGroupMetadata,
    worker_id: usize,
    cursor: PageCursor,
    /// Data pages actually emitted so far. Diverges from the cursor's count
    /// only for a decode slice, which drops pages outside its row range; the
    /// emitted index keeps each decoder's page slots dense from zero.
    emitted_data_pages: usize,
}

impl ColumnPageBuilder {
    /// The filter mask for a data page spanning `[row_offset, row_end)`, or
    /// `None` to keep every row. For a decode slice, a page entirely outside
    /// the slice's row range yields `Err(())`: the page is not emitted at all.
    fn data_page_mask(&self, row_offset: u32, row_end: u32) -> Result<Option<FilterMask>, ()> {
        if let Some(slice) = &self.query_row_group_metadata.decode_slice {
            let (keep_start, keep_end) = slice.row_range;
            if row_end <= keep_start || row_offset >= keep_end {
                return Err(());
            }
            if keep_start <= row_offset && row_end <= keep_end {
                return Ok(None);
            }
            return Ok(Some(FilterMask::from_range(
                row_offset, row_end, keep_start, keep_end,
            )));
        }
        Ok(self
            .query_row_group_metadata
            .filtered_indices()
            .as_ref()
            .map(|f| FilterMask::new(row_offset, row_end, f)))
    }

    /// Build one `CompressedPage`, attaching the current query's filter mask
    /// and advancing the cursor. Returns `None` for a data page that a decode
    /// slice drops entirely (outside its row range).
    fn build_page(
        &mut self,
        file_offset: usize,
        span: usize,
        header: PageHeader,
        payload: PagePayload,
    ) -> Option<CompressedPage> {
        let (_, row_offset) = self.cursor.advance(&header);
        let is_data_page = header.r#type == PageType::DATA_PAGE;
        let filter_mask = if is_data_page {
            match self.data_page_mask(
                row_offset,
                row_offset + header.data_page_num_values() as u32,
            ) {
                Ok(mask) => mask,
                Err(()) => return None,
            }
        } else {
            None
        };
        let page_idx = self.emitted_data_pages;
        if is_data_page {
            self.emitted_data_pages += 1;
        }
        let (data, decompressed) = match payload {
            PagePayload::Compressed(bytes) => (bytes, None),
            PagePayload::Decompressed(bytes) => (Vec::new(), Some(bytes)),
        };

        Some(CompressedPage {
            // This is the claimer, not necessarily this worker: the indexer
            // often runs on a stealing sibling, but the decode must return to
            // the worker that claimed the row group (it owns the decoder
            // state and the claim accounting).
            worker_id: self.worker_id,
            row_group: self.query_row_group_metadata.clone(),
            column_idx: self.col_idx,
            file_offset,
            span,
            page_idx,
            data,
            decompressed,
            filter_mask,
            header,
        })
    }

    /// Walk a `Compressed` part's raw byte stream and Thrift-parse it into pages,
    /// appending each to `pages`. Each iteration reads a page header, then slices out
    /// the compressed payload via [`MultiBufferReader::copy_out_buffers`] (zero-copy).
    fn parse_compressed_pages(
        &mut self,
        part_offset: usize,
        buffers: &[Bytes],
        pages: &mut Vec<CompressedPage>,
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

            if let Some(page) = self.build_page(
                part_offset + page_start,
                span,
                header,
                PagePayload::Compressed(data),
            ) {
                pages.push(page);
            }

            if reader.remaining_in_cur() == 0
                && reader.position().buffer_index == buffers_length - 1
            {
                break;
            }
        }
        Ok(())
    }
}

/// Build one column's pages, in file order, from its resolved parts.
fn build_column_pages(
    col_idx: usize,
    query_row_group_metadata: QueryRowGroupMetadata,
    parts: Vec<ColumnPart>,
    worker_id: usize,
) -> Result<Vec<CompressedPage>, ParquetError> {
    let mut builder = ColumnPageBuilder {
        col_idx,
        query_row_group_metadata,
        worker_id,
        cursor: PageCursor::default(),
        emitted_data_pages: 0,
    };
    let mut pages = Vec::with_capacity(128);
    for part in parts {
        match part {
            ColumnPart::Compressed { offset, bytes } => {
                builder.parse_compressed_pages(offset, &bytes, &mut pages)?
            }
            ColumnPart::Decompressed {
                offset,
                span,
                header,
                data,
            } => {
                if let Some(page) =
                    builder.build_page(offset, span, *header, PagePayload::Decompressed(data))
                {
                    pages.push(page);
                }
            }
        }
    }
    Ok(pages)
}

/// Emit page lists onto the LIFO channel: data pages interleaved so the list
/// with the fewest emitted rows goes next (the decoder can cut batches as
/// early as possible), dictionary pages last so they arrive first.
fn emit_pages<S: Sender<CompressedPage>>(
    mut pages_per_column: Vec<Vec<CompressedPage>>,
    sender: &mut S,
) -> dispatch::UnaryResult<()> {
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

/// Split a fanned-out row group into row-range decode slices, each presented
/// to a different worker as a dense mini row group: only the pages overlapping
/// the slice's range are emitted (boundary pages carry a range mask), page
/// indices are renumbered per slice, and every slice gets its own copy of the
/// dictionary pages. Slices share the claim via [`SliceRelease`]; the
/// decompressed cache dedupes the byte work of pages sent to two slices.
fn fan_out_row_group<S: Sender<CompressedPage>>(
    buffer: RowGroupBuffer,
    worker_count: usize,
    slices: usize,
    sender: &mut S,
) -> dispatch::UnaryResult<()> {
    let num_rows = buffer.metadata.num_rows() as u32;
    let release = Arc::new(SliceRelease {
        remaining: AtomicUsize::new(slices),
        claimer_worker_id: buffer.worker_id,
    });
    let rows_per_slice = num_rows.div_ceil(slices as u32);
    let worker_stride = (worker_count / slices).max(1);

    // Parse each column's parts into pages once; the slices below take cheap
    // clones (shared `Bytes`) of the pages overlapping their row range instead
    // of re-parsing the Thrift headers per slice.
    let base_pages_per_column: Vec<Vec<CompressedPage>> = buffer
        .columns
        .into_iter()
        .enumerate()
        .map(|(col_idx, parts)| {
            build_column_pages(col_idx, buffer.metadata.clone(), parts, buffer.worker_id)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(crate::parquet::op_err)?;

    let mut pages_per_slice_column = Vec::with_capacity(slices * base_pages_per_column.len());
    for slice_idx in 0..slices {
        let keep_start = slice_idx as u32 * rows_per_slice;
        let keep_end = (keep_start + rows_per_slice).min(num_rows);
        let slice_metadata = buffer.metadata.for_decode_slice(DecodeSlice {
            row_range: (keep_start, keep_end),
            release: release.clone(),
        });
        let slice_worker = (buffer.worker_id + slice_idx * worker_stride) % worker_count;
        for base_pages in &base_pages_per_column {
            let mut slice_pages = Vec::with_capacity(base_pages.len() / slices + 2);
            let mut row_offset = 0u32;
            let mut emitted_data_pages = 0;
            for page in base_pages {
                let is_data_page = page.header.r#type == PageType::DATA_PAGE;
                let (filter_mask, page_idx) = if is_data_page {
                    let row_end = row_offset + page.header.data_page_num_values() as u32;
                    let page_range = (row_offset, row_end);
                    row_offset = row_end;
                    if page_range.1 <= keep_start || page_range.0 >= keep_end {
                        continue;
                    }
                    let mask = (keep_start > page_range.0 || page_range.1 > keep_end).then(|| {
                        FilterMask::from_range(page_range.0, page_range.1, keep_start, keep_end)
                    });
                    emitted_data_pages += 1;
                    (mask, emitted_data_pages - 1)
                } else {
                    (None, emitted_data_pages)
                };
                slice_pages.push(CompressedPage {
                    worker_id: slice_worker,
                    row_group: slice_metadata.clone(),
                    column_idx: page.column_idx,
                    file_offset: page.file_offset,
                    span: page.span,
                    page_idx,
                    header: page.header.clone(),
                    data: page.data.clone(),
                    decompressed: page.decompressed.clone(),
                    filter_mask,
                });
            }
            pages_per_slice_column.push(slice_pages);
        }
    }
    emit_pages(pages_per_slice_column, sender)
}

impl Unary<RowGroupBuffer, CompressedPage> for Indexer {
    fn consume<S: Sender<CompressedPage>>(
        &mut self,
        buffer: RowGroupBuffer,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if let Some(plan) = buffer.metadata.decode_fan_out
            && buffer.metadata.filtered_indices().is_none()
        {
            let slices = plan.slice_count(buffer.metadata.num_rows() as u32);
            if slices > 1 {
                return fan_out_row_group(buffer, plan.worker_count, slices, sender);
            }
        }

        let pages_per_column: Vec<_> = buffer
            .columns
            .into_iter()
            .enumerate()
            .map(|(col_idx, parts)| {
                build_column_pages(col_idx, buffer.metadata.clone(), parts, buffer.worker_id)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::parquet::op_err)?;

        emit_pages(pages_per_column, sender)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::test_utils::dummy_metadata;
    use crate::parquet::types::thrift::general::Encoding;
    use crate::parquet::types::thrift::headers::{
        DataPageHeader as ThriftDataPageHeader, DictionaryPageHeader as ThriftDictionaryPageHeader,
    };
    use crate::parquet::types::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
    use bytes::Bytes;
    use dispatch::test_utils::run_unary;

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
    fn make_column_buffer(pages: &[(PageHeader, Vec<u8>)]) -> Vec<ColumnPart> {
        let mut buf = Vec::new();
        for (header, payload) in pages {
            buf.extend_from_slice(&serialize_header(header));
            buf.extend_from_slice(payload);
        }
        vec![ColumnPart::Compressed {
            offset: 0,
            bytes: vec![Bytes::from(buf)],
        }]
    }

    fn make_row_group_buffer(
        columns: Vec<Vec<ColumnPart>>,
        filtered_indices: Option<Vec<u32>>,
    ) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: dummy_metadata(filtered_indices),
            columns,
            worker_id: 0,
        }
    }

    /// Single data page → 1 CompressedPage with correct column_idx, page_idx, and payload.
    #[test]
    fn test_single_data_page() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let header = data_page_header(10, payload.len() as i32);
        let col = make_column_buffer(&[(header, payload.clone())]);
        let buffer = make_row_group_buffer(vec![col], None);

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
        let buffer = make_row_group_buffer(vec![col], None);

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
        let buffer = make_row_group_buffer(vec![col], None);

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
        let buffer = make_row_group_buffer(vec![col_a, col_b], None);

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
        let buffer = make_row_group_buffer(vec![col_a, col_b], None);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        let col_ids: Vec<usize> = pages.iter().map(|p| p.column_idx).collect();
        assert_eq!(col_ids, vec![0, 1, 0]);
    }

    /// With filtered_indices, data pages get a FilterMask; dict pages don't.
    #[test]
    fn test_filter_mask_on_data_pages_only() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(10, 1), vec![0xAA]),
        ]);
        let buffer = make_row_group_buffer(vec![col], Some(vec![2, 5]));

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
        let buffer = make_row_group_buffer(vec![col], None);

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
        let buffer = make_row_group_buffer(vec![col], None);

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
        let col = vec![ColumnPart::Decompressed {
            offset: 42,
            span: 20,
            header: Box::new(header),
            data: data.clone(),
        }];
        let buffer = make_row_group_buffer(vec![col], None);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].file_offset, 42);
        assert_eq!(pages[0].span, 20);
        assert!(pages[0].data.is_empty());
        assert_eq!(pages[0].decompressed, Some(data));
    }

    fn metadata_with_rows(num_rows: i64) -> QueryRowGroupMetadata {
        use crate::parquet::types::metadata::RowGroupMetadata;
        use crate::parquet::types::table::ParquetTable;
        let dummy = crate::parquet::test_utils::dummy_row_group();
        let table = ParquetTable::new(vec![std::sync::Arc::new(RowGroupMetadata {
            open_file: dummy.open_file.clone(),
            schema: dummy.schema.clone(),
            columns: vec![],
            num_rows,
            file_row_group_idx: 0,
            live_decompressed_pages: std::sync::Arc::new(AtomicUsize::new(0)),
        })]);
        QueryRowGroupMetadata::new(&table, 0, None)
    }

    /// Two decode slices split a column's pages by row range: interior pages
    /// go to exactly one slice unmasked, the page straddling the boundary goes
    /// to both with complementary masks, and the dictionary page goes to both.
    #[test]
    fn fan_out_splits_pages_into_row_range_slices() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(100, 1), vec![0xA0]),
            (data_page_header(100, 1), vec![0xA1]),
            (data_page_header(100, 1), vec![0xA2]),
        ]);
        let buffer = RowGroupBuffer {
            metadata: metadata_with_rows(300),
            columns: vec![col],
            worker_id: 3,
        };
        let mut sender = dispatch::test_utils::CollectSender::default();

        fan_out_row_group(buffer, 4, 2, &mut sender).unwrap();

        let pages = sender.items;
        let dicts: Vec<_> = pages
            .iter()
            .filter(|p| p.header.r#type == PageType::DICTIONARY_PAGE)
            .collect();
        assert_eq!(dicts.len(), 2);
        let slice_of = |p: &CompressedPage| p.row_group.decode_slice.clone().unwrap();
        let first_slice: Vec<_> = pages
            .iter()
            .filter(|p| p.header.r#type == PageType::DATA_PAGE && slice_of(p).row_range == (0, 150))
            .collect();
        let second_slice: Vec<_> = pages
            .iter()
            .filter(|p| {
                p.header.r#type == PageType::DATA_PAGE && slice_of(p).row_range == (150, 300)
            })
            .collect();
        assert_eq!(first_slice.len(), 2);
        assert_eq!(second_slice.len(), 2);
        let mask_rows = |pages: &[&CompressedPage]| -> usize {
            pages
                .iter()
                .map(|p| {
                    p.filter_mask
                        .as_ref()
                        .map(|m| m.rows())
                        .unwrap_or(p.header.data_page_num_values() as usize)
                })
                .sum()
        };
        assert_eq!(mask_rows(&first_slice), 150);
        assert_eq!(mask_rows(&second_slice), 150);
        let workers = |pages: &[&CompressedPage]| -> Vec<usize> {
            let mut w: Vec<_> = pages.iter().map(|p| p.worker_id).collect();
            w.dedup();
            w
        };
        assert_ne!(workers(&first_slice), workers(&second_slice));
        let mut first_indices: Vec<_> = first_slice.iter().map(|p| p.page_idx).collect();
        first_indices.sort();
        assert_eq!(first_indices, vec![0, 1]);
        assert_eq!(
            slice_of(first_slice[0])
                .release
                .remaining
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    /// `page_idx` and filter-mask row offsets stay continuous across a
    /// `Compressed` part followed by a `Decompressed` one in the same column.
    #[test]
    fn test_cursor_continuous_across_mixed_parts() {
        let payload = vec![0xA0];
        let mut parts =
            make_column_buffer(&[(data_page_header(10, payload.len() as i32), payload)]);
        parts.push(ColumnPart::Decompressed {
            offset: 1000,
            span: 5,
            header: Box::new(data_page_header(10, 0)),
            data: vec![Bytes::from(vec![1u8])],
        });
        let buffer = make_row_group_buffer(vec![parts], Some(vec![15]));

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
}
