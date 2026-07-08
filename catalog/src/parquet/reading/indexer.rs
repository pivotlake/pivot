//! Splits raw column buffers into individual compressed pages and sends them in an order
//! optimised for downstream decoding.
//!
//! The Indexer sits between IO (which delivers whole column chunks) and the Decompressor
//! (which expects individual pages). For each column it walks the byte stream, parsing Thrift
//! page headers and slicing out the compressed payload. When filtered indices are present it
//! attaches a [`FilterMask`] to each data page so the decoder knows which rows to keep.
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
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::page::CompressedPage;
use crate::parquet::types::requests::RowGroupBuffer;
use crate::parquet::types::thrift::general::PageType;
use crate::parquet::types::thrift::headers::PageHeader;
use crate::parquet::types::thrift::parquet_thrift::{ParquetError, ThriftReadInputProtocol};
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{MultiBufferReader, ReaderPosition};
use dispatch::worker::WORKER_IDX;

pub type IndexerFactory = DefaultUnaryFactory<Indexer>;

/// Splits a [`RowGroupBuffer`] into individual [`CompressedPage`]s.
#[derive(Default)]
pub struct Indexer {}

/// Walk a single column's byte stream and parse it into compressed pages.
///
/// Each iteration reads a Thrift page header, then slices out the compressed payload via
/// [`MultiBufferReader::copy_out_buffers`] (zero-copy).
/// For data pages with filtered indices, a [`FilterMask`] scoped to the page's
/// row range is attached.
fn create_compressed_pages(
    col_idx: usize,
    column_chunk_offset: usize,
    query_row_group_metadata: QueryRowGroupMetadata,
    buffers: &[Bytes],
) -> Result<Vec<CompressedPage>, ParquetError> {
    let mut position = ReaderPosition::default();
    let buffers_length = buffers.len();
    let mut reader = MultiBufferReader::new(buffers, &mut position);
    let mut pages = Vec::with_capacity(128);
    let mut data_page_idx = 0;
    let mut row_offset = 0;
    loop {
        // Bytes consumed so far = this page's offset within the chunk, which the
        // chunk's file offset turns into the page's absolute file offset.
        let page_start = reader.consumed();
        let header = {
            let mut prot = ThriftReadInputProtocol::new(&mut reader);
            PageHeader::read_thrift_without_stats(&mut prot)?
        };

        let data = reader.copy_out_buffers(header.compressed_page_size as usize);

        let page = CompressedPage {
            worker_id: WORKER_IDX.get(),
            row_group: query_row_group_metadata.clone(),
            column_idx: col_idx,
            file_offset: column_chunk_offset + page_start,
            page_idx: data_page_idx,
            data,
            filter_mask: if header.r#type == PageType::DATA_PAGE
                && let Some(f) = query_row_group_metadata.filtered_indices()
            {
                Some(FilterMask::new(
                    row_offset,
                    row_offset + header.data_page_num_values() as u32,
                    f,
                ))
            } else {
                None
            },
            header,
        };

        if page.header.r#type == PageType::DATA_PAGE {
            data_page_idx += 1;
            row_offset += page.header.data_page_num_values() as u32;
        }

        pages.push(page);

        if reader.remaining_in_cur() == 0 && reader.position().buffer_index == buffers_length - 1 {
            break;
        }
    }
    Ok(pages)
}

impl Unary<RowGroupBuffer, CompressedPage> for Indexer {
    fn consume<S: Sender<CompressedPage>>(
        &mut self,
        buffer: RowGroupBuffer,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let mut pages_per_column: Vec<_> = buffer
            .columns
            .iter()
            .enumerate()
            .map(|(col_idx, b)| {
                create_compressed_pages(
                    col_idx,
                    buffer.column_offsets[col_idx],
                    buffer.metadata.clone(),
                    b,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::parquet::op_err)?;

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
                encoding: Encoding::PLAIN,
                is_sorted: None,
            }),
            data_page_header_v2: None,
        }
    }

    /// Concatenate serialized page headers + payloads into a single column buffer.
    fn make_column_buffer(pages: &[(PageHeader, Vec<u8>)]) -> Vec<Bytes> {
        let mut buf = Vec::new();
        for (header, payload) in pages {
            buf.extend_from_slice(&serialize_header(header));
            buf.extend_from_slice(payload);
        }
        vec![Bytes::from(buf)]
    }

    fn make_row_group_buffer(
        columns: Vec<Vec<Bytes>>,
        filtered_indices: Option<Vec<u32>>,
    ) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: dummy_metadata(filtered_indices),
            column_offsets: (0..columns.len()).collect(),
            columns,
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
}
