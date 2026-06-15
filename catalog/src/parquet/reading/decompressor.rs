//! Snappy-decompresses individual compressed Parquet pages.
//!
//! Each [`CompressedPage`] is decompressed into a [`DecompressedPage`] using the Ring buffer
//! pool for output allocation. These pages can be backed by multiple underlying buffers, and will be
//! decompressed into multiple buffers (from the memory ring).
//!
//! Pages whose [`FilterMask`] is entirely false are short-circuited
//! into [`DecompressedPageType::SkippedData`] without touching the decompressor.
//!
//! [`FilterMask`]: crate::parquet::types::filter_mask::FilterMask

use crate::parquet::types::page::{
    CompressedPage, DataPage, DecompressedPage, DecompressedPageType,
};
use crate::parquet::types::thrift::general::PageType;
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{BUFFER_SIZE, memory_ctx};
use snap::raw::Decoder;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Snappy(#[from] snap::Error),
    #[error("{0}")]
    UnsupportedPageType(PageType),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Factory for creating [`Decompressor`] instances, one per worker thread.
pub type DecompressorFactory = DefaultUnaryFactory<Decompressor>;

/// Snappy decompressor that converts [`CompressedPage`]s into [`DecompressedPage`]s.
///
/// Holds a reusable [`snap::raw::Decoder`] to avoid per-page allocation of decoder state.
pub struct Decompressor {
    decoder: Decoder,
}

impl Default for Decompressor {
    fn default() -> Self {
        Self {
            decoder: Decoder::new(),
        }
    }
}

impl Decompressor {
    fn decompress(&mut self, page: CompressedPage) -> Result<DecompressedPage> {
        let input: Vec<&[u8]> = page.data.iter().map(|b: &Bytes| b.as_ref()).collect();

        let uncompressed_size = page.header.uncompressed_page_size as usize;
        let num_buffers = uncompressed_size.div_ceil(BUFFER_SIZE);
        let mut write_buffers: Vec<_> = (0..num_buffers)
            .map(|_| memory_ctx().get_write_buffer(false))
            .collect();

        let output_bufs: Vec<&mut [u8]> = write_buffers.iter_mut().map(|b| b.as_mut()).collect();

        self.decoder.decompress_scattered(&input, output_bufs)?;

        let mut data = Vec::with_capacity(num_buffers);
        let mut remaining = uncompressed_size;
        for write_buffer in write_buffers {
            let chunk_len = remaining.min(BUFFER_SIZE);
            data.push(Bytes::from_owner(write_buffer).slice(..chunk_len));
            remaining -= chunk_len;
        }

        Ok(DecompressedPage {
            worker_id: page.worker_id,
            query_row_group_metadata: page.row_group,
            column_idx: page.column_idx,
            data: match page.header.r#type {
                PageType::DATA_PAGE => DecompressedPageType::Data(DataPage {
                    header: page.header.data_page_header.unwrap(),
                    data,
                    filter_mask: page.filter_mask,
                }),
                PageType::DICTIONARY_PAGE => DecompressedPageType::Dict {
                    header: page.header.dictionary_page_header.unwrap(),
                    data,
                },
                _ => return Err(Error::UnsupportedPageType(page.header.r#type)),
            },
            idx: page.page_idx,
        })
    }
}

impl Unary<CompressedPage, DecompressedPage> for Decompressor {
    fn consume<OP: Sender<DecompressedPage>>(
        &mut self,
        page: CompressedPage,
        output: &mut OP,
    ) -> dispatch::UnaryResult<()> {
        // A data page whose row group was pruned downstream (dictionary pushdown
        // set the shared flag) is never read: the decoder has already dropped the
        // row group and ignores its late-arriving pages (via `closed_row_groups`).
        // Drop it here entirely — not even a SkippedData marker — saving the
        // marker allocation and the channel round-trip for every data page of
        // every pruned row group. Completion is stream-based (no page-count
        // wait), so a dropped page can't stall it. Dictionary pages carry no
        // `data_page_header` and still decompress (needed for the membership
        // scan).
        if page.header.data_page_header.is_some() && page.row_group.is_pruned() {
            return Ok(());
        }

        // Skip snappy for a data page whose rows are all filtered out by a filter
        // mask: the decoder still needs the page slot for accounting, so emit a
        // SkippedData marker instead of paying decompression.
        let skip = page.header.data_page_header.is_some()
            && page.filter_mask.as_ref().is_some_and(|f| f.all_false());
        let decompressed_page = if skip {
            DecompressedPage {
                worker_id: page.worker_id,
                query_row_group_metadata: page.row_group,
                column_idx: page.column_idx,
                idx: page.page_idx,
                data: DecompressedPageType::SkippedData {
                    header: page.header.data_page_header.unwrap(),
                },
            }
        } else {
            self.decompress(page).map_err(crate::parquet::op_err)?
        };

        output.send(decompressed_page)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::test_utils::dummy_metadata;
    use crate::parquet::types::filter_mask::FilterMask;
    use crate::parquet::types::thrift::general::Encoding;
    use crate::parquet::types::thrift::headers::{
        DataPageHeader, DictionaryPageHeader, PageHeader,
    };
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::{CollectSender, run_unary};
    use snap::raw::Encoder;

    // -- Helpers --

    /// Snappy-compress `data` and return a CompressedPage with a data page header.
    fn compressed_data_page(
        data: &[u8],
        num_values: i32,
        filter_mask: Option<FilterMask>,
    ) -> CompressedPage {
        let compressed = Encoder::new().compress_vec(data).unwrap();
        CompressedPage {
            worker_id: 0,
            row_group: dummy_metadata(None),
            column_idx: 0,
            page_idx: 0,
            header: PageHeader {
                r#type: PageType::DATA_PAGE,
                uncompressed_page_size: data.len() as i32,
                compressed_page_size: compressed.len() as i32,
                crc: None,
                data_page_header: Some(DataPageHeader {
                    num_values,
                    encoding: Encoding::PLAIN,
                    definition_level_encoding: Encoding::RLE,
                    repetition_level_encoding: Encoding::RLE,
                    statistics: None,
                }),
                index_page_header: None,
                dictionary_page_header: None,
                data_page_header_v2: None,
            },
            data: vec![Bytes::from(compressed)],
            filter_mask,
        }
    }

    fn compressed_dict_page(data: &[u8], num_values: i32) -> CompressedPage {
        let compressed = Encoder::new().compress_vec(data).unwrap();
        CompressedPage {
            worker_id: 0,
            row_group: dummy_metadata(None),
            column_idx: 0,
            page_idx: 0,
            header: PageHeader {
                r#type: PageType::DICTIONARY_PAGE,
                uncompressed_page_size: data.len() as i32,
                compressed_page_size: compressed.len() as i32,
                crc: None,
                data_page_header: None,
                index_page_header: None,
                dictionary_page_header: Some(DictionaryPageHeader {
                    num_values,
                    encoding: Encoding::PLAIN_DICTIONARY,
                    is_sorted: None,
                }),
                data_page_header_v2: None,
            },
            data: vec![Bytes::from(compressed)],
            filter_mask: None,
        }
    }

    // -- Tests --

    /// All-false filter mask → SkippedData without decompression.
    #[test]
    fn test_skip_on_all_false_filter_mask() {
        let mask = FilterMask::new(0, 10, &[]);
        let page = compressed_data_page(b"irrelevant", 10, Some(mask));

        let out = run_unary(Decompressor::default(), vec![page]);

        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0].data,
            DecompressedPageType::SkippedData { .. }
        ));
    }

    /// SkippedData preserves the original data page header.
    #[test]
    fn test_skipped_data_preserves_header() {
        let mask = FilterMask::new(0, 5, &[]);
        let page = compressed_data_page(b"irrelevant", 42, Some(mask));

        let out = run_unary(Decompressor::default(), vec![page]);

        let DecompressedPageType::SkippedData { header } = &out[0].data else {
            panic!("expected SkippedData");
        };
        assert_eq!(header.num_values, 42);
    }

    /// Metadata fields (worker_id, column_idx, page_idx) are preserved through decompression.
    #[test]
    fn test_metadata_preserved() {
        init_test_free_pool(4);
        let payload = vec![1u8; 128];
        let mut page = compressed_data_page(&payload, 5, None);
        page.worker_id = 7;
        page.column_idx = 3;
        page.page_idx = 12;

        let out = run_unary(Decompressor::default(), vec![page]);

        let out = &out[0];
        assert_eq!(out.worker_id, 7);
        assert_eq!(out.column_idx, 3);
        assert_eq!(out.idx, 12);
    }

    /// Data page → DecompressedPageType::Data with correct decompressed bytes.
    #[test]
    fn test_data_page_decompression() {
        init_test_free_pool(4);
        let payload = vec![0xABu8; 256];
        let page = compressed_data_page(&payload, 100, None);

        let out = run_unary(Decompressor::default(), vec![page]);

        assert_eq!(out.len(), 1);
        let DecompressedPageType::Data(data_page) = &out[0].data else {
            panic!("expected Data");
        };
        let decompressed: Vec<u8> = data_page.data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(decompressed, payload);
    }

    /// Dictionary page → DecompressedPageType::Dict with correct decompressed bytes.
    #[test]
    fn test_dict_page_decompression() {
        init_test_free_pool(4);
        let payload = vec![0xDDu8; 64];
        let page = compressed_dict_page(&payload, 50);

        let out = run_unary(Decompressor::default(), vec![page]);

        assert_eq!(out.len(), 1);
        let DecompressedPageType::Dict { header, data } = &out[0].data else {
            panic!("expected Dict");
        };
        assert_eq!(header.num_values, 50);
        let decompressed: Vec<u8> = data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(decompressed, payload);
    }

    /// Non-all-false filter mask → page is decompressed and filter mask is forwarded.
    #[test]
    fn test_filter_mask_passed_through() {
        init_test_free_pool(4);
        let mask = FilterMask::new(0, 10, &[3]);
        let page = compressed_data_page(&[0u8; 32], 10, Some(mask));

        let out = run_unary(Decompressor::default(), vec![page]);

        let DecompressedPageType::Data(data_page) = &out[0].data else {
            panic!("expected Data");
        };
        assert!(data_page.filter_mask.is_some());
        assert_eq!(data_page.filter_mask.as_ref().unwrap().rows(), 1);
    }

    /// Unsupported page type → error.
    #[test]
    fn test_unsupported_page_type() {
        init_test_free_pool(4);
        let compressed = Encoder::new().compress_vec(&[0u8; 16]).unwrap();
        let page = CompressedPage {
            worker_id: 0,
            row_group: dummy_metadata(None),
            column_idx: 0,
            page_idx: 0,
            header: PageHeader {
                r#type: PageType::INDEX_PAGE,
                uncompressed_page_size: 16,
                compressed_page_size: compressed.len() as i32,
                crc: None,
                data_page_header: None,
                index_page_header: None,
                dictionary_page_header: None,
                data_page_header_v2: None,
            },
            data: vec![Bytes::from(compressed)],
            filter_mask: None,
        };

        let mut sender = CollectSender::new();
        let result = Decompressor::default().consume(page, &mut sender);

        assert!(result.is_err());
    }
}
