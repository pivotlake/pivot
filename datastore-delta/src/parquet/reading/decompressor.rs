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
use crate::parquet::types::thrift::headers::PageHeader;
use crate::parquet::types::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{BlockKey, memory_ctx};
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
    fn decompress(&mut self, mut page: CompressedPage) -> Result<DecompressedPage> {
        let is_dict = match page.header.r#type {
            PageType::DATA_PAGE => false,
            PageType::DICTIONARY_PAGE => true,
            other => return Err(Error::UnsupportedPageType(other)),
        };

        // A page already resolved from the decompressed cache before this row
        // group's IO was even issued (see `requests::ColumnPart::Decompressed`)
        // carries its bytes here directly - no cache lookup needed, and none of the
        // races one would otherwise have to reason about between that resolution
        // and now.
        let data = if let Some(decompressed) = page.decompressed.take() {
            decompressed
        } else {
            // The decompressed bytes depend only on the compressed input, not on the
            // per-query filter mask, so a cache hit reuses them across queries; the
            // mask is attached to the `DataPage` below either way.
            let key = BlockKey {
                open_file: page.row_group.get_metadata().open_file.clone(),
                offset: page.file_offset,
                len: page.span,
            };
            match memory_ctx().decompressed_cache().get(&key) {
                Some(cached) => cached,
                // An empty page decompresses to no bytes, and the snappy decoder
                // panics if handed zero output buffers. This happens for the
                // dictionary page of an all-null column, e.g. the unused `value`
                // fallback leaf of a shredded variant path (every value went to
                // its typed leaf). Nothing to cache either.
                None if page.header.uncompressed_page_size == 0 => Vec::new(),
                None => {
                    // Reserve the page's output region in the cache (packed into
                    // shared slots via this worker's fill cursor), decompress
                    // straight into it, and insert. The header is stored
                    // alongside it, opaque to the cache, so a later `get_range`
                    // hit can hand a caller a full page identity without touching
                    // disk (see `requests::parse_page_header`).
                    let uncompressed_size = page.header.uncompressed_page_size as usize;
                    let mut reservation = memory_ctx()
                        .decompressed_cache()
                        .reserve(&key, uncompressed_size);
                    let input: Vec<&[u8]> = page.data.iter().map(|b: &Bytes| b.as_ref()).collect();
                    self.decoder
                        .decompress_scattered(&input, reservation.as_mut_slices())?;
                    memory_ctx().decompressed_cache().insert(
                        key,
                        serialize_header(&page.header),
                        reservation,
                        page.row_group
                            .get_metadata()
                            .live_decompressed_pages
                            .clone(),
                    )
                }
            }
        };

        let payload = if is_dict {
            DecompressedPageType::Dict {
                header: page.header.dictionary_page_header.unwrap(),
                data,
            }
        } else {
            DecompressedPageType::Data(DataPage {
                header: page.header.data_page_header.unwrap(),
                data,
                filter_mask: page.filter_mask,
            })
        };

        Ok(DecompressedPage {
            worker_id: page.worker_id,
            query_row_group_metadata: page.row_group,
            column_idx: page.column_idx,
            data: payload,
            idx: page.page_idx,
        })
    }
}

/// Serialize a page's header back to Thrift compact bytes, to store alongside its
/// decompressed bytes in the cache (opaque to the cache itself - see
/// `requests::parse_page_header`, which reverses this on a hit). Cheap: a few dozen
/// bytes, entirely in memory, no relation to the (possibly much larger) page payload.
fn serialize_header(header: &PageHeader) -> Vec<Bytes> {
    let mut buf = Vec::new();
    let mut prot = ThriftCompactOutputProtocol::new(&mut buf);
    header
        .write_thrift(&mut prot)
        .expect("writing to an in-memory buffer cannot fail");
    vec![Bytes::from(buf)]
}

impl Unary<CompressedPage, DecompressedPage> for Decompressor {
    fn consume(
        &mut self,
        page: CompressedPage,
        output: &mut dyn Sender<DecompressedPage>,
        _io: &mut dispatch::OperatorIO,
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
            file_offset: 0,
            span: compressed.len(),
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
            decompressed: None,
            filter_mask,
        }
    }

    fn compressed_dict_page(data: &[u8], num_values: i32) -> CompressedPage {
        let compressed = Encoder::new().compress_vec(data).unwrap();
        CompressedPage {
            worker_id: 0,
            row_group: dummy_metadata(None),
            column_idx: 0,
            file_offset: 0,
            span: compressed.len(),
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
            decompressed: None,
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

    /// A page whose `decompressed` field is already set (a decompressed-cache hit
    /// resolved before this row group's IO was issued) is used directly - no
    /// decompression, no cache lookup.
    #[test]
    fn test_decompressed_field_skips_decompression() {
        init_test_free_pool(1);
        let mut page = compressed_data_page(b"never touched", 10, None);
        // Garbage compressed bytes that would fail to decompress, proving they're
        // never read: `decompressed` takes priority.
        page.data = vec![Bytes::from(vec![0xFFu8; 4])];
        page.decompressed = Some(vec![Bytes::from(vec![7u8, 8, 9])]);

        let out = run_unary(Decompressor::default(), vec![page]);

        let DecompressedPageType::Data(data_page) = &out[0].data else {
            panic!("expected Data");
        };
        let bytes: Vec<u8> = data_page.data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(bytes, vec![7, 8, 9]);
    }

    /// Caching a decompressed page counts it live on its row group's metadata,
    /// the signal the scan feed uses to schedule cache-covered row groups first.
    #[test]
    fn test_cached_page_counts_live_on_its_row_group() {
        init_test_free_pool(4);
        let metadata = dummy_metadata(None);
        let mut page = compressed_data_page(&[7u8; 64], 10, None);
        page.row_group = metadata.clone();

        let _out = run_unary(Decompressor::default(), vec![page]);

        assert_eq!(
            metadata
                .get_metadata()
                .live_decompressed_pages
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    /// Decompressing the same page identity twice reuses the cached bytes
    /// (same ring address) instead of decompressing again.
    #[test]
    fn test_same_page_is_cached_and_reused() {
        init_test_free_pool(8);
        let payload = vec![0x5Au8; 200];
        let metadata = dummy_metadata(None);
        let make_page = || {
            let mut page = compressed_data_page(&payload, 10, None);
            page.row_group = metadata.clone();
            page
        };

        let first = run_unary(Decompressor::default(), vec![make_page()]);
        let second = run_unary(Decompressor::default(), vec![make_page()]);

        let data = |out: &DecompressedPage| {
            let DecompressedPageType::Data(d) = &out.data else {
                panic!("expected Data");
            };
            d.data.clone()
        };
        let decompressed: Vec<u8> = data(&second[0]).iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(decompressed, payload);
        assert_eq!(
            data(&first[0])[0].as_ptr(),
            data(&second[0])[0].as_ptr(),
            "second read should reuse the cached ring buffer",
        );
    }

    /// A cache hit attaches the *current* query's filter mask, not the one from
    /// the read that populated the cache.
    #[test]
    fn test_cache_hit_attaches_the_query_filter_mask() {
        init_test_free_pool(8);
        let payload = vec![0u8; 64];
        let metadata = dummy_metadata(None);
        let mut first_page = compressed_data_page(&payload, 10, None);
        first_page.row_group = metadata.clone();
        run_unary(Decompressor::default(), vec![first_page]);

        let mut second_page =
            compressed_data_page(&payload, 10, Some(FilterMask::new(0, 10, &[3])));
        second_page.row_group = metadata.clone();
        let out = run_unary(Decompressor::default(), vec![second_page]);

        let DecompressedPageType::Data(data_page) = &out[0].data else {
            panic!("expected Data");
        };
        assert_eq!(data_page.filter_mask.as_ref().unwrap().rows(), 1);
    }

    /// Regression: the cache keys on the physical column-chunk offset, not the
    /// projection-relative `column_idx`. Two different physical columns that share
    /// a `column_idx` (across queries with different projections) must NOT collide.
    /// That bug returned column A's bytes for column B (an out-of-bounds at decode).
    #[test]
    fn test_distinct_columns_sharing_column_idx_do_not_collide() {
        init_test_free_pool(8);
        let metadata = dummy_metadata(None);
        let mut col_a = compressed_data_page(&[0xAAu8; 64], 10, None);
        col_a.row_group = metadata.clone();
        col_a.file_offset = 100;
        col_a.column_idx = 0;
        let mut col_b = compressed_data_page(&[0xBBu8; 64], 10, None);
        col_b.row_group = metadata.clone();
        col_b.file_offset = 200; // different physical column...
        col_b.column_idx = 0; // ...but same projection-relative index

        run_unary(Decompressor::default(), vec![col_a]);
        let out = run_unary(Decompressor::default(), vec![col_b]);

        let DecompressedPageType::Data(d) = &out[0].data else {
            panic!("expected Data");
        };
        let bytes: Vec<u8> = d.data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(
            bytes,
            vec![0xBBu8; 64],
            "must not return column A's cached bytes"
        );
    }

    /// The same physical column reached at a different projection position (same
    /// `file_offset`, different `column_idx`) still hits the cache.
    #[test]
    fn test_same_offset_different_column_idx_hits() {
        init_test_free_pool(8);
        let metadata = dummy_metadata(None);
        let payload = vec![0xC7u8; 80];
        let mut first = compressed_data_page(&payload, 10, None);
        first.row_group = metadata.clone();
        first.file_offset = 500;
        first.column_idx = 0;
        let mut second = compressed_data_page(&payload, 10, None);
        second.row_group = metadata.clone();
        second.file_offset = 500;
        second.column_idx = 3;

        let a = run_unary(Decompressor::default(), vec![first]);
        let b = run_unary(Decompressor::default(), vec![second]);

        let ptr = |o: &DecompressedPage| {
            let DecompressedPageType::Data(d) = &o.data else {
                panic!("expected Data");
            };
            d.data[0].as_ptr()
        };
        assert_eq!(
            ptr(&a[0]),
            ptr(&b[0]),
            "same physical column should reuse cached bytes"
        );
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
            file_offset: 0,
            span: compressed.len(),
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
            decompressed: None,
            filter_mask: None,
        };

        let mut sender = CollectSender::new();
        let result = Decompressor::default().consume(
            page,
            &mut sender,
            &mut dispatch::TestOperatorIO::default().io(),
        );

        assert!(result.is_err());
    }
}
