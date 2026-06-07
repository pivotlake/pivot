//! [`TypedColumnDecoder`] — the generic, type-parameterised implementation of
//! [`ColumnDecoder`].
//!
//! This struct is parameterised over three traits that together describe how to
//! decode a particular Parquet column type:
//! - `D: Dict` — builds and queries the dictionary (if the column uses
//!   dictionary encoding).
//! - `B: ArrayBuilder` — accumulates decoded values into an Arrow array.
//! - `P: DecodePlain` — reads plain-encoded values from raw page bytes.
//!
//! Concrete column decoders (e.g. `PrimitiveColumnDecoder`, `BytesViewDecoder`)
//! are type aliases over `TypedColumnDecoder` with the appropriate type
//! parameters.

use crate::parquet::decoding::column_decoders::levels::decode_def_levels;
use crate::parquet::decoding::column_decoders::rle::RleDecoder;
use crate::parquet::decoding::column_decoders::{
    ArrayBuilder, ColumnDecoder, DecodePlain, Dict, Error, Result,
};
use crate::parquet::types::filter_mask::RunningFilterMask;
use crate::parquet::types::page::{DataPage, DecompressedPage, DecompressedPageType};
use crate::parquet::types::thrift::general::Encoding;
use crate::parquet::types::thrift::headers::DataPageHeader;
use arrow_array::ArrayRef;
use bytes::Bytes;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};
use std::marker::PhantomData;

/// Which encoding strategy to use for a data page's values.
pub enum ValueDecoder<P: DecodePlain> {
    Plain(P),
    Rle(RleDecoder),
}

/// State for the page currently being read.
///
/// Tracks the value decoder, the optional filter mask cursor, and how many
/// rows remain in the page.
pub struct ReadPage<
    D: Dict<Builder = B, Item = B::Element>,
    B: ArrayBuilder,
    P: DecodePlain<Builder = B>,
> {
    /// Plain or RLE decoder for this page's values.
    decoder: ValueDecoder<P>,
    /// Filter mask cursor; `None` when the full page is kept.
    running_filter_mask_opt: Option<RunningFilterMask>,
    /// Rows left to decode in this page.
    pub(crate) remaining: usize,
    phantom_data: PhantomData<(D, B)>,
}

impl<D: Dict<Builder = B, Item = B::Element>, B: ArrayBuilder, P: DecodePlain<Builder = B>>
    ReadPage<D, B, P>
{
    /// Decodes up to `size` rows from this page into `builder`.
    ///
    /// When a [`RunningFilterMask`] is present, false runs are skipped and
    /// only true runs are decoded, so the actual number of values pushed may
    /// be less than `size`.
    pub fn read_into(&mut self, dict: &Option<D>, builder: &mut B, size: usize) {
        let current_len = builder.len();
        let mut read_left = size.min(self.remaining);
        while read_left > 0 {
            let (keep, next_run) = match self.running_filter_mask_opt.as_mut() {
                Some(m) => m.next_run(read_left),
                None => (true, read_left),
            };

            match &mut self.decoder {
                ValueDecoder::Plain(p) => {
                    if keep {
                        p.read(builder, next_run)
                    } else {
                        p.skip(next_run)
                    }
                }
                ValueDecoder::Rle(r) => {
                    if keep {
                        r.read(
                            builder,
                            dict.as_ref().expect("No dict available!"),
                            next_run,
                        )
                    } else {
                        r.skip(next_run)
                    }
                }
            }

            if keep {
                read_left -= next_run;
            }
        }
        self.remaining -= builder.len() - current_len;
    }
}

/// Tracks whether a page slot holds decodable data or was fully filtered out.
#[allow(clippy::large_enum_variant)]
enum PageSlot {
    Skipped,
    Data(DataPage),
}

/// Generic column decoder parameterised by dictionary, builder, and plain
/// decoder types.
///
/// Accumulates [`DecompressedPage`]s and decodes them into Arrow arrays on
/// demand. See the [module docs](self) for how the type parameters fit
/// together.
pub struct TypedColumnDecoder<
    D: Dict<Builder = B, Item = B::Element>,
    B: ArrayBuilder,
    P: DecodePlain<Builder = B>,
> {
    /// Indexed by page number. `None` means the page hasn't arrived yet.
    pages: Vec<Option<PageSlot>>,
    /// Index of the next page to decode.
    page_idx: usize,
    /// Maximum definition level for this column (0 = non-nullable).
    max_def_level: i16,
    /// The page currently being consumed, if any.
    read_page: Option<ReadPage<D, B, P>>,
    /// Dictionary built from a dictionary page, if one has been received.
    dict: Option<D>,
    /// Pushed-down equality constant for dictionary pruning, if any.
    eq_const: Option<B::Element>,
    /// Cached result of scanning the dictionary for [`Self::eq_const`]:
    /// `Some(true)` when the constant is absent, `Some(false)` when present,
    /// `None` before the dictionary is built (or when no constant is set).
    dict_excludes: Option<bool>,
    phantom_data: PhantomData<B>,
}

impl<D: Dict<Builder = B, Item = B::Element>, B: ArrayBuilder, P: DecodePlain<Builder = B>>
    TypedColumnDecoder<D, B, P>
{
    /// Creates a new decoder for a column with the given maximum definition
    /// level. Use `0` for non-nullable columns.
    pub fn new(max_def_level: i16) -> Self {
        Self {
            pages: Vec::new(),
            page_idx: 0,
            max_def_level,
            read_page: None,
            dict: None,
            eq_const: None,
            dict_excludes: None,
            phantom_data: Default::default(),
        }
    }

    /// Installs a pushed-down equality constant. When the dictionary is later
    /// built, it is scanned once for this value; if absent, the enclosing row
    /// group can be pruned (see [`ColumnDecoder::dict_excludes_constant`]).
    pub fn set_eq_constant(&mut self, value: B::Element) {
        self.eq_const = Some(value);
    }

    /// Selects the appropriate [`ValueDecoder`] (plain or RLE-dictionary)
    /// based on the page header's encoding.
    fn create_decoder(
        &self,
        header: DataPageHeader,
        data: Vec<Bytes>,
        mut position: ReaderPosition,
    ) -> Result<ValueDecoder<P>> {
        match header.encoding {
            Encoding::PLAIN => Ok(ValueDecoder::Plain(P::new(data, position))),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                if self.dict.is_none() {
                    return Err(Error::NoPagesReady);
                }
                let bit_width = {
                    let mut reader = MultiBufferReader::new(&data, &mut position);
                    reader.read_u8()
                };
                Ok(ValueDecoder::Rle(RleDecoder::new(
                    data, position, bit_width,
                )))
            }
            r => Err(Error::UnsupportedEncoding(r)),
        }
    }

    /// Advances to the next buffered data page and prepares it for reading.
    ///
    /// Skipped pages are silently stepped over. Returns `Ok(None)` when there
    /// are no more pages to consume.
    fn create_next_read_page(&mut self) -> Result<Option<()>> {
        if self.page_idx >= self.pages.len() {
            return Ok(None);
        }

        let page = loop {
            let page = self.pages[self.page_idx]
                .take()
                .ok_or(Error::NoPagesReady)?;
            if let PageSlot::Data(d) = page {
                break d;
            }
            self.page_idx += 1;
        };

        let mut position = ReaderPosition::default();

        let non_null_count = if self.max_def_level > 0 {
            // Skip past the definition level bytes so `position` points at the values.
            let mut reader = MultiBufferReader::new(&page.data, &mut position);
            let def_level_byte_len = reader.read_u32_le() as usize;
            let def_levels = decode_def_levels(&mut reader, page.rows(), def_level_byte_len);
            let non_null = def_levels.iter().filter(|&&v| v).count();
            if non_null != page.rows() {
                return Err(Error::NullableColumnsNotSupported);
            }
            non_null
        } else {
            page.rows()
        };

        self.read_page = Some(ReadPage {
            decoder: self.create_decoder(page.header, page.data, position)?,
            running_filter_mask_opt: page.filter_mask.map(RunningFilterMask::new),
            remaining: non_null_count,
            phantom_data: Default::default(),
        });

        Ok(Some(()))
    }
}

impl<D: Dict<Builder = B, Item = B::Element>, B: ArrayBuilder, P: DecodePlain<Builder = B>>
    ColumnDecoder for TypedColumnDecoder<D, B, P>
where
    B::Element: PartialEq,
{
    fn dict_excludes_constant(&self) -> Option<bool> {
        self.dict_excludes
    }

    fn available(&self) -> usize {
        let (mut available, mut page_idx) = if let Some(s) = &self.read_page {
            (s.remaining, self.page_idx + 1)
        } else {
            (0, self.page_idx)
        };

        while page_idx < self.pages.len()
            && let Some(p) = &self.pages[page_idx]
        {
            available += match p {
                PageSlot::Data(d) => {
                    if self.dict.is_none()
                        && matches!(
                            d.header.encoding,
                            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                        )
                    {
                        return available;
                    }
                    d.rows()
                }
                PageSlot::Skipped => 0,
            };
            page_idx += 1;
        }
        available
    }

    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        match page.data {
            DecompressedPageType::Dict { header, data } => {
                let dict = D::new(data, header.num_values as usize, allocator);
                if let Some(needle) = self.eq_const.as_ref() {
                    let present = (0..dict.len()).any(|i| dict.entry(i) == *needle);
                    self.dict_excludes = Some(!present);
                }
                self.dict = Some(dict);
            }
            DecompressedPageType::Data(data) => {
                let idx = page.idx;
                if idx >= self.pages.len() {
                    self.pages.resize_with(idx + 1, || None);
                }
                self.pages[idx] = Some(PageSlot::Data(data));
            }
            DecompressedPageType::SkippedData { .. } => {
                let idx = page.idx;
                if idx >= self.pages.len() {
                    self.pages.resize_with(idx + 1, || None);
                }
                self.pages[idx] = Some(PageSlot::Skipped);
            }
        }
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let mut builder = B::with_capacity(allocator, size);
        if let Some(d) = self.dict.as_ref() {
            d.register_onto(&mut builder)
        }
        while builder.len() < size {
            if self.read_page.is_none() {
                match self.create_next_read_page()? {
                    Some(()) => {}
                    None => break,
                }
            }

            let remaining = size - builder.len();
            let read_page = self.read_page.as_mut().unwrap();
            read_page.read_into(&self.dict, &mut builder, remaining);

            if read_page.remaining == 0 {
                self.read_page = None;
                self.page_idx += 1;
            }
        }

        Ok(builder.into_array(None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::decoding::column_decoders::primitive::PrimitiveColumnDecoder;
    use crate::parquet::test_utils::dummy_metadata;
    use crate::parquet::types::filter_mask::FilterMask;
    use crate::parquet::types::page::{DataPage, DecompressedPageType};
    use crate::parquet::types::thrift::general::Encoding;
    use crate::parquet::types::thrift::headers::PageHeader;
    use arrow_array::Int32Array;
    use arrow_array::types::Int32Type;
    use dispatch::memory::{SlabAllocator, init_test_free_pool};

    // -- Helpers --

    fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn extract_i32s(arr: &ArrayRef) -> Vec<i32> {
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn data_page(data: Vec<u8>, num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn filtered_data_page(
        data: Vec<u8>,
        num_values: usize,
        idx: usize,
        mask: FilterMask,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: Some(mask),
            }),
        }
    }

    fn skipped_page(num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::SkippedData {
                header: header.data_page_header.unwrap(),
            },
        }
    }

    fn dict_page(data: Vec<u8>, num_values: usize) -> DecompressedPage {
        let header = PageHeader::for_dict_page(num_values as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(data)],
            },
        }
    }

    type Dec = PrimitiveColumnDecoder<Int32Type>;

    // -- available --

    /// No pages inserted → nothing available.
    #[test]
    fn test_available_empty() {
        let dec = Dec::new(0);

        assert_eq!(dec.available(), 0);
    }

    /// Single page with 3 values → 3 available.
    #[test]
    fn test_available_single_page() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0), &mut alloc);

        assert_eq!(dec.available(), 3);
    }

    /// Two pages sum their rows for availability.
    #[test]
    fn test_available_two_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[3, 4, 5]), 3, 1), &mut alloc);

        assert_eq!(dec.available(), 5);
    }

    /// Skipped pages contribute 0 rows to availability.
    #[test]
    fn test_available_skipped_page_contributes_zero() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(10, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 1), &mut alloc);

        assert_eq!(dec.available(), 2);
    }

    /// RLE-dict page without a dictionary → 0 available.
    #[test]
    fn test_available_dict_page_missing() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x00, 0x00])],
                filter_mask: None,
            }),
        };
        dec.insert_page(page, &mut alloc);

        assert_eq!(dec.available(), 0);
    }

    // -- Dictionary pruning (pushed-down equality constant) --

    /// No constant set → never reports a pruning decision.
    #[test]
    fn test_dict_excludes_none_without_constant() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert_eq!(dec.dict_excludes_constant(), None);
    }

    /// Constant set but dictionary not yet loaded → no decision.
    #[test]
    fn test_dict_excludes_none_before_dict_loaded() {
        let mut dec = Dec::new(0);
        dec.set_eq_constant(20);

        assert_eq!(dec.dict_excludes_constant(), None);
    }

    /// Constant present in the dictionary → `Some(false)` (cannot prune).
    #[test]
    fn test_dict_excludes_false_when_present() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(20);
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert_eq!(dec.dict_excludes_constant(), Some(false));
    }

    /// Constant absent from the dictionary → `Some(true)` (row group prunable).
    #[test]
    fn test_dict_excludes_true_when_absent() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(99);
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert_eq!(dec.dict_excludes_constant(), Some(true));
    }

    // -- Page insertion order --

    /// Pages inserted out of order are read in index order.
    #[test]
    fn test_out_of_order_insertion() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[30, 40]), 2, 1), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 0), &mut alloc);

        let result = dec.read(&mut alloc, 4).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20, 30, 40]);
    }

    // -- Skipped pages --

    /// A skipped page is stepped over; read produces values from the next data page.
    #[test]
    fn test_skipped_page_stepped_over() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 1), &mut alloc);

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20]);
    }

    /// Only skipped pages → no rows available.
    #[test]
    fn test_only_skipped_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0), &mut alloc);

        assert_eq!(dec.available(), 0);
    }

    // -- Incremental reads --

    /// Reading fewer rows than available leaves the rest for the next read.
    #[test]
    fn test_incremental_read_within_page() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(
            data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0),
            &mut alloc,
        );

        let r1 = dec.read(&mut alloc, 2).unwrap();
        let r2 = dec.read(&mut alloc, 3).unwrap();

        assert_eq!(extract_i32s(&r1), vec![10, 20]);
        assert_eq!(extract_i32s(&r2), vec![30, 40, 50]);
    }

    /// Incremental reads spanning multiple pages.
    #[test]
    fn test_incremental_read_across_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[4, 5]), 2, 1), &mut alloc);

        let r1 = dec.read(&mut alloc, 4).unwrap();
        let r2 = dec.read(&mut alloc, 1).unwrap();

        assert_eq!(extract_i32s(&r1), vec![1, 2, 3, 4]);
        assert_eq!(extract_i32s(&r2), vec![5]);
    }

    // -- Filter mask --

    /// Filtered page keeps only the matching rows.
    #[test]
    fn test_filter_mask_keeps_matching_rows() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[1, 3]);
        dec.insert_page(
            filtered_data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, mask),
            &mut alloc,
        );

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![20, 40]);
    }

    /// All-false filter mask → 0 rows available.
    #[test]
    fn test_filter_mask_all_false() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[]);
        dec.insert_page(
            filtered_data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, mask),
            &mut alloc,
        );

        assert_eq!(dec.available(), 0);
    }

    // -- Dict-encoded --

    /// Dict page followed by RLE data page decodes correctly.
    #[test]
    fn test_dict_then_data() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(dict_page(encode_i32s(&[100, 200, 300]), 3), &mut alloc);

        // bit_width=2, 1 group of 8, indices [0,1,2,0,0,1,2,0]
        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page, &mut alloc);

        let result = dec.read(&mut alloc, 8).unwrap();

        assert_eq!(
            extract_i32s(&result),
            vec![100, 200, 300, 100, 100, 200, 300, 100]
        );
    }

    /// available() returns 0 until dict page arrives, then reports the full count.
    #[test]
    fn test_dict_page_unblocks_availability() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page, &mut alloc);
        assert_eq!(dec.available(), 0);

        dec.insert_page(dict_page(encode_i32s(&[100, 200, 300]), 3), &mut alloc);

        assert_eq!(dec.available(), 8);
    }
}
