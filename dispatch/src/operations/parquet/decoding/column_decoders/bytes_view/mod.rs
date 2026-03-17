//! Column decoder for variable-length string and binary types, producing
//! Arrow `StringViewArray`s.
//!
//! Arrow's *view* layout stores each value as a 128-bit view: short strings
//! (≤ 12 bytes) are inlined, while longer ones reference a `(block_id,
//! offset)` into an external buffer list. This module provides the three
//! components needed by [`TypedColumnDecoder`]:
//!
//! - [`ViewsBuilder`] — the [`ArrayBuilder`](super::ArrayBuilder) that
//!   accumulates views and data blocks.
//! - [`PlainPageDecoder`] — the [`DecodePlain`](super::DecodePlain) that reads
//!   length-prefixed byte arrays from plain-encoded pages.
//! - [`ViewDict`] — the [`Dict`](super::Dict) that pre-parses dictionary
//!   entries into views for O(1) lookup.
//!
//! [`BytesViewDecoder`] ties them together as a ready-to-use type alias.

use crate::operations::parquet::decoding::column_decoders::TypedColumnDecoder;
use crate::operations::parquet::decoding::column_decoders::bytes_view::dict::ViewDict;
use crate::operations::parquet::decoding::column_decoders::bytes_view::plain_page_decoder::PlainPageDecoder;
use crate::operations::parquet::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
pub(crate) mod views_builder;

pub(crate) mod dict;
mod plain_page_decoder;

/// Ready-to-use column decoder for variable-length string/binary types.
pub type BytesViewDecoder = TypedColumnDecoder<ViewDict, ViewsBuilder, PlainPageDecoder>;

#[cfg(test)]
mod tests {
    use crate::memory::SlabAllocator;
    use crate::memory::init_test_free_pool;
    use crate::operations::parquet::decoding::column_decoders::ColumnDecoder;
    use crate::operations::parquet::decoding::column_decoders::bytes_view::BytesViewDecoder;
    use crate::operations::parquet::test_utils::dummy_metadata;
    use crate::operations::parquet::types::page::{
        DataPage, DecompressedPage, DecompressedPageType,
    };
    use crate::operations::parquet::types::thrift::general::Encoding;
    use crate::operations::parquet::types::thrift::headers::PageHeader;
    use arrow_array::{Array, ArrayRef, StringViewArray};
    use bytes::Bytes;

    fn encode_plain_strings(strings: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for s in strings {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        data
    }

    fn make_data_page(
        data: Vec<u8>,
        num_values: usize,
        encoding: Encoding,
        idx: usize,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
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

    fn make_dict_page(entries: &[&str]) -> DecompressedPage {
        let header = PageHeader::for_dict_page(entries.len() as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(encode_plain_strings(entries))],
            },
        }
    }

    fn extract_strings(array: &ArrayRef) -> Vec<String> {
        let sv = array.as_any().downcast_ref::<StringViewArray>().unwrap();
        (0..sv.len()).map(|i| sv.value(i).to_string()).collect()
    }

    #[test]
    fn test_plain_short_strings() {
        init_test_free_pool(4);
        let page = make_data_page(
            encode_plain_strings(&["hi", "bye", "ok"]),
            3,
            Encoding::PLAIN,
            0,
        );

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(0);
        dec.insert_page(page, &mut allocator);

        let result = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_strings(&result), vec!["hi", "bye", "ok"]);
    }

    #[test]
    fn test_plain_long_strings() {
        init_test_free_pool(4);
        let long1 = "abcdefghijklmnop"; // 16 bytes, exceeds 12-byte inline limit
        let long2 = "qrstuvwxyz123456";
        let page = make_data_page(encode_plain_strings(&[long1, long2]), 2, Encoding::PLAIN, 0);

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(0);
        dec.insert_page(page, &mut allocator);

        let result = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_strings(&result), vec![long1, long2]);
    }

    #[test]
    fn test_dict_encoded() {
        init_test_free_pool(4);
        let dict_page = make_dict_page(&["foo", "bar", "baz"]);

        // bit_width=2, bit-packed 1 group: indices [2, 0, 1, 0]
        // header=(1<<1)|1=3, packed: byte0=0x12, byte1=0x00
        let mut rle_data = vec![2u8]; // bit_width
        rle_data.extend_from_slice(&[3, 0x12, 0x00]); // header + packed bytes
        let data_page = make_data_page(rle_data, 4, Encoding::RLE_DICTIONARY, 0);

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(0);
        dec.insert_page(dict_page, &mut allocator);
        dec.insert_page(data_page, &mut allocator);

        let result = dec.read(&mut allocator, 4).unwrap();
        assert_eq!(extract_strings(&result), vec!["baz", "foo", "bar", "foo"]);
    }

    #[test]
    fn test_multiple_plain_pages() {
        init_test_free_pool(4);
        let page0 = make_data_page(encode_plain_strings(&["a", "b"]), 2, Encoding::PLAIN, 0);
        let page1 = make_data_page(encode_plain_strings(&["c", "d"]), 2, Encoding::PLAIN, 1);

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(0);
        dec.insert_page(page0, &mut allocator);
        dec.insert_page(page1, &mut allocator);
        let result = dec.read(&mut allocator, 4).unwrap();

        assert_eq!(extract_strings(&result), vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_partial_read() {
        init_test_free_pool(4);
        let page = make_data_page(
            encode_plain_strings(&["x", "y", "z"]),
            3,
            Encoding::PLAIN,
            0,
        );

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(0);
        dec.insert_page(page, &mut allocator);

        let r1 = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_strings(&r1), vec!["x", "y"]);

        let r2 = dec.read(&mut allocator, 1).unwrap();
        assert_eq!(extract_strings(&r2), vec!["z"]);
    }

    /// PLAIN page with def levels (max_def_level=1, all values non-null).
    /// The page data is: [4-byte def-level length][def levels][plain strings].
    /// PlainPageDecoder must start reading after the def levels, not from byte 0.
    #[test]
    fn test_plain_with_def_levels() {
        init_test_free_pool(4);
        let mut page_data = Vec::new();

        // Def level section: RLE-encoded, 3 values all present (def=1)
        // RLE header = (3 << 1) | 0 = 6, value byte = 1
        let def_levels = vec![6u8, 1];
        page_data.extend_from_slice(&(def_levels.len() as u32).to_le_bytes());
        page_data.extend_from_slice(&def_levels);

        // PLAIN encoded strings after the def levels
        page_data.extend_from_slice(&encode_plain_strings(&["hi", "bye", "ok"]));

        let page = make_data_page(page_data, 3, Encoding::PLAIN, 0);

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(1); // max_def_level = 1
        dec.insert_page(page, &mut allocator);
        let result = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_strings(&result), vec!["hi", "bye", "ok"]);
    }

    /// Helper: build a PLAIN data page with def levels prepended.
    /// `def_bools` indicates present (true) / null (false) for each value.
    /// `strings` are the non-null values in order.
    fn make_plain_page_with_def_levels(def_bools: &[bool], strings: &[&str]) -> DecompressedPage {
        let num_values = def_bools.len();
        let mut page_data = Vec::new();

        // RLE-encode def levels (bit_width=1): one bit-packed group
        // Bit-packed header: (num_groups << 1) | 1
        let num_groups = num_values.div_ceil(8);
        let header = ((num_groups as u8) << 1) | 1;
        let mut def_bytes = vec![header];
        for group in 0..num_groups {
            let mut byte = 0u8;
            for bit in 0..8 {
                let idx = group * 8 + bit;
                if idx < num_values && def_bools[idx] {
                    byte |= 1 << bit;
                }
            }
            def_bytes.push(byte);
        }

        // 4-byte length prefix + def level bytes
        page_data.extend_from_slice(&(def_bytes.len() as u32).to_le_bytes());
        page_data.extend_from_slice(&def_bytes);

        // PLAIN encoded non-null strings
        page_data.extend_from_slice(&encode_plain_strings(strings));

        make_data_page(page_data, num_values, Encoding::PLAIN, 0)
    }

    /// Nullable column, all values present — output should have no nulls.
    #[test]
    fn test_nullable_all_present() {
        init_test_free_pool(4);
        let page = make_plain_page_with_def_levels(&[true, true, true], &["aa", "bb", "cc"]);

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(1);
        dec.insert_page(page, &mut allocator);

        let result = dec.read(&mut allocator, 3).unwrap();
        let sv = result.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.null_count(), 0);
        assert_eq!(sv.value(0), "aa");
        assert_eq!(sv.value(1), "bb");
        assert_eq!(sv.value(2), "cc");
    }

    /// Nullable column with nulls — output must have correct length,
    /// null positions, and non-null values in the right slots.
    #[ignore]
    #[test]
    fn test_nullable_with_nulls() {
        init_test_free_pool(4);
        // 4 total values: present, null, present, null
        let page = make_plain_page_with_def_levels(
            &[true, false, true, false],
            &["hello", "world"], // only 2 non-null values encoded
        );

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(1);
        dec.insert_page(page, &mut allocator);

        let result = dec.read(&mut allocator, 4).unwrap();
        let sv = result.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(sv.len(), 4);
        assert_eq!(sv.null_count(), 2);
        assert!(!sv.is_null(0));
        assert!(sv.is_null(1));
        assert!(!sv.is_null(2));
        assert!(sv.is_null(3));
        assert_eq!(sv.value(0), "hello");
        assert_eq!(sv.value(2), "world");
    }

    /// Nullable column where all values are null.
    #[ignore]
    #[test]
    fn test_nullable_all_null() {
        init_test_free_pool(4);
        let page = make_plain_page_with_def_levels(
            &[false, false, false],
            &[], // no non-null values
        );

        let mut allocator = SlabAllocator::new(true);
        let mut dec = BytesViewDecoder::new(1);
        dec.insert_page(page, &mut allocator);

        let result = dec.read(&mut allocator, 3).unwrap();
        let sv = result.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.null_count(), 3);
        assert!(sv.is_null(0));
        assert!(sv.is_null(1));
        assert!(sv.is_null(2));
    }
}
