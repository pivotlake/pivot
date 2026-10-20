//! Column decoder for BOOLEAN leaves.
//!
//! A PLAIN boolean page stores one bit per value, least significant bit first,
//! which is arrow's own bitmap layout. An RLE boolean page stores the same
//! values as a 4-byte little-endian length followed by an RLE/bit-packed
//! hybrid stream of bit width 1, the stream a dictionary page's indices use,
//! so it decodes through the shared [`RleDecoder`] against a two-entry table
//! of `false` and `true`.
//!
//! [`BooleanLeaf`] names these as the column's [`LeafType`], and
//! [`BooleanLeafDecoder`] is the ready-to-use [`TypedLeafDecoder`] over it.

use std::ops::Index;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use bytes::Bytes;
use dispatch::arrays::BooleanBuilder;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};

use crate::reading::decoding::leaf_decoders::rle::RleDecoder;
use crate::reading::decoding::leaf_decoders::{
    ArrayBuilder, DecodeEncoding, DecodePlain, Dict, DictFromBytes, DictFromVecBytes, LeafType,
    TypedLeafDecoder, Unsupported,
};

/// [`DecodePlain`] implementation for booleans: one bit per value, least
/// significant bit first.
///
/// A read or skip may stop partway through a byte, so the byte begun last is
/// kept with how many of its bits were consumed, and the next call carries on
/// from the bit after.
pub struct BooleanPlainDecoder {
    data: Vec<Bytes>,
    position: ReaderPosition,
    /// The byte the next value's bit sits in, once any of its bits were read.
    begun_byte: u8,
    /// How many of `begun_byte`'s bits were consumed: the next value is that
    /// bit of it, and 8 means the next value starts a new byte.
    bits_consumed: u8,
}

impl BooleanPlainDecoder {
    /// Decodes `out.len()` values into `out`.
    fn read_into(&mut self, out: &mut [bool]) {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        let mut written = 0;
        while written < out.len() && self.bits_consumed < 8 {
            out[written] = (self.begun_byte >> self.bits_consumed) & 1 == 1;
            self.bits_consumed += 1;
            written += 1;
        }
        while out.len() - written >= 8 {
            let byte = reader.read_u8();
            for bit in 0..8 {
                out[written + bit] = (byte >> bit) & 1 == 1;
            }
            written += 8;
        }
        if written < out.len() {
            self.begun_byte = reader.read_u8();
            self.bits_consumed = 0;
            while written < out.len() {
                out[written] = (self.begun_byte >> self.bits_consumed) & 1 == 1;
                self.bits_consumed += 1;
                written += 1;
            }
        }
    }
}

impl DecodePlain for BooleanPlainDecoder {
    type Builder = BooleanBuilder;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            data,
            position,
            begun_byte: 0,
            bits_consumed: 8,
        }
    }

    fn read(&mut self, builder: &mut BooleanBuilder, size: usize) {
        self.read_into(builder.spare_mut(size));
    }

    fn skip(&mut self, size: usize) {
        let in_begun_byte = (8 - self.bits_consumed as usize).min(size);
        self.bits_consumed += in_begun_byte as u8;
        let mut left = size - in_begun_byte;
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        reader.skip(left / 8);
        left %= 8;
        if left > 0 {
            self.begun_byte = reader.read_u8();
            self.bits_consumed = left as u8;
        }
    }
}

/// The table an RLE boolean page's bits index: bit 0 is `false`, bit 1 `true`.
const BIT_VALUES: BooleanDict<[bool; 2]> = BooleanDict {
    entries: [false, true],
    len: 2,
};

/// [`DecodeEncoding`] implementation for an RLE boolean page: the values as
/// an RLE/bit-packed stream of width 1, read through the dictionary decoder
/// with [`BIT_VALUES`] standing in for the dictionary.
pub struct BooleanRleDecoder {
    runs: RleDecoder,
}

impl DecodeEncoding for BooleanRleDecoder {
    type Builder = BooleanBuilder;

    fn new(data: Vec<Bytes>, mut position: ReaderPosition) -> Option<Self> {
        // The stream's byte length leads it. The page's value count already
        // bounds the reads, so the length is stepped over.
        MultiBufferReader::new(&data, &mut position).read_u32_le();
        Some(Self {
            runs: RleDecoder::new(data, position, 1),
        })
    }

    fn read(&mut self, builder: &mut BooleanBuilder, size: usize) {
        self.runs.read(builder, &BIT_VALUES, size);
    }

    fn skip(&mut self, size: usize) {
        self.runs.skip(size);
    }
}

/// [`Dict`] implementation for booleans, generic over what holds the entries:
/// the constant [`BIT_VALUES`] table for an RLE page, or a vector for a
/// dictionary page.
///
/// No writer dictionary-encodes a boolean column (the format reserves the
/// encoding for the other types), but a dictionary page of PLAIN booleans is
/// a well-defined thing to read, and reading it keeps this decoder on the
/// same page machinery as every other leaf. Such a dictionary holds two
/// entries at most, so it lives on the heap rather than on a slab.
pub struct BooleanDict<B> {
    entries: B,
    len: usize,
}

impl<B: Index<usize, Output = bool> + Send + Sync + 'static> Dict for BooleanDict<B> {
    type Builder = BooleanBuilder;
    type Item = bool;
    type EqConstant = bool;

    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<bool> {
        let (array, _) = scalar.get();
        let booleans = array.as_boolean_opt()?;
        (booleans.len() == 1 && !booleans.is_null(0)).then(|| booleans.value(0))
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> bool {
        self.entries[idx]
    }
}

impl BooleanDict<Vec<bool>> {
    fn from_plain_page(data: Vec<Bytes>, size: usize) -> Self {
        let mut entries = vec![false; size];
        BooleanPlainDecoder::new(data, ReaderPosition::default()).read_into(&mut entries);
        Self { entries, len: size }
    }
}

impl DictFromBytes for BooleanDict<Vec<bool>> {
    fn new_from_bytes(data: Bytes, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self::from_plain_page(vec![data], size)
    }
}

impl DictFromVecBytes for BooleanDict<Vec<bool>> {
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self::from_plain_page(data, size)
    }
}

/// The [`LeafType`] of a boolean column: its pages are PLAIN bits, runs, or
/// dictionary indices.
pub struct BooleanLeaf;

impl LeafType for BooleanLeaf {
    type Builder = BooleanBuilder;

    type Plain = BooleanPlainDecoder;
    type DeltaBinaryPacked = Unsupported<BooleanBuilder>;
    type DeltaLengthByteArray = Unsupported<BooleanBuilder>;
    type Rle = BooleanRleDecoder;

    type ContiguousDict = BooleanDict<Vec<bool>>;
    type ScatteredDict = BooleanDict<Vec<bool>>;
}

/// Ready-to-use column decoder for BOOLEAN leaves.
pub type BooleanLeafDecoder = TypedLeafDecoder<BooleanLeaf>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::LeafDecoder;
    use crate::test_utils::dummy_metadata;
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use crate::types::filter_mask::FilterMask;
    use crate::types::metadata::RowSelection;
    use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
    use arrow_array::BooleanArray;
    use dispatch::memory::init_test_free_pool;

    fn data_page(
        data: Vec<u8>,
        num_values: usize,
        encoding: Encoding,
        filter_mask: Option<FilterMask>,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask,
            }),
        }
    }

    /// PLAIN booleans: one bit per value, least significant bit first.
    fn pack_bits(values: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; values.len().div_ceil(8)];
        for (i, &value) in values.iter().enumerate() {
            bytes[i / 8] |= (value as u8) << (i % 8);
        }
        bytes
    }

    fn pattern(count: usize) -> Vec<bool> {
        (0..count).map(|i| i % 3 == 0 || i % 7 == 0).collect()
    }

    fn values_of(array: &ArrayRef) -> Vec<bool> {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap())
            .collect()
    }

    #[test]
    fn plain_page_reads_in_pieces_that_split_bytes() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(45);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(pack_bits(&values), 45, Encoding::PLAIN, None));

        let first = values_of(&decoder.read(&mut allocator, 5).unwrap());
        let second = values_of(&decoder.read(&mut allocator, 21).unwrap());
        let third = values_of(&decoder.read(&mut allocator, 19).unwrap());

        assert_eq!([first, second, third].concat(), values);
    }

    #[test]
    fn plain_page_skips_filtered_rows_mid_byte() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(40);
        let kept = [3u32, 4, 5, 13, 30, 31, 39];
        let mask = FilterMask::new(0, 40, &kept);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(
            pack_bits(&values),
            40,
            Encoding::PLAIN,
            Some(mask),
        ));

        let read = values_of(&decoder.read(&mut allocator, kept.len()).unwrap());

        let expected: Vec<bool> = kept.iter().map(|&row| values[row as usize]).collect();
        assert_eq!(read, expected);
    }

    #[test]
    fn rle_page_reads_runs_and_packed_groups() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // A bit-packed group of eight values, then a run of five `true`s, behind
        // the stream's byte length.
        let packed_group = pattern(8);
        let mut stream = vec![(1 << 1) | 1];
        stream.extend(pack_bits(&packed_group));
        stream.extend([5 << 1, 1]);
        let mut page = (stream.len() as u32).to_le_bytes().to_vec();
        page.extend(stream);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(page, 13, Encoding::RLE, None));

        let read = values_of(&decoder.read(&mut allocator, 13).unwrap());

        assert_eq!(read, [packed_group, vec![true; 5]].concat());
    }
}
