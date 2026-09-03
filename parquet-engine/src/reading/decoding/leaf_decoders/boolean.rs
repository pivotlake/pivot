//! Decoder for Parquet `BOOLEAN` leaves.
//!
//! PLAIN booleans are packed least-significant bit first, unlike the byte- and
//! word-aligned primitive types. This module supplies the boolean-specific
//! builder and PLAIN decoder to the shared [`TypedLeafDecoder`].

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Datum, RecordBatch, Scalar};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use bytes::Bytes;
use dispatch::arrays::{ArrayBuilder, SlabColumn};
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};

use super::{DecodeDelta, DecodePlain, Dict, DictFromBytes, DictFromVecBytes, TypedLeafDecoder};
use crate::thrift::general::Encoding;

/// Builds an Arrow bitmap from decoded boolean values. The temporary byte-per-
/// value column lives in slab memory; only the finished bitmap is retained.
pub(crate) struct BooleanBuilder {
    values: SlabColumn<bool>,
}

impl ArrayBuilder for BooleanBuilder {
    type Element = bool;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            values: SlabColumn::with_capacity(allocator, capacity),
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn push(&mut self, element: &bool, amount: usize) {
        self.values.spare_mut(amount).fill(*element);
    }

    fn spare_mut(&mut self, count: usize) -> &mut [bool] {
        self.values.spare_mut(count)
    }

    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef {
        let len = self.values.len();
        let values = BooleanBuffer::collect_bool(len, |i| self.values.as_slice()[i]);
        let nulls = null_buffer
            .map(|buffer| NullBuffer::new(BooleanBuffer::new(buffer, 0, len)))
            .filter(|nulls| nulls.null_count() != 0);
        Arc::new(BooleanArray::new(values, nulls))
    }
}

/// Cursor over a PLAIN boolean stream. Pages begin byte-aligned; reads and
/// skips may stop at any bit and resume from the saved byte on the next call.
pub(crate) struct BooleanPlainDecoder {
    data: Vec<Bytes>,
    position: ReaderPosition,
    current_byte: u8,
    next_bit: u8,
}

impl BooleanPlainDecoder {
    #[inline(always)]
    fn next(&mut self) -> bool {
        if self.next_bit == 8 {
            self.current_byte = MultiBufferReader::new(&self.data, &mut self.position).read_u8();
            self.next_bit = 0;
        }
        let value = self.current_byte & (1 << self.next_bit) != 0;
        self.next_bit += 1;
        value
    }
}

impl DecodePlain for BooleanPlainDecoder {
    type Builder = BooleanBuilder;
    type Delta = BooleanDeltaDecoder;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            data,
            position,
            current_byte: 0,
            next_bit: 8,
        }
    }

    fn read(&mut self, builder: &mut BooleanBuilder, size: usize) {
        for slot in builder.spare_mut(size) {
            *slot = self.next();
        }
    }

    fn skip(&mut self, size: usize) {
        for _ in 0..size {
            self.next();
        }
    }
}

/// BOOLEAN has no delta encoding. The shared decoder requires an associated
/// delta type, whose constructor rejects any such page as unsupported.
pub(crate) struct BooleanDeltaDecoder;

impl DecodeDelta for BooleanDeltaDecoder {
    type Builder = BooleanBuilder;

    const ENCODING: Encoding = Encoding::DELTA_BINARY_PACKED;

    fn new(_data: Vec<Bytes>, _position: ReaderPosition) -> Option<Self> {
        None
    }

    fn read(&mut self, _builder: &mut BooleanBuilder, _size: usize) {
        unreachable!("a BOOLEAN delta decoder cannot be constructed")
    }

    fn skip(&mut self, _size: usize) {
        unreachable!("a BOOLEAN delta decoder cannot be constructed")
    }
}

/// Dictionary encoding is forbidden for BOOLEAN, but the generic decoder's
/// type parameters still require a dictionary implementation. Keeping this
/// small implementation also makes malformed-but-decodable files fail less
/// mysteriously if another writer emits one.
pub(crate) struct BooleanDict {
    entries: Vec<bool>,
}

impl Dict for BooleanDict {
    type Builder = BooleanBuilder;
    type Item = bool;
    type EqConstant = bool;

    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<bool> {
        let (array, _) = Datum::get(scalar);
        let boolean = array.as_any().downcast_ref::<BooleanArray>()?;
        (boolean.len() == 1 && boolean.is_valid(0)).then(|| boolean.value(0))
    }

    fn maybe_contains(data: &[Bytes], size: usize, needle: &bool) -> bool {
        decode_values(data, size).contains(needle)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn entry(&self, idx: usize) -> bool {
        self.entries[idx]
    }

    fn filter_record_batch_by_const(
        &self,
        batch: RecordBatch,
        _column: usize,
        _needle: &bool,
    ) -> RecordBatch {
        batch
    }
}

impl DictFromBytes for BooleanDict {
    fn new_from_bytes(data: Bytes, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self {
            entries: decode_values(&[data], size),
        }
    }
}

impl DictFromVecBytes for BooleanDict {
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self {
            entries: decode_values(&data, size),
        }
    }
}

fn decode_values(data: &[Bytes], size: usize) -> Vec<bool> {
    let mut position = ReaderPosition::default();
    let mut reader = MultiBufferReader::new(data, &mut position);
    let mut values = Vec::with_capacity(size);
    while values.len() < size {
        let byte = reader.read_u8();
        for bit in 0..8 {
            if values.len() == size {
                break;
            }
            values.push(byte & (1 << bit) != 0);
        }
    }
    values
}

pub(crate) type BooleanLeafDecoder =
    TypedLeafDecoder<BooleanDict, BooleanDict, BooleanBuilder, BooleanPlainDecoder>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_lsb_first_across_bytes() {
        let values = decode_values(&[Bytes::from_static(&[0b0100_1101, 0b0000_0001])], 9);

        assert_eq!(
            values,
            vec![true, false, true, true, false, false, true, false, true]
        );
    }
}
