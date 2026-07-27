//! Entry points that let `benches/decode.rs` drive one decoder on its own.
//!
//! Scanning a column end to end spends most of its time on everything around
//! the decoders: fetching pages, snappy-decompressing them, and materialising
//! Arrow arrays. A column decoded `DELTA_BINARY_PACKED` costs about a tenth of
//! that scan, so a change inside the decoder barely moves the end-to-end
//! number even when it halves the decoder's own work. These hooks hand the
//! benchmark the decode step by itself, where such a change is visible.
//!
//! The same holds for the encoders on the write side, which the file writer
//! wraps in a dataflow, snappy and footer assembly.
//!
//! Not part of the crate's interface, and not used outside benchmarks.

use bytes::Bytes;
use dispatch::arrays::ArrayBuilder;
use dispatch::memory::{ReaderPosition, SlabAllocator};

use crate::parquet::reading::decoding::column_decoders::bytes_view::delta_length_page_decoder::DeltaLengthPageDecoder;
use crate::parquet::reading::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
use crate::parquet::reading::decoding::column_decoders::{DecodeDelta, DeltaDecoder};

/// Decode `values` `i64`s from one `DELTA_BINARY_PACKED` page and return their
/// sum, so nothing the decoder produced can be optimised away.
///
/// `page` holds the page body, starting at its header, in the buffers a scan
/// would hand the decoder. It is passed already built, and cloned here for the
/// reference counts alone, so a measurement covers the decoding rather than a
/// copy of the page. The caller supplies the allocator for the same reason.
pub fn decode_delta_binary_packed(
    page: &[Bytes],
    values: usize,
    allocator: &mut SlabAllocator,
) -> i64 {
    let data = page.to_vec();
    let mut decoder =
        DeltaDecoder::<arrow_array::types::Int64Type>::new(data, ReaderPosition::default())
            .expect("an i64 column decodes delta-packed pages");
    let mut builder =
        <dispatch::arrays::PrimitiveBuilder<arrow_array::types::Int64Type> as ArrayBuilder>::with_capacity(
            allocator, values,
        );

    decoder.read(&mut builder, values);

    let array = builder.into_array(None);
    let decoded = array
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("the builder produces an Int64Array");
    decoded
        .values()
        .iter()
        .fold(0i64, |acc, v| acc.wrapping_add(*v))
}

/// Decode `values` strings from one `DELTA_LENGTH_BYTE_ARRAY` page and return
/// the total number of bytes they hold, so nothing decoded can be optimised
/// away.
pub fn decode_delta_length_byte_array(
    page: &[Bytes],
    values: usize,
    allocator: &mut SlabAllocator,
) -> usize {
    let data = page.to_vec();
    let mut decoder = DeltaLengthPageDecoder::<arrow_array::types::StringViewType>::new(
        data,
        ReaderPosition::default(),
    )
    .expect("a string column decodes delta-length pages");
    let mut builder =
        <ViewsBuilder<arrow_array::types::StringViewType> as ArrayBuilder>::with_capacity(
            allocator, values,
        );

    decoder.read(&mut builder, values);

    let array = builder.into_array(None);
    let decoded = array
        .as_any()
        .downcast_ref::<arrow_array::StringViewArray>()
        .expect("the builder produces a StringViewArray");
    // Sum the lengths straight off the views rather than resolving each value:
    // resolving walks the block list per value and would cost more than the
    // decoding this is here to measure.
    decoded
        .views()
        .iter()
        .map(|view| *view as u32 as usize)
        .sum()
}

/// Encode `values` as one `DELTA_BINARY_PACKED` page body and return its size.
pub fn encode_delta_binary_packed(values: &[i64], out: &mut Vec<u8>) -> usize {
    out.clear();
    crate::parquet::writing::encoder::delta::encode_binary_packed(values, out);
    out.len()
}

/// Encode a byte-array column as one `DELTA_LENGTH_BYTE_ARRAY` page body and
/// return its size.
pub fn encode_delta_length_byte_array(
    values: &arrow_array::StringViewArray,
    out: &mut Vec<u8>,
) -> usize {
    out.clear();
    crate::parquet::writing::encoder::delta::encode_length_byte_array(values, out)
        .expect("a string column encodes as delta lengths");
    out.len()
}
