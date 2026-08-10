//! The delta encode path: the two encodings that store a column as differences
//! rather than as whole values.
//!
//! `DELTA_BINARY_PACKED` writes a starting value and then, for each block of
//! values, the smallest difference in it followed by every difference minus
//! that smallest one, bit-packed to the narrowest width each group of them
//! needs. `DELTA_LENGTH_BYTE_ARRAY` writes a byte-array column's lengths that
//! way and then the value bytes back to back.
//!
//! Both are the reason this path exists: a column whose values are too many and
//! too varied to dictionary-encode falls back to PLAIN, which spends the full
//! width on every value and four bytes on every byte-array length. A key column
//! is the usual case, and packing its differences roughly halves it.
//!
//! The block layout is [`VALUES_PER_BLOCK`] and [`MINIBLOCKS_PER_BLOCK`]; the
//! count per miniblock has to stay a multiple of 32, which is what keeps each
//! one starting on a byte boundary.

use arrow_array::{
    Array, BinaryViewArray, Date32Array, Decimal64Array, Decimal128Array, Int8Array, Int16Array,
    Int32Array, Int64Array, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use std::borrow::Cow;
use thriftparquet::general::Encoding;

use crate::parquet::{DecimalWriteStorage, decimal_write_storage};

use dispatch::memory::SlabAllocator;

use super::super::error::WriteResult;
use super::super::types::EncodedPage;
use super::leaves::Leaf;
use super::pages::{self, PageKind, PageRange};

/// Values a block holds, and how many miniblocks it is cut into.
///
/// A block header (its smallest difference, plus a width per miniblock) is
/// parsed on read whatever the block holds, so a wide block spreads that cost
/// over more values, and a wide miniblock spreads the per-miniblock setup over
/// more values still. The cost of going wide is that one outlying difference
/// widens every value it shares a miniblock with, which is why this is a
/// balance rather than the largest block the format allows.
const VALUES_PER_BLOCK: usize = 2048;
const MINIBLOCKS_PER_BLOCK: usize = 8;
const VALUES_PER_MINIBLOCK: usize = VALUES_PER_BLOCK / MINIBLOCKS_PER_BLOCK;

/// Delta-encode a leaf, or `None` when its type has no delta form: floats are
/// not whole numbers, and a decimal wide enough to need fixed-length bytes is
/// stored as bytes rather than as an integer.
///
/// Which of the two encodings a leaf takes follows from its type, so the caller
/// gets it back rather than having to ask again.
pub(super) fn try_encode_chunk(
    leaf: &Leaf,
    allocator: &mut SlabAllocator,
) -> WriteResult<Option<(Encoding, Vec<EncodedPage>)>> {
    let Some(encoding) = encoding_for(leaf.values.data_type()) else {
        return Ok(None);
    };
    let pages = pages::page_ranges(leaf)?
        .into_iter()
        .map(|range| encode_data_page(leaf, range, encoding, allocator))
        .collect::<WriteResult<Vec<_>>>()?;
    Ok(Some((encoding, pages)))
}

/// The delta encoding a column of this type takes, or `None` for one that has
/// none.
///
/// An unsigned column is deliberately absent, though it is stored as an integer
/// and read back from a delta page fine: the deltas here are accumulated in
/// `i64`, so an unsigned value past its signed maximum would need more bits than
/// the physical type it is annotated as declares. Such a column takes a
/// dictionary or PLAIN instead, which store its bits at their declared width.
fn encoding_for(data_type: &DataType) -> Option<Encoding> {
    match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Date32
        | DataType::Timestamp(TimeUnit::Microsecond, None) => Some(Encoding::DELTA_BINARY_PACKED),
        DataType::Decimal64(precision, _) | DataType::Decimal128(precision, _) => {
            match decimal_write_storage(*precision) {
                DecimalWriteStorage::Int32 | DecimalWriteStorage::Int64 => {
                    Some(Encoding::DELTA_BINARY_PACKED)
                }
                DecimalWriteStorage::FixedLen => None,
            }
        }
        DataType::Utf8 | DataType::Utf8View | DataType::BinaryView => {
            Some(Encoding::DELTA_LENGTH_BYTE_ARRAY)
        }
        _ => None,
    }
}

/// Encode one delta data page: the page's rows' definition levels, then its
/// values in whichever delta form the leaf's type takes.
fn encode_data_page(
    leaf: &Leaf,
    range: PageRange,
    encoding: Encoding,
    allocator: &mut SlabAllocator,
) -> WriteResult<EncodedPage> {
    let num_rows = range.rows.len();
    let values = leaf.values.slice(range.values.start, range.values.len());
    let mut encoded = Vec::new();
    match encoding {
        Encoding::DELTA_BINARY_PACKED => {
            encode_binary_packed(&integers(values.as_ref())?, &mut encoded)
        }
        _ => encode_length_byte_array(values.as_ref(), &mut encoded)?,
    }

    let levels = leaf.def_levels.as_ref().map(|levels| &levels[range.rows]);
    let raw = pages::data_page_body(levels, leaf.max_def_level, encoded);
    pages::assemble_page(
        num_rows as i64,
        raw,
        PageKind::Data {
            num_values: num_rows,
            encoding,
        },
        allocator,
    )
}

/// An integer leaf's stored values, widened to `i64` so one encoder covers
/// every integer width. A decimal contributes its raw unscaled integer, which
/// is what its storage holds.
///
/// A column already stored as `i64` is handed over as it lies, which is the
/// common case for a key or a money column: only the narrower and wider
/// storages have to be walked into a buffer.
fn integers(array: &dyn Array) -> WriteResult<Cow<'_, [i64]>> {
    let len = array.len();
    // A storage of another width is walked into an owned buffer at `i64`.
    macro_rules! widened {
        ($arr:ty) => {{
            let a = downcast::<$arr>(array)?;
            Cow::Owned((0..len).map(|i| a.value(i) as i64).collect())
        }};
    }
    Ok(match array.data_type() {
        DataType::Int64 => Cow::Borrowed(downcast::<Int64Array>(array)?.values()),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Cow::Borrowed(downcast::<TimestampMicrosecondArray>(array)?.values())
        }
        DataType::Decimal64(_, _) => Cow::Borrowed(downcast::<Decimal64Array>(array)?.values()),
        DataType::Int8 => widened!(Int8Array),
        DataType::Int16 => widened!(Int16Array),
        DataType::Int32 => widened!(Int32Array),
        DataType::Date32 => widened!(Date32Array),
        DataType::Decimal128(_, _) => widened!(Decimal128Array),
        other => {
            return Err(super::super::error::WriteError::UnsupportedType(
                other.clone(),
            ));
        }
    })
}

/// Write `values` as `DELTA_BINARY_PACKED`: the header, then a block per
/// [`VALUES_PER_BLOCK`] values holding the block's smallest difference, one
/// width per miniblock, and the packed differences.
///
/// A block's differences are built into a fixed buffer rather than a vector as
/// long as the column: a block is the same width whatever the page holds, so
/// the same buffer serves the whole page and the work stays in cache.
pub(crate) fn encode_binary_packed(values: &[i64], out: &mut Vec<u8>) {
    put_uvarint(out, VALUES_PER_BLOCK as u64);
    put_uvarint(out, MINIBLOCKS_PER_BLOCK as u64);
    put_uvarint(out, values.len() as u64);
    put_zigzag(out, values.first().copied().unwrap_or(0));
    if values.len() < 2 {
        return;
    }

    let mut deltas = [0i64; VALUES_PER_BLOCK];
    let mut previous = values[0];
    let mut rest = &values[1..];
    while !rest.is_empty() {
        let count = rest.len().min(VALUES_PER_BLOCK);

        // The differences and the smallest of them come out of one pass. They
        // are taken modulo 2^64, as the format defines them, so a column
        // spanning the whole integer range still encodes.
        let mut min_delta = i64::MAX;
        for (delta, value) in deltas[..count].iter_mut().zip(&rest[..count]) {
            *delta = value.wrapping_sub(previous);
            previous = *value;
            min_delta = min_delta.min(*delta);
        }
        put_zigzag(out, min_delta);

        // Every miniblock's width goes in the header, including the ones past
        // the end of a part-full block: they hold nothing, and a width of zero
        // is what says so. Carrying each miniblock's largest difference through
        // the pass above, to save this one, measured slower: the index division
        // it needs per value costs more than this fold, which the compiler
        // vectorises.
        let mut widths = [0u8; MINIBLOCKS_PER_BLOCK];
        for (index, width) in widths.iter_mut().enumerate() {
            *width = miniblock(&deltas[..count], index)
                .iter()
                .fold(0u8, |widest, delta| {
                    let bits = 64 - delta.wrapping_sub(min_delta).leading_zeros() as u8;
                    widest.max(bits)
                });
        }
        out.extend_from_slice(&widths);

        for (index, width) in widths.iter().enumerate() {
            pack_miniblock(miniblock(&deltas[..count], index), min_delta, *width, out);
        }
        rest = &rest[count..];
    }
}

/// The `index`th miniblock of a block, empty once the block runs out.
fn miniblock(block: &[i64], index: usize) -> &[i64] {
    let start = (index * VALUES_PER_MINIBLOCK).min(block.len());
    let end = (start + VALUES_PER_MINIBLOCK).min(block.len());
    &block[start..end]
}

/// Append one miniblock's differences at `width` bits each, padded to the full
/// count so the miniblock occupies the whole span a reader steps over.
fn pack_miniblock(deltas: &[i64], min_delta: i64, width: u8, out: &mut Vec<u8>) {
    // Specialising on the width makes every shift below a constant and lets the
    // widths that fit a 64-bit accumulator avoid a 128-bit one. The dispatch
    // happens once per miniblock, not once per value.
    macro_rules! widths {
        ($($w:literal),* $(,)?) => {
            match width {
                0 => {}
                $($w => pack_width::<$w>(deltas, min_delta, out),)*
                _ => unreachable!("a delta bit width is at most 64"),
            }
        };
    }
    widths!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    )
}

/// The packing loop for one width.
///
/// The differences shift through an accumulator that is emptied eight bytes at
/// a time, rather than one: a byte-at-a-time drain spends a shift and a
/// single-byte store on every byte of the output, which a profile showed as
/// about half the encoder's time.
#[inline]
fn pack_width<const W: usize>(deltas: &[i64], min_delta: i64, out: &mut Vec<u8>) {
    // Zeroed, so a part-full miniblock pads itself. The tail beyond the packed
    // bytes is room for the last store, which writes a whole accumulator.
    let mut packed = [0u8; VALUES_PER_MINIBLOCK * 8 + 8];
    let mut at = 0usize;
    let mut accumulator = 0u64;
    let mut bits = 0usize;
    for delta in deltas {
        let value = delta.wrapping_sub(min_delta) as u64;
        accumulator |= value << bits;
        if bits + W >= 64 {
            packed[at..at + 8].copy_from_slice(&accumulator.to_le_bytes());
            at += 8;
            // What did not fit starts the next accumulator. A value exactly
            // filling it leaves nothing behind, and shifting by its own width
            // would be undefined, so that case is taken separately.
            let carried = 64 - bits;
            accumulator = if carried >= 64 { 0 } else { value >> carried };
            bits = W - carried;
        } else {
            bits += W;
        }
    }
    packed[at..at + 8].copy_from_slice(&accumulator.to_le_bytes());
    out.extend_from_slice(&packed[..VALUES_PER_MINIBLOCK * W / 8]);
}

/// Write a byte-array leaf as `DELTA_LENGTH_BYTE_ARRAY`: the lengths delta
/// packed, then the values back to back with nothing between them.
pub(crate) fn encode_length_byte_array(array: &dyn Array, out: &mut Vec<u8>) -> WriteResult<()> {
    let len = array.len();
    let mut lengths = Vec::with_capacity(len);
    // The lengths are written before the bytes, so the values are walked twice:
    // once to measure them, once to lay them down. Gathering the bytes on the
    // first pass instead would copy every one of them an extra time.
    macro_rules! encode {
        ($arr:ty, |$value:ident| $body:expr) => {{
            let a = downcast::<$arr>(array)?;
            for i in 0..len {
                let $value = a.value(i);
                let value: &[u8] = $body;
                lengths.push(value.len() as i64);
            }
            encode_binary_packed(&lengths, out);
            out.reserve(lengths.iter().sum::<i64>() as usize);
            for i in 0..len {
                let $value = a.value(i);
                let value: &[u8] = $body;
                out.extend_from_slice(value);
            }
        }};
    }
    match array.data_type() {
        DataType::Utf8 => encode!(StringArray, |value| value.as_bytes()),
        DataType::Utf8View => encode!(StringViewArray, |value| value.as_bytes()),
        DataType::BinaryView => encode!(BinaryViewArray, |value| value),
        other => {
            return Err(super::super::error::WriteError::UnsupportedType(
                other.clone(),
            ));
        }
    }
    Ok(())
}

fn put_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        out.push(if value == 0 { byte } else { byte | 0x80 });
        if value == 0 {
            return;
        }
    }
}

/// Write a signed value zigzagged, so a small negative one stays small.
fn put_zigzag(out: &mut Vec<u8>, value: i64) {
    put_uvarint(out, ((value << 1) ^ (value >> 63)) as u64);
}

fn downcast<A: 'static>(array: &dyn Array) -> WriteResult<&A> {
    array
        .as_any()
        .downcast_ref::<A>()
        .ok_or(super::super::error::WriteError::Downcast {
            expected: std::any::type_name::<A>(),
        })
}
