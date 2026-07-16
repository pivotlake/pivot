//! The PLAIN encode path: cut a leaf into pages and PLAIN-encode each.
//!
//! A page's body is its definition levels (for a leaf that has any) followed by
//! the stored values back to back — fixed-width values little-endian, BYTE_ARRAY
//! values a 4-byte LE length prefix then the bytes. [`encode_into`] is also
//! reused by [`dictionary`](super::dictionary) to encode a dictionary page's
//! distinct values.

use arrow_array::{
    Array, BinaryViewArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow_schema::DataType;
use thriftparquet::general::Encoding;

use super::super::error::{WriteError, WriteResult};
use super::super::types::EncodedPage;
use super::leaves::Leaf;
use super::pages::{self, PageKind, PageRange};

/// PLAIN-encode a leaf: cut it into pages and encode each.
pub(super) fn encode_chunk(leaf: &Leaf) -> WriteResult<Vec<EncodedPage>> {
    pages::page_ranges(leaf)?
        .into_iter()
        .map(|range| encode_data_page(leaf, range))
        .collect()
}

/// Encode one PLAIN data page: the page's rows' definition levels, then its
/// stored values. A page counts its rows, not its values — the absent rows have
/// a level but nothing in the value stream.
fn encode_data_page(leaf: &Leaf, range: PageRange) -> WriteResult<EncodedPage> {
    let num_rows = range.rows.len();
    let values = leaf.values.slice(range.values.start, range.values.len());
    let mut encoded = Vec::new();
    encode_into(values.as_ref(), &mut encoded)?;

    let levels = leaf.def_levels.as_ref().map(|levels| &levels[range.rows]);
    let raw = pages::data_page_body(levels, leaf.max_def_level, encoded);
    pages::assemble_page(
        num_rows as i64,
        raw,
        PageKind::Data {
            num_values: num_rows,
            encoding: Encoding::PLAIN,
        },
    )
}

/// Append `array`'s PLAIN-encoded values to `out`: fixed-width values
/// little-endian, BYTE_ARRAY values a length-prefixed copy.
///
/// Parquet stores only the values that are present, so a leaf's absent rows are
/// already dropped by the time they reach here (see
/// [`leaves`](super::leaves)) — a null left in the array would mean a field
/// nullable in the data but required in the schema, which would silently write
/// the wrong values, so it errors instead.
pub(super) fn encode_into(array: &dyn Array, out: &mut Vec<u8>) -> WriteResult<()> {
    if array.null_count() > 0 {
        return Err(WriteError::NullsInRequiredColumn {
            nulls: array.null_count(),
        });
    }
    let len = array.len();
    // Fixed-width values: their native little-endian bytes, back to back.
    macro_rules! fixed {
        ($arr:ty) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }};
    }
    // BYTE_ARRAY values: a 4-byte LE length prefix then the bytes.
    macro_rules! byte_array {
        ($arr:ty, |$value:ident| $bytes:expr) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| {
                let $value = a.value(i);
                let bytes: &[u8] = $bytes;
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(bytes);
            });
        }};
    }
    match array.data_type() {
        DataType::Int32 => fixed!(Int32Array),
        DataType::Int64 => fixed!(Int64Array),
        DataType::Float32 => fixed!(Float32Array),
        DataType::Float64 => fixed!(Float64Array),
        DataType::Utf8 => byte_array!(StringArray, |value| value.as_bytes()),
        DataType::Utf8View => byte_array!(StringViewArray, |value| value.as_bytes()),
        DataType::BinaryView => byte_array!(BinaryViewArray, |value| value),
        other => return Err(WriteError::UnsupportedType(other.clone())),
    }
    Ok(())
}

fn downcast<A: 'static>(array: &dyn Array) -> WriteResult<&A> {
    array
        .as_any()
        .downcast_ref::<A>()
        .ok_or(WriteError::Downcast {
            expected: std::any::type_name::<A>(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A string-view value longer than 12 bytes lives in an external buffer, not
    /// inline in the view; the encoder must follow the view to it and write the
    /// full bytes (length prefix + data), not the inline prefix.
    #[test]
    fn encodes_long_string_view_from_its_buffer() {
        let short = "ab";
        let long = "this string is well past twelve bytes";
        let array = StringViewArray::from(vec![short, long]);

        let mut out = Vec::new();
        encode_into(&array, &mut out).unwrap();

        let mut expected = Vec::new();
        for s in [short, long] {
            expected.extend_from_slice(&(s.len() as u32).to_le_bytes());
            expected.extend_from_slice(s.as_bytes());
        }
        assert_eq!(out, expected);
    }

    /// A variant's `metadata`/`value` leaves are binary, and encode as
    /// length-prefixed bytes just like strings do.
    #[test]
    fn encodes_binary_view_values() {
        let array = BinaryViewArray::from(vec![b"\x01\x02".as_slice(), b"\xff".as_slice()]);

        let mut out = Vec::new();
        encode_into(&array, &mut out).unwrap();

        assert_eq!(out, vec![2, 0, 0, 0, 1, 2, 1, 0, 0, 0, 255]);
    }
}
