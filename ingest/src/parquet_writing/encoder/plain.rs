//! The PLAIN encode path: cut a column chunk into pages and PLAIN-encode each.
//!
//! A page's body is the column's values written back to back — fixed-width values
//! little-endian, BYTE_ARRAY values a 4-byte LE length prefix then the bytes.
//! [`encode_into`] is also reused by [`dictionary`](super::dictionary) to encode
//! a dictionary page's distinct values.

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow_schema::DataType;
use thriftparquet::general::Encoding;

use super::super::error::{WriteError, WriteResult};
use super::super::types::EncodedPage;
use super::pages::{self, PageKind};

/// PLAIN-encode a column chunk: cut it into pages and encode each.
pub(super) fn encode_chunk(values: &ArrayRef) -> WriteResult<Vec<EncodedPage>> {
    pages::page_slices(values)
        .iter()
        .map(encode_data_page)
        .collect()
}

/// Encode one PLAIN data page from a column slice.
fn encode_data_page(values: &ArrayRef) -> WriteResult<EncodedPage> {
    let num_rows = values.len();
    let mut raw = Vec::new();
    encode_into(values.as_ref(), &mut raw)?;
    pages::assemble_page(
        num_rows as i64,
        raw,
        PageKind::Data {
            num_values: num_rows,
            encoding: Encoding::PLAIN,
        },
    )
}

/// Append a required column's PLAIN-encoded values to `out`: fixed-width values
/// little-endian, BYTE_ARRAY values a length-prefixed copy.
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
        ($arr:ty) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| {
                let bytes = a.value(i).as_bytes();
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
        DataType::Utf8 => byte_array!(StringArray),
        DataType::Utf8View => byte_array!(StringViewArray),
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
}
