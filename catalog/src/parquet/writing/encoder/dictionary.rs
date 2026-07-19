//! The dictionary encode path: a PLAIN dictionary page of the distinct values
//! plus one data page of RLE/bit-packed indices into it.
//!
//! Mirrors arrow's policy — try a dictionary, fall back to PLAIN once the
//! dictionary's distinct values would reach [`DICTIONARY_PAGE_SIZE_LIMIT`] — so
//! low-cardinality columns dictionary-encode and high-cardinality ones (e.g.
//! timestamps) stay PLAIN.

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use thriftparquet::general::Encoding;

use super::super::error::WriteResult;
use super::super::types::EncodedPage;
use super::pages::{self, PageKind};
use super::{plain, rle};

/// Fall back from dictionary to PLAIN once the dictionary's distinct values would
/// reach this PLAIN-encoded size — arrow's default `dictionary_page_size_limit`.
const DICTIONARY_PAGE_SIZE_LIMIT: usize = 1024 * 1024;

/// Dictionary-encode `values` if it pays: build the dictionary, and while its
/// distinct values stay under [`DICTIONARY_PAGE_SIZE_LIMIT`], return the PLAIN
/// dictionary page and the RLE-encoded index page. `None` means fall back to
/// PLAIN — the dictionary grew too large, or the value type doesn't
/// dictionary-cast.
pub(super) fn try_encode(values: &ArrayRef) -> WriteResult<Option<(EncodedPage, EncodedPage)>> {
    let dict_type = DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(values.data_type().clone()),
    );
    let Ok(dictionary) = arrow_cast::cast(values, &dict_type) else {
        return Ok(None);
    };
    let dictionary = dictionary.as_dictionary::<Int32Type>();
    let distinct = dictionary.values();

    // The PLAIN-encoded distinct values are the dictionary page body; their size
    // is arrow's fallback signal.
    let mut dict_raw = Vec::new();
    plain::encode_into(distinct.as_ref(), &mut dict_raw)?;
    if dict_raw.len() >= DICTIONARY_PAGE_SIZE_LIMIT {
        return Ok(None);
    }

    let indices: Vec<u32> = dictionary
        .keys()
        .values()
        .iter()
        .map(|&k| k as u32)
        .collect();
    Ok(Some((
        dictionary_page(dict_raw, distinct.len())?,
        index_page(&indices, index_bit_width(distinct.len()))?,
    )))
}

/// Bits to encode an index into a dictionary of `len` entries (at least 1, so a
/// single-entry dictionary still carries a real bit-width byte for the reader).
fn index_bit_width(len: usize) -> u8 {
    if len <= 1 {
        1
    } else {
        (usize::BITS - (len - 1).leading_zeros()) as u8
    }
}

/// Encode the dictionary page: the distinct values, already PLAIN-encoded into
/// `raw`. A dictionary page has no rows of its own.
fn dictionary_page(raw: Vec<u8>, num_values: usize) -> WriteResult<EncodedPage> {
    pages::assemble_page(0, raw, PageKind::Dictionary { num_values })
}

/// Encode the index data page: a one-byte index bit-width followed by the
/// RLE/bit-packed indices.
fn index_page(indices: &[u32], bit_width: u8) -> WriteResult<EncodedPage> {
    let mut raw = Vec::with_capacity(1 + indices.len());
    raw.push(bit_width);
    raw.extend_from_slice(&rle::encode_indices(indices, bit_width));
    pages::assemble_page(
        indices.len() as i64,
        raw,
        PageKind::Data {
            num_values: indices.len(),
            encoding: Encoding::RLE_DICTIONARY,
        },
    )
}
