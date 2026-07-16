//! The dictionary encode path: a PLAIN dictionary page of the distinct values
//! plus one data page of RLE/bit-packed indices into it.
//!
//! Mirrors arrow's policy — try a dictionary, fall back to PLAIN once the
//! dictionary's distinct values would reach [`DICTIONARY_PAGE_SIZE_LIMIT`] — so
//! low-cardinality columns dictionary-encode and high-cardinality ones (e.g.
//! timestamps) stay PLAIN.

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use thriftparquet::general::Encoding;

use super::super::error::WriteResult;
use super::super::types::EncodedPage;
use super::leaves::Leaf;
use super::pages::{self, PageKind};
use super::{plain, rle};

/// Fall back from dictionary to PLAIN once the dictionary's distinct values would
/// reach this PLAIN-encoded size — arrow's default `dictionary_page_size_limit`.
const DICTIONARY_PAGE_SIZE_LIMIT: usize = 1024 * 1024;

/// Dictionary-encode a leaf if it pays: build the dictionary, and while its
/// distinct values stay under [`DICTIONARY_PAGE_SIZE_LIMIT`], return the PLAIN
/// dictionary page and the RLE-encoded index page. `None` means fall back to
/// PLAIN — the dictionary grew too large, or the value type doesn't
/// dictionary-cast.
pub(super) fn try_encode(leaf: &Leaf) -> WriteResult<Option<(EncodedPage, EncodedPage)>> {
    let values = &leaf.values;
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
        index_page(leaf, &indices, index_bit_width(distinct.len()))?,
    )))
}

/// Bits to encode an index into a dictionary of `len` entries — the width of its
/// largest index.
fn index_bit_width(len: usize) -> u8 {
    rle::bit_width(len.saturating_sub(1))
}

/// Encode the dictionary page: the distinct values, already PLAIN-encoded into
/// `raw`. A dictionary page has no rows of its own.
fn dictionary_page(raw: Vec<u8>, num_values: usize) -> WriteResult<EncodedPage> {
    pages::assemble_page(0, raw, PageKind::Dictionary { num_values })
}

/// Encode the index data page: the leaf's definition levels, then a one-byte
/// index bit-width followed by the RLE/bit-packed indices. One page covers the
/// whole leaf, so it carries every row's level; the indices cover only the rows
/// that store a value.
fn index_page(leaf: &Leaf, indices: &[u32], bit_width: u8) -> WriteResult<EncodedPage> {
    let mut encoded = Vec::with_capacity(1 + indices.len());
    encoded.push(bit_width);
    encoded.extend_from_slice(&rle::encode_indices(indices, bit_width));

    let num_rows = leaf.rows();
    let raw = pages::data_page_body(leaf.def_levels.as_deref(), leaf.max_def_level, encoded);
    pages::assemble_page(
        num_rows as i64,
        raw,
        PageKind::Data {
            num_values: num_rows,
            encoding: Encoding::RLE_DICTIONARY,
        },
    )
}
