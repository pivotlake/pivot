//! Page mechanics shared by both encode strategies ([`plain`](super::plain) and
//! [`dictionary`](super::dictionary)): cut a column chunk into ~1 MiB pages, and
//! frame an encoded page body into an [`EncodedPage`].
//!
//! A page's values are a zero-copy slice of the column, so cutting copies nothing.

use arrow_array::{Array, ArrayRef, StringArray, StringViewArray};
use arrow_schema::DataType;
use snap::raw::Encoder;
use thriftparquet::general::{Encoding, PageType};
use thriftparquet::headers::{DataPageHeader, DictionaryPageHeader, PageHeader};
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

use super::super::error::WriteResult;
use super::super::types::EncodedPage;

/// Target uncompressed size of one data page. Matches Parquet's usual ~1 MiB
/// data page size: large enough to amortize per-page overhead and compress
/// well, small enough to be a useful unit of parallel work.
const TARGET_PAGE_SIZE: usize = 1024 * 1024;

/// Bytes a PLAIN-encoded BYTE_ARRAY value spends before its data: a little-endian
/// `u32` length prefix (matches the BYTE_ARRAY encoding in [`plain`](super::plain)).
/// Counted per value when estimating a page's encoded size.
const BYTE_ARRAY_LEN_PREFIX: usize = size_of::<u32>();

/// Split one column's values into ~1 MiB pages, each a zero-copy slice in row
/// order.
pub(super) fn page_slices(values: &ArrayRef) -> Vec<ArrayRef> {
    page_ranges(values, TARGET_PAGE_SIZE)
        .into_iter()
        .map(|(start, len)| values.slice(start, len))
        .collect()
}

/// Split a column into contiguous `(start, len)` row ranges, each ~`target`
/// uncompressed bytes when PLAIN-encoded.
fn page_ranges(values: &ArrayRef, target: usize) -> Vec<(usize, usize)> {
    let len = values.len();
    if len == 0 {
        return Vec::new();
    }
    match plain_fixed_width(values.data_type()) {
        Some(width) => fixed_ranges(len, (target / width).max(1)),
        None => variable_ranges(values, target),
    }
}

fn fixed_ranges(len: usize, rows_per_page: usize) -> Vec<(usize, usize)> {
    (0..len)
        .step_by(rows_per_page)
        .map(|start| (start, rows_per_page.min(len - start)))
        .collect()
}

/// Walk a BYTE_ARRAY column's values, cutting a page once the accumulated PLAIN
/// size (length prefix + bytes per value) reaches `target`. Always at least one
/// row per page.
fn variable_ranges(values: &ArrayRef, target: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut page_start = 0;
    let mut row = 0;
    let mut acc = 0usize;
    for_each_value_len(values.as_ref(), |value_len| {
        acc += BYTE_ARRAY_LEN_PREFIX + value_len;
        row += 1;
        if acc >= target {
            ranges.push((page_start, row - page_start));
            page_start = row;
            acc = 0;
        }
    });
    if page_start < row {
        ranges.push((page_start, row - page_start));
    }
    ranges
}

/// PLAIN byte size of one fixed-width value, or `None` for variable-width types.
fn plain_fixed_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Int32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::Float64 => Some(8),
        _ => None,
    }
}

/// Call `f` with each value's byte length (for BYTE_ARRAY size accounting).
fn for_each_value_len(array: &dyn Array, mut f: impl FnMut(usize)) {
    match array.data_type() {
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            (0..a.len()).for_each(|i| f(a.value(i).len()));
        }
        DataType::Utf8View => {
            let a = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            (0..a.len()).for_each(|i| f(a.value(i).len()));
        }
        // Unsupported here; the encoder rejects it during encoding.
        _ => (0..array.len()).for_each(|_| f(0)),
    }
}

/// Which page to build — everything in the page header except the body sizes,
/// which [`assemble_page`] fills in after compressing.
pub(super) enum PageKind {
    Data {
        num_values: usize,
        encoding: Encoding,
    },
    Dictionary {
        num_values: usize,
    },
}

/// Snappy-compress `raw` (an encoded page body), prepend the page header, and
/// tally the sizes into an [`EncodedPage`].
pub(super) fn assemble_page(
    num_rows: i64,
    raw: Vec<u8>,
    kind: PageKind,
) -> WriteResult<EncodedPage> {
    let compressed = Encoder::new().compress_vec(&raw)?;
    let (page_type, data_page_header, dictionary_page_header) = match kind {
        PageKind::Data {
            num_values,
            encoding,
        } => (
            PageType::DATA_PAGE,
            Some(DataPageHeader {
                num_values: num_values as i32,
                encoding,
                // Unused for required columns (the reader skips levels when
                // max_def_level == 0), but the fields are required.
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: None,
            }),
            None,
        ),
        PageKind::Dictionary { num_values } => (
            PageType::DICTIONARY_PAGE,
            None,
            Some(DictionaryPageHeader {
                num_values: num_values as i32,
                encoding: Encoding::PLAIN,
                is_sorted: None,
            }),
        ),
    };
    let header = PageHeader {
        r#type: page_type,
        uncompressed_page_size: raw.len() as i32,
        compressed_page_size: compressed.len() as i32,
        crc: None,
        data_page_header,
        index_page_header: None,
        dictionary_page_header,
        data_page_header_v2: None,
    };

    // The page on the wire is the thrift header followed by the compressed body.
    let mut bytes = Vec::new();
    header.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut bytes))?;
    let header_len = bytes.len();
    bytes.extend_from_slice(&compressed);

    Ok(EncodedPage {
        num_rows,
        uncompressed_size: raw.len(),
        header_len,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use std::sync::Arc;

    #[test]
    fn fixed_width_pages_split_by_row_count() {
        let column: ArrayRef = Arc::new(Int64Array::from((0..250).collect::<Vec<i64>>()));

        // 8 bytes/value, target 800 bytes => 100 rows/page => 100, 100, 50.
        let ranges = page_ranges(&column, 800);

        assert_eq!(ranges, vec![(0, 100), (100, 100), (200, 50)]);
    }

    #[test]
    fn variable_width_pages_split_by_byte_size() {
        // Each value encodes as 4 + 6 = 10 bytes ("abcdef").
        let column: ArrayRef = Arc::new(StringArray::from(vec!["abcdef"; 10]));

        // Target 25 bytes => cut after 3 values (30 >= 25): 3, 3, 3, 1.
        let ranges = page_ranges(&column, 25);

        assert_eq!(ranges, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
    }
}
