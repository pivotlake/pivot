//! Hand-rolled, **page-parallel** Parquet encoder — the write-side mirror of
//! dispatch's hand-rolled reader.
//!
//! Like the reader, this avoids the upstream `parquet` crate: it reuses the
//! shared [`thriftparquet`] metadata/codec layer and the in-house snappy
//! (`snap`). The work is split exactly where the reader parallelizes — at the
//! **page** level, with pages sized to a ~1 MiB target (the usual Parquet data
//! page size):
//!
//! 1. [`build_page_jobs`] treats the flush's batches as one row group and cuts
//!    each column into ~1 MiB [`PageJob`]s. A page is a list of zero-copy
//!    slices into the original batch arrays (a page may span a batch boundary),
//!    so building the jobs copies nothing.
//! 2. [`encode_page`] (run in parallel across the worker pool, one job each)
//!    PLAIN-encodes the page's slices into one buffer, snappy-compresses it, and
//!    prepends the page header — producing self-contained [`EncodedPage`] bytes.
//!    The PLAIN encode is the single copy of the data, and it is parallel.
//! 3. [`assemble_parquet`] stitches the pages into one file: pages of a column
//!    are laid out contiguously as that column's chunk (recording its
//!    offset/size), then the footer is written. Cheap and serial.
//!
//! Parallelism therefore scales with data size (a few-MB flush → a handful of
//! pages; a large flush → many), just like the reader parallelizes over however
//! many pages a file contains.
//!
//! Format targeted (the reader's supported subset): DATA_PAGE v1, PLAIN
//! encoding, SNAPPY compression (the reader always snappy-decompresses), and
//! **required** columns only — no def/rep levels, so a page is just the
//! concatenated PLAIN values. The footer carries only the fields the reader
//! reads, so files are readable by pivot's reader but are not fully
//! spec-compliant (no per-chunk `codec`/`type`/`num_values`), which other
//! engines like DuckDB may reject — a follow-up that only needs extra fields on
//! `thriftparquet`'s `ColumnMetaData`/`SchemaElement`.

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, SchemaRef};
use snap::raw::Encoder;
use thriftparquet::footer::{ColumnChunk, ColumnMetaData, FileMetaData, RowGroup, SchemaElement};
use thriftparquet::general::{Encoding, PageType, Type};
use thriftparquet::headers::{DataPageHeader, PageHeader};
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Target uncompressed size of one data page. Matches Parquet's usual ~1 MiB
/// data page size: large enough to amortize per-page overhead and compress
/// well, small enough to be a useful unit of parallel work.
const TARGET_PAGE_SIZE: usize = 1024 * 1024;

/// One page to encode: the slices of a column (in row order) that make up the
/// page. More than one slice when the page spans a batch boundary.
pub struct PageJob {
    pub column: usize,
    /// Order of this page within its column chunk.
    pub page_index: usize,
    pub num_rows: i64,
    pub pieces: Vec<ArrayRef>,
}

/// An encoded data page (header + snappy-compressed values), tagged with its
/// position so [`assemble_parquet`] can place it after the parallel encode.
pub struct EncodedPage {
    pub column: usize,
    pub page_index: usize,
    pub num_rows: i64,
    pub bytes: Vec<u8>,
}

/// Cut a flush's batches into ~1 MiB page jobs (one row group). Each column's
/// values are walked in row order and split into pages; a page references the
/// original arrays by zero-copy slice, so nothing is copied here.
pub fn build_page_jobs(batches: &[RecordBatch]) -> Vec<PageJob> {
    let num_columns = batches[0].num_columns();
    let mut jobs = Vec::new();
    for column in 0..num_columns {
        let arrays: Vec<ArrayRef> = batches.iter().map(|b| b.column(column).clone()).collect();
        let ranges = page_ranges(&arrays, arrays[0].data_type(), TARGET_PAGE_SIZE);
        for (page_index, (start, len)) in ranges.into_iter().enumerate() {
            jobs.push(PageJob {
                column,
                page_index,
                num_rows: len as i64,
                pieces: slice_range(&arrays, start, len),
            });
        }
    }
    jobs
}

/// Encode one page: PLAIN-encode its slices into one buffer, snappy-compress,
/// prepend the page header. Runs in parallel (one call per [`PageJob`]).
pub fn encode_page(job: PageJob) -> Result<EncodedPage, String> {
    let mut raw = Vec::new();
    for piece in &job.pieces {
        encode_plain_into(piece.as_ref(), &mut raw)?;
    }
    let compressed = Encoder::new()
        .compress_vec(&raw)
        .map_err(|e| format!("snappy compress: {e}"))?;

    let header = PageHeader {
        r#type: PageType::DATA_PAGE,
        uncompressed_page_size: raw.len() as i32,
        compressed_page_size: compressed.len() as i32,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: job.num_rows as i32,
            encoding: Encoding::PLAIN,
            // Unused for required columns (the reader skips levels when
            // max_def_level == 0), but the fields are required.
            definition_level_encoding: Encoding::RLE,
            repetition_level_encoding: Encoding::RLE,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };

    let mut bytes = Vec::with_capacity(compressed.len() + 32);
    write_thrift(&header, &mut bytes)?;
    bytes.extend_from_slice(&compressed);

    Ok(EncodedPage {
        column: job.column,
        page_index: job.page_index,
        num_rows: job.num_rows,
        bytes,
    })
}

/// Stitch encoded pages into one Parquet file (a single row group of
/// `total_rows` rows). Pages may arrive in any order (work-stealing); each
/// column's pages are reordered by `page_index` and laid out contiguously as
/// that column's chunk.
pub fn assemble_parquet(
    schema: &SchemaRef,
    total_rows: i64,
    pages: Vec<EncodedPage>,
) -> Result<Vec<u8>, String> {
    let num_columns = schema.fields().len();

    let mut by_column: Vec<Vec<EncodedPage>> = (0..num_columns).map(|_| Vec::new()).collect();
    for page in pages {
        by_column[page.column].push(page);
    }
    for column in &mut by_column {
        column.sort_by_key(|p| p.page_index);
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(PARQUET_MAGIC);

    let mut columns = Vec::with_capacity(num_columns);
    for (col, pages) in by_column.into_iter().enumerate() {
        if pages.is_empty() {
            return Err(format!("no pages produced for column {col}"));
        }
        let data_page_offset = out.len() as i64;
        for page in pages {
            out.extend_from_slice(&page.bytes);
        }
        columns.push(ColumnChunk {
            meta_data: Some(ColumnMetaData {
                total_compressed_size: out.len() as i64 - data_page_offset,
                data_page_offset,
                dictionary_page_offset: None,
                statistics: None,
            }),
        });
    }
    let row_groups = vec![RowGroup {
        columns,
        num_rows: total_rows,
    }];

    // Schema: a root group element followed by one leaf per column.
    let mut schema_elements = Vec::with_capacity(num_columns + 1);
    schema_elements.push(SchemaElement {
        physical_type: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(num_columns as i32),
        converted_type: None,
        logical_type: None,
    });
    for field in schema.fields() {
        schema_elements.push(SchemaElement {
            physical_type: Some(physical_type(field.data_type())?),
            repetition_type: Some(REPETITION_REQUIRED),
            name: field.name().clone(),
            num_children: None,
            converted_type: None,
            logical_type: None,
        });
    }

    let file_meta = FileMetaData {
        schema: schema_elements,
        row_groups,
    };

    // Footer: [FileMetaData][u32 LE footer length][PAR1].
    let mut footer = Vec::new();
    write_thrift(&file_meta, &mut footer)?;
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(PARQUET_MAGIC);

    Ok(out)
}

// --- page splitting -------------------------------------------------------

/// Split a column (given as its per-batch arrays) into contiguous `(start,
/// len)` row ranges over the logical concatenation, each ~`target` uncompressed
/// bytes when PLAIN-encoded.
fn page_ranges(arrays: &[ArrayRef], data_type: &DataType, target: usize) -> Vec<(usize, usize)> {
    let total: usize = arrays.iter().map(|a| a.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    match plain_fixed_width(data_type) {
        Some(width) => fixed_ranges(total, (target / width).max(1)),
        None => variable_ranges(arrays, target),
    }
}

fn fixed_ranges(len: usize, rows_per_page: usize) -> Vec<(usize, usize)> {
    (0..len)
        .step_by(rows_per_page)
        .map(|start| (start, rows_per_page.min(len - start)))
        .collect()
}

/// Walk a BYTE_ARRAY column's values across its arrays, cutting a page once the
/// accumulated PLAIN size (4-byte length prefix + bytes per value) reaches
/// `target`. Always at least one row per page.
fn variable_ranges(arrays: &[ArrayRef], target: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut page_start = 0;
    let mut row = 0;
    let mut acc = 0usize;
    for array in arrays {
        for_each_value_len(array.as_ref(), |value_len| {
            acc += 4 + value_len;
            row += 1;
            if acc >= target {
                ranges.push((page_start, row - page_start));
                page_start = row;
                acc = 0;
            }
        });
    }
    if page_start < row {
        ranges.push((page_start, row - page_start));
    }
    ranges
}

/// Map a `(start, len)` row range over the logical concatenation of `arrays`
/// to the zero-copy slices that cover it.
fn slice_range(arrays: &[ArrayRef], start: usize, len: usize) -> Vec<ArrayRef> {
    let end = start + len;
    let mut pieces = Vec::new();
    let mut cursor = 0; // global index of the current array's first row
    for array in arrays {
        let (array_start, array_end) = (cursor, cursor + array.len());
        cursor = array_end;
        let lo = start.max(array_start);
        let hi = end.min(array_end);
        if lo < hi {
            pieces.push(array.slice(lo - array_start, hi - lo));
        }
    }
    pieces
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
        // Unsupported here; `encode_plain_into` rejects it during encoding.
        _ => (0..array.len()).for_each(|_| f(0)),
    }
}

// --- value encoding -------------------------------------------------------

/// Map an Arrow type to its Parquet physical type id (matching the reader's
/// `convert_physical_to_arrow`). Errors on types the encoder does not handle.
fn physical_type(data_type: &DataType) -> Result<i32, String> {
    Ok(match data_type {
        DataType::Int32 => Type::INT32 as i32,
        DataType::Int64 => Type::INT64 as i32,
        DataType::Float32 => Type::FLOAT as i32,
        DataType::Float64 => Type::DOUBLE as i32,
        DataType::Utf8 | DataType::Utf8View => Type::BYTE_ARRAY as i32,
        other => return Err(format!("unsupported column type for parquet write: {other:?}")),
    })
}

/// Append a required column slice's PLAIN-encoded values to `out`, little-endian
/// and with no levels. BYTE_ARRAY values are a 4-byte LE length prefix followed
/// by the bytes.
fn encode_plain_into(array: &dyn Array, out: &mut Vec<u8>) -> Result<(), String> {
    if array.null_count() > 0 {
        return Err(format!(
            "parquet write: column has {} nulls but only required columns are supported",
            array.null_count()
        ));
    }
    let len = array.len();
    match array.data_type() {
        DataType::Int32 => {
            let a = downcast::<Int32Array>(array)?;
            out.reserve(len * 4);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Int64 => {
            let a = downcast::<Int64Array>(array)?;
            out.reserve(len * 8);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Float32 => {
            let a = downcast::<Float32Array>(array)?;
            out.reserve(len * 4);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Float64 => {
            let a = downcast::<Float64Array>(array)?;
            out.reserve(len * 8);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Utf8 => {
            let a = downcast::<StringArray>(array)?;
            (0..len).for_each(|i| encode_byte_array(a.value(i).as_bytes(), out));
        }
        DataType::Utf8View => {
            let a = downcast::<StringViewArray>(array)?;
            (0..len).for_each(|i| encode_byte_array(a.value(i).as_bytes(), out));
        }
        other => return Err(format!("unsupported column type for parquet write: {other:?}")),
    }
    Ok(())
}

fn encode_byte_array(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn write_thrift<T: WriteThrift>(value: &T, out: &mut Vec<u8>) -> Result<(), String> {
    let mut prot = ThriftCompactOutputProtocol::new(out);
    value
        .write_thrift(&mut prot)
        .map_err(|e| format!("thrift encode: {e}"))
}

fn downcast<A: 'static>(array: &dyn Array) -> Result<&A, String> {
    array.as_any().downcast_ref::<A>().ok_or_else(|| {
        format!(
            "parquet write: array downcast to {} failed",
            std::any::type_name::<A>()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn arrays<const N: usize>(arrs: [ArrayRef; N]) -> Vec<ArrayRef> {
        arrs.into_iter().collect()
    }

    #[test]
    fn fixed_width_pages_split_by_row_count() {
        let column = arrays([Arc::new(Int64Array::from((0..250).collect::<Vec<i64>>()))]);

        // 8 bytes/value, target 800 bytes => 100 rows/page => 100, 100, 50.
        let ranges = page_ranges(&column, &DataType::Int64, 800);

        assert_eq!(ranges, vec![(0, 100), (100, 100), (200, 50)]);
    }

    #[test]
    fn variable_width_pages_split_by_byte_size() {
        // Each value encodes as 4 + 6 = 10 bytes ("abcdef").
        let column = arrays([Arc::new(StringArray::from(vec!["abcdef"; 10]))]);

        // Target 25 bytes => cut after 3 values (30 >= 25): 3, 3, 3, 1.
        let ranges = page_ranges(&column, &DataType::Utf8, 25);

        assert_eq!(ranges, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
    }

    #[test]
    fn pages_and_slices_span_batch_boundaries() {
        // Three batches of 3 rows; 8 bytes/value, target 16 bytes => 2 rows/page,
        // so page boundaries fall mid-batch.
        let column = arrays([
            Arc::new(Int64Array::from(vec![0, 1, 2])),
            Arc::new(Int64Array::from(vec![3, 4, 5])),
            Arc::new(Int64Array::from(vec![6, 7, 8])),
        ]);

        let ranges = page_ranges(&column, &DataType::Int64, 16);
        let second_page = slice_range(&column, ranges[1].0, ranges[1].1);

        // The second page (rows 2..4) straddles batches 0 and 1: tail of batch 0
        // (row 2) + head of batch 1 (row 3).
        assert_eq!(ranges, vec![(0, 2), (2, 2), (4, 2), (6, 2), (8, 1)]);
        assert_eq!(
            second_page.iter().map(|a| a.len()).collect::<Vec<_>>(),
            vec![1, 1]
        );
    }
}
