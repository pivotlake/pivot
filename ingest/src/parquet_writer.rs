//! Hand-rolled, **page-parallel** Parquet encoder — the write-side mirror of
//! dispatch's hand-rolled reader.
//!
//! Like the reader, this avoids the upstream `parquet` crate: it reuses the
//! shared [`thriftparquet`] metadata/codec layer and the in-house snappy
//! (`snap`). The work is split exactly where the reader parallelizes — at the
//! **page** level, with pages sized to a ~1 MiB target (the usual Parquet data
//! page size):
//!
//! 1. [`build_page_jobs`] concatenates each column across the flush's batches
//!    (one row group) and splits it into ~1 MiB [`PageJob`]s.
//! 2. [`encode_page`] (run in parallel across the worker pool, one job each)
//!    PLAIN-encodes a page's values, snappy-compresses them, and prepends the
//!    page header — producing self-contained [`EncodedPage`] bytes.
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

/// One page to encode: a contiguous slice of one column's values.
pub struct PageJob {
    pub column: usize,
    /// Order of this page within its column chunk.
    pub page_index: usize,
    pub num_rows: i64,
    pub array: ArrayRef,
}

/// An encoded data page (header + snappy-compressed values), tagged with its
/// position so [`assemble_parquet`] can place it after the parallel encode.
pub struct EncodedPage {
    pub column: usize,
    pub page_index: usize,
    pub num_rows: i64,
    pub bytes: Vec<u8>,
}

/// Turn a flush's batches into ~1 MiB page jobs. Each column is concatenated
/// across all batches (forming one row group) and split into pages; the
/// concatenation shares buffers by `Arc`, so the batches can be dropped after.
pub fn build_page_jobs(batches: &[RecordBatch]) -> Result<Vec<PageJob>, String> {
    let num_columns = batches[0].num_columns();
    let mut jobs = Vec::new();
    for column in 0..num_columns {
        let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(column).as_ref()).collect();
        let merged = arrow_select::concat::concat(&arrays)
            .map_err(|e| format!("concat column {column}: {e}"))?;
        for (page_index, (start, len)) in page_ranges(merged.as_ref(), TARGET_PAGE_SIZE)
            .into_iter()
            .enumerate()
        {
            jobs.push(PageJob {
                column,
                page_index,
                num_rows: len as i64,
                array: merged.slice(start, len),
            });
        }
    }
    Ok(jobs)
}

/// Encode one page: PLAIN values + snappy + page header. Runs in parallel
/// (one call per [`PageJob`]) on the worker pool.
pub fn encode_page(job: PageJob) -> Result<EncodedPage, String> {
    let raw = encode_plain(job.array.as_ref())?;
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
/// `total_rows` rows). `schema` is the batches' Arrow schema. Pages may arrive
/// in any order (work-stealing); each column's pages are reordered by
/// `page_index` and laid out contiguously as that column's chunk.
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

/// Split `array` into contiguous `(offset, len)` page ranges, each ~`target`
/// uncompressed bytes when PLAIN-encoded. Fixed-width types split by row count;
/// variable-width (BYTE_ARRAY) accumulates the per-value encoded size.
fn page_ranges(array: &dyn Array, target: usize) -> Vec<(usize, usize)> {
    let len = array.len();
    if len == 0 {
        return Vec::new();
    }
    if let Some(width) = plain_fixed_width(array.data_type()) {
        return fixed_ranges(len, (target / width).max(1));
    }
    match array.data_type() {
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            var_ranges(len, target, |i| a.value(i).len())
        }
        DataType::Utf8View => {
            let a = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            var_ranges(len, target, |i| a.value(i).len())
        }
        // Unsupported: one page; `encode_plain` will reject it.
        _ => vec![(0, len)],
    }
}

/// PLAIN byte size of one fixed-width value, or `None` for variable-width types.
fn plain_fixed_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Int32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::Float64 => Some(8),
        _ => None,
    }
}

fn fixed_ranges(len: usize, rows_per_page: usize) -> Vec<(usize, usize)> {
    (0..len)
        .step_by(rows_per_page)
        .map(|start| (start, rows_per_page.min(len - start)))
        .collect()
}

/// Walk values, cutting a page once the accumulated PLAIN size (4-byte length
/// prefix + bytes per value) reaches `target`. Always at least one row per page.
fn var_ranges(len: usize, target: usize, value_len: impl Fn(usize) -> usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut acc = 0usize;
    for i in 0..len {
        acc += 4 + value_len(i);
        if acc >= target {
            ranges.push((start, i + 1 - start));
            start = i + 1;
            acc = 0;
        }
    }
    if start < len {
        ranges.push((start, len - start));
    }
    ranges
}

/// Serialize a [`WriteThrift`] value (compact protocol) onto `out`.
fn write_thrift<T: WriteThrift>(value: &T, out: &mut Vec<u8>) -> Result<(), String> {
    let mut prot = ThriftCompactOutputProtocol::new(out);
    value
        .write_thrift(&mut prot)
        .map_err(|e| format!("thrift encode: {e}"))
}

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

/// PLAIN-encode a required column's values, little-endian, with no levels.
/// BYTE_ARRAY values are a 4-byte LE length prefix followed by the bytes.
fn encode_plain(array: &dyn Array) -> Result<Vec<u8>, String> {
    if array.null_count() > 0 {
        return Err(format!(
            "parquet write: column has {} nulls but only required columns are supported",
            array.null_count()
        ));
    }
    let len = array.len();
    let mut out = Vec::new();
    match array.data_type() {
        DataType::Int32 => {
            let a = downcast::<Int32Array>(array)?;
            out.reserve(len * 4);
            for i in 0..len {
                out.extend_from_slice(&a.value(i).to_le_bytes());
            }
        }
        DataType::Int64 => {
            let a = downcast::<Int64Array>(array)?;
            out.reserve(len * 8);
            for i in 0..len {
                out.extend_from_slice(&a.value(i).to_le_bytes());
            }
        }
        DataType::Float32 => {
            let a = downcast::<Float32Array>(array)?;
            out.reserve(len * 4);
            for i in 0..len {
                out.extend_from_slice(&a.value(i).to_le_bytes());
            }
        }
        DataType::Float64 => {
            let a = downcast::<Float64Array>(array)?;
            out.reserve(len * 8);
            for i in 0..len {
                out.extend_from_slice(&a.value(i).to_le_bytes());
            }
        }
        DataType::Utf8 => {
            let a = downcast::<StringArray>(array)?;
            for i in 0..len {
                encode_byte_array(a.value(i).as_bytes(), &mut out);
            }
        }
        DataType::Utf8View => {
            let a = downcast::<StringViewArray>(array)?;
            for i in 0..len {
                encode_byte_array(a.value(i).as_bytes(), &mut out);
            }
        }
        other => return Err(format!("unsupported column type for parquet write: {other:?}")),
    }
    Ok(out)
}

fn encode_byte_array(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
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

    #[test]
    fn fixed_width_pages_split_by_row_count() {
        let array = Int64Array::from((0..250).collect::<Vec<i64>>());

        // 8 bytes/value, target 800 bytes => 100 rows/page => 100, 100, 50.
        let ranges = page_ranges(&array, 800);

        assert_eq!(ranges, vec![(0, 100), (100, 100), (200, 50)]);
    }

    #[test]
    fn variable_width_pages_split_by_byte_size() {
        // Each value encodes as 4 + 6 = 10 bytes ("abcdef").
        let array = StringArray::from(vec!["abcdef"; 10]);

        // Target 25 bytes => cut after 3 values (30 >= 25): 3, 3, 3, 1.
        let ranges = page_ranges(&array, 25);

        assert_eq!(ranges, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
    }
}
