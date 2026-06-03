//! Hand-rolled, **page-parallel** Parquet encoder — the write-side mirror of
//! dispatch's hand-rolled reader.
//!
//! Like the reader, this avoids the upstream `parquet` crate: it reuses the
//! shared [`thriftparquet`] metadata/codec layer and the in-house snappy
//! (`snap`). The work is split exactly where the reader parallelizes — at the
//! **page** level:
//!
//! 1. [`build_page_jobs`] turns a set of [`RecordBatch`]es into one
//!    [`PageJob`] per `(row group, column)` — each job is one data page.
//! 2. [`encode_page`] (run in parallel across the worker pool, one job each)
//!    PLAIN-encodes a column's values, snappy-compresses them, and prepends the
//!    page header — producing self-contained [`EncodedPage`] bytes.
//! 3. [`assemble_parquet`] stitches the encoded pages into one file: it lays
//!    them out back-to-back (recording each column chunk's offset/size) and
//!    writes the footer. Cheap and serial — just concatenation plus thrift.
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

/// One page to encode: a single column's values for a single row group.
pub struct PageJob {
    pub row_group: usize,
    pub column: usize,
    pub num_rows: i64,
    pub array: ArrayRef,
}

/// An encoded data page (header + snappy-compressed values), tagged with its
/// position so [`assemble_parquet`] can place it after the parallel encode.
pub struct EncodedPage {
    pub row_group: usize,
    pub column: usize,
    pub num_rows: i64,
    pub bytes: Vec<u8>,
}

/// Split `batches` into one [`PageJob`] per `(row group, column)`. Each batch
/// becomes one row group; each column of it, one page. Column arrays are shared
/// by `Arc` clone, so this is cheap and the batches can be dropped afterwards.
pub fn build_page_jobs(batches: &[RecordBatch]) -> Vec<PageJob> {
    let mut jobs = Vec::new();
    for (row_group, batch) in batches.iter().enumerate() {
        let num_rows = batch.num_rows() as i64;
        for (column, array) in batch.columns().iter().enumerate() {
            jobs.push(PageJob {
                row_group,
                column,
                num_rows,
                array: array.clone(),
            });
        }
    }
    jobs
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
        row_group: job.row_group,
        column: job.column,
        num_rows: job.num_rows,
        bytes,
    })
}

/// Stitch encoded pages into one Parquet file. `schema` is the batches' Arrow
/// schema; `num_row_groups` is the batch count. Pages may arrive in any order
/// (work-stealing) — they are placed by their `(row group, column)` tag.
pub fn assemble_parquet(
    schema: &SchemaRef,
    num_row_groups: usize,
    pages: Vec<EncodedPage>,
) -> Result<Vec<u8>, String> {
    let num_columns = schema.fields().len();

    // Bucket pages into a row-group × column grid, recording each row group's
    // row count.
    let mut grid: Vec<Vec<Option<EncodedPage>>> = (0..num_row_groups)
        .map(|_| (0..num_columns).map(|_| None).collect())
        .collect();
    let mut rg_rows = vec![0i64; num_row_groups];
    for page in pages {
        let (rg, col) = (page.row_group, page.column);
        rg_rows[rg] = page.num_rows;
        grid[rg][col] = Some(page);
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(PARQUET_MAGIC);

    let mut row_groups = Vec::with_capacity(num_row_groups);
    for (rg, columns_grid) in grid.into_iter().enumerate() {
        let mut columns = Vec::with_capacity(num_columns);
        for (col, page) in columns_grid.into_iter().enumerate() {
            let page = page.ok_or_else(|| format!("missing page for row group {rg} column {col}"))?;
            let data_page_offset = out.len() as i64;
            out.extend_from_slice(&page.bytes);
            columns.push(ColumnChunk {
                meta_data: Some(ColumnMetaData {
                    total_compressed_size: out.len() as i64 - data_page_offset,
                    data_page_offset,
                    dictionary_page_offset: None,
                    statistics: None,
                }),
            });
        }
        row_groups.push(RowGroup {
            columns,
            num_rows: rg_rows[rg],
        });
    }

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
