//! Assembles encoded column chunks into one Parquet file per output file.
//!
//! A plain `Unary` map: the [`partition`](super::partition) stage routes every
//! column chunk of a file to `file_id % worker_count`, so all of a file's chunks
//! arrive at one [`FileAssembler`]. It gathers a row group's chunks (by
//! `row_group_id`, one per schema column), assembles the row group — laying each
//! chunk's dictionary page (if any) and data pages out contiguously and stamping
//! sort columns' footer `Statistics` — and once a `file_id` has all its
//! `n_row_groups`, builds the file and emits it as an [`EncodedFile`] tagged with
//! the partition tuple and `sort_bounds` to record in the manifest. Nothing
//! crosses workers and there is no finish phase: every file completes in
//! `consume`. (The upstream [`partition`](super::partition) breaker is what makes
//! this possible — it hands down file-sized units with a known row-group count.)
//!
//! The footer is fully populated (column `type`/`encodings`/`path_in_schema`/
//! `codec`/`num_values`/sizes/stats, `dictionary_page_offset` for dict chunks, row
//! group `total_byte_size`, file `version`/`num_rows`, string columns marked
//! UTF8), so the output round-trips through pivot's reader and through strict
//! readers like arrow-rs and DuckDB.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, Field, SchemaRef};
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use thriftparquet::footer::{
    ColumnChunk, ColumnMetaData, FileMetaData, RowGroup, SchemaElement, Statistics,
};
use thriftparquet::general::{Encoding, Type};
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

use super::error::{WriteError, WriteResult};
use super::types::{
    EncodedColumnChunk, EncodedFile, FileId, RowGroupHeader, RowGroupId, RowGroupSortStats,
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Parquet `CompressionCodec::SNAPPY`.
const SNAPPY_CODEC: i32 = 1;
/// Parquet `ConvertedType::UTF8` — marks a BYTE_ARRAY column as a string.
const CONVERTED_TYPE_UTF8: i32 = 0;
/// Parquet format version written into the footer.
const PARQUET_VERSION: i32 = 1;

pub(super) type FileAssemblerFactory = DefaultUnaryFactory<FileAssembler>;

pub(super) fn factories(worker_count: usize) -> Vec<FileAssemblerFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

/// Items collected under one key until all `remaining` have arrived, carrying the
/// row group `header` they share. Used twice: a row group's column chunks (keyed
/// by row-group id) and a file's assembled row groups (keyed by file id).
struct Gathering<V> {
    remaining: usize,
    header: Arc<RowGroupHeader>,
    items: Vec<V>,
}

impl<V> Gathering<V> {
    fn new(remaining: usize, header: Arc<RowGroupHeader>) -> Self {
        Self {
            remaining,
            header,
            items: Vec::new(),
        }
    }

    /// Add `item`; returns `true` once the last expected item has arrived.
    fn push(&mut self, item: V) -> bool {
        self.items.push(item);
        self.remaining -= 1;
        self.remaining == 0
    }
}

/// Per-worker consumer: gather a file's column chunks (all routed here by
/// `file_id`) and emit the finished file once they have all arrived.
#[derive(Default)]
pub(super) struct FileAssembler {
    chunks_by_row_group: HashMap<RowGroupId, Gathering<EncodedColumnChunk>>,
    row_groups_by_file: HashMap<FileId, Gathering<AssembledRowGroup>>,
}

impl Unary<EncodedColumnChunk, EncodedFile> for FileAssembler {
    fn consume<S: Sender<EncodedFile>>(
        &mut self,
        chunk: EncodedColumnChunk,
        sender: &mut S,
    ) -> UnaryResult<()> {
        // Gather this row group's column chunks (one per schema column).
        let row_group_id = chunk.header.row_group_id;
        let columns = chunk.header.schema.fields().len();
        let header = chunk.header.clone();
        if !self
            .chunks_by_row_group
            .entry(row_group_id)
            .or_insert_with(|| Gathering::new(columns, header))
            .push(chunk)
        {
            return Ok(());
        }

        // Row group done: assemble it with its sort columns' footer statistics.
        let Gathering {
            header,
            items: chunks,
            ..
        } = self.chunks_by_row_group.remove(&row_group_id).unwrap();
        let stats = build_stats(&header.tag.sort_stats);
        let group = assemble_row_group(&header.schema, chunks, &stats)?;

        // Add it to its file; emit the file once all its row groups are in.
        let file_id = header.tag.file_id;
        let n_row_groups = header.tag.n_row_groups;
        if !self
            .row_groups_by_file
            .entry(file_id)
            .or_insert_with(|| Gathering::new(n_row_groups, header))
            .push(group)
        {
            return Ok(());
        }

        let Gathering {
            header,
            items: groups,
            ..
        } = self.row_groups_by_file.remove(&file_id).unwrap();
        let bytes = build_file(&header.schema, groups)?;
        sender.send(EncodedFile {
            bytes,
            partition: header.tag.partition.clone(),
            sort_bounds: header.tag.sort_bounds.clone(),
        })?;
        Ok(())
    }
}

/// Build the column-index → footer `Statistics` map for a row group's sort
/// columns (min/max as the bytes the reader decodes; skip any column whose type
/// has no encodable stats).
fn build_stats(sort_stats: &RowGroupSortStats) -> HashMap<usize, Statistics> {
    let mut map = HashMap::new();
    for col in &sort_stats.cols {
        let (Some(min_value), Some(max_value)) = (stat_bytes(&col.min), stat_bytes(&col.max))
        else {
            continue;
        };
        map.insert(
            col.column,
            Statistics {
                max: None,
                min: None,
                null_count: Some(col.null_count),
                distinct_count: None,
                max_value: Some(max_value),
                min_value: Some(min_value),
            },
        );
    }
    map
}

/// Encode a single-element stats array (a sort column's min or max, computed by
/// the partition stage) into the Parquet `min_value`/`max_value` bytes. The
/// encoding must match the reader's `decode_scalar` exactly — little-endian for
/// primitives, raw UTF-8 for strings — or stats-based row-group pruning would
/// silently drop rows.
fn stat_bytes(value: &ArrayRef) -> Option<Vec<u8>> {
    macro_rules! le_bytes {
        ($arr:ty) => {
            value
                .as_any()
                .downcast_ref::<$arr>()?
                .value(0)
                .to_le_bytes()
                .to_vec()
        };
    }
    macro_rules! raw_bytes {
        ($arr:ty) => {
            value
                .as_any()
                .downcast_ref::<$arr>()?
                .value(0)
                .as_bytes()
                .to_vec()
        };
    }
    Some(match value.data_type() {
        DataType::Int32 => le_bytes!(Int32Array),
        DataType::Int64 => le_bytes!(Int64Array),
        DataType::Float32 => le_bytes!(Float32Array),
        DataType::Float64 => le_bytes!(Float64Array),
        DataType::Utf8 => raw_bytes!(StringArray),
        DataType::Utf8View => raw_bytes!(StringViewArray),
        _ => return None,
    })
}

/// One fully-encoded row group: its column-chunk bytes plus the chunk metadata,
/// with offsets relative to the start of `bytes` (rebased by [`build_file`]).
struct AssembledRowGroup {
    bytes: Vec<u8>,
    columns: Vec<ColumnChunk>,
}

/// Assemble one row group from its column chunks (one per schema column, any
/// order). `stats` maps a column index to the footer `Statistics` to write for it
/// (sort columns).
fn assemble_row_group(
    schema: &SchemaRef,
    mut chunks: Vec<EncodedColumnChunk>,
    stats: &HashMap<usize, Statistics>,
) -> WriteResult<AssembledRowGroup> {
    // Footer column chunks must be in schema order; work-stealing delivers them
    // in any order.
    chunks.sort_by_key(|c| c.column);

    let mut bytes = Vec::new();
    let mut columns = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let col = chunk.column;
        columns.push(write_column_chunk(
            &mut bytes,
            schema.field(col),
            chunk,
            stats.get(&col).cloned(),
        )?);
    }
    Ok(AssembledRowGroup { bytes, columns })
}

/// Stitch several assembled row groups into one Parquet file, rebasing each row
/// group's column offsets to its position in the file.
fn build_file(schema: &SchemaRef, groups: Vec<AssembledRowGroup>) -> WriteResult<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(PARQUET_MAGIC);

    let mut row_groups = Vec::with_capacity(groups.len());
    let mut num_rows = 0i64;
    for group in groups {
        let base = out.len() as i64;
        out.extend_from_slice(&group.bytes);

        // Rebase each column's page offsets to the file, and tally the row
        // group's size and (from any column — all cover the same rows) row count.
        let mut total_byte_size = 0i64;
        let mut group_rows = 0i64;
        let columns: Vec<ColumnChunk> = group
            .columns
            .into_iter()
            .map(|mut chunk| {
                chunk.file_offset += base;
                if let Some(meta) = chunk.meta_data.as_mut() {
                    meta.data_page_offset += base;
                    if let Some(dict_offset) = meta.dictionary_page_offset.as_mut() {
                        *dict_offset += base;
                    }
                    total_byte_size += meta.total_uncompressed_size;
                    group_rows = meta.num_values;
                }
                chunk
            })
            .collect();
        num_rows += group_rows;
        row_groups.push(RowGroup {
            columns,
            total_byte_size,
            num_rows: group_rows,
        });
    }

    let file_meta = FileMetaData {
        version: PARQUET_VERSION,
        schema: build_schema_elements(schema)?,
        num_rows,
        row_groups,
        created_by: Some("pivotdb-ingest".to_string()),
    };
    write_footer(&mut out, &file_meta)?;
    Ok(out)
}

/// Append one column chunk to `out`: its dictionary page (if any) followed by its
/// data pages, contiguously. Returns the chunk metadata with offsets relative to
/// `out` (rebased to the file by [`build_file`]).
fn write_column_chunk(
    out: &mut Vec<u8>,
    field: &Field,
    chunk: EncodedColumnChunk,
    statistics: Option<Statistics>,
) -> WriteResult<ColumnChunk> {
    if chunk.data_pages.is_empty() {
        return Err(WriteError::MissingPages {
            column: chunk.column,
        });
    }
    let chunk_start = out.len() as i64;
    let mut uncompressed = 0i64;

    // The dictionary page, if any, precedes the data pages.
    let dictionary_page_offset = chunk.dictionary_page.as_ref().map(|dict| {
        let offset = out.len() as i64;
        uncompressed += (dict.header_len + dict.uncompressed_size) as i64;
        out.extend_from_slice(&dict.bytes);
        offset
    });

    let data_page_offset = out.len() as i64;
    let mut num_values = 0i64;
    for page in &chunk.data_pages {
        num_values += page.num_rows;
        uncompressed += (page.header_len + page.uncompressed_size) as i64;
        out.extend_from_slice(&page.bytes);
    }
    let compressed = out.len() as i64 - chunk_start;

    // A dictionary chunk encodes its data pages as RLE_DICTIONARY indices and its
    // dictionary page as PLAIN; a plain chunk is all PLAIN.
    let encodings = if dictionary_page_offset.is_some() {
        vec![Encoding::PLAIN as i32, Encoding::RLE_DICTIONARY as i32]
    } else {
        vec![Encoding::PLAIN as i32]
    };
    Ok(ColumnChunk {
        file_offset: chunk_start,
        meta_data: Some(ColumnMetaData {
            physical_type: physical_type(field.data_type())?,
            encodings,
            path_in_schema: vec![field.name().clone()],
            codec: SNAPPY_CODEC,
            num_values,
            total_uncompressed_size: uncompressed,
            total_compressed_size: compressed,
            data_page_offset,
            dictionary_page_offset,
            statistics,
            encoding_stats: None,
        }),
    })
}

/// Build the footer schema: a root group element followed by one leaf per
/// column (string columns marked UTF8).
fn build_schema_elements(schema: &SchemaRef) -> WriteResult<Vec<SchemaElement>> {
    let mut elements = Vec::with_capacity(schema.fields().len() + 1);
    elements.push(SchemaElement {
        physical_type: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(schema.fields().len() as i32),
        converted_type: None,
        logical_type: None,
    });
    for field in schema.fields() {
        elements.push(SchemaElement {
            physical_type: Some(physical_type(field.data_type())?),
            repetition_type: Some(REPETITION_REQUIRED),
            name: field.name().clone(),
            num_children: None,
            converted_type: converted_type(field.data_type()),
            logical_type: None,
        });
    }
    Ok(elements)
}

/// Write the trailing footer: `[FileMetaData][u32 LE footer length][PAR1]`.
fn write_footer(out: &mut Vec<u8>, file_meta: &FileMetaData) -> WriteResult<()> {
    let mut footer = Vec::new();
    file_meta.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut footer))?;
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(PARQUET_MAGIC);
    Ok(())
}

/// Map an Arrow type to its Parquet physical type id (matching the reader's
/// `convert_physical_to_arrow`). Errors on types the encoder does not handle.
fn physical_type(data_type: &DataType) -> WriteResult<i32> {
    Ok(match data_type {
        DataType::Int32 => Type::INT32 as i32,
        DataType::Int64 => Type::INT64 as i32,
        DataType::Float32 => Type::FLOAT as i32,
        DataType::Float64 => Type::DOUBLE as i32,
        DataType::Utf8 | DataType::Utf8View => Type::BYTE_ARRAY as i32,
        other => return Err(WriteError::UnsupportedType(other.clone())),
    })
}

/// The Parquet converted type for a column, if any. Marks string columns as
/// UTF8 so readers surface them as text rather than opaque bytes.
fn converted_type(data_type: &DataType) -> Option<i32> {
    match data_type {
        DataType::Utf8 | DataType::Utf8View => Some(CONVERTED_TYPE_UTF8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::writing::encoder::encode_column_chunk;
    use crate::parquet::writing::types::{EncodedColumnChunk, PartitionTag};
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::Arc;

    /// A minimal header for assembling one standalone row group in a test.
    fn header(schema: &SchemaRef) -> Arc<RowGroupHeader> {
        Arc::new(RowGroupHeader {
            row_group_id: 0,
            dest_worker: 0,
            schema: schema.clone(),
            tag: Arc::new(PartitionTag {
                file_id: 0,
                n_row_groups: 1,
                partition: None,
                sort_bounds: None,
                sort_stats: RowGroupSortStats { cols: vec![] },
            }),
        })
    }

    /// Encode a batch the way the pipeline does — encode each column into a chunk
    /// (dictionary or PLAIN, the encoder's choice) → `assemble_row_group` →
    /// `build_file` — and return the bytes (a single-row-group file, no stats).
    fn encode(batch: &RecordBatch) -> Vec<u8> {
        let header = header(&batch.schema());
        let chunks: Vec<EncodedColumnChunk> = (0..batch.num_columns())
            .map(|column| {
                let (dictionary_page, data_pages) = encode_column_chunk(batch.column(column))?;
                Ok(EncodedColumnChunk {
                    header: header.clone(),
                    column,
                    dictionary_page,
                    data_pages,
                })
            })
            .collect::<WriteResult<_>>()
            .unwrap();
        let group = assemble_row_group(&batch.schema(), chunks, &HashMap::new()).unwrap();
        build_file(&batch.schema(), vec![group]).unwrap()
    }

    /// Write `batch` and read it back through arrow-rs's strict reader (our
    /// DuckDB-readability proxy).
    fn round_trip(batch: &RecordBatch) -> RecordBatch {
        let bytes = encode(batch);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        std::fs::write(&path, &bytes).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let read: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(read.len(), 1);
        read.into_iter().next().unwrap()
    }

    /// A small low-cardinality batch dictionary-encodes; the output round-trips.
    #[test]
    fn output_is_readable_by_a_strict_parquet_reader() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "bb", "ccc"])),
            ],
        )
        .unwrap();

        let got = round_trip(&batch);
        assert_eq!(
            got.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
            &Int64Array::from(vec![1, 2, 3])
        );
        assert_eq!(
            got.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap(),
            &StringArray::from(vec!["a", "bb", "ccc"])
        );
    }

    /// A many-row, few-distinct column exercises the dictionary path's RLE runs
    /// (long repeats) and bit-packed groups, and round-trips through arrow-rs.
    #[test]
    fn dictionary_encoded_column_round_trips() {
        // 900 rows over 3 distinct strings, with both long runs and scattered
        // values, so the index stream uses RLE and bit-packed runs.
        let services = ["api", "db", "cache"];
        let values: Vec<&str> = (0..900).map(|i| services[(i / 50 + i % 3) % 3]).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("svc", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(values.clone())) as ArrayRef],
        )
        .unwrap();

        let got = round_trip(&batch);
        assert_eq!(
            got.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap(),
            &StringArray::from(values)
        );
    }
}
