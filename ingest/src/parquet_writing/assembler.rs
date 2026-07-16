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
//!
//! A row group holds one column chunk per *leaf*, while a job (and so an arriving
//! [`EncodedColumnChunk`]) covers one top-level column: a flat column is its own
//! leaf, and a shredded variant is a group of them. So the gathering still counts
//! one chunk per schema column, and a chunk lays its leaves down in the
//! depth-first order the footer schema numbers them in.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, FieldRef, SchemaRef};
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use thriftparquet::footer::{
    ColumnChunk, ColumnMetaData, FileMetaData, LogicalType, RowGroup, SchemaElement, Statistics,
};
use thriftparquet::general::Encoding;
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

use super::error::{WriteError, WriteResult};
use super::types::{
    EncodedColumnChunk, EncodedFile, EncodedLeaf, FileId, RowGroupHeader, RowGroupId,
    RowGroupSortStats,
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Parquet repetition type for an optional (nullable) field — a field whose rows
/// carry definition levels.
const REPETITION_OPTIONAL: i32 = 1;
/// Parquet `CompressionCodec::SNAPPY`.
const SNAPPY_CODEC: i32 = 1;
/// Parquet format version written into the footer.
const PARQUET_VERSION: i32 = 1;
/// Parquet `ConvertedType::UTF8` — marks a BYTE_ARRAY column as a string so
/// readers surface it as text rather than opaque bytes.
const CONVERTED_UTF8: i32 = 0;

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
        let group = assemble_row_group(chunks, &stats)?;

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
    mut chunks: Vec<EncodedColumnChunk>,
    stats: &HashMap<usize, Statistics>,
) -> WriteResult<AssembledRowGroup> {
    // Footer column chunks must be in schema order; work-stealing delivers them
    // in any order. Within a column, its leaves are already in footer order.
    chunks.sort_by_key(|c| c.column);

    let mut bytes = Vec::new();
    let mut columns = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        // Only sort columns get footer statistics, and a sort key is always a
        // flat column — one leaf, which takes them. A wider column (a variant)
        // is never a sort key, so it gets none.
        let mut statistics = (chunk.leaves.len() == 1)
            .then(|| stats.get(&chunk.column).cloned())
            .flatten();
        for leaf in chunk.leaves {
            columns.push(write_leaf_chunk(&mut bytes, leaf, statistics.take())?);
        }
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

/// Append one leaf's column chunk to `out`: its dictionary page (if any) followed
/// by its data pages, contiguously. Returns the chunk metadata with offsets
/// relative to `out` (rebased to the file by [`build_file`]).
fn write_leaf_chunk(
    out: &mut Vec<u8>,
    leaf: EncodedLeaf,
    statistics: Option<Statistics>,
) -> WriteResult<ColumnChunk> {
    if leaf.data_pages.is_empty() {
        return Err(WriteError::MissingPages {
            path: leaf.path.join("."),
        });
    }
    let chunk_start = out.len() as i64;
    let mut uncompressed = 0i64;

    // The dictionary page, if any, precedes the data pages.
    let dictionary_page_offset = leaf.dictionary_page.as_ref().map(|dict| {
        let offset = out.len() as i64;
        uncompressed += (dict.header_len + dict.uncompressed_size) as i64;
        out.extend_from_slice(&dict.bytes);
        offset
    });

    let data_page_offset = out.len() as i64;
    let mut num_values = 0i64;
    for page in &leaf.data_pages {
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
            physical_type: leaf.physical_type,
            encodings,
            path_in_schema: leaf.path,
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

/// Build the footer schema: a root group element followed by every field's,
/// depth-first — the flat, pre-order element list Parquet stores a schema tree
/// as, and the order the reader rebuilds it from.
fn build_schema_elements(schema: &SchemaRef) -> WriteResult<Vec<SchemaElement>> {
    let mut elements = vec![SchemaElement {
        physical_type: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(schema.fields().len() as i32),
        converted_type: None,
        logical_type: None,
    }];
    for field in schema.fields() {
        push_schema_element(field, &mut elements)?;
    }
    Ok(elements)
}

/// Append `field`'s schema element, and for a struct its children's after it. A
/// nullable field is OPTIONAL, which is what tells the reader its leaves carry
/// definition levels.
fn push_schema_element(field: &FieldRef, elements: &mut Vec<SchemaElement>) -> WriteResult<()> {
    let repetition_type = Some(if field.is_nullable() {
        REPETITION_OPTIONAL
    } else {
        REPETITION_REQUIRED
    });
    match field.data_type() {
        DataType::Struct(children) => {
            elements.push(SchemaElement {
                physical_type: None,
                repetition_type,
                name: field.name().clone(),
                num_children: Some(children.len() as i32),
                converted_type: None,
                // The VARIANT annotation is the whole difference between a
                // variant column and a plain struct of binary leaves: it is what
                // a reader keys off to treat the group as semi-structured.
                logical_type: catalog::parquet::is_variant_field(field)
                    .then_some(LogicalType::Variant),
            });
            for child in children {
                push_schema_element(child, elements)?;
            }
        }
        data_type => elements.push(SchemaElement {
            physical_type: Some(catalog::parquet::arrow_to_parquet_physical(data_type)?),
            repetition_type,
            name: field.name().clone(),
            num_children: None,
            converted_type: match data_type {
                DataType::Utf8 | DataType::Utf8View => Some(CONVERTED_UTF8),
                _ => None,
            },
            logical_type: None,
        }),
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_writing::encoder::encode_column_chunk;
    use crate::parquet_writing::types::{EncodedColumnChunk, PartitionTag};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Array, Int64Array, RecordBatch, StringArray, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields, Schema};
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

    /// Encode a batch the way the pipeline does — flatten each column into leaves
    /// and encode them (dictionary or PLAIN, the encoder's choice) →
    /// `assemble_row_group` → `build_file` — and return the bytes (a
    /// single-row-group file, no stats).
    fn encode(batch: &RecordBatch) -> Vec<u8> {
        let schema = batch.schema();
        let header = header(&schema);
        let chunks: Vec<EncodedColumnChunk> = (0..batch.num_columns())
            .map(|column| {
                Ok(EncodedColumnChunk {
                    header: header.clone(),
                    column,
                    leaves: encode_column_chunk(schema.field(column), batch.column(column))?,
                })
            })
            .collect::<WriteResult<_>>()
            .unwrap();
        let group = assemble_row_group(chunks, &HashMap::new()).unwrap();
        build_file(&schema, vec![group]).unwrap()
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

    /// A nullable column and a nullable struct round-trip through arrow-rs: the
    /// footer carries a real group with OPTIONAL fields, and the definition
    /// levels place the nulls back where they were. This is the shape a shredded
    /// variant is built out of.
    #[test]
    fn a_nested_nullable_column_round_trips() {
        let inner = Fields::from(vec![Field::new("a", DataType::Int64, true)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Struct(inner.clone()), true),
        ]));
        // Row 1's struct is absent entirely; row 2 has a struct whose field is
        // null — two different depths of absence, one level apart.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(StructArray::new(
                    inner,
                    vec![Arc::new(Int64Array::from(vec![Some(10), None, None]))],
                    Some(NullBuffer::from(vec![true, false, true])),
                )),
            ],
        )
        .unwrap();

        let got = round_trip(&batch);

        assert_eq!(
            got.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(1), None, Some(3)])
        );
        let structs = got.column(1).as_struct();
        assert!(structs.is_null(1), "row 1's struct is absent");
        assert_eq!(
            structs.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(10), None, None])
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
