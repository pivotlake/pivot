//! Assembles encoded column chunks into complete Parquet files.
//!
//! Every encoded chunk carries an assembly-worker id. Routing all chunks for a
//! file to that worker lets [`FileAssembler`] gather them without shared state.
//! It first assembles each row group, then orders the completed row groups and
//! writes the file footer. A file is emitted from `consume` as soon as its last
//! expected row group arrives.
//!
//! An [`EncodedColumnChunk`] represents one top-level schema column and may
//! contain several primitive leaves. Leaves are written in schema order, with
//! dictionary pages before data pages, and their offsets and statistics are
//! recorded in the footer.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, FieldRef, SchemaRef};
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use thriftparquet::footer::{
    ColumnChunk, ColumnMetaData, FileMetaData, LogicalType, RowGroup, SchemaElement,
};
use thriftparquet::general::Encoding;
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

use dispatch::memory::{FileBytes, Slab, SlabAllocator};

use super::error::{WriteError, WriteResult};
use super::types::{
    AssembledFile, EncodedColumnChunk, EncodedLeaf, FileId, RowGroupContext, RowGroupId,
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Parquet repetition type for an optional (nullable) field — a field whose rows
/// carry definition levels.
const REPETITION_OPTIONAL: i32 = 1;
/// Parquet format version written into the footer.
const PARQUET_VERSION: i32 = 1;

pub(super) type FileAssemblerFactory = DefaultUnaryFactory<FileAssembler>;

pub(super) fn factories(worker_count: usize) -> Vec<FileAssemblerFactory> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

/// Items accumulated until an expected count is reached. The same helper is
/// used for columns within a row group and row groups within a file.
struct Gathering<V> {
    remaining: usize,
    context: Arc<RowGroupContext>,
    items: Vec<V>,
}

impl<V> Gathering<V> {
    fn new(remaining: usize, context: Arc<RowGroupContext>) -> Self {
        Self {
            remaining,
            context,
            items: Vec::new(),
        }
    }

    /// Returns `true` when this item completes the collection.
    fn push(&mut self, item: V) -> bool {
        self.items.push(item);
        self.remaining -= 1;
        self.remaining == 0
    }
}

/// Per-worker state for row groups and files that are still being assembled.
#[derive(Default)]
pub(super) struct FileAssembler {
    chunks_by_row_group: HashMap<RowGroupId, Gathering<EncodedColumnChunk>>,
    row_groups_by_file: HashMap<FileId, Gathering<AssembledRowGroup>>,
}

impl Unary<EncodedColumnChunk, AssembledFile> for FileAssembler {
    fn consume(
        &mut self,
        chunk: EncodedColumnChunk,
        sender: &mut dyn Sender<AssembledFile>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        // A row group is ready once one encoded chunk has arrived for every
        // top-level schema column.
        let row_group_id = chunk.context.row_group_id;
        let column_count = chunk.context.schema.fields().len();
        let context = chunk.context.clone();
        if !self
            .chunks_by_row_group
            .entry(row_group_id)
            .or_insert_with(|| Gathering::new(column_count, context))
            .push(chunk)
        {
            return Ok(());
        }

        let Gathering {
            context,
            items: chunks,
            ..
        } = self.chunks_by_row_group.remove(&row_group_id).unwrap();
        let group = AssembledRowGroup::new(row_group_id, chunks)?;

        let file_id = context.file_info.file_id;
        let row_group_count = context.file_info.row_group_count;
        if !self
            .row_groups_by_file
            .entry(file_id)
            .or_insert_with(|| Gathering::new(row_group_count, context))
            .push(group)
        {
            return Ok(());
        }

        let Gathering {
            context,
            items: mut groups,
            ..
        } = self.row_groups_by_file.remove(&file_id).unwrap();
        // Parallel encoding may complete row groups out of order.
        groups.sort_by_key(|group| group.row_group_id);
        let (bytes, metadata) = build_file(&context.schema, groups)?;
        sender.send(AssembledFile {
            bytes,
            metadata,
            partition: context.file_info.partition.clone(),
        })?;
        Ok(())
    }
}

/// Encoded bytes and footer column metadata for one row group. Column offsets
/// are relative to the row group until `build_file` rebases them.
struct AssembledRowGroup {
    row_group_id: RowGroupId,
    bytes: FileBytes,
    columns: Vec<ColumnChunk>,
}

impl AssembledRowGroup {
    /// Writes a row group's encoded columns in schema order.
    fn new(row_group_id: RowGroupId, mut chunks: Vec<EncodedColumnChunk>) -> WriteResult<Self> {
        // Workers may return columns in any order. Leaves within each column
        // are already ordered by the encoder.
        chunks.sort_by_key(|chunk| chunk.column_index);

        let mut bytes = FileBytes::new();
        let mut columns = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            for leaf in chunk.leaves {
                columns.push(write_leaf_chunk(&mut bytes, leaf)?);
            }
        }
        Ok(Self {
            row_group_id,
            bytes,
            columns,
        })
    }
}

/// Stitch several assembled row groups into one Parquet file, rebasing each row
/// group's column offsets to its position in the file. Returns the bytes and the
/// footer metadata just written into them, so a consumer that needs the file's
/// row-group metadata can take it directly instead of parsing the footer back.
fn build_file(
    schema: &SchemaRef,
    groups: Vec<AssembledRowGroup>,
) -> WriteResult<(FileBytes, FileMetaData)> {
    let mut out = FileBytes::new();
    out.push(run_of(PARQUET_MAGIC)?);

    let mut row_groups = Vec::with_capacity(groups.len());
    let mut num_rows = 0i64;
    for group in groups {
        let base = out.len() as i64;
        out.append(group.bytes);

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
        created_by: Some("pivotdb".to_string()),
    };
    write_footer(&mut out, &file_meta)?;
    Ok((out, file_meta))
}

/// Append one leaf's column chunk to `out`: its dictionary page (if any) followed
/// by its data pages, contiguously. Returns the chunk metadata with offsets
/// relative to `out` (rebased to the file by [`build_file`]).
fn write_leaf_chunk(out: &mut FileBytes, leaf: EncodedLeaf) -> WriteResult<ColumnChunk> {
    if leaf.data_pages.is_empty() {
        return Err(WriteError::MissingPages {
            path: leaf.path.join("."),
        });
    }
    let chunk_start = out.len() as i64;
    let mut uncompressed = 0i64;

    // The dictionary page, if any, precedes the data pages.
    let dictionary_page_offset = leaf.dictionary_page.map(|dict| {
        let offset = out.len() as i64;
        uncompressed += (dict.header_len + dict.uncompressed_size) as i64;
        out.push(dict.bytes);
        offset
    });

    let data_page_offset = out.len() as i64;
    let mut num_values = 0i64;
    for page in leaf.data_pages {
        num_values += page.num_rows;
        uncompressed += (page.header_len + page.uncompressed_size) as i64;
        out.push(page.bytes);
    }
    let compressed = out.len() as i64 - chunk_start;

    // A dictionary chunk encodes its data pages as RLE_DICTIONARY indices and
    // its dictionary page as PLAIN; any other chunk reports the one encoding
    // its data pages carry.
    let encodings = if dictionary_page_offset.is_some() {
        vec![Encoding::PLAIN as i32, Encoding::RLE_DICTIONARY as i32]
    } else {
        vec![leaf.data_page_encoding as i32]
    };
    Ok(ColumnChunk {
        file_offset: chunk_start,
        meta_data: Some(ColumnMetaData {
            physical_type: leaf.physical_type,
            encodings,
            path_in_schema: leaf.path,
            codec: leaf.compression.codec(),
            num_values,
            total_uncompressed_size: uncompressed,
            total_compressed_size: compressed,
            data_page_offset,
            dictionary_page_offset,
            statistics: Some(leaf.statistics),
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
        type_length: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(schema.fields().len() as i32),
        converted_type: None,
        scale: None,
        precision: None,
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
                type_length: None,
                repetition_type,
                name: field.name().clone(),
                num_children: Some(children.len() as i32),
                converted_type: None,
                scale: None,
                precision: None,
                // The VARIANT annotation is the whole difference between a
                // variant column and a plain struct of binary leaves: it is what
                // a reader keys off to treat the group as semi-structured.
                logical_type: crate::parquet::is_variant_field(field)
                    .then_some(LogicalType::Variant),
            });
            for child in children {
                push_schema_element(child, elements)?;
            }
        }
        data_type => {
            // A leaf is described by its physical type plus whatever annotation
            // its type needs, which the two write-path halves of `arrow_map`
            // give together.
            let annotation = crate::parquet::arrow_to_annotation(data_type);
            elements.push(SchemaElement {
                physical_type: Some(crate::parquet::arrow_to_parquet_physical(data_type)?),
                type_length: annotation.type_length,
                repetition_type,
                name: field.name().clone(),
                num_children: None,
                converted_type: annotation.converted_type,
                scale: annotation.scale,
                precision: annotation.precision,
                logical_type: annotation.logical_type,
            })
        }
    }
    Ok(())
}

/// Write the trailing footer: `[FileMetaData][u32 LE footer length][PAR1]`. It
/// is one more run of the file, small enough to build in one piece.
fn write_footer(out: &mut FileBytes, file_meta: &FileMetaData) -> WriteResult<()> {
    let mut footer = Vec::new();
    file_meta.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut footer))?;
    footer.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    footer.extend_from_slice(PARQUET_MAGIC);
    out.push(run_of(&footer)?);
    Ok(())
}

/// A run holding a copy of `bytes`, for the small pieces of a file that are not
/// pages: the leading magic, and the footer.
fn run_of(bytes: &[u8]) -> WriteResult<Slab> {
    let mut run = SlabAllocator::new(false).get_slab_of_size(bytes.len(), false);
    run.as_mut_slice().copy_from_slice(bytes);
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::ParquetTable;
    use crate::parquet::types::arrow_map::CONVERTED_DECIMAL;
    use crate::parquet::writing::encoder::encode_column_chunk;
    use crate::parquet::writing::types::{EncodedColumnChunk, FileAssemblyInfo};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal64Type, Int64Type};
    use arrow_array::{
        Array, ArrayRef, Datum, Decimal64Array, Int64Array, RecordBatch, StructArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields, Schema};
    use dispatch::Dispatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::Arc;

    fn context(schema: &SchemaRef) -> Arc<RowGroupContext> {
        Arc::new(RowGroupContext {
            row_group_id: 0,
            assembly_worker: 0,
            schema: schema.clone(),
            file_info: Arc::new(FileAssemblyInfo {
                file_id: 0,
                row_group_count: 1,
                partition: None,
            }),
        })
    }

    /// Encode a batch the way the pipeline does — flatten each column into leaves
    /// and encode them (dictionary or PLAIN, the encoder's choice) →
    /// `assemble_row_group` → `build_file` — and return the bytes (a
    /// single-row-group file, no stats).
    fn encode(batch: &RecordBatch) -> Vec<u8> {
        // Pages are written into ring memory, so this needs a memory context of
        // its own: a thread can hold only one, and a test may also spin up a
        // Dispatch. The file is copied out before the thread ends, which is
        // where its slabs are released.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    dispatch::memory::init_test_free_pool(64);
                    let mut allocator = SlabAllocator::new(false);
                    let schema = batch.schema();
                    let context = context(&schema);
                    let chunks: Vec<EncodedColumnChunk> = (0..batch.num_columns())
                        .map(|column| {
                            Ok(EncodedColumnChunk {
                                context: context.clone(),
                                column_index: column,
                                leaves: encode_column_chunk(
                                    schema.field(column),
                                    batch.column(column),
                                    &mut allocator,
                                )?,
                            })
                        })
                        .collect::<WriteResult<_>>()
                        .unwrap();
                    let group = AssembledRowGroup::new(0, chunks).unwrap();
                    build_file(&schema, vec![group])
                        .unwrap()
                        .0
                        .runs()
                        .flatten()
                        .copied()
                        .collect()
                })
                .join()
                .unwrap()
        })
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

    /// A decimal leaf's schema element carries the full decimal description:
    /// the precision-chosen storage (INT64 here, with no type_length),
    /// precision/scale, the Decimal logical type, and the legacy DECIMAL
    /// converted type. A wide decimal stores as a fixed 16-byte array instead.
    #[test]
    fn a_decimal_column_annotates_its_schema_element() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Decimal64(10, 2), false),
            Field::new("total", DataType::Decimal128(20, 2), false),
        ]));

        let elements = build_schema_elements(&schema).unwrap();

        let narrow = &elements[1];
        assert_eq!(
            narrow.physical_type,
            Some(thriftparquet::general::Type::INT64 as i32)
        );
        assert_eq!(narrow.type_length, None);
        assert_eq!(narrow.converted_type, Some(CONVERTED_DECIMAL));
        assert_eq!(narrow.precision, Some(10));
        assert_eq!(narrow.scale, Some(2));
        assert_eq!(
            narrow.logical_type,
            Some(LogicalType::Decimal {
                scale: 2,
                precision: 10
            })
        );
        let wide = &elements[2];
        assert_eq!(
            wide.physical_type,
            Some(thriftparquet::general::Type::FIXED_LEN_BYTE_ARRAY as i32)
        );
        assert_eq!(wide.type_length, Some(16));
    }

    /// A decimal column's row-group statistics decode into bounds at the
    /// column's own type, negative values included, which is what lets a decimal
    /// predicate skip a row group.
    #[test]
    fn a_decimal_columns_statistics_decode_into_bounds() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal64(10, 2),
            false,
        )]));
        let values = Decimal64Array::from(vec![12345_i64, -67890, 100])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        std::fs::write(&path, encode(&batch)).unwrap();
        let dispatch = Dispatch::spin_up(2, 64, None);

        let table =
            Arc::new(ParquetTable::from_files(dispatch.dispatcher(), &[path], &[]).unwrap());

        let stats = table.row_groups()[0].column_statistics(0).unwrap();
        let (min, _) = stats.min.as_ref().unwrap().get();
        let (max, _) = stats.max.as_ref().unwrap().get();
        assert_eq!(min.as_primitive::<Decimal64Type>().value(0), -67890);
        assert_eq!(max.as_primitive::<Decimal64Type>().value(0), 12345);
        dispatch.exit();
    }
}
