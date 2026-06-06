//! Assembles a row group's pages and packs row groups into Parquet files.
//!
//! A pipeline breaker (the [`OrderByLimit`](dispatch) shape): each worker
//! collects the pages of the row groups routed to it, assembles each row group
//! once all its pages arrive, and emits a finished Parquet file every
//! `target_row_groups`. At finish, leftover row groups go to worker 0 over a
//! side `mpsc` channel, which packs them into the final file(s).
//!
//! The assembly itself is cheap and serial: a row group's pages are laid out
//! column by column into a contiguous byte block (offsets relative to the
//! block), then several such blocks are stitched into one file with their
//! offsets rebased and the footer written. The footer is fully populated
//! (column `type`/`encodings`/`path_in_schema`/`codec`/`num_values`/sizes, row
//! group `total_byte_size`, file `version`/`num_rows`, string columns marked
//! UTF8), so the output round-trips through pivot's reader and through strict
//! readers like arrow-rs and DuckDB.

use std::collections::HashMap;
use std::mem;
use std::sync::mpsc::{self, Receiver, Sender as StdSender, TryRecvError};

use arrow_schema::{DataType, Field, SchemaRef};
use dispatch::{Consumer, Outputter, PipelineBreaker, Sender, UnaryFactory, UnaryResult};
use thriftparquet::footer::{ColumnChunk, ColumnMetaData, FileMetaData, RowGroup, SchemaElement};
use thriftparquet::general::{Encoding, Type};

use super::types::{EncodedPage, PipeEncodedPage, RowGroupId};
use super::{to_arrow, write_thrift};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Parquet `CompressionCodec::SNAPPY`.
const SNAPPY_CODEC: i32 = 1;
/// Parquet `ConvertedType::UTF8` — marks a BYTE_ARRAY column as a string.
const CONVERTED_TYPE_UTF8: i32 = 0;
/// Parquet format version written into the footer.
const PARQUET_VERSION: i32 = 1;

/// Factory for one worker's [`FileAssembler`]. Only worker 0 receives the
/// side-channel receiver.
pub(super) struct FileAssemblerFactory {
    target_row_groups: usize,
    leftover_tx: StdSender<(SchemaRef, AssembledRowGroup)>,
    leftover_rx: Option<Receiver<(SchemaRef, AssembledRowGroup)>>,
}

impl UnaryFactory<PipeEncodedPage, Vec<u8>> for FileAssemblerFactory {
    type Unary = PipelineBreaker<PipeEncodedPage, Vec<u8>, FileAssembler>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(FileAssembler {
            target_row_groups: self.target_row_groups,
            partial: HashMap::new(),
            ready: Vec::new(),
            schema: None,
            leftover_tx: self.leftover_tx,
            leftover_rx: self.leftover_rx,
        })
    }
}

/// One factory per worker, sharing the worker-0 side channel.
pub(super) fn factories(
    target_row_groups: usize,
    worker_count: usize,
) -> Vec<FileAssemblerFactory> {
    let (tx, rx) = mpsc::channel();
    let mut rx = Some(rx);
    (0..worker_count)
        .map(|_| FileAssemblerFactory {
            target_row_groups,
            leftover_tx: tx.clone(),
            leftover_rx: rx.take(),
        })
        .collect()
}

/// Per-worker consumer: assemble owned row groups and pack them into files.
pub(super) struct FileAssembler {
    target_row_groups: usize,
    /// Pages collected per row group: `(expected count, pages so far)`.
    partial: HashMap<RowGroupId, (usize, Vec<EncodedPage>)>,
    /// Assembled row groups awaiting a file.
    ready: Vec<AssembledRowGroup>,
    schema: Option<SchemaRef>,
    leftover_tx: StdSender<(SchemaRef, AssembledRowGroup)>,
    leftover_rx: Option<Receiver<(SchemaRef, AssembledRowGroup)>>,
}

impl Consumer<PipeEncodedPage, Vec<u8>> for FileAssembler {
    type Outputter = FileAssemblerOutputter;

    fn consume<S: Sender<Vec<u8>>>(
        &mut self,
        page: PipeEncodedPage,
        sender: &mut S,
    ) -> UnaryResult<()> {
        self.schema.get_or_insert_with(|| page.schema.clone());
        let entry = self
            .partial
            .entry(page.rg_id)
            .or_insert_with(|| (page.n_pages, Vec::new()));
        entry.1.push(page.page);
        if entry.1.len() != entry.0 {
            return Ok(());
        }
        // Row group complete: assemble it.
        let (_, pages) = self.partial.remove(&page.rg_id).unwrap();
        let group = assemble_row_group(&page.schema, pages).map_err(to_arrow)?;
        self.ready.push(group);
        if self.ready.len() >= self.target_row_groups {
            let file = build_file(&page.schema, mem::take(&mut self.ready)).map_err(to_arrow)?;
            sender.send(file)?;
        }
        Ok(())
    }

    fn into_outputter(mut self) -> UnaryResult<Option<Self::Outputter>> {
        if let Some(schema) = self.schema.clone() {
            for group in self.ready.drain(..) {
                let _ = self.leftover_tx.send((schema.clone(), group));
            }
        }
        Ok(self.leftover_rx.map(|rx| FileAssemblerOutputter {
            rx,
            target_row_groups: self.target_row_groups,
            schema: None,
            pending: Vec::new(),
        }))
    }
}

/// Worker 0's output phase: gather every worker's leftover row groups and pack
/// them into files (`target_row_groups` each, the remainder as the final file).
pub(super) struct FileAssemblerOutputter {
    rx: Receiver<(SchemaRef, AssembledRowGroup)>,
    target_row_groups: usize,
    schema: Option<SchemaRef>,
    pending: Vec<AssembledRowGroup>,
}

impl Outputter<Vec<u8>> for FileAssemblerOutputter {
    fn output<S: Sender<Vec<u8>>>(&mut self, sender: &mut S) -> UnaryResult<bool> {
        match self.rx.try_recv() {
            Ok((schema, group)) => {
                self.schema.get_or_insert(schema);
                self.pending.push(group);
                if self.pending.len() >= self.target_row_groups {
                    self.emit(sender)?;
                }
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.pending.is_empty() {
                    self.emit(sender)?;
                }
                Ok(true)
            }
        }
    }
}

impl FileAssemblerOutputter {
    fn emit<S: Sender<Vec<u8>>>(&mut self, sender: &mut S) -> UnaryResult<()> {
        let schema = self.schema.clone().expect("schema set before emit");
        let file = build_file(&schema, mem::take(&mut self.pending)).map_err(to_arrow)?;
        sender.send(file)?;
        Ok(())
    }
}

/// One fully-encoded row group: its column-chunk bytes plus the chunk metadata,
/// with offsets relative to the start of `bytes` (rebased by [`build_file`]).
struct AssembledRowGroup {
    bytes: Vec<u8>,
    columns: Vec<ColumnChunk>,
    num_rows: i64,
    total_byte_size: i64,
}

/// Assemble one row group from all of its encoded pages (any order). The pages
/// must belong to a single row group; columns are laid out contiguously.
fn assemble_row_group(
    schema: &SchemaRef,
    pages: Vec<EncodedPage>,
) -> Result<AssembledRowGroup, String> {
    let by_column = group_pages_by_column(schema.fields().len(), pages);
    let mut bytes = Vec::new();
    let mut columns = Vec::with_capacity(by_column.len());
    let mut total_byte_size = 0i64;
    for (col, pages) in by_column.into_iter().enumerate() {
        if pages.is_empty() {
            return Err(format!("no pages produced for column {col}"));
        }
        let (chunk, uncompressed) = write_column_chunk(&mut bytes, schema.field(col), pages)?;
        total_byte_size += uncompressed;
        columns.push(chunk);
    }
    // Every column chunk covers the same rows.
    let num_rows = columns[0]
        .meta_data
        .as_ref()
        .map(|m| m.num_values)
        .unwrap_or(0);
    Ok(AssembledRowGroup {
        bytes,
        columns,
        num_rows,
        total_byte_size,
    })
}

/// Stitch several assembled row groups into one Parquet file, rebasing each row
/// group's column offsets to its position in the file.
fn build_file(schema: &SchemaRef, groups: Vec<AssembledRowGroup>) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(PARQUET_MAGIC);

    let mut row_groups = Vec::with_capacity(groups.len());
    let mut num_rows = 0i64;
    for group in groups {
        let base = out.len() as i64;
        out.extend_from_slice(&group.bytes);
        let columns = group
            .columns
            .into_iter()
            .map(|mut chunk| {
                chunk.file_offset += base;
                if let Some(meta) = chunk.meta_data.as_mut() {
                    meta.data_page_offset += base;
                }
                chunk
            })
            .collect();
        num_rows += group.num_rows;
        row_groups.push(RowGroup {
            columns,
            total_byte_size: group.total_byte_size,
            num_rows: group.num_rows,
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

/// Bucket pages by their column and order each bucket by `page_index` (pages
/// arrive in any order from work-stealing).
fn group_pages_by_column(num_columns: usize, pages: Vec<EncodedPage>) -> Vec<Vec<EncodedPage>> {
    let mut by_column: Vec<Vec<EncodedPage>> = (0..num_columns).map(|_| Vec::new()).collect();
    for page in pages {
        by_column[page.column].push(page);
    }
    for column in &mut by_column {
        column.sort_by_key(|p| p.page_index);
    }
    by_column
}

/// Append a column's ordered pages to `out` as one contiguous column chunk.
/// Returns the chunk metadata (offsets relative to `out`) and its uncompressed
/// size (for the row-group tally).
fn write_column_chunk(
    out: &mut Vec<u8>,
    field: &Field,
    pages: Vec<EncodedPage>,
) -> Result<(ColumnChunk, i64), String> {
    let data_page_offset = out.len() as i64;
    let mut num_values = 0i64;
    let mut uncompressed = 0i64;
    for page in pages {
        num_values += page.num_rows;
        uncompressed += (page.header_len + page.uncompressed_size) as i64;
        out.extend_from_slice(&page.bytes);
    }
    let compressed = out.len() as i64 - data_page_offset;
    let chunk = ColumnChunk {
        file_offset: data_page_offset,
        meta_data: Some(ColumnMetaData {
            physical_type: physical_type(field.data_type())?,
            encodings: vec![Encoding::PLAIN as i32],
            path_in_schema: vec![field.name().clone()],
            codec: SNAPPY_CODEC,
            num_values,
            total_uncompressed_size: uncompressed,
            total_compressed_size: compressed,
            data_page_offset,
            dictionary_page_offset: None,
            statistics: None,
            encoding_stats: None,
        }),
    };
    Ok((chunk, uncompressed))
}

/// Build the footer schema: a root group element followed by one leaf per
/// column (string columns marked UTF8).
fn build_schema_elements(schema: &SchemaRef) -> Result<Vec<SchemaElement>, String> {
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
fn write_footer(out: &mut Vec<u8>, file_meta: &FileMetaData) -> Result<(), String> {
    let mut footer = Vec::new();
    write_thrift(file_meta, &mut footer)?;
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(PARQUET_MAGIC);
    Ok(())
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
        other => {
            return Err(format!(
                "unsupported column type for parquet write: {other:?}"
            ));
        }
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
    use crate::parquet_writing::{encoder::encode_page, planner::build_page_jobs};
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema;
    use std::sync::Arc;

    /// Encode a batch the way the pipeline does — `build_page_jobs` →
    /// `encode_page` (here serial) → `assemble_row_group` → `build_file` — and
    /// return the bytes (a single-row-group file).
    fn encode(batch: &RecordBatch) -> Vec<u8> {
        let pages: Vec<EncodedPage> = build_page_jobs(std::slice::from_ref(batch))
            .into_iter()
            .map(encode_page)
            .collect::<Result<_, _>>()
            .unwrap();
        let group = assemble_row_group(&batch.schema(), pages).unwrap();
        build_file(&batch.schema(), vec![group]).unwrap()
    }

    /// The written file is spec-compliant enough for arrow-rs's strict reader
    /// (our DuckDB-readability proxy): it reads back with the original values.
    #[test]
    fn output_is_readable_by_a_strict_parquet_reader() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

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

        let bytes = encode(&batch);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        std::fs::write(&path, &bytes).unwrap();

        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let read: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();

        assert_eq!(read.len(), 1);
        let got = &read[0];
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
}
