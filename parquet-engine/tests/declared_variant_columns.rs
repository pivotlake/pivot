//! A variant column declared the way a table format that resolves columns by
//! id declares one, nullable and under a Parquet field id: written through the
//! whole write pipeline and read back with arrow-rs's strict reader, the
//! oracle for "another engine can read this".

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{ArrowError, Schema};
use dispatch::{BUFFER_SIZE, Dispatch, values_input};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet_engine::PARQUET_FIELD_ID_METADATA_KEY;
use parquet_engine::writing::{
    AssembledFile, Partitioning, encode_record_batches_spec, unshred_batches_spec,
};
use parquet_variant_compute::json_to_variant;

/// A 64 MiB file-cache ring.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

/// The column name every test writes its documents under.
const COLUMN: &str = "attrs";

/// The Parquet field id [`DeclaredJsonItem`] declares its column under.
const DECLARED_FIELD_ID: i32 = 7;

/// Documents under a column declared the way a table format that resolves
/// columns by id declares one: nullable, under a Parquet field id. A row
/// without a document holds NULL.
struct DeclaredJsonItem(Vec<Option<&'static str>>);

impl DeclaredJsonItem {
    fn into_batch(self) -> Result<RecordBatch, ArrowError> {
        let json: ArrayRef = Arc::new(StringArray::from(self.0));
        let variants = json_to_variant(&json)?;
        let field = variants.field(COLUMN).with_nullable(true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            PARQUET_FIELD_ID_METADATA_KEY.to_string(),
            DECLARED_FIELD_ID.to_string(),
        );
        let schema = Schema::new(vec![field.with_metadata(metadata)]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(variants.into_inner())])
    }
}

/// Run the write pipeline over `items` (one unpartitioned, unsorted stream,
/// so one output file) and return the finished files' bytes.
/// `rows_per_group` caps a row group's rows.
fn write(items: Vec<DeclaredJsonItem>, rows_per_group: usize) -> Vec<Vec<u8>> {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let batches: Vec<RecordBatch> = items
        .into_iter()
        .map(|item| item.into_batch().unwrap())
        .collect();
    let schema = batches[0].schema();
    let spec = unshred_batches_spec(values_input(dispatch.dispatcher(), batches).record_batches());
    // Copy worker-owned ring memory before returning the bytes to the test
    // thread.
    let files: Vec<Vec<u8>> = encode_record_batches_spec(
        spec,
        schema,
        Partitioning::by_columns(&[]),
        Arc::from([]),
        rows_per_group,
        usize::MAX,
    )
    .map_each(|file: AssembledFile| Ok(file.bytes.runs().flatten().copied().collect::<Vec<u8>>()))
    .collect()
    .unwrap();
    dispatch.exit();
    files
}

/// Read a file back through arrow-rs's strict reader, as another engine would.
fn read_back(bytes: &[u8]) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|batch| batch.unwrap()).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// A file's leaf column paths, dotted: the shape the writer stamped into the
/// footer.
fn leaf_paths(bytes: &[u8]) -> Vec<String> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|column| column.path().string())
        .collect()
}

#[test]
fn a_shredded_column_keeps_the_field_id_it_was_declared_under() {
    let documents = vec![Some(r#"{"id": 1}"#), Some(r#"{"id": 2}"#)];

    let files = write(vec![DeclaredJsonItem(documents)], 128 * 1024);

    let footer = SerializedFileReader::new(bytes::Bytes::copy_from_slice(&files[0])).unwrap();
    let schema = footer.metadata().file_metadata().schema_descr();
    let column = schema.root_schema().get_fields()[0].get_basic_info();
    assert!(leaf_paths(&files[0]).contains(&"attrs.typed_value.id.typed_value".to_string()));
    assert_eq!(
        column.has_id().then(|| column.id()),
        Some(DECLARED_FIELD_ID)
    );
}

#[test]
fn a_null_document_among_shredded_ones_reads_back_null() {
    let documents = vec![Some(r#"{"id": 1}"#), None, Some(r#"{"id": 3}"#)];

    let files = write(vec![DeclaredJsonItem(documents)], 128 * 1024);

    let batch = read_back(&files[0]);
    let is_null: Vec<bool> = (0..3).map(|row| batch.column(0).is_null(row)).collect();
    assert!(leaf_paths(&files[0]).contains(&"attrs.typed_value.id.typed_value".to_string()));
    assert_eq!(is_null, [false, true, false]);
}
