#![allow(dead_code)]

use std::sync::Once;

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use tempfile::TempDir;

use dispatch::ParquetTable;

static INIT: Once = Once::new();

pub fn init() {
    init_with_workers(core_affinity::get_core_ids().unwrap().len());
}

pub fn init_with_workers(num_workers: usize) {
    INIT.call_once(|| {
        dispatch::init(num_workers);
    });
}

pub fn parquet_table(batches: &[RecordBatch]) -> (TempDir, Arc<ParquetTable>) {
    parquet_table_with_opts(batches, true)
}

pub fn parquet_table_with_opts(
    batches: &[RecordBatch],
    dictionary: bool,
) -> (TempDir, Arc<ParquetTable>) {
    let dir = TempDir::new().unwrap();
    let schema = batches[0].schema();
    let path = dir.path().join("data.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(dictionary)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    let table = Arc::new(ParquetTable::from_directory(dir.path()).unwrap());
    (dir, table)
}

pub fn strings_and_ints(names: &[&str], values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringViewArray::from(names.to_vec())),
            Arc::new(Int64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

pub fn extract_count(batches: &[RecordBatch]) -> u64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

pub fn collect_strings(batches: &[RecordBatch], col: usize) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i).to_string())
        })
        .collect()
}

pub fn collect_i64s(batches: &[RecordBatch], col: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b.column(col).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}

pub fn collect_u64s(batches: &[RecordBatch], col: usize) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}
