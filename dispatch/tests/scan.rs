mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::*;
use dispatch::{ParquetTable, Projection, table_input};

#[test]
fn scan_all_columns() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(
        &["a", "b", "c", "d", "e"],
        &[1, 2, 3, 4, 5],
    )]);

    let results = table_input(&table, Projection::all(2), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert_eq!(results[0].num_columns(), 2);
}

#[test]
fn scan_column_subset() {
    init();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&[batch]);

    let results = table_input(&table, Projection::columns([1]), false)
        .collect()
        .unwrap();

    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn scan_multiple_parquet_files() {
    init();
    let dir = TempDir::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    for (i, batch) in [
        strings_and_ints(&["a", "b"], &[1, 2]),
        strings_and_ints(&["c", "d", "e"], &[3, 4, 5]),
    ]
    .iter()
    .enumerate()
    {
        let file = std::fs::File::create(dir.path().join(format!("part{i}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), Some(props.clone())).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
    }
    let table = Arc::new(ParquetTable::from_directory(dir.path()).unwrap());

    let results = table_input(&table, Projection::all(2), false)
        .count()
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn scan_empty_table() {
    init();
    let (_dir, table) = parquet_table(&[strings_and_ints(&[], &[])]);

    let results = table_input(&table, Projection::all(0), false)
        .count()
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 0);
}
