mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::*;
use dispatch::Projection;
use catalog::parquet::{ParquetTable, table_input};

#[test]
fn scan_all_columns() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(
        &dispatch,
        &[strings_and_ints(
            &["a", "b", "c", "d", "e"],
            &[1, 2, 3, 4, 5],
        )],
        true,
    );

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert_eq!(results[0].num_columns(), 2);
}

#[test]
fn scan_column_subset() {
    let dispatch = dispatch(1);
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
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);

    let results = table_input(&dispatch, &table, Projection::columns([1]), false)
        .collect()
        .unwrap();

    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn scan_multiple_parquet_files() {
    let dispatch = dispatch(1);
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
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .count()
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn materialize_rejects_corrupt_footer_without_panicking() {
    // A file whose trailing `[footer_len][PAR1]` claims a footer larger than the
    // whole file. The footer reader must surface a clean error, not underflow the
    // offset math (`size - 8 - footer_len`) into a wild cache read.
    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corrupt.parquet");
    // 16 bytes: 8 filler + footer_len = u32::MAX + "PAR1".
    let mut bytes = vec![0u8; 8];
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(b"PAR1");
    std::fs::write(&path, &bytes).unwrap();

    let result = ParquetTable::from_files(&dispatch, &[&path]);
    assert!(
        result.is_err(),
        "a footer longer than the file must error, not panic"
    );
}

#[test]
fn scan_empty_table() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[strings_and_ints(&[], &[])], true);

    let results = table_input(&dispatch, &table, Projection::all(0), false)
        .count()
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 0);
}
