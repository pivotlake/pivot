mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Int64Array, RecordBatch, StringViewArray, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::parquet::{ParquetTable, table_input};
use common::*;
use dispatch::{AggregationKind, AggregationSlot, Projection};

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

// Regression: a projection with no data columns must still emit one row per
// table row. The column-driven page pipeline produces nothing for zero columns
// (no pages → no decoder), so a bare empty projection silently returned zero
// rows; it now routes to the dedicated empty-projection scan, and the row count
// survives CopyOut (which feeds `collect`).
#[test]
fn scan_empty_projection_emits_row_count() {
    let dispatch = dispatch(1);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);

    let results = table_input(&dispatch, &table, Projection::all(0), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert!(results.iter().all(|b| b.num_columns() == 0));
}

#[test]
fn scan_nested_struct_column() {
    let dispatch = dispatch(1);
    // A struct column `user{id, name}` alongside a scalar `ts`, written by
    // arrow-rs (a trusted nested writer) and read back through pivot.
    let inner = Fields::from(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
    ]);
    let user = StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as _,
            Arc::new(StringViewArray::from(vec!["a", "bb", "ccc"])) as _,
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("user", DataType::Struct(inner), false),
        Field::new("ts", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(user) as _,
            Arc::new(Int64Array::from(vec![10, 20, 30])) as _,
        ],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    // Three leaves (`user.id`, `user.name`, `ts`) reassemble into two columns.
    let results = table_input(&dispatch, &table, Projection::all(3), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let got = &results[0];
    assert_eq!(got.num_columns(), 2);
    let user = got.column(0).as_struct();
    assert_eq!(
        user.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![1, 2, 3])
    );
    assert_eq!(
        user.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["a", "bb", "ccc"])
    );
    assert_eq!(
        got.column(1).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20, 30])
    );
}

/// A batch laid out `a, s{x, y}, b`: four leaves where a struct sits between
/// scalars, so leaf (column-chunk) indices and top-level field indices diverge.
fn struct_between_scalars_batch() -> RecordBatch {
    let inner = Fields::from(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Utf8View, false),
    ]);
    let s = StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int64Array::from(vec![10, 20])) as _,
            Arc::new(StringViewArray::from(vec!["p", "q"])) as _,
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("s", DataType::Struct(inner), false),
        Field::new("b", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as _,
            Arc::new(s) as _,
            Arc::new(Int64Array::from(vec![100, 200])) as _,
        ],
    )
    .unwrap()
}

#[test]
fn scan_struct_between_scalars_keeps_leaf_order() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    let results = table_input(&dispatch, &table, Projection::all(4), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 3);
    let s = got.column(1).as_struct();
    assert_eq!(
        got.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![1, 2])
    );
    assert_eq!(
        s.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20])
    );
    assert_eq!(
        s.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["p", "q"])
    );
    assert_eq!(
        got.column(2).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![100, 200])
    );
}

#[test]
fn scan_scalar_after_struct() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    // `b` is top-level field 2 but leaf 3; projecting its leaf must decode
    // that chunk, not the struct's second leaf.
    let results = table_input(&dispatch, &table, Projection::columns([3]), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 1);
    assert_eq!(got.schema().field(0).name(), "b");
    assert_eq!(collect_i64s(&results, 0), vec![100, 200]);
}

#[test]
fn scan_struct_only() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    // The struct's two leaves project as one nested column.
    let results = table_input(&dispatch, &table, Projection::columns([1, 2]), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 1);
    let s = got.column(0).as_struct();
    assert_eq!(
        s.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20])
    );
    assert_eq!(
        s.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["p", "q"])
    );
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
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
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
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 0);
}
