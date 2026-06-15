//! Scan-side dynamic-filter pruning: a [`RowGroupFilter`] passed to
//! `table_input_with_filter` is consulted as each row group is stolen, and a
//! row group it rejects is never read. We make pruning observable by writing
//! one row group per three rows and asserting a rejected group's values are
//! absent from the scan output.

mod common;

use std::fs::File;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::*;
use dispatch::Projection;
use catalog::parquet::table_input_with_filter;
use catalog::parquet::{RowGroupFilter, RowGroupMetadata};

/// Write a single Int64 column with one row group per three rows, so each row
/// group carries distinct min/max statistics.
fn row_group_per_three(
    dispatch: &DispatchGuard,
    rows: &[i64],
) -> (TempDir, Arc<catalog::parquet::ParquetTable>) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(rows.to_vec()))],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(3))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let table = parquet_table_from_dir(dispatch, dir.path());
    (dir, table)
}

/// Read column 0's min statistic, if present.
fn min_i64(rg: &RowGroupMetadata, col: usize) -> Option<i64> {
    let min = rg.column_statistics(col)?.min.as_ref()?;
    let (arr, _) = arrow_array::Datum::get(min);
    let arr = arr.as_any().downcast_ref::<Int64Array>()?;
    (!arr.is_null(0)).then(|| arr.value(0))
}

#[test]
fn row_group_filter_skips_rejected_groups() {
    let dispatch = dispatch(1);
    // Three row groups: [100,101,102], [1,2,3], [200,201,202].
    let (_dir, table) = row_group_per_three(&dispatch, &[100, 101, 102, 1, 2, 3, 200, 201, 202]);

    // Keep only groups whose min is >= 100 — i.e. prune the middle [1,2,3] group.
    let filter: RowGroupFilter =
        Arc::new(|rg: &RowGroupMetadata| min_i64(rg, 0).unwrap_or(0) >= 100);

    let results =
        table_input_with_filter(&dispatch, &table, Projection::all(1), false, Some(filter))
            .collect()
            .unwrap();
    let mut vals = collect_i64s(&results, 0);
    vals.sort_unstable();

    assert_eq!(vals, vec![100, 101, 102, 200, 201, 202]);
}

#[test]
fn no_filter_reads_every_group() {
    let dispatch = dispatch(1);
    let (_dir, table) = row_group_per_three(&dispatch, &[100, 101, 102, 1, 2, 3, 200, 201, 202]);

    // Control: same scan with no filter reads all nine rows.
    let results = table_input_with_filter(&dispatch, &table, Projection::all(1), false, None)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 9);
}
