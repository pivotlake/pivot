//! End-to-end tests of the staged scan: a pushed
//! [`ScanFilter`](planner::catalog::ScanFilter) makes the scan fetch the
//! filter columns first and request the remaining columns only for row groups
//! with surviving rows. The contract under test: a staged scan followed by the
//! (unchanged) filter operator produces exactly the same rows as an unstaged
//! scan followed by that filter — across selectivities (nothing / some groups /
//! everything), null filter results, multi-page row groups, and the fallback
//! shapes where staging must not engage at all.

mod common;

use std::fs::File;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::*;
use dispatch::Projection;
use goose::parquet::table_input_with_filter_and_eq_predicates;
use planner::catalog::{ScanFilter, ScanFilterEval};

/// Schema used throughout: an Int64 filter column plus two payload columns,
/// so staging always has "remaining" columns whose fetch it can skip.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
        Field::new("val", DataType::Int64, false),
    ]))
}

/// One batch of `rows` consecutive ids starting at `first_id`.
fn batch(first_id: i64, rows: usize) -> RecordBatch {
    let ids: Vec<i64> = (first_id..first_id + rows as i64).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("name-{i}")).collect();
    let vals: Vec<i64> = ids.iter().map(|i| i * 100).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringViewArray::from(
                names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(vals)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Write `batches` so each becomes its own row group; `page_rows` bounds the
/// rows per data page (small values force multi-page chunks).
fn write_groups(batches: &[RecordBatch], page_rows: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let rows_per_group = batches[0].num_rows();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(rows_per_group))
        .set_data_page_row_count_limit(page_rows)
        .set_write_batch_size(page_rows)
        .build();
    let path = dir.path().join("data.parquet");
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        batches[0].schema(),
        Some(props),
    )
    .unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    dir
}

/// A `ScanFilter` over column 0 of the projection keeping ids in
/// `[lo, hi)` — the staged equivalent of `WHERE id >= lo AND id < hi`.
fn id_range_filter(lo: i64, hi: i64) -> ScanFilter {
    ScanFilter {
        columns: vec![0],
        evaluator: Arc::new(move || {
            Box::new(move |batch: &RecordBatch| {
                let ids = batch.column(0).as_primitive::<Int64Type>();
                BooleanArray::from_iter(
                    ids.iter()
                        .map(|v| Some(v.is_some_and(|v| v >= lo && v < hi))),
                )
            }) as ScanFilterEval
        }),
    }
}

/// The per-row predicate matching [`id_range_filter`], for the filter operator
/// above the scan and for computing expected rows.
fn id_in_range(lo: i64, hi: i64) -> impl Fn(i64) -> bool {
    move |id| id >= lo && id < hi
}

/// Run a full-projection scan (staged when `scan_filter` is `Some`) with the
/// `WHERE lo <= id < hi` filter operator above it, returning the surviving
/// `(id, name, val)` rows sorted by id.
fn scan_and_filter(
    workers: usize,
    dir: &TempDir,
    scan_filter: Option<ScanFilter>,
    lo: i64,
    hi: i64,
) -> Vec<(i64, String, i64)> {
    let dispatch = dispatch_with_buffers(workers, 64);
    let table = parquet_table_from_dir(&dispatch, dir.path());
    let pred = id_in_range(lo, hi);
    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        Projection::all(3),
        false,
        None,
        None,
        Arc::new(Vec::new()),
        scan_filter,
    )
    .filter(move || {
        let pred = id_in_range(lo, hi);
        move |batch: &RecordBatch| {
            let ids = batch.column(0).as_primitive::<Int64Type>();
            BooleanArray::from_iter(ids.iter().map(|v| Some(v.is_some_and(&pred))))
        }
    })
    .collect()
    .unwrap();
    let _ = pred;
    let mut rows: Vec<(i64, String, i64)> = results
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_primitive::<Int64Type>();
            let names = b.column(1).as_string_view();
            let vals = b.column(2).as_primitive::<Int64Type>();
            (0..b.num_rows())
                .map(|i| (ids.value(i), names.value(i).to_string(), vals.value(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    rows.sort();
    rows
}

/// Four 5-row row groups: ids 0..5, 5..10, 10..15, 15..20.
fn four_groups() -> TempDir {
    write_groups(
        &[batch(0, 5), batch(5, 5), batch(10, 5), batch(15, 5)],
        1024,
    )
}

/// Staged == unstaged when the filter kills some row groups entirely and
/// keeps parts of others (ids 3..12 span groups 0–2; group 3 has no survivor).
#[test]
fn staged_matches_unstaged_with_partial_groups() {
    let dir = four_groups();
    let staged = scan_and_filter(2, &dir, Some(id_range_filter(3, 12)), 3, 12);
    let unstaged = scan_and_filter(2, &dir, None, 3, 12);
    assert_eq!(staged, unstaged);
    assert_eq!(staged.len(), 9);
    assert_eq!(staged[0], (3, "name-3".to_string(), 300));
    assert_eq!(staged[8], (11, "name-11".to_string(), 1100));
}

/// A filter no row matches: every row group dies in phase A, nothing is
/// emitted, and the dataflow still terminates cleanly.
#[test]
fn staged_filter_matches_nothing() {
    let dir = four_groups();
    let staged = scan_and_filter(2, &dir, Some(id_range_filter(100, 200)), 100, 200);
    assert!(staged.is_empty());
}

/// A filter every row matches: phase B degrades to plain full-group reads and
/// the staged scan returns the whole table.
#[test]
fn staged_filter_matches_everything() {
    let dir = four_groups();
    let staged = scan_and_filter(2, &dir, Some(id_range_filter(-100, 100)), -100, 100);
    let unstaged = scan_and_filter(2, &dir, None, -100, 100);
    assert_eq!(staged, unstaged);
    assert_eq!(staged.len(), 20);
}

/// Multi-page row groups with survivors clustered in one page: phase B must
/// reassemble exactly the surviving rows (pages with no survivor are skipped
/// internally via their filter masks).
#[test]
fn staged_with_multiple_pages_per_group() {
    // Two 64-row groups, 8 rows per page; survivors are rows 40..44 of the
    // first group only.
    let dir = write_groups(&[batch(0, 64), batch(64, 64)], 8);
    let staged = scan_and_filter(2, &dir, Some(id_range_filter(40, 44)), 40, 44);
    let unstaged = scan_and_filter(2, &dir, None, 40, 44);
    assert_eq!(staged, unstaged);
    assert_eq!(staged.len(), 4);
}

/// Null filter results drop the row, exactly like the filter operator above.
#[test]
fn staged_filter_with_nulls_drops_rows() {
    let dir = four_groups();
    // Keep even ids; odd ids get a null verdict instead of `false`.
    let nully = ScanFilter {
        columns: vec![0],
        evaluator: Arc::new(|| {
            Box::new(|batch: &RecordBatch| {
                let ids = batch.column(0).as_primitive::<Int64Type>();
                BooleanArray::from_iter(ids.iter().map(|v| {
                    if v.unwrap() % 2 == 0 {
                        Some(true)
                    } else {
                        None
                    }
                }))
            }) as ScanFilterEval
        }),
    };
    let dispatch = dispatch_with_buffers(2, 64);
    let table = parquet_table_from_dir(&dispatch, dir.path());
    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        Projection::all(3),
        false,
        None,
        None,
        Arc::new(Vec::new()),
        Some(nully),
    )
    .collect()
    .unwrap();
    let mut ids: Vec<i64> = results
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect();
    ids.sort();
    assert_eq!(ids, (0..20).filter(|i| i % 2 == 0).collect::<Vec<i64>>());
}

/// An empty filter-column set can't stage (nothing to evaluate phase A on):
/// the scan must fall back to the unstaged pipeline and emit every row.
#[test]
fn empty_filter_columns_fall_back_to_unstaged() {
    let dir = four_groups();
    let no_columns = ScanFilter {
        columns: vec![],
        evaluator: Arc::new(|| {
            Box::new(|batch: &RecordBatch| BooleanArray::from(vec![false; batch.num_rows()]))
                as ScanFilterEval
        }),
    };
    let dispatch = dispatch_with_buffers(1, 64);
    let table = parquet_table_from_dir(&dispatch, dir.path());
    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        Projection::all(3),
        false,
        None,
        None,
        Arc::new(Vec::new()),
        Some(no_columns),
    )
    .collect()
    .unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 20);
}

/// A filter reading every projected column can't save any I/O: the scan must
/// fall back to the unstaged pipeline (its evaluator never runs, so all rows
/// are emitted).
#[test]
fn full_width_filter_falls_back_to_unstaged() {
    let dir = four_groups();
    let full_width = ScanFilter {
        columns: vec![0, 1, 2],
        evaluator: Arc::new(|| {
            Box::new(|batch: &RecordBatch| BooleanArray::from(vec![false; batch.num_rows()]))
                as ScanFilterEval
        }),
    };
    let dispatch = dispatch_with_buffers(1, 64);
    let table = parquet_table_from_dir(&dispatch, dir.path());
    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        Projection::all(3),
        false,
        None,
        None,
        Arc::new(Vec::new()),
        Some(full_width),
    )
    .collect()
    .unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 20);
}

/// A metadata-emitting (late materialization) scan never stages, even when a
/// scan filter is supplied: every row comes back, tagged with the row-group
/// metadata columns the materializer needs.
#[test]
fn metadata_emitting_scan_ignores_scan_filter() {
    let dir = four_groups();
    let dispatch = dispatch_with_buffers(1, 64);
    let table = parquet_table_from_dir(&dispatch, dir.path());
    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        Projection::columns([0]),
        true,
        None,
        None,
        Arc::new(Vec::new()),
        Some(id_range_filter(0, 1)),
    )
    .collect()
    .unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 20, "late-mat scan must not pre-filter");
    // id column + the appended row-group-idx / row-idx metadata pair.
    assert_eq!(results[0].num_columns(), 3);
    assert_eq!(
        dispatch::trailing_metadata_columns(results[0].schema_ref()),
        2
    );
}

/// Single worker, many groups: exercises sequential phase A → phase B
/// hand-off on one thread (no work stealing to mask ordering bugs).
#[test]
fn staged_single_worker() {
    let dir = four_groups();
    let staged = scan_and_filter(1, &dir, Some(id_range_filter(5, 10)), 5, 10);
    let unstaged = scan_and_filter(1, &dir, None, 5, 10);
    assert_eq!(staged, unstaged);
    assert_eq!(staged.len(), 5);
}
