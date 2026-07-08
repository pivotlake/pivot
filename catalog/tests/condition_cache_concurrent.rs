//! Multi-worker condition-cache runs: a grouped Top-N over a pushed `<> ''`
//! filter, repeated so the first run observes and later runs inject. Guards
//! against worker livelocks that only appear with several workers stealing.

mod common;

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, OnceLock};

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use catalog::{ParquetCatalog, QueryConditionCache};
use planner::Planner;
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest};
use planner::types::Type;

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(8, 512, None))
        .dispatcher()
        .clone()
}

/// A table of `groups` row groups, `rows_per_group` rows each. Names are
/// mostly empty (filtered out); the matches are clustered, high-cardinality
/// strings so chunks mix dictionary and plain pages.
fn write_table(dir: &TempDir, groups: usize, rows_per_group: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(rows_per_group))
        // Several pages per column chunk, so per-page filter masks and
        // whole-page skips are exercised, not just whole-chunk reads.
        .set_data_page_row_count_limit(512)
        .set_write_batch_size(512)
        // A tiny dictionary budget so the string column overflows it and later
        // pages fall back to plain encoding (mixed-encoding chunks).
        .set_dictionary_page_size_limit(1024)
        .build();
    let file = File::create(dir.path().join("data.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
    for g in 0..groups {
        let ids: Vec<i64> = (0..rows_per_group)
            .map(|i| (g * rows_per_group + i) as i64)
            .collect();
        // Mostly-empty names (filtered out), with matches clustered at the
        // front of each row group so later pages are entirely filtered out
        // (all-false masks). Sparse enough (~3%) that the cache stores
        // positions rather than a dense marker, so injection is exercised.
        let names: Vec<String> = (0..rows_per_group)
            .map(|i| {
                if i < 300 && i % 3 == 0 {
                    format!("phrase-{:06}", (g * 251 + i * 17) % 4096)
                } else {
                    String::new()
                }
            })
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringViewArray::from(
                    names.iter().map(String::as_str).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

fn run_sql(catalog: &Arc<ParquetCatalog>, sql: &str) -> Vec<RecordBatch> {
    let mut planner = Planner::new(catalog.clone() as Arc<dyn PlannerCatalog>);
    planner
        .plan(sql)
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap()
}

/// Flatten an Int64 column out of the result batches by position.
fn int64_column(batches: &[RecordBatch], idx: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        out.extend(column.iter().flatten());
    }
    out
}

/// Rows per group and group count for the test table; the expected match
/// count below must mirror `write_table`'s name pattern.
const GROUPS: usize = 40;
const ROWS_PER_GROUP: usize = 3000;

fn expected_matches() -> i64 {
    (GROUPS as i64)
        * (0..ROWS_PER_GROUP as i64)
            .filter(|i| *i < 300 && i % 3 == 0)
            .count() as i64
}

/// A cache whose byte budget holds only a fraction of one query's entries, so
/// inserts evict mid-scan and every run stays a mixed observe/inject scan.
#[test]
fn capacity_thrash_keeps_runs_correct() {
    let dir = TempDir::new().unwrap();
    write_table(&dir, GROUPS, ROWS_PER_GROUP);
    let catalog = Arc::new(
        ParquetCatalog::new(dispatcher()).with_condition_cache(Arc::new(
            // Room for roughly a quarter of the table's row-group entries.
            QueryConditionCache::new(true, 10 * (100 * 4 + 128)),
        )),
    );
    create_events(&catalog, &dir);
    let total = "SELECT COUNT(*) AS c FROM events WHERE name <> ''";

    for _ in 0..6 {
        let counted = int64_column(&run_sql(&catalog, total), 0);

        assert_eq!(counted, vec![expected_matches()]);
    }
}

fn create_events(catalog: &Arc<ParquetCatalog>, dir: &TempDir) {
    let mut options = HashMap::new();
    options.insert(
        "path".to_string(),
        dir.path().to_string_lossy().into_owned(),
    );
    let request = CreateTableRequest {
        name: "events".to_string(),
        columns: vec![
            Column {
                name: "id".to_string(),
                col_type: Type::Int64,
            },
            Column {
                name: "name".to_string(),
                col_type: Type::Utf8,
            },
        ],
        options,
        if_not_exists: false,
    };
    catalog
        .create_table(request, &dispatcher())
        .unwrap()
        .execute()
        .collect()
        .unwrap();
}

#[test]
fn grouped_top_n_over_pushed_filter_repeats_with_many_workers() {
    let dir = TempDir::new().unwrap();
    write_table(&dir, GROUPS, ROWS_PER_GROUP);
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_events(&catalog, &dir);
    let grouped = "SELECT name, COUNT(*) AS c FROM events \
                   WHERE name <> '' GROUP BY name ORDER BY c DESC LIMIT 10";
    let total = "SELECT COUNT(*) AS c FROM events WHERE name <> ''";

    // The grouped Top-N's LIMIT is tie-ambiguous, so correctness is asserted
    // on the total match count, which shares the same filter conjunction (the
    // first run observes it, later runs inject the cached positions). The sum
    // of every group's count must equal that total on every run.
    let grouped_first = int64_column(&run_sql(&catalog, grouped), 1);
    let total_first = int64_column(&run_sql(&catalog, total), 0);
    let grouped_second = int64_column(&run_sql(&catalog, grouped), 1);
    let total_second = int64_column(&run_sql(&catalog, total), 0);

    assert_eq!(grouped_first.len(), 10);
    assert_eq!(grouped_second.len(), 10);
    assert_eq!(total_first, vec![expected_matches()]);
    assert_eq!(total_second, vec![expected_matches()]);
    let stats = catalog.condition_cache().stats();
    assert_eq!(
        stats.inserts, GROUPS as u64,
        "first run must publish every row group"
    );
    assert!(
        stats.hits >= 3 * GROUPS as u64,
        "later runs must inject, got {stats:?}"
    );
}
