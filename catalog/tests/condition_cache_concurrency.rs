//! Multi-worker regression for the condition cache's replay path: a selective
//! filter feeding a GROUP BY + ORDER BY ... LIMIT (the ClickBench q21/q22 shape).
//! Replay scans only the few row groups that have survivors, so across many
//! workers most get no work — the empty-partition case that stalls the pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::ParquetCatalog;
use dispatch::{DataFlowDispatcher, Dispatch};
use planner::Planner;
use planner::catalog::{Catalog, Column, CreateTableRequest};
use planner::types::Type;

/// Write a SNAPPY file of one string column `s` with `groups` row groups; only a
/// few contain `"google"`, the rest are all `"x"`. So a `LIKE '%google%'` filter
/// leaves survivors in just a handful of row groups.
fn write_table(dir: &std::path::Path, groups: usize, rows_per_group: usize) {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, false)]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(rows_per_group))
        // Small data pages → many pages per row group, like a real wide string
        // column, so a sparse filter leaves most pages with no survivors.
        .set_data_page_row_count_limit(512)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(dir.join("data.parquet")).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();
    let total = groups * rows_per_group;
    // High-cardinality (mostly unique) values like URLs, so the dictionary falls
    // back to PLAIN encoding across many pages; "google" appears in only a few
    // scattered rows.
    let owned: Vec<String> = (0..total)
        .map(|i| {
            if i % 7919 == 0 {
                format!("http://google.com/path/{i}")
            } else {
                format!("http://site-{i}.example.com/page/{}", i * 31 + 7)
            }
        })
        .collect();
    let values: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringViewArray::from(values))]).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn create_request(name: &str, path: &std::path::Path) -> CreateTableRequest {
    let mut options = HashMap::new();
    options.insert("path".to_string(), path.to_string_lossy().into_owned());
    CreateTableRequest {
        name: name.to_string(),
        columns: vec![Column {
            name: "s".to_string(),
            col_type: Type::Utf8,
        }],
        options,
        if_not_exists: false,
    }
}

fn run(catalog: &Arc<ParquetCatalog>, dispatcher: &DataFlowDispatcher, sql: &str) -> usize {
    let mut planner = Planner::new(catalog.clone() as Arc<dyn Catalog>);
    planner
        .plan(sql)
        .unwrap()
        .compile(dispatcher)
        .unwrap()
        .collect()
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

#[test]
fn replay_under_group_by_order_by_limit_across_workers_completes() {
    let dispatch = Dispatch::spin_up(8, 256, None);
    let d = dispatch.dispatcher().clone();

    let dir = TempDir::new().unwrap();
    write_table(dir.path(), 50, 4000); // 50 row groups, many pages each
    let mut catalog = ParquetCatalog::new(d.clone());
    catalog.set_condition_cache_enabled(true);
    let catalog = Arc::new(catalog);
    catalog
        .create_table(create_request("t", dir.path()), &d)
        .unwrap()
        .execute()
        .collect()
        .unwrap();

    let sql =
        "SELECT s, COUNT(*) AS c FROM t WHERE s LIKE '%google%' GROUP BY s ORDER BY c DESC LIMIT 10";

    let first = run(&catalog, &d, sql); // populate
    let replayed = run(&catalog, &d, sql); // replay must not stall

    assert!(first > 0);
    assert_eq!(first, replayed);
}
