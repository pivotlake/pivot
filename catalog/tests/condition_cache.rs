//! End-to-end condition-cache behavior: the first run of a filtered query
//! publishes each fully-scanned row group's surviving positions, and repeat
//! runs answer row groups from the cache (reading only the surviving rows,
//! skipping empty row groups outright) with identical results.

mod common;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use catalog::store::ObjectPath;
use catalog::{ParquetCatalog, QueryConditionCache};
use planner::Planner;
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest};
use planner::types::Type;

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 128, None))
        .dispatcher()
        .clone()
}

/// Write `rows` as a parquet file with one row per row group, so per-row-group
/// cache entries (including empty ones) are observable.
fn write_one_row_per_group(dir: &Path, file_name: &str, rows: &[(i64, &str)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let file = File::create(dir.join(file_name)).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
    for (id, name) in rows {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![*id])),
                Arc::new(StringViewArray::from(vec![*name])),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

/// A catalog with one table `t` over `rows` (one row group each), swapping in
/// `cache` as its condition cache.
fn catalog_with_table(
    dir: &TempDir,
    rows: &[(i64, &str)],
    cache: Arc<QueryConditionCache>,
) -> Arc<ParquetCatalog> {
    write_one_row_per_group(dir.path(), "data.parquet", rows);
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()).with_condition_cache(cache));
    let mut options = HashMap::new();
    options.insert(
        "path".to_string(),
        dir.path().to_string_lossy().into_owned(),
    );
    let request = CreateTableRequest {
        name: "t".to_string(),
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
    catalog
}

fn enabled_cache() -> Arc<QueryConditionCache> {
    Arc::new(QueryConditionCache::new(true, 1 << 20))
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

/// Flatten and sort the `id` column across the result batches.
fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for batch in batches {
        let idx = batch.schema().index_of("id").unwrap();
        let column = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        out.extend(column.iter().flatten());
    }
    out.sort_unstable();
    out
}

#[test]
fn repeated_contains_filter_hits_the_cache_with_equal_results() {
    let dir = TempDir::new().unwrap();
    let cache = enabled_cache();
    let rows = [(1, "alice"), (2, "bob"), (3, "charlie")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());

    let first = run_sql(&catalog, "SELECT id FROM t WHERE name LIKE '%ob%'");
    let inserts_after_first = cache.stats().inserts;
    let second = run_sql(&catalog, "SELECT id FROM t WHERE name LIKE '%ob%'");

    assert_eq!(ids(&first), vec![2]);
    assert_eq!(ids(&second), vec![2]);
    // Every row group published on the first run (including the two with no
    // survivors) and answered from the cache on the second.
    assert_eq!(inserts_after_first, 3);
    assert_eq!(cache.stats().hits, 3);
}

#[test]
fn pushed_and_residual_conditions_cache_as_one_conjunction() {
    let dir = TempDir::new().unwrap();
    let cache = enabled_cache();
    let rows = [(1, "alice"), (2, "bob"), (3, "charlie")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());

    let sql = "SELECT id FROM t WHERE contains(name, 'l') AND id <> 3";
    let first = run_sql(&catalog, sql);
    let second = run_sql(&catalog, sql);

    assert_eq!(ids(&first), vec![1]);
    assert_eq!(ids(&second), vec![1]);
    assert!(cache.stats().hits > 0);
}

#[test]
fn a_different_constant_does_not_reuse_the_entry() {
    let dir = TempDir::new().unwrap();
    let cache = enabled_cache();
    let rows = [(1, "alice"), (2, "bob"), (3, "bobby")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());

    let bob = run_sql(&catalog, "SELECT id FROM t WHERE contains(name, 'bob')");
    let alice = run_sql(&catalog, "SELECT id FROM t WHERE contains(name, 'alice')");

    assert_eq!(ids(&bob), vec![2, 3]);
    assert_eq!(ids(&alice), vec![1]);
    assert_eq!(cache.stats().hits, 0);
}

#[test]
fn a_file_appended_between_runs_is_scanned_and_then_cached() {
    let dir = TempDir::new().unwrap();
    let cache = enabled_cache();
    let rows = [(1, "alice"), (2, "bob")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());
    let sql = "SELECT id FROM t WHERE contains(name, 'b')";

    let first = run_sql(&catalog, sql);
    let extra = TempDir::new().unwrap();
    write_one_row_per_group(extra.path(), "extra.parquet", &[(4, "abel"), (5, "carol")]);
    append(&catalog, "t", &extra.path().join("extra.parquet"));
    let second = run_sql(&catalog, sql);
    let third = run_sql(&catalog, sql);

    assert_eq!(ids(&first), vec![2]);
    // The new file's row groups miss (fresh keys) while the old ones hit; the
    // union is complete either way.
    assert_eq!(ids(&second), vec![2, 4]);
    assert_eq!(ids(&third), vec![2, 4]);
}

#[test]
fn select_star_with_order_by_limit_repeats_correctly() {
    let dir = TempDir::new().unwrap();
    let cache = enabled_cache();
    let rows = [(1, "alice"), (2, "bob"), (3, "abigail"), (4, "carol")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());
    let sql = "SELECT * FROM t WHERE contains(name, 'a') ORDER BY id LIMIT 2";

    let first = run_sql(&catalog, sql);
    let second = run_sql(&catalog, sql);

    assert_eq!(ids(&first), vec![1, 3]);
    assert_eq!(ids(&second), vec![1, 3]);
}

#[test]
fn disabled_cache_changes_nothing_and_records_nothing() {
    let dir = TempDir::new().unwrap();
    let cache = Arc::new(QueryConditionCache::new(false, 1 << 20));
    let rows = [(1, "alice"), (2, "bob")];
    let catalog = catalog_with_table(&dir, &rows, cache.clone());
    let sql = "SELECT id FROM t WHERE contains(name, 'b')";

    let first = run_sql(&catalog, sql);
    let second = run_sql(&catalog, sql);

    assert_eq!(ids(&first), vec![2]);
    assert_eq!(ids(&second), vec![2]);
    assert_eq!(cache.stats().inserts, 0);
    assert_eq!(cache.stats().hits, 0);
}

/// Commit an already-written parquet file into the table, as ingest would.
fn append(catalog: &ParquetCatalog, name: &str, path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let relative = ObjectPath::new(path.file_name().unwrap().to_string_lossy());
    catalog
        .table_handle(name)
        .expect("table exists")
        .append_data_file(relative, &bytes, None, None)
        .unwrap()
}
