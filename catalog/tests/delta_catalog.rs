//! The catalog over tables whose durable format is a Delta Lake transaction
//! log: opening a database that contains one, and the background sync
//! (`refresh_catalog`) materializing externally written delta commits into the
//! in-memory table set.

use std::fs::File;
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use catalog::ParquetCatalog;
use catalog::store::ObjectPath;
use deltalake_core::kernel::transaction::CommitBuilder;
use deltalake_core::kernel::{Action, Add, DataType as DeltaType, PrimitiveType, StructField};
use deltalake_core::operations::create::CreateBuilder;
use deltalake_core::protocol::{DeltaOperation, SaveMode};
use planner::catalog::Catalog as PlannerCatalog;
use planner::types::Type;

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 64, None))
        .dispatcher()
        .clone()
}

/// The tokio runtime the test's own delta writes run on (the external Delta
/// writer's role; the catalog under test brings its own).
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().unwrap())
}

/// Write a one-column parquet file (`id` Int64, three rows) at
/// `<table_dir>/<name>` and return its size in bytes.
fn write_parquet(table_dir: &std::path::Path, name: &str) -> u64 {
    std::fs::create_dir_all(table_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
    )
    .unwrap();
    let path = table_dir.join(name);
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    std::fs::metadata(&path).unwrap().len()
}

/// The `add` action an external delta writer records for one committed file:
/// a `part` partition value and `id` min/max stats.
fn add_action(name: &str, size: u64, part: &str, modification_time: i64) -> Action {
    Action::Add(Add {
        path: name.to_string(),
        partition_values: [("part".to_string(), Some(part.to_string()))].into(),
        size: size as i64,
        modification_time,
        data_change: true,
        stats: Some(r#"{"minValues":{"id":1},"maxValues":{"id":3}}"#.to_string()),
        ..Default::default()
    })
}

/// Create the delta table at `table_dir` (schema `id` long + `part` string,
/// partitioned by `part`, sorted by `id`) committing `actions` as version 0.
fn create_delta_table(table_dir: &std::path::Path, actions: Vec<Action>) {
    let uri = url::Url::from_directory_path(table_dir)
        .unwrap()
        .to_string();
    runtime()
        .block_on(
            CreateBuilder::new()
                .with_location(uri)
                .with_columns(vec![
                    StructField::new("id", DeltaType::Primitive(PrimitiveType::Long), false),
                    StructField::new("part", DeltaType::Primitive(PrimitiveType::String), false),
                ])
                .with_partition_columns(["part"])
                .with_configuration([("pivot.sortBy".to_string(), Some(r#"["id"]"#.to_string()))])
                .with_raise_if_key_not_exists(false)
                .with_save_mode(SaveMode::ErrorIfExists)
                .with_actions(actions)
                .into_future(),
        )
        .unwrap();
}

/// Commit `actions` as the delta table's next version, as a concurrent
/// external writer would.
fn commit_to_delta_table(table_dir: &std::path::Path, actions: Vec<Action>) {
    let uri = url::Url::from_directory_path(table_dir).unwrap();
    runtime()
        .block_on(async move {
            let table = deltalake_core::table::builder::DeltaTableBuilder::from_url(uri)?
                .load()
                .await?;
            CommitBuilder::default()
                .with_actions(actions)
                .build(
                    Some(table.snapshot()?),
                    table.log_store(),
                    DeltaOperation::Write {
                        mode: SaveMode::Append,
                        partition_by: None,
                        predicate: None,
                    },
                )
                .await
        })
        .unwrap();
}

/// The database index naming `events` at the `events` location, as the
/// catalog's `_pivot_manifest.json` records it.
fn write_database_manifest(root: &std::path::Path) {
    let manifest = serde_json::json!({
        "version": 1,
        "tables": [{"name": "events", "location": "events"}],
    });
    std::fs::write(root.join("_pivot_manifest.json"), manifest.to_string()).unwrap();
}

/// Opening a database whose table location holds a delta log loads the
/// table's schema, specs, and file metadata from that log.
#[test]
fn open_loads_a_delta_table_from_its_log() {
    let root = TempDir::new().unwrap();
    let table_dir = root.path().join("events");
    let size = write_parquet(&table_dir, "a.parquet");
    create_delta_table(&table_dir, vec![add_action("a.parquet", size, "x", 1)]);
    write_database_manifest(root.path());

    let catalog = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();

    let table = catalog.table_handle("events").unwrap();
    let columns = table.columns();
    assert_eq!(columns.len(), 2);
    assert_eq!(
        (columns[0].name.as_str(), &columns[0].col_type),
        ("id", &Type::Int64)
    );
    assert_eq!(
        (columns[1].name.as_str(), &columns[1].col_type),
        ("part", &Type::Utf8)
    );
    assert_eq!(table.partition_by(), ["part"]);
    assert_eq!(table.sort_by(), ["id"]);
    let partitions = table.file_partitions();
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].0.as_str(), "a.parquet");
    assert_eq!(partitions[0].1, Some(serde_json::json!({"part": "x"})));
    assert!(
        table.file_sort_bounds()[0].1.is_none(),
        "sort bounds aren't decoded while the write path recording them is disabled"
    );
}

/// The background sync picks up a delta table registered after the database
/// was opened, and advances it as an external writer commits new versions.
#[test]
fn background_refresh_follows_external_delta_commits() {
    let root = TempDir::new().unwrap();
    let catalog = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();
    let table_dir = root.path().join("events");
    let size_a = write_parquet(&table_dir, "a.parquet");
    create_delta_table(&table_dir, vec![add_action("a.parquet", size_a, "x", 1)]);
    write_database_manifest(root.path());

    assert!(catalog.refresh_catalog().unwrap());

    let table = catalog.table_handle("events").unwrap();
    assert_eq!(table.version(), 0);
    assert_eq!(table.file_refs().len(), 1);

    let size_b = write_parquet(&table_dir, "b.parquet");
    commit_to_delta_table(&table_dir, vec![add_action("b.parquet", size_b, "y", 2)]);

    assert!(catalog.refresh_catalog().unwrap());

    let table = catalog.table_handle("events").unwrap();
    assert_eq!(table.version(), 1);
    let paths: Vec<String> = table
        .file_refs()
        .iter()
        .map(|f| f.path.as_str().to_string())
        .collect();
    assert_eq!(paths, ["a.parquet", "b.parquet"]);
    let refreshed_again = catalog.refresh_catalog().unwrap();
    assert!(!refreshed_again, "no new commit means no change");
}

/// CREATE TABLE persists the table as a delta table: its transaction log
/// lands under the table's location, the declared schema round-trips through
/// the exact pivot type in the field metadata, and the table reloads whole
/// from the log on a fresh open.
#[test]
fn create_table_commits_a_delta_log_that_reloads() {
    let root = TempDir::new().unwrap();
    let table_dir = root.path().join("events");
    write_parquet(&table_dir, "pivot.parquet");
    let catalog = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();
    catalog
        .create_table(
            planner::catalog::CreateTableRequest {
                name: "events".to_string(),
                columns: vec![planner::catalog::Column {
                    name: "id".to_string(),
                    col_type: Type::UInt64,
                }],
                options: Default::default(),
                if_not_exists: false,
            },
            &dispatcher(),
        )
        .unwrap()
        .execute()
        .collect()
        .unwrap();

    assert!(
        table_dir
            .join("_delta_log")
            .join("00000000000000000000.json")
            .is_file()
    );
    let reopened = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();
    let table = reopened.table_handle("events").unwrap();
    let columns = table.columns();
    assert_eq!(
        (columns[0].name.as_str(), &columns[0].col_type),
        ("id", &Type::UInt64),
        "the exact pivot type survives the delta round trip"
    );
    let paths: Vec<String> = table
        .file_refs()
        .iter()
        .map(|f| f.path.as_str().to_string())
        .collect();
    assert_eq!(paths, ["pivot.parquet"]);
}

/// Data writes are disabled while the write paths are reworked for the delta
/// format: an append errors explicitly and commits nothing.
#[test]
fn data_writes_are_disabled() {
    let root = TempDir::new().unwrap();
    let table_dir = root.path().join("events");
    let size = write_parquet(&table_dir, "a.parquet");
    create_delta_table(&table_dir, vec![add_action("a.parquet", size, "x", 1)]);
    write_database_manifest(root.path());
    let catalog = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();

    let mut table = catalog.table_handle("events").unwrap();
    let result = table.append_data_file(ObjectPath::new("b.parquet"), b"bytes", None, None);

    assert!(result.unwrap_err().to_string().contains("disabled"));
    assert_eq!(table.version(), 0, "nothing was committed");
}

/// A partition-equality filter builds a scan view holding only the files whose
/// recorded partition value can match; no filter keeps every file.
#[test]
fn partition_filter_builds_only_the_matching_partitions_files() {
    let root = TempDir::new().unwrap();
    let table_dir = root.path().join("events");
    let keep_size = write_parquet(&table_dir, "keep.parquet");
    let drop_size = write_parquet(&table_dir, "drop.parquet");
    create_delta_table(
        &table_dir,
        vec![
            add_action("keep.parquet", keep_size, "x", 1),
            add_action("drop.parquet", drop_size, "y", 2),
        ],
    );
    write_database_manifest(root.path());
    let catalog = ParquetCatalog::open(root.path().to_str().unwrap(), &dispatcher()).unwrap();

    let table = catalog.table_handle("events").unwrap();
    let filter = catalog::PartitionEqFilter {
        column: "part".to_string(),
        value: serde_json::json!("x"),
    };
    let kept = table
        .build_scan_view(std::slice::from_ref(&filter))
        .unwrap();
    let all = table.build_scan_view(&[]).unwrap();

    // Only keep.parquet's row group survives the filter; drop.parquet's
    // recorded `part` value can't match. No filter keeps both files.
    assert_eq!(kept.row_groups().len(), 1);
    assert_eq!(all.row_groups().len(), 2);
}
