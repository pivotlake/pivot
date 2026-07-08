//! Blackbox compaction tests: land small files through the catalog's `INSERT`
//! path, run the [`Compacter`], and assert the merged file replaces them in
//! the catalog with its rows, partition tuple, and sort bounds intact.

use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use compact::{Compacter, DEFAULT_MIN_FILES_TO_MERGE};
use dispatch::{DataFlowDispatcher, Dispatch, Projection, values_input};
use tempfile::TempDir;

use catalog::ParquetCatalog;
use catalog::parquet::table_input;
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest};
use planner::types::Type;

/// A shared dispatch pool for the whole test binary. Oversized ring: the fused
/// scan→encode dataflow keeps decode and encode in flight together, so
/// decompressed pages queue while workers encode.
fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(2, 256, None))
        .dispatcher()
        .clone()
}

/// `CREATE TABLE <name> (id BIGINT, name VARCHAR)` at `dir` (when given, the
/// way the server would run it with an explicit path; otherwise store-relative
/// under the database root), plus any extra `WITH` options.
fn create_table(
    catalog: &Arc<ParquetCatalog>,
    name: &str,
    dir: Option<&Path>,
    options: &[(&str, &str)],
) {
    let mut opts = std::collections::HashMap::new();
    if let Some(dir) = dir {
        opts.insert("path".to_string(), dir.to_str().unwrap().to_string());
    }
    for (key, value) in options {
        opts.insert(key.to_string(), value.to_string());
    }
    let request = CreateTableRequest {
        name: name.to_string(),
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
        options: opts,
        if_not_exists: false,
    };
    catalog
        .create_table(request, &dispatcher())
        .unwrap()
        .execute()
        .collect()
        .unwrap();
}

/// Land one `(id, name)` batch as its own committed Parquet file (one insert =
/// one file for these row counts).
fn insert_rows(catalog: &Arc<ParquetCatalog>, table: &str, ids: Vec<i64>, name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let names: Vec<&str> = std::iter::repeat_n(name, ids.len()).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
        ],
    )
    .unwrap();
    let source = values_input(&dispatcher(), vec![batch]).record_batches();
    let ctx = catalog.query_context();
    catalog
        .table(table)
        .unwrap()
        .insert(source, ctx.as_ref())
        .unwrap()
        .collect()
        .unwrap();
}

/// Run one full compaction round sized so the table's current files all merge.
fn compact_all_files(catalog: &Arc<ParquetCatalog>, table: &str) {
    let total: u64 = catalog
        .table_files(table)
        .unwrap()
        .iter()
        .map(|f| f.size)
        .sum();
    let compacter = Compacter::new(
        total,
        DEFAULT_MIN_FILES_TO_MERGE,
        Duration::from_secs(1),
        catalog.clone(),
    );
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(compacter.compact_all());
}

/// The table's current committed rows, decoded through the engine's scan.
fn read_rows(catalog: &Arc<ParquetCatalog>, name: &str) -> Vec<(i64, String)> {
    let mut table = catalog.table_handle(name).unwrap();
    table.refresh().unwrap();
    let parquet = table.parquet(&[]).unwrap();
    let batches = table_input(&dispatcher(), &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), names.value(i).to_string()));
        }
    }
    rows.sort();
    rows
}

/// End-to-end compaction: several inserted small files merge into one (a
/// single scan→encode dataflow on the dispatch pool), the catalog swaps to the
/// merged file in one log commit, and the merged file decodes back to all the
/// original rows.
#[test]
fn compaction_merges_inserted_files_and_swaps_catalog() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, "events", Some(dir.path()), &[]);
    insert_rows(&catalog, "events", vec![1, 2, 3], "svc");
    insert_rows(&catalog, "events", vec![4, 5], "svc");
    insert_rows(&catalog, "events", vec![6, 7, 8, 9], "svc");
    assert_eq!(catalog.table_files("events").unwrap().len(), 3);

    compact_all_files(&catalog, "events");

    // The merge is a 3→1 swap in the catalog; the inputs linger on disk as
    // deferred-deletion orphans until their version is pruned.
    let files = catalog.table_files("events").unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].path.as_str().contains("compacted"));
    assert_eq!(
        read_rows(&catalog, "events"),
        (1..=9)
            .map(|id| (id, "svc".to_string()))
            .collect::<Vec<_>>(),
    );
}

/// Compacting a partitioned + sorted table preserves each merged file's
/// partition tuple and recomputes its sort bounds - it merges within one
/// partition and re-applies the table's spec, rather than dropping the
/// metadata.
#[test]
fn compaction_preserves_partition_and_sort_metadata() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(
        &catalog,
        "partitioned",
        Some(dir.path()),
        &[("partition_by", "name"), ("sort_by", "id")],
    );
    insert_rows(&catalog, "partitioned", vec![3, 1], "svc");
    insert_rows(&catalog, "partitioned", vec![2], "svc");
    insert_rows(&catalog, "partitioned", vec![5, 4], "svc");
    assert_eq!(catalog.table_files("partitioned").unwrap().len(), 3);

    compact_all_files(&catalog, "partitioned");

    assert_eq!(catalog.table_files("partitioned").unwrap().len(), 1);
    let mut table = catalog.table_handle("partitioned").unwrap();
    table.refresh().unwrap();
    let partitions = table.file_partitions();
    assert_eq!(partitions.len(), 1);
    assert_eq!(
        partitions[0]
            .1
            .as_ref()
            .expect("merged file keeps a partition tuple")["name"],
        "svc"
    );
    assert!(
        table.file_sort_bounds()[0].1.is_some(),
        "merged file keeps recomputed sort bounds"
    );
}

/// Compaction is location-agnostic: a **store-relative** table (data under the
/// database root, resolved through the store - the same path a remote `s3://`
/// root takes) compacts through the exact same code, with writes and deletes
/// going through the table's store handle.
#[test]
fn compaction_works_on_store_relative_tables() {
    let db = TempDir::new().unwrap();
    let catalog =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    // No `path` option: the table lives at `relative_events` under the root.
    create_table(&catalog, "relative_events", None, &[]);
    insert_rows(&catalog, "relative_events", vec![1, 2], "svc");
    insert_rows(&catalog, "relative_events", vec![3], "svc");

    compact_all_files(&catalog, "relative_events");

    // One merged object replaced the two inputs, in the store and the catalog.
    let files = catalog.table_files("relative_events").unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].path.as_str().contains("compacted"));
    // Deferred deletion: the merged object is written to the store; the inputs
    // remain on disk as orphans until their version is pruned.
    let on_disk: Vec<String> = std::fs::read_dir(db.path().join("relative_events"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".parquet"))
        .collect();
    assert!(on_disk.contains(&files[0].path.as_str().to_string()));
    assert_eq!(
        read_rows(&catalog, "relative_events"),
        vec![
            (1, "svc".to_string()),
            (2, "svc".to_string()),
            (3, "svc".to_string()),
        ],
    );
}

/// The compacter can live in a **separate process**: it holds nothing but a
/// catalog handle, and each poll round reloads the table from its log. Here
/// the "server" inserts through one catalog instance while the compacter works
/// through a second instance over the same database root - and the server sees
/// the swap at its next bind.
#[test]
fn compacter_in_another_process_compacts_the_servers_inserts() {
    let db = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    let server_catalog =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    create_table(&server_catalog, "shared_events", Some(data_dir.path()), &[]);
    insert_rows(&server_catalog, "shared_events", vec![1, 2, 3], "svc");
    insert_rows(&server_catalog, "shared_events", vec![4, 5], "svc");
    insert_rows(&server_catalog, "shared_events", vec![6, 7, 8, 9], "svc");

    // "Compacter" process: a separate catalog over the same root. Its poll
    // round reloads the table from the log before scanning.
    let compacter_catalog =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    compact_all_files(&compacter_catalog, "shared_events");

    // The server's next query reloads to the compacted version.
    let files = server_catalog.table_files("shared_events").unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].path.as_str().contains("compacted"));
    assert_eq!(read_rows(&server_catalog, "shared_events").len(), 9);
}
