//! Blackbox tests for the catalog's `INSERT` path: resolving a table and
//! compiling an insert through `Table::insert` yields a dataflow that runs a
//! `RecordBatch` source through the write pipeline, lands committed Parquet
//! files, and emits the row count — with the rows readable through the
//! engine's own scan the moment the dataflow finishes.

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch, Projection, values_input};
use tempfile::TempDir;

use catalog::ParquetCatalog;
use catalog::parquet::table_input;
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest};
use planner::types::Type;

/// A shared single-worker dispatch pool for the whole test binary; inserts
/// encode on it and reads decode on it.
fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 128, None))
        .dispatcher()
        .clone()
}

/// `CREATE TABLE <name> (id BIGINT, name VARCHAR) WITH (path = dir, ...)`, the
/// way the server would run it.
fn create_table(
    catalog: &Arc<ParquetCatalog>,
    name: &str,
    dir: &TempDir,
    options: &[(&str, &str)],
) {
    let mut opts = std::collections::HashMap::from([(
        "path".to_string(),
        dir.path().to_str().unwrap().to_string(),
    )]);
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

/// Run `batches` through an insert as one statement, the way the compiled
/// route does: resolve the table, compile the insert dataflow, execute it, and
/// read the row count off its single output batch.
fn insert(catalog: &Arc<ParquetCatalog>, name: &str, batches: Vec<RecordBatch>) -> u64 {
    let source = values_input(&dispatcher(), batches).record_batches();
    let table = catalog.table(name).unwrap();
    let ctx = catalog.query_context();
    let counts = table
        .insert(source, ctx.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    counts
        .iter()
        .map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0)
        })
        .sum()
}

/// An `(id, name)` batch. Names are plain `Utf8` on purpose: the insert path
/// must conform them to the table's physical `Utf8View`.
fn batch(ids: Vec<i64>, names: Vec<&str>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
        ],
    )
    .unwrap()
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

#[test]
fn insert_commits_rows_readable_on_return() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, "events", &dir, &[]);

    let rows = insert(
        &catalog,
        "events",
        vec![batch(vec![1, 2, 3], vec!["a", "b", "c"])],
    );

    assert_eq!(rows, 3);
    assert_eq!(catalog.table_files("events").unwrap().len(), 1);
    assert_eq!(
        read_rows(&catalog, "events"),
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string()),
        ],
    );
}

#[test]
fn inserts_accumulate_across_statements() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, "accumulating", &dir, &[]);

    insert(&catalog, "accumulating", vec![batch(vec![1], vec!["a"])]);
    insert(&catalog, "accumulating", vec![batch(vec![2], vec!["b"])]);

    assert_eq!(catalog.table_files("accumulating").unwrap().len(), 2);
    assert_eq!(
        read_rows(&catalog, "accumulating"),
        vec![(1, "a".to_string()), (2, "b".to_string())],
    );
}

#[test]
fn insert_into_partitioned_sorted_table_records_metadata() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(
        &catalog,
        "partitioned",
        &dir,
        &[("partition_by", "name"), ("sort_by", "id")],
    );

    insert(
        &catalog,
        "partitioned",
        vec![batch(vec![5, 1, 3], vec!["svc", "web", "svc"])],
    );

    // One file per partition, each stamped with its tuple and sort bounds.
    let mut table = catalog.table_handle("partitioned").unwrap();
    table.refresh().unwrap();
    let mut partitions: Vec<String> = table
        .file_partitions()
        .into_iter()
        .map(|(_, tuple)| tuple.expect("inserted file keeps a partition tuple")["name"].to_string())
        .collect();
    partitions.sort();
    assert_eq!(partitions, vec!["\"svc\"", "\"web\""]);
    assert!(
        table
            .file_sort_bounds()
            .iter()
            .all(|(_, bounds)| bounds.is_some()),
        "inserted files keep recomputed sort bounds"
    );
    assert_eq!(
        read_rows(&catalog, "partitioned"),
        vec![
            (1, "web".to_string()),
            (3, "svc".to_string()),
            (5, "svc".to_string()),
        ],
    );
}

#[test]
fn empty_insert_writes_nothing() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, "untouched", &dir, &[]);

    let rows = insert(&catalog, "untouched", vec![]);

    assert_eq!(rows, 0);
    assert!(catalog.table_files("untouched").unwrap().is_empty());
}

#[test]
fn insert_into_missing_table_has_no_table_to_compile_against() {
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));

    let resolved = catalog.table("missing");

    assert!(resolved.is_none());
}
