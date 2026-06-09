//! End-to-end tests for a persisted database: a `ParquetCatalog` opened on a
//! directory keeps a durable table manifest, so `CREATE TABLE` survives a
//! restart (a fresh `open`). Uses a local directory — the same path an
//! object-store database would take, minus the network.

mod common;
use common::*;

use std::collections::HashMap;
use std::path::Path;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use dispatch::Projection;
use goose::ParquetCatalog;
use goose::parquet::table_input;
use planner::catalog::{Catalog, Column, CreateTableRequest, Result as CatalogResult};
use planner::types::Type;

fn write_parquet(path: &Path, batch: &arrow_array::RecordBatch) {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "value".to_string(),
            col_type: Type::Int64,
        },
    ]
}

/// A `CREATE TABLE … WITH (path = '<dir>')` request.
fn path_request(name: &str, path: &Path, columns: Vec<Column>) -> CreateTableRequest {
    let mut options = HashMap::new();
    options.insert("path".to_string(), path.to_string_lossy().into_owned());
    CreateTableRequest {
        name: name.to_string(),
        columns,
        options,
        if_not_exists: false,
    }
}

/// A `CREATE TABLE name (cols)` request with no explicit path (lives under the
/// database root).
fn rooted_request(name: &str, columns: Vec<Column>) -> CreateTableRequest {
    CreateTableRequest {
        name: name.to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    }
}

/// Mirror the server's `CREATE TABLE`: `create_table` reads the footers (on the
/// coordinator) and returns the plan that writes the table, which we execute.
fn create(
    dispatch: &DispatchGuard,
    catalog: &ParquetCatalog,
    request: CreateTableRequest,
) -> CatalogResult<()> {
    catalog
        .create_table(request, dispatch)?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))
}

#[test]
fn create_table_with_path_scans_rows() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    write_parquet(
        &data.path().join("a.parquet"),
        &strings_and_ints(&["a", "b", "c"], &[1, 2, 3]),
    );

    let catalog = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create(
        &dispatch,
        &catalog,
        path_request("events", data.path(), columns()),
    )
    .unwrap();

    let table = catalog.parquet_table("events").expect("table created");
    assert_eq!(table.columns.len(), 2);
    let parquet = table.parquet.clone();
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn tables_persist_across_reopen() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    write_parquet(
        &data.path().join("a.parquet"),
        &strings_and_ints(&["a", "b"], &[10, 20]),
    );
    let db_uri = db.path().to_str().unwrap();

    // Create the table, then drop the catalog — only the on-disk manifest remains.
    {
        let catalog = ParquetCatalog::open(db_uri, &dispatch).unwrap();
        create(
            &dispatch,
            &catalog,
            path_request("events", data.path(), columns()),
        )
        .unwrap();
    }

    // Reopening (as a restart would) reloads the table and its data.
    let reopened = ParquetCatalog::open(db_uri, &dispatch).unwrap();
    let table = reopened
        .parquet_table("events")
        .expect("table reloaded from the manifest");
    assert_eq!(table.columns.len(), 2);
    let parquet = table.parquet.clone();
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
}

#[test]
fn rooted_table_is_created_empty_under_the_db_root_and_persists() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let db_uri = db.path().to_str().unwrap();

    {
        let catalog = ParquetCatalog::open(db_uri, &dispatch).unwrap();
        create(&dispatch, &catalog, rooted_request("t", columns())).unwrap();

        // A data-less table: registered, with no row groups (its data lives under
        // `<root>/t`, which fills in once data is written there).
        let t = catalog.parquet_table("t").unwrap();
        assert_eq!(t.location, "t");
        assert!(t.parquet.row_groups().is_empty());
    }

    // And it survives a reopen.
    let reopened = ParquetCatalog::open(db_uri, &dispatch).unwrap();
    assert!(reopened.parquet_table("t").is_some());
}

#[test]
fn create_runs_the_fetch_and_commit_dataflow_across_workers() {
    // Production runs CREATE TABLE with workers = cores: the files are stolen
    // and their footers read by many workers, the row groups fan in to worker 0,
    // and only worker 0 runs the commit. Cover that shape (the other tests are
    // single-worker).
    let dispatch = dispatch_with_buffers(4, 32);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    for i in 0..8i64 {
        write_parquet(
            &data.path().join(format!("{i}.parquet")),
            &strings_and_ints(&["a", "b", "c"], &[i, i + 1, i + 2]),
        );
    }

    let catalog = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create(
        &dispatch,
        &catalog,
        path_request("events", data.path(), columns()),
    )
    .unwrap();

    let table = catalog.parquet_table("events").unwrap();
    assert_eq!(table.parquet.row_groups().len(), 8);
    let parquet = table.parquet.clone();
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 24);

    // The no-data case drives the same fan-in with nothing to fetch: every
    // worker's sink finishes empty and worker 0 still commits.
    create(&dispatch, &catalog, rooted_request("empty", columns())).unwrap();
    let empty = catalog.parquet_table("empty").unwrap();
    assert!(empty.parquet.row_groups().is_empty());
}

#[test]
fn rejects_an_object_store_scheme_in_a_table_path() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let catalog = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    let err = create(
        &dispatch,
        &catalog,
        path_request("t", Path::new("s3://bucket/data"), columns()),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("not a URL"),
        "expected scheme rejection: {err}"
    );
}
