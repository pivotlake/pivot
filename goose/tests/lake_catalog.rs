//! End-to-end tests for the goose object-store catalog: attaching a table from
//! a snapshot and querying its data files, and CAS-creating a new catalog.
//! Uses a local catalog root (a temp dir with `_goose_log/` + parquet files),
//! exercising the same path an `s3://`/`gs://` root would, minus the network.

mod common;
use common::*;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use dispatch::Projection;
use goose::ParquetCatalog;
use goose::metadata::{
    CatalogSnapshot, Column as MetaColumn, DataFile, FORMAT_VERSION, Schema as MetaSchema,
    Table as MetaTable,
};
use goose::parquet::table_input;
use planner::catalog::{Catalog, Column, CreateTableRequest, Result as CatalogResult};
use planner::types::Type;

fn write_parquet(path: &std::path::Path, batch: &arrow_array::RecordBatch) {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(fs::File::create(path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn lake_request(name: &str, url: &std::path::Path, columns: Vec<Column>) -> CreateTableRequest {
    let mut options = HashMap::new();
    options.insert("url".to_string(), url.to_string_lossy().into_owned());
    CreateTableRequest {
        name: name.to_string(),
        columns,
        options,
        if_not_exists: false,
    }
}

/// Mirror the server's `CREATE TABLE`: `create_table` reads the footers (on the
/// coordinator) and returns the plan that writes the table, which we execute.
fn create(
    dispatch: &DispatchGuard,
    catalog: &Arc<ParquetCatalog>,
    request: CreateTableRequest,
) -> CatalogResult<()> {
    catalog
        .create_table(request, dispatch)?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))
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

#[test]
fn attach_existing_lake_table_and_scan_rows() {
    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();

    // A goose catalog root: one parquet data file plus a snapshot referencing it
    // by a path relative to the root.
    fs::create_dir_all(dir.path().join("_goose_data")).unwrap();
    let batch = strings_and_ints(&["a", "b", "c"], &[1, 2, 3]);
    let data_path = dir.path().join("_goose_data/a.parquet");
    write_parquet(&data_path, &batch);
    let data_size = fs::metadata(&data_path).unwrap().len();

    let snapshot = CatalogSnapshot {
        format_version: FORMAT_VERSION,
        version: 1,
        schemas: vec![MetaSchema {
            name: "main".to_string(),
            tables: vec![MetaTable {
                name: "events".to_string(),
                columns: vec![
                    MetaColumn {
                        name: "name".into(),
                        type_sql: "VARCHAR".into(),
                    },
                    MetaColumn {
                        name: "value".into(),
                        type_sql: "BIGINT".into(),
                    },
                ],
                files: vec![DataFile {
                    location: "_goose_data/a.parquet".to_string(),
                    size: data_size,
                    row_count: 3,
                }],
            }],
        }],
    };
    fs::create_dir_all(dir.path().join("_goose_log")).unwrap();
    fs::write(
        dir.path().join("_goose_log/00000000000000000001.json"),
        snapshot.to_vec(),
    )
    .unwrap();

    let catalog = Arc::new(ParquetCatalog::new());
    create(
        &dispatch,
        &catalog,
        lake_request("events", dir.path(), columns()),
    )
    .unwrap();

    let table = catalog
        .parquet_table("events")
        .expect("table resolved from the snapshot");
    assert_eq!(table.columns.len(), 2);
    // The row-group metadata was materialized when the table was created.
    let parquet = table.parquet.clone();
    assert!(
        !parquet.row_groups().is_empty(),
        "the data file's footer was parsed into row groups"
    );

    // Scan it end-to-end and check the rows came back.
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn create_new_lake_table_cas_commits_a_snapshot() {
    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap(); // empty: no _goose_log yet

    let catalog = Arc::new(ParquetCatalog::new());
    create(
        &dispatch,
        &catalog,
        lake_request("t", dir.path(), columns()),
    )
    .unwrap();

    // Exactly one snapshot was committed via create-if-absent.
    let logs: Vec<_> = fs::read_dir(dir.path().join("_goose_log"))
        .unwrap()
        .collect();
    assert_eq!(logs.len(), 1, "one snapshot committed");

    // The table resolves (empty — no data files committed yet).
    let t = catalog.parquet_table("t").expect("new table resolves");
    assert!(t.parquet.row_groups().is_empty());

    // The committed snapshot records the table and its declared columns,
    // round-tripped to SQL type spellings.
    let store = goose::store::open_store(dir.path().to_str().unwrap()).unwrap();
    let snap = goose::store::latest_snapshot(store.as_ref()).unwrap();
    assert_eq!(snap.version, 1);
    let committed = snap.table("main", "t").expect("table is in the snapshot");
    assert_eq!(committed.columns.len(), 2);
    assert_eq!(committed.columns[1].type_sql, "BIGINT");
}
