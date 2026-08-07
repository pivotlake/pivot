//! Vacuum deletes files the current table version no longer references once their
//! storage mtime is older than the retention window, and never touches a live
//! file. Age is judged against `now_ms`, threaded in so the window is exercised
//! without waiting on the wall clock.

mod common;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::{DispatchGuard, commit_datastore_transaction, dispatch};
use datastore::DatastoreTransaction as _;
use datastore_delta::{DEFAULT_VACUUM_POLL, DeltaDatastore, Vacuumer};
use planner::catalog::{Column, CreateTableRequest};

const EIGHT_DAYS_MS: u64 = 8 * 24 * 60 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Write a small SNAPPY Parquet file (single Int64 column) at `path`.
fn write_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "Timestamp",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1i64, 2, 3])) as _],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// `CREATE TABLE events (Timestamp Int64) WITH (path = dir)`, adopting whatever
/// Parquet files already sit under `dir` as the table's initial files.
fn create_events_table(datastore: &Arc<DeltaDatastore>, dispatch: &DispatchGuard, dir: &Path) {
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "events".to_string(),
        columns: vec![Column {
            name: "Timestamp".to_string(),
            col_type: planner::types::Type::Int64,
        }],
        options: HashMap::from([("path".to_string(), dir.to_string_lossy().into_owned())]),
        if_not_exists: false,
    };
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(request)
        .unwrap()
        .compile(dispatch)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
}

#[test]
fn unreferenced_file_past_retention_is_deleted_and_the_live_file_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let table_dir = db.path().join("events");
    std::fs::create_dir_all(&table_dir).unwrap();
    write_parquet(&table_dir.join("live.parquet"));
    let datastore = DeltaDatastore::open_local(db.path(), &dispatch).unwrap();
    create_events_table(&datastore, &dispatch, &table_dir);
    // Dropped in after CREATE, so no Add references it: an unreferenced orphan.
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms() + EIGHT_DAYS_MS);

    assert!(
        !table_dir.join("orphan.parquet").exists(),
        "an unreferenced file past the retention window is deleted"
    );
    assert!(
        table_dir.join("live.parquet").exists(),
        "the current version's live file is kept, however old"
    );
}

#[test]
fn unreferenced_file_within_retention_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let table_dir = db.path().join("events");
    std::fs::create_dir_all(&table_dir).unwrap();
    write_parquet(&table_dir.join("live.parquet"));
    let datastore = DeltaDatastore::open_local(db.path(), &dispatch).unwrap();
    create_events_table(&datastore, &dispatch, &table_dir);
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms());

    assert!(
        table_dir.join("orphan.parquet").exists(),
        "an unreferenced file still within the retention window is kept"
    );
}
