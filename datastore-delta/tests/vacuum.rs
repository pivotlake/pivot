//! Vacuum deletes files the current table version no longer references once their
//! storage mtime is older than the retention window, and never touches a live
//! file. Age is judged against `now_ms`, threaded in so the window is exercised
//! without waiting on the wall clock.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::{DispatchGuard, commit_datastore_transaction, dispatch, table_dir};
use datastore::DatastoreTransaction as _;
use datastore_delta::{DEFAULT_VACUUM_POLL, DeltaDatastore, Vacuumer};
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};

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

/// `CREATE TABLE events (Timestamp Int64)`, adopting the Parquet files already
/// under `dir`, and return the table's own directory: where its log lives, and
/// where files written into it land.
fn create_events_table(
    datastore: &Arc<DeltaDatastore>,
    dispatch: &DispatchGuard,
    db: &Path,
    dir: &Path,
) -> PathBuf {
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "events".to_string(),
        columns: vec![Column {
            name: "Timestamp".to_string(),
            col_type: planner::types::Type::Int64,
        }],
        options: HashMap::from([(
            "with_pre_existing_parquets".to_string(),
            dir.to_string_lossy().into_owned(),
        )]),
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
    table_dir(db, datastore, "events")
}

#[test]
fn unreferenced_file_past_retention_is_deleted_and_the_live_file_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    // Dropped into the table's own directory after CREATE, so no Add references
    // it: an unreferenced orphan.
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms() + EIGHT_DAYS_MS);

    assert!(
        !table_dir.join("orphan.parquet").exists(),
        "an unreferenced file past the retention window is deleted"
    );
    assert!(
        adopted_dir.join("live.parquet").exists(),
        "the current version's live file is kept, however old"
    );
}

#[test]
fn unreferenced_file_within_retention_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms());

    assert!(
        table_dir.join("orphan.parquet").exists(),
        "an unreferenced file still within the retention window is kept"
    );
}

/// The adopted directory is never swept: those files are the user's, so an
/// unreferenced one there survives however old it is.
#[test]
fn a_file_in_the_adopted_directory_is_never_deleted() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    write_parquet(&adopted_dir.join("theirs.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms() + EIGHT_DAYS_MS);

    assert!(adopted_dir.join("theirs.parquet").exists());
}

/// Compaction merges adopted files into the table's own storage and drops them
/// from the log, and the vacuum sweep that follows still leaves the files
/// themselves alone, however far past the retention window: they are the
/// directory owner's, not the table's.
#[test]
fn compaction_merges_adopted_files_without_deleting_them() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    for name in ["a.parquet", "b.parquet"] {
        write_parquet(&adopted_dir.join(name));
    }
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    let name = SchemaQualifiedTableName::in_default_schema("events");
    let inputs = datastore.table_files(&name).unwrap();
    assert_eq!(inputs.len(), 2, "both adopted files are live");

    let id = datastore.table_handle(&name).unwrap().id();
    datastore_delta::compact_table_files(&datastore, id, &inputs, 128 * 1024).unwrap();
    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()))
        .vacuum_all(now_ms() + EIGHT_DAYS_MS);

    // One merged file in the table's own directory, holding every row.
    let merged = datastore.table_files(&name).unwrap();
    assert_eq!(merged.len(), 1);
    assert!(table_dir.join(merged[0].path.as_str()).exists());
    // And both originals still sit where their owner left them.
    for name in ["a.parquet", "b.parquet"] {
        assert!(
            adopted_dir.join(name).exists(),
            "compaction must not delete {name}"
        );
    }
}
