//! `DROP TABLE` on a managed table: the catalog forgets the table at commit,
//! its storage stays until vacuum reclaims it after the retention window, and
//! a recreated name is a new table.

mod iceberg_support;

use std::sync::Arc;

use datastore_pivotlake::{DEFAULT_VACUUM_POLL, Vacuumer};
use iceberg_support::{Database, FIVE_HOURS_MS, now_ms, write_int_parquet};
use planner::catalog::SchemaQualifiedTableName;

fn vacuum(db: &Database, now_ms: u64) {
    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, db.datastore.clone()))
        .vacuum_all(now_ms)
        .unwrap();
}

#[test]
fn a_drop_survives_a_reopen() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("DROP TABLE t");

    let reopened = Database::reopen(db.close());

    assert!(
        !reopened
            .datastore
            .contains_table(&SchemaQualifiedTableName::in_default_schema("t"))
    );
}

#[test]
fn dropping_a_missing_table_fails_unless_if_exists() {
    let mut db = Database::new();

    let error = db.try_execute("DROP TABLE missing").unwrap_err();
    db.execute("DROP TABLE IF EXISTS missing");

    assert!(error.contains("missing"), "{error}");
}

#[test]
fn recreating_a_dropped_name_mints_a_fresh_identity() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let first_directory = db.table_directory("t");

    db.execute("DROP TABLE t");
    db.execute("CREATE TABLE t (value BIGINT)");

    assert_ne!(db.table_directory("t"), first_directory);
    assert!(db.path().join(first_directory).join("metadata").is_dir());
}

#[test]
fn vacuum_reclaims_a_dropped_tables_storage_only_after_retention() {
    let mut db = Database::new();
    let adopted_dir = db.path().join("adopted");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_int_parquet(&adopted_dir.join("live.parquet"), &[1]);
    db.execute("CREATE TABLE t (value BIGINT) WITH (with_pre_existing_parquets = 'adopted')");
    let table_dir = db.table_path("t");
    db.execute("DROP TABLE t");

    vacuum(&db, now_ms());
    assert!(table_dir.exists(), "inside the window the storage is kept");
    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(
        !table_dir.exists(),
        "past the window the storage is deleted"
    );
    assert!(adopted_dir.join("live.parquet").exists());
}

#[test]
fn refresh_removes_a_table_another_process_dropped() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let name = SchemaQualifiedTableName::in_default_schema("t");
    // Rewrite the index the way another process's drop would, leaving this
    // datastore's in-memory set holding the table.
    let index_path = db.path().join("_pivot_manifest.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
    let schema_entry = index["schemas"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|schema| schema["name"] == planner::DEFAULT_SCHEMA_NAME)
        .unwrap();
    let id = schema_entry["table_ids"]
        .as_object_mut()
        .unwrap()
        .remove("t")
        .unwrap();
    index["table_locations"]
        .as_object_mut()
        .unwrap()
        .remove(id.as_str().unwrap());
    index["version"] = (index["version"].as_u64().unwrap() + 1).into();
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let changed = db.datastore.refresh_from_store().unwrap();

    assert!(changed);
    assert!(!db.datastore.contains_table(&name));
}

#[test]
fn a_database_written_before_the_format_field_is_delta_and_refuses_to_open_as_iceberg() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("_pivot_manifest.json"),
        br#"{"version":1,"schemas":[{"name":"main","table_ids":{}}]}"#,
    )
    .unwrap();
    let dispatcher = iceberg_support::shared_dispatcher();

    let error = datastore_pivotlake::PivotlakeDatastore::open_with_format(
        dir.path().to_str().unwrap(),
        &dispatcher,
        Some(datastore_pivotlake::TableFormat::Iceberg),
    )
    .expect_err("a Delta database must not open as an Iceberg one");
    let opened =
        datastore_pivotlake::PivotlakeDatastore::open(dir.path().to_str().unwrap(), &dispatcher)
            .unwrap();

    assert!(error.to_string().contains("delta"), "{error}");
    assert_eq!(
        opened.table_format(),
        datastore_pivotlake::TableFormat::Delta
    );
}
