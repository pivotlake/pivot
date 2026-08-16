//! `DROP SCHEMA` at the datastore level: the drop removes the schema (durably,
//! and only at commit), a non-empty schema needs `CASCADE` to take its tables
//! with it, and a refresh removes a schema another process dropped from the
//! shared manifest.

mod common;
use common::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction as _;
use datastore_delta::{DEFAULT_VACUUM_POLL, DeltaDatastore, Vacuumer};
use planner::catalog::{
    Column, CreateSchemaRequest, CreateTableRequest, DropSchemaRequest, Result as CatalogResult,
    SchemaQualifiedTableName,
};
use planner::types::Type;

const EIGHT_DAYS_MS: u64 = 8 * 24 * 60 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Mirror the server's `CREATE SCHEMA`: run the staging dataflow, then commit
/// the transaction that persists and publishes the staged schema.
fn create_schema(
    dispatch: &DispatchGuard,
    datastore: &Arc<DeltaDatastore>,
    name: &str,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_schema(CreateSchemaRequest {
            datastore_name: None,
            name: name.to_string(),
            if_not_exists: false,
        })?
        .compile(dispatch)?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

/// Mirror the server's `CREATE TABLE` of an empty table inside `schema`.
fn create_table(
    dispatch: &DispatchGuard,
    datastore: &Arc<DeltaDatastore>,
    schema: &str,
    name: &str,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: Some(schema.to_string()),
            name: name.to_string(),
            columns: vec![Column {
                name: "value".to_string(),
                col_type: Type::Int64,
            }],
            options: HashMap::new(),
            if_not_exists: false,
        })?
        .compile(dispatch)?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

fn drop_request(name: &str, if_exists: bool, cascade: bool) -> DropSchemaRequest {
    DropSchemaRequest {
        datastore_name: None,
        name: name.to_string(),
        if_exists,
        cascade,
    }
}

/// Drop a schema, mirroring how the server runs `DROP SCHEMA`: execute the
/// staging dataflow, then commit the transaction that unregisters the schema.
fn drop_schema(
    dispatch: &DispatchGuard,
    datastore: &Arc<DeltaDatastore>,
    request: DropSchemaRequest,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_schema(request)?
        .compile(dispatch)?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

#[test]
fn dropped_schema_is_gone_from_the_catalog_and_the_manifest() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    drop_schema(
        &dispatch,
        &datastore,
        drop_request("analytics", false, false),
    )
    .unwrap();

    assert!(!datastore.contains_schema("analytics"));
    drop(datastore);
    let reopened = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    assert!(!reopened.contains_schema("analytics"));
}

#[test]
fn drop_is_visible_only_after_commit() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_schema(drop_request("analytics", false, false))
        .unwrap()
        .compile(&dispatch)
        .unwrap()
        .execute()
        .collect()
        .unwrap();

    assert!(datastore.contains_schema("analytics"));

    commit_datastore_transaction(transaction).unwrap();

    assert!(!datastore.contains_schema("analytics"));
}

#[test]
fn rolling_back_a_drop_keeps_the_schema() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_schema(drop_request("analytics", false, false))
        .unwrap()
        .compile(&dispatch)
        .unwrap()
        .execute()
        .collect()
        .unwrap();

    transaction.rollback();

    assert!(datastore.contains_schema("analytics"));
}

#[test]
fn dropping_a_missing_schema_fails() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();

    let err = drop_schema(&dispatch, &datastore, drop_request("missing", false, false))
        .unwrap_err()
        .to_string();

    assert!(err.contains("does not exist"), "{err}");
}

#[test]
fn dropping_a_missing_schema_with_if_exists_succeeds() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();

    drop_schema(&dispatch, &datastore, drop_request("missing", true, false)).unwrap();
}

#[test]
fn dropping_a_non_empty_schema_without_cascade_fails() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    create_table(&dispatch, &datastore, "analytics", "t").unwrap();

    let err = drop_schema(
        &dispatch,
        &datastore,
        drop_request("analytics", false, false),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("not empty"), "{err}");
    assert!(datastore.contains_schema("analytics"));
}

#[test]
fn cascade_drops_the_schemas_tables_but_keeps_their_storage() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    create_table(&dispatch, &datastore, "analytics", "t").unwrap();
    let name = SchemaQualifiedTableName::new("analytics", "t");
    let table_dir = db
        .path()
        .join(datastore.table_handle(&name).unwrap().location());

    drop_schema(
        &dispatch,
        &datastore,
        drop_request("analytics", false, true),
    )
    .unwrap();

    assert!(!datastore.contains_schema("analytics"));
    assert!(!datastore.contains_table(&name));
    assert!(
        table_dir.join("_delta_log").exists(),
        "a query planned before the drop may still read the table, so its storage stays for vacuum"
    );
}

#[test]
fn vacuum_reclaims_a_cascade_dropped_tables_storage_after_retention() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    create_table(&dispatch, &datastore, "analytics", "t").unwrap();
    let table_dir = db.path().join(
        datastore
            .table_handle(&SchemaQualifiedTableName::new("analytics", "t"))
            .unwrap()
            .location(),
    );
    drop_schema(
        &dispatch,
        &datastore,
        drop_request("analytics", false, true),
    )
    .unwrap();
    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));

    vacuumer.vacuum_all(now_ms());
    assert!(
        table_dir.exists(),
        "within the retention window the dropped table's storage is kept"
    );

    vacuumer.vacuum_all(now_ms() + EIGHT_DAYS_MS);
    assert!(
        !table_dir.exists(),
        "past the retention window the dropped table's storage is deleted"
    );
}

#[test]
fn recreating_a_dropped_schema_starts_empty() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    create_table(&dispatch, &datastore, "analytics", "t").unwrap();

    drop_schema(
        &dispatch,
        &datastore,
        drop_request("analytics", false, true),
    )
    .unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    assert!(datastore.contains_schema("analytics"));
    assert!(!datastore.contains_table(&SchemaQualifiedTableName::new("analytics", "t")));
}

#[test]
fn refresh_removes_a_schema_another_process_dropped() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    // Rewrite the manifest the way another process's drop would, leaving this
    // datastore's in-memory index holding the schema.
    let manifest_path = db.path().join("_pivot_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["schemas"]
        .as_array_mut()
        .unwrap()
        .retain(|schema| schema["name"] != "analytics");
    manifest["version"] = (manifest["version"].as_u64().unwrap() + 1).into();
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let changed = datastore.refresh_from_store().unwrap();

    assert!(changed);
    assert!(!datastore.contains_schema("analytics"));
}
