//! Schemas of a managed datastore: created through SQL, recorded in the
//! index, holding tables apart by name, and back after a reopen.

mod iceberg_support;

use iceberg_support::{Database, collect_i64s, single_i64};
use planner::catalog::SchemaQualifiedTableName;

#[test]
fn a_new_datastore_defines_only_the_default_schema() {
    let db = Database::new();

    assert!(db.datastore.contains_schema(planner::DEFAULT_SCHEMA_NAME));
    assert!(!db.datastore.contains_schema("logs"));
}

#[test]
fn a_created_schema_holds_tables_and_survives_a_reopen() {
    let mut db = Database::new();
    db.execute("CREATE SCHEMA logs");
    db.execute("CREATE TABLE logs.events (value BIGINT)");
    db.execute("INSERT INTO logs.events VALUES (7)");

    let mut reopened = Database::reopen(db.close());

    assert!(reopened.datastore.contains_schema("logs"));
    let rows = reopened.execute("SELECT value FROM logs.events");
    assert_eq!(single_i64(&rows), 7);
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(reopened.path().join("_pivot_manifest.json")).unwrap(),
    )
    .unwrap();
    let schemas: Vec<&str> = index["schemas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|schema| schema["name"].as_str().unwrap())
        .collect();
    assert!(schemas.contains(&"logs"), "{schemas:?}");
}

#[test]
fn creating_an_existing_schema_is_rejected_unless_if_not_exists() {
    let mut db = Database::new();
    db.execute("CREATE SCHEMA logs");

    let error = db.try_execute("CREATE SCHEMA logs").unwrap_err();
    db.execute("CREATE SCHEMA IF NOT EXISTS logs");

    assert!(error.contains("logs"), "{error}");
}

#[test]
fn creating_a_table_in_an_unknown_schema_is_rejected() {
    let mut db = Database::new();

    let error = db
        .try_execute("CREATE TABLE nowhere.t (value BIGINT)")
        .unwrap_err();

    assert!(error.contains("nowhere"), "{error}");
}

#[test]
fn the_same_table_name_in_two_schemas_resolves_separately() {
    let mut db = Database::new();
    db.execute("CREATE SCHEMA a");
    db.execute("CREATE SCHEMA b");
    db.execute("CREATE TABLE a.t (value BIGINT)");
    db.execute("CREATE TABLE b.t (value BIGINT)");
    db.execute("INSERT INTO a.t VALUES (1)");
    db.execute("INSERT INTO b.t VALUES (2), (3)");

    let a = db.execute("SELECT value FROM a.t ORDER BY value");
    let b = db.execute("SELECT value FROM b.t ORDER BY value");

    assert_eq!(collect_i64s(&a, 0), [1]);
    assert_eq!(collect_i64s(&b, 0), [2, 3]);
    let in_a = db
        .datastore
        .table_handle(&SchemaQualifiedTableName::new("a", "t"))
        .unwrap();
    let in_b = db
        .datastore
        .table_handle(&SchemaQualifiedTableName::new("b", "t"))
        .unwrap();
    assert_ne!(in_a.id(), in_b.id(), "each table has storage of its own");
}
