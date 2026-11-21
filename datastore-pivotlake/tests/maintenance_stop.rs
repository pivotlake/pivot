//! A datastore whose maintenance has been stopped, or told to stop, runs no
//! compaction again: a `COMPACT` issued afterwards fails instead of starting
//! a round.

mod iceberg_support;

use catalog::{DEFAULT_DATASTORE_NAME, Datastore};
use iceberg_support::Database;
use planner::catalog::SchemaQualifiedTableName;

#[test]
fn compact_after_stop_fails_as_the_compacter_stopped() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (v BIGINT)");
    let table = SchemaQualifiedTableName::in_default_schema("t");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let compacted = runtime.block_on(async {
        db.stop().await;
        let transaction = db.begin_transaction();
        transaction
            .compact(DEFAULT_DATASTORE_NAME, &table, false)
            .await
    });

    let error = compacted.unwrap_err().to_string();
    assert!(error.contains("compaction actor stopped"), "{error}");
}

/// The first `COMPACT` spawns the compaction actor, so the second finds its
/// handle and is answered by an actor that was only told to stop.
#[test]
fn compact_after_a_requested_stop_fails_as_the_compacter_stopped() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (v BIGINT)");
    let table = SchemaQualifiedTableName::in_default_schema("t");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let compacted = runtime.block_on(async {
        let transaction = db.begin_transaction();
        transaction
            .compact(DEFAULT_DATASTORE_NAME, &table, false)
            .await
            .unwrap();
        db.datastore.request_stop();
        transaction
            .compact(DEFAULT_DATASTORE_NAME, &table, false)
            .await
    });

    let error = compacted.unwrap_err().to_string();
    assert!(error.contains("compaction actor stopped"), "{error}");
}
