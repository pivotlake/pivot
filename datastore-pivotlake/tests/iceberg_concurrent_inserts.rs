//! Several INSERT statements committing to one managed table at once, each
//! through a transaction of its own: every one lands, as its own version,
//! and none is lost to another's commit.

mod iceberg_support;

use std::sync::Arc;

use catalog::datastore::DatastoreTransaction as _;
use datastore_pivotlake::test_support::metadata_versions;
use iceberg_support::{Database, collect_i64s, insert_rows, int_batch, single_i64};
use planner::catalog::SchemaQualifiedTableName;

#[test]
fn parallel_inserts_into_one_table_all_land() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");

    let inserts: Vec<_> = (1..=8i64)
        .map(|value| {
            let datastore = Arc::clone(&db.datastore);
            std::thread::spawn(move || insert_rows(&datastore, "events", vec![int_batch(&[value])]))
        })
        .collect();
    for insert in inserts {
        insert.join().unwrap();
    }

    let count = db.execute("SELECT count(*) FROM events");
    let rows = db.execute("SELECT value FROM events ORDER BY value");
    assert_eq!(single_i64(&count), 8);
    assert_eq!(collect_i64s(&rows, 0), (1..=8).collect::<Vec<_>>());
    assert_eq!(
        metadata_versions(&db.root_uri(), &db.table_directory("events")).unwrap(),
        (1..=9).collect::<Vec<_>>(),
        "the create and one version per insert"
    );
}

#[test]
fn an_insert_appends_onto_a_commit_made_after_its_transaction_began() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");
    let early = Arc::clone(&db.datastore).begin_transaction();

    db.insert_rows("events", vec![int_batch(&[1])]);
    // The transaction opened before that insert still sees the empty
    // table, yet its own insert commits on top of the latest version.
    let table = early
        .bind_table(
            planner::DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("events"),
        )
        .unwrap()
        .unwrap();
    let dispatcher = iceberg_support::shared_dispatcher();
    let rows = dispatch::values_input(&dispatcher, vec![int_batch(&[2])]).record_batches();
    table
        .compile_insert(rows, &dispatcher)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(early.commit())
        .unwrap();

    let rows = db.execute("SELECT value FROM events ORDER BY value");
    assert_eq!(collect_i64s(&rows, 0), [1, 2]);
}
