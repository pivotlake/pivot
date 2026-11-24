//! Several INSERT statements committing to one managed table at once, each
//! through a transaction of its own: every one commits, none is lost to
//! another's commit, and those that wait for the table together share a
//! version.

mod iceberg_support;

use std::sync::{Arc, Barrier};
use std::time::Duration;

use catalog::datastore::DatastoreTransaction as _;
use datastore_pivotlake::test_support::metadata_versions;
use iceberg_support::{Database, collect_i64s, insert_rows, int_batch, single_i64, stage_insert};
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
    let versions = metadata_versions(&db.root_uri(), &db.table_directory("events")).unwrap();
    assert_eq!(single_i64(&count), 8);
    assert_eq!(collect_i64s(&rows, 0), (1..=8).collect::<Vec<_>>());
    assert_eq!(
        versions,
        (1..=versions.len() as u64).collect::<Vec<_>>(),
        "the versions follow one another"
    );
    assert!(
        (2..=9).contains(&versions.len()),
        "the create and at most one version per insert, got {versions:?}"
    );
}

#[test]
fn inserts_that_wait_for_the_table_together_share_a_version() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");
    let start = Arc::new(Barrier::new(64));

    let inserts: Vec<_> = (1..=64i64)
        .map(|value| {
            let datastore = Arc::clone(&db.datastore);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                insert_rows(&datastore, "events", vec![int_batch(&[value])])
            })
        })
        .collect();
    for insert in inserts {
        insert.join().unwrap();
    }

    let rows = db.execute("SELECT value FROM events ORDER BY value");
    let versions = metadata_versions(&db.root_uri(), &db.table_directory("events")).unwrap();
    assert_eq!(collect_i64s(&rows, 0), (1..=64).collect::<Vec<_>>());
    assert!(
        versions.len() < 65,
        "64 inserts at once made a version each: {versions:?}"
    );
}

#[test]
fn more_inserts_than_the_runtime_has_blocking_threads_all_commit() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(2)
        .enable_time()
        .build()
        .unwrap();
    let transactions: Vec<_> = (1..=16i64)
        .map(|value| stage_insert(&db.datastore, "events", vec![int_batch(&[value])]))
        .collect();

    let commits: Vec<_> = transactions
        .into_iter()
        .map(|transaction| runtime.spawn(async move { transaction.commit().await }))
        .collect();
    for commit in commits {
        let committed =
            runtime.block_on(async { tokio::time::timeout(Duration::from_secs(60), commit).await });
        committed
            .expect("the commit is made")
            .expect("the commit does not panic")
            .unwrap();
    }

    let rows = db.execute("SELECT value FROM events ORDER BY value");
    assert_eq!(collect_i64s(&rows, 0), (1..=16).collect::<Vec<_>>());
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
