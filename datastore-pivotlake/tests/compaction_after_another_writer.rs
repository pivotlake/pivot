//! A compaction round merges the files another process committed to a table
//! since the datastore last read it from the store, whichever format keeps
//! the table.

mod iceberg_support;

use std::sync::Arc;
use std::time::Duration;

use datastore_pivotlake::{CompactionConfig, PivotlakeDatastore, TableFormat};
use iceberg_support::{Database, collect_i64s, insert_rows, int_batch, shared_dispatcher};
use iceberg_tables::test_support::FaultyStore;
use object_storage::{LocalStore, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

/// Sizes under which a final sweep merges any two small files.
fn build_pair_merge_config() -> CompactionConfig {
    CompactionConfig {
        target_bytes: 64 * 1024 * 1024,
        merge_target_bytes: 5 * 64 * 1024 * 1024,
        min_files: 2,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    }
}

/// A datastore of its own over the directory of `db`, as another process
/// serving the same database holds one: opened through a store that takes
/// no lock on the directory.
fn open_from_another_process(db: &Database, format: TableFormat) -> Arc<PivotlakeDatastore> {
    let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
        inner: Arc::new(LocalStore::new(db.path()).unwrap()),
        faults: Arc::default(),
    });
    PivotlakeDatastore::from_store(store, &shared_dispatcher(), None, Some(format)).unwrap()
}

/// Two files another process inserted, which the compacting datastore's
/// copy of the table does not list, are merged by its next round.
fn check_a_round_merges_the_files_another_process_inserted(format: TableFormat) {
    let mut db = Database::in_format_with_compaction(format, build_pair_merge_config());
    db.execute("CREATE TABLE t (value BIGINT)");
    let other_process = open_from_another_process(&db, format);
    insert_rows(&other_process, "t", vec![int_batch(&[1])]);
    insert_rows(&other_process, "t", vec![int_batch(&[2])]);

    db.compact("t", true);

    let files = db
        .datastore
        .table_files(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap();
    assert_eq!(files.len(), 1, "{files:?}");
    let rows = db.execute("SELECT value FROM t ORDER BY value");
    assert_eq!(collect_i64s(&rows, 0), [1, 2]);
}

#[test]
fn a_round_merges_the_files_another_process_inserted_into_a_delta_table() {
    check_a_round_merges_the_files_another_process_inserted(TableFormat::Delta);
}

#[test]
fn a_round_merges_the_files_another_process_inserted_into_an_iceberg_table() {
    check_a_round_merges_the_files_another_process_inserted(TableFormat::Iceberg);
}
