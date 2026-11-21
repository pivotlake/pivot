//! A compaction round merges the files another process committed to a table
//! once the datastore has refreshed from the store, whichever format keeps
//! the table.

mod iceberg_support;

use std::sync::Arc;
use std::time::Duration;

use datastore_pivotlake::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_MIN_FILES_TO_MERGE, PivotlakeDatastore,
    TableFormat, default_max_merge_input_bytes,
};
use iceberg_support::{Database, collect_i64s, insert_rows, int_batch, shared_dispatcher};
use iceberg_tables::table_metadata_cache::{
    DEFAULT_TABLE_METADATA_CACHE_BYTES, TableMetadataCache,
};
use iceberg_tables::test_support::FaultyStore;
use object_storage::{LocalStore, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

/// The default sizes, under which a routine round merges six files of a
/// few bytes each.
fn build_default_sizes_config() -> CompactionConfig {
    CompactionConfig {
        target_bytes: DEFAULT_COMPACT_BYTES,
        max_merge_input_bytes: default_max_merge_input_bytes(DEFAULT_COMPACT_BYTES),
        min_files: DEFAULT_MIN_FILES_TO_MERGE,
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
    let table_metadata_cache = TableMetadataCache::new(DEFAULT_TABLE_METADATA_CACHE_BYTES);
    PivotlakeDatastore::from_store(
        store,
        &shared_dispatcher(),
        None,
        Some(format),
        table_metadata_cache,
    )
    .unwrap()
}

/// Six files another process inserted are merged by the round that follows
/// the compacting datastore's refresh from the store, the sweep its
/// background refresh runs.
fn check_a_round_after_a_refresh_merges_the_files_another_process_inserted(format: TableFormat) {
    let mut db = Database::in_format_with_compaction(format, build_default_sizes_config());
    db.execute("CREATE TABLE t (value BIGINT)");
    let other_process = open_from_another_process(&db, format);
    for value in 1..=6 {
        insert_rows(&other_process, "t", vec![int_batch(&[value])]);
    }
    db.datastore.refresh_from_store().unwrap();

    db.compact("t", false);

    let files = db
        .datastore
        .table_files(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap();
    assert_eq!(files.len(), 1, "{files:?}");
    let rows = db.execute("SELECT value FROM t ORDER BY value");
    assert_eq!(collect_i64s(&rows, 0), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn a_round_after_a_refresh_merges_the_files_another_process_inserted_into_a_delta_table() {
    check_a_round_after_a_refresh_merges_the_files_another_process_inserted(TableFormat::Delta);
}

#[test]
fn a_round_after_a_refresh_merges_the_files_another_process_inserted_into_an_iceberg_table() {
    check_a_round_after_a_refresh_merges_the_files_another_process_inserted(TableFormat::Iceberg);
}
