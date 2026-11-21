//! Routine compaction rounds, one after another over one table, whichever
//! format keeps it: a round merges what the commits since the round before
//! added, in the partitions they added it to, and leaves the rest of the
//! table as it was, whether those commits were its own merges, an insert's
//! or another writer's.

mod iceberg_support;

use std::sync::Arc;
use std::time::Duration;

use datastore_pivotlake::test_support::{commit_external_append, commit_external_replace};
use datastore_pivotlake::{
    CompacterHandle, CompactionConfig, FileRef, PivotlakeDatastore, TableFormat,
};
use iceberg_support::{
    Database, collect_i64s, insert_rows, int_batch, int_parquet_bytes, shared_dispatcher,
};
use iceberg_tables::test_support::FaultyStore;
use object_storage::{LocalStore, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

const FORMATS: [TableFormat; 2] = [TableFormat::Delta, TableFormat::Iceberg];

/// Sizes under which any two small files are a merge.
fn build_pair_merge_config() -> CompactionConfig {
    CompactionConfig {
        target_bytes: 64 * 1024 * 1024,
        merge_target_bytes: 1,
        min_files: 2,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    }
}

/// The files of the table `t`, in the order the table holds them.
fn list_files(database: &Database) -> Vec<FileRef> {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    database.datastore.table_files(&name).unwrap().unwrap()
}

/// A datastore of its own over the directory of `database`, as another
/// process serving the same database holds one: opened through a store that
/// takes no lock on the directory.
fn open_from_another_process(database: &Database, format: TableFormat) -> Arc<PivotlakeDatastore> {
    let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
        inner: Arc::new(LocalStore::new(database.path()).unwrap()),
        faults: Arc::default(),
    });
    PivotlakeDatastore::from_store(store, &shared_dispatcher(), None, Some(format)).unwrap()
}

/// Run one routine compaction round over the table `name` of `datastore`,
/// under the sizes any two small files are a merge under.
fn run_round(datastore: &Arc<PivotlakeDatastore>, name: &str) {
    let config = build_pair_merge_config();
    let (compacter, actor) = CompacterHandle::new(
        config.target_bytes,
        config.merge_target_bytes,
        config.min_files,
        config.max_concurrent_merges,
        config.layout_clique_size,
        None,
        Arc::clone(datastore),
    );
    let name = SchemaQualifiedTableName::in_default_schema(name);
    let id = datastore.table_handle(&name).unwrap().id();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let running = runtime.spawn(actor.run());
    runtime
        .block_on(compacter.compact(name, id, false))
        .unwrap();
    running.abort();
}

#[test]
fn a_round_merges_the_files_added_since_the_round_before_and_no_other() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (v BIGINT, p BIGINT) PARTITIONED BY (p)");
        database.execute("INSERT INTO t VALUES (1, 1)");
        database.execute("INSERT INTO t VALUES (2, 2)");
        database.compact("t", false);
        let settled = list_files(&database);
        database.execute("INSERT INTO t VALUES (3, 2)");

        let passes = database.compact("t", false);

        let files = list_files(&database);
        let kept: Vec<&FileRef> = files.iter().filter(|file| settled.contains(file)).collect();
        assert_eq!(
            passes, 2,
            "{format}: one merge, then a pass that finds none"
        );
        assert_eq!(files.len(), 2, "{format}: {files:?}");
        assert_eq!(kept.len(), 1, "{format}: the other partition's file stays");
        let rows = database.execute("SELECT v FROM t WHERE p = 2 ORDER BY v");
        assert_eq!(collect_i64s(&rows, 0), [2, 3], "{format}");
    }
}

#[test]
fn a_round_over_a_table_nothing_was_committed_to_examines_nothing() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (v BIGINT)");
        database.execute("INSERT INTO t VALUES (1)");
        database.execute("INSERT INTO t VALUES (2)");
        database.compact("t", false);
        let settled = list_files(&database);

        let passes = database.compact("t", false);

        assert_eq!(passes, 1, "{format}");
        assert_eq!(list_files(&database), settled, "{format}");
    }
}

#[test]
fn a_round_merges_the_outputs_of_its_own_merges() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (value BIGINT)");
        for value in 1..=4 {
            database.execute(&format!("INSERT INTO t VALUES ({value})"));
        }

        let passes = database.compact("t", false);

        let files = list_files(&database);
        assert_eq!(
            passes, 4,
            "{format}: three merges, then a pass that finds none"
        );
        assert_eq!(files.len(), 1, "{format}: {files:?}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 2, 3, 4], "{format}");
    }
}

#[test]
fn a_round_after_a_merge_and_an_insert_merges_the_output_with_the_inserted_file() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (value BIGINT)");
        database.execute("INSERT INTO t VALUES (1)");
        database.execute("INSERT INTO t VALUES (2)");
        database.compact("t", false);
        let merged = list_files(&database);
        database.execute("INSERT INTO t VALUES (3)");

        let passes = database.compact("t", false);

        let files = list_files(&database);
        assert_eq!(merged.len(), 1, "{format}: {merged:?}");
        assert_eq!(
            passes, 2,
            "{format}: one merge, then a pass that finds none"
        );
        assert_eq!(files.len(), 1, "{format}: {files:?}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 2, 3], "{format}");
    }
}

#[test]
fn a_round_after_a_merge_and_another_processes_commit_merges_the_output_with_its_file() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (value BIGINT)");
        database.execute("INSERT INTO t VALUES (1)");
        database.execute("INSERT INTO t VALUES (2)");
        database.compact("t", false);
        let merged = list_files(&database);
        let other_process = open_from_another_process(&database, format);
        insert_rows(&other_process, "t", vec![int_batch(&[3])]);
        database.datastore.refresh_from_store().unwrap();

        let passes = database.compact("t", false);

        let files = list_files(&database);
        assert_eq!(merged.len(), 1, "{format}: {merged:?}");
        assert_eq!(
            passes, 2,
            "{format}: one merge, then a pass that finds none"
        );
        assert_eq!(files.len(), 1, "{format}: {files:?}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 2, 3], "{format}");
    }
}

#[test]
fn a_round_after_another_process_rewrote_its_merges_output_names_the_rewrite_alone() {
    for format in FORMATS {
        let mut database = Database::in_format_with_compaction(format, build_pair_merge_config());
        database.execute("CREATE TABLE t (value BIGINT)");
        database.execute("INSERT INTO t VALUES (1)");
        database.execute("INSERT INTO t VALUES (2)");
        database.compact("t", false);
        let merged = list_files(&database);
        let other_process = open_from_another_process(&database, format);
        insert_rows(&other_process, "t", vec![int_batch(&[3])]);
        run_round(&other_process, "t");
        database.datastore.refresh_from_store().unwrap();

        let passes = database.compact("t", false);

        let files = list_files(&database);
        assert_eq!(passes, 1, "{format}: one file is no merge");
        assert_eq!(files.len(), 1, "{format}: {files:?}");
        assert!(!merged.contains(&files[0]), "{format}: {files:?}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 2, 3], "{format}");
    }
}

#[test]
fn a_round_merges_what_another_writer_left_of_the_files_the_round_before_saw() {
    // The two files inserted here stay under the merge target; the one of
    // them another writer leaves reaches it with the two that writer adds.
    let outside_file = int_parquet_bytes(&[3]);
    let mut database = Database::with_compaction(CompactionConfig {
        merge_target_bytes: 2 * outside_file.len() as u64,
        min_files: usize::MAX,
        ..build_pair_merge_config()
    });
    database.execute("CREATE TABLE t (value BIGINT)");
    database.execute("INSERT INTO t VALUES (1)");
    database.execute("INSERT INTO t VALUES (2)");
    database.compact("t", false);
    let retired = list_files(&database).remove(0).path;
    let (root, directory) = (database.root_uri(), database.table_directory("t"));
    commit_external_replace(
        &root,
        &directory,
        &[retired.as_str()],
        "outside-3.parquet",
        &outside_file,
        1,
    )
    .unwrap();
    commit_external_append(
        &root,
        &directory,
        "outside-4.parquet",
        &int_parquet_bytes(&[4]),
        1,
    )
    .unwrap();
    database.datastore.refresh_from_store().unwrap();

    database.compact("t", false);

    let files = list_files(&database);
    assert_eq!(files.len(), 1, "{files:?}");
    let rows = database.execute("SELECT count(*) FROM t");
    assert_eq!(collect_i64s(&rows, 0), [3]);
}
