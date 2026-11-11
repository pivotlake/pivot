//! The numbered metadata files a managed table keeps its versions in: how a
//! version and the hint are named, which commits write the hint and what a
//! store that starts from it finds, how a commit settles a write the store
//! did not answer for and a race it lost, what a commit and a refresh read
//! of the store, and how long the metadata log names a version.

mod iceberg_support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use datastore_pivotlake::test_support::{
    commit_external_append, hinted_version, live_data_file_names, logged_metadata_versions,
    metadata_versions, set_table_property,
};
use datastore_pivotlake::{
    CatalogTable, DEFAULT_VACUUM_POLL, Error, PivotlakeDatastore, TableFormat, Vacuumer,
};
use iceberg_support::{
    Database, FIVE_HOURS_MS, append_version, int_batch, int_parquet_bytes, names_in, now_ms,
    shared_dispatcher, single_i64,
};
use iceberg_tables::test_support::{Faults, FaultyStore};
use object_storage::{LocalStore, ObjectPath, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;
use tempfile::TempDir;

/// A database in a directory of its own, read and written through a store
/// that counts its requests and fails the ones the faults name.
fn open_faulty_database() -> (TempDir, Arc<Faults>, Database) {
    let directory = TempDir::new().unwrap();
    let faults = Arc::new(Faults::default());
    let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
        inner: Arc::new(LocalStore::new(directory.path()).unwrap()),
        faults: Arc::clone(&faults),
    });
    let root = directory.path().to_string_lossy().into_owned();
    (directory, faults, Database::over_store(root, store))
}

/// The copy of the table `name` the datastore of `db` holds now.
fn hold_table(db: &Database, name: &str) -> CatalogTable {
    db.datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
        .expect("the table exists")
}

/// Bring the table `name` of `db` to `version` without a commit: its
/// newest metadata file is copied under the name of every version after it
/// up to `version`, as commits that change nothing leave the directory.
fn copy_newest_version_up_to(db: &Database, name: &str, version: u64) {
    let metadata_dir = db.table_path(name).join("metadata");
    let versions = metadata_versions(&db.root_uri(), &db.table_directory(name)).unwrap();
    let newest = *versions.last().expect("the table has a version");
    let locate = |version: u64| metadata_dir.join(format!("v{version:020}.metadata.json"));
    for next in newest + 1..=version {
        std::fs::copy(locate(newest), locate(next)).unwrap();
    }
}

/// The metadata directory of the table `name` of `db`, a database rooted
/// at `root`.
fn locate_metadata_dir(root: &Path, db: &Database, name: &str) -> PathBuf {
    root.join(db.table_directory(name)).join("metadata")
}

/// How many manifests and manifest lists `metadata_dir` holds.
fn count_avro_objects(metadata_dir: &Path) -> usize {
    names_in(metadata_dir)
        .iter()
        .filter(|name| name.ends_with(".avro"))
        .count()
}

/// Run one vacuum round as of `now_ms` from another process serving the
/// database of `db`: a datastore of its own over the same directory, opened
/// through a store that takes no lock on it.
fn vacuum_from_another_process(db: &Database, now_ms: u64) {
    let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
        inner: Arc::new(LocalStore::new(db.path()).unwrap()),
        faults: Arc::default(),
    });
    let other_process = PivotlakeDatastore::from_store(
        store,
        &shared_dispatcher(),
        None,
        Some(TableFormat::Iceberg),
    )
    .unwrap();
    Vacuumer::new(DEFAULT_VACUUM_POLL, other_process)
        .vacuum_all(now_ms)
        .unwrap();
}

#[test]
fn a_new_table_is_version_one_with_no_hint() {
    let mut db = Database::new();

    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let created_at = hold_table(&db, "t").version();
    let reopened = Database::reopen(db.close());

    assert_eq!(created_at, 1);
    assert_eq!(
        metadata_versions(&reopened.root_uri(), &directory).unwrap(),
        [1]
    );
    assert_eq!(
        hinted_version(&reopened.root_uri(), &directory).unwrap(),
        None
    );
    assert_eq!(hold_table(&reopened, "t").version(), 1);
}

#[test]
fn metadata_file_names_are_padded_and_parse_back_strictly() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let metadata_dir = db.table_path("t").join("metadata");
    for stray in [
        "v12.metadata.json",
        "v00000000000000000012.metadata.json.tmp",
    ] {
        std::fs::write(metadata_dir.join(stray), b"not a version").unwrap();
    }

    db.execute("INSERT INTO t VALUES (1)");
    let versions = metadata_versions(&db.root_uri(), &directory).unwrap();
    let mut reopened = Database::reopen(db.close());
    let count = reopened.execute("SELECT count(*) FROM t");

    assert!(
        metadata_dir
            .join("v00000000000000000001.metadata.json")
            .is_file()
    );
    assert!(
        metadata_dir
            .join("v00000000000000000002.metadata.json")
            .is_file()
    );
    assert_eq!(versions, [1, 2]);
    assert_eq!(single_i64(&count), 1);
}

#[test]
fn padded_names_sort_in_version_order() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let metadata_dir = db.table_path("t").join("metadata");
    for file in 0..10 {
        append_version(&db, &directory, &format!("f{file}.parquet"));
    }

    let mut names: Vec<String> = names_in(&metadata_dir)
        .into_iter()
        .filter(|name| name.ends_with(".metadata.json"))
        .collect();
    names.sort();
    let reopened = Database::reopen(db.close());

    let in_version_order: Vec<String> = (1..=11)
        .map(|version| format!("v{version:020}.metadata.json"))
        .collect();
    assert_eq!(names, in_version_order);
    assert_eq!(hold_table(&reopened, "t").version(), 11);
}

#[test]
fn a_hint_is_read_as_the_padded_digits_and_rejects_anything_else() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let hint_path = db.table_path("t").join("metadata/pivot-version-hint.text");

    std::fs::write(&hint_path, "00000000000000000001\n").unwrap();
    let with_a_newline = hinted_version(&db.root_uri(), &directory).unwrap();
    std::fs::write(&hint_path, "1").unwrap();
    let unpadded = hinted_version(&db.root_uri(), &directory);

    assert_eq!(with_a_newline, Some(1));
    assert!(matches!(
        unpadded,
        Err(Error::Iceberg(
            iceberg_tables::Error::MalformedVersionHint { .. }
        ))
    ));
}

#[test]
fn a_commit_whose_write_succeeded_behind_a_lost_answer_reports_success() {
    let (directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let table_directory = db.table_directory("t");
    let metadata_dir = locate_metadata_dir(directory.path(), &db, "t");
    faults.lost_put_answers.store(1, Ordering::SeqCst);

    db.execute("INSERT INTO t VALUES (1)");

    assert_eq!(hold_table(&db, "t").version(), 2);
    assert_eq!(
        live_data_file_names(&db.root_uri(), &table_directory)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(count_avro_objects(&metadata_dir), 2);
}

#[test]
fn a_commit_whose_write_never_answers_keeps_its_objects_and_says_so() {
    let (directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let metadata_dir = locate_metadata_dir(directory.path(), &db, "t");
    let mut table = hold_table(&db, "t");
    faults.fail_metadata_puts.store(true, Ordering::SeqCst);

    let error = table
        .append_data_file(ObjectPath::new("a.parquet"), &int_parquet_bytes(&[1]), None)
        .expect_err("the commit's outcome is unknown");

    assert!(error.may_have_committed());
    assert!(error.to_string().contains("outcome is unknown"), "{error}");
    assert_eq!(count_avro_objects(&metadata_dir), 2);
}

#[test]
fn an_insert_reads_of_the_store_the_manifest_list_it_is_built_on_and_the_one_it_wrote() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    faults.take_requests();

    db.execute("INSERT INTO t VALUES (2)");

    let requests = faults.take_requests();
    assert_eq!(hold_table(&db, "t").version(), 3);
    assert_eq!(requests.object_reads, 2);
    assert_eq!(requests.listings, 0);
}

#[test]
fn an_insert_whose_version_cannot_be_loaded_succeeds_and_keeps_its_file() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let table_directory = db.table_directory("t");
    faults.fail_gets.store(true, Ordering::SeqCst);

    db.execute("INSERT INTO t VALUES (1)");
    let held_after_insert = hold_table(&db, "t").version();
    faults.fail_gets.store(false, Ordering::SeqCst);
    db.datastore.refresh_from_store().unwrap();
    let count = db.execute("SELECT count(*) FROM t");

    assert_eq!(held_after_insert, 1);
    assert_eq!(hold_table(&db, "t").version(), 2);
    assert_eq!(single_i64(&count), 1);
    assert_eq!(
        live_data_file_names(&db.root_uri(), &table_directory)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn a_writer_that_lost_the_race_leaves_no_manifest_behind() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let metadata_dir = db.table_path("t").join("metadata");
    // The datastore still holds version 1, so its insert is built on it.
    append_version(&db, &directory, "winner.parquet");
    let objects_of_winner = count_avro_objects(&metadata_dir);

    db.insert_rows("t", vec![int_batch(&[1])]);

    // The insert lost version 2 and landed as version 3: the manifest and
    // the manifest list of that commit are the only objects it added.
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3]
    );
    assert_eq!(count_avro_objects(&metadata_dir), objects_of_winner + 2);
}

#[test]
fn an_unchanged_table_is_reported_unchanged_with_one_listing_and_no_read() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let mut table = hold_table(&db, "t");
    faults.take_requests();

    let advanced = table.refresh().unwrap();

    let requests = faults.take_requests();
    assert!(!advanced);
    assert_eq!(requests.object_reads, 0);
    assert_eq!(requests.listings, 1);
}

#[test]
fn a_version_committed_through_another_store_is_found_by_reading_only_it() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let table_directory = db.table_directory("t");
    let mut table = hold_table(&db, "t");
    let committed = commit_external_append(
        &db.root_uri(),
        &table_directory,
        "a.parquet",
        &int_parquet_bytes(&[1]),
        1,
    )
    .unwrap();
    faults.take_requests();

    let advanced = table.refresh().unwrap();

    let requests = faults.take_requests();
    assert!(advanced);
    assert_eq!(table.version(), committed);
    // The version's metadata file and its manifest list.
    assert_eq!(requests.object_reads, 2);
    assert_eq!(requests.listings, 1);
}

#[test]
fn tables_sharing_a_store_refresh_to_the_latest_version_from_wherever_they_are() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let mut pinned = hold_table(&db, "t");
    db.execute("INSERT INTO t VALUES (1)");
    let mut latest = hold_table(&db, "t");

    let advanced = pinned.refresh().unwrap();
    let advanced_again = latest.refresh().unwrap();

    assert!(advanced);
    assert_eq!(pinned.version(), latest.version());
    assert_eq!(pinned.file_refs().unwrap().len(), 1);
    assert!(!advanced_again);
}

#[test]
fn the_version_a_store_committed_is_its_latest_without_a_read() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    let mut held_before = hold_table(&db, "t");
    db.execute("INSERT INTO t VALUES (1)");
    faults.take_requests();

    let advanced = held_before.refresh().unwrap();

    let requests = faults.take_requests();
    assert!(advanced);
    assert_eq!(held_before.version(), 2);
    // The metadata is the one the store wrote: only the version's manifest
    // list is read.
    assert_eq!(requests.object_reads, 1);
    assert_eq!(requests.listings, 1);
}

#[test]
fn a_store_whose_latest_seen_version_was_deleted_finds_the_current_one() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let mut table = hold_table(&db, "t");
    let seen = table.version();
    let committed = commit_external_append(
        &db.root_uri(),
        &directory,
        "a.parquet",
        &int_parquet_bytes(&[1]),
        1,
    )
    .unwrap();
    let seen_file = format!("metadata/v{seen:020}.metadata.json");
    std::fs::remove_file(db.table_path("t").join(seen_file)).unwrap();

    let advanced = table.refresh().unwrap();

    assert_eq!(seen, 1);
    assert!(advanced);
    assert_eq!(table.version(), committed);
}

#[test]
fn a_commit_drops_no_metadata_log_entry_however_many_there_are() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");

    for value in 0..120 {
        db.insert_rows("t", vec![int_batch(&[value])]);
    }

    let every_earlier_version: Vec<u64> = (1..=120).collect();
    assert_eq!(
        logged_metadata_versions(&db.root_uri(), &directory).unwrap(),
        every_earlier_version
    );
}

#[test]
fn a_table_that_caps_its_previous_versions_at_100_keeps_every_log_entry_across_commits() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let cap_property = "write.metadata.previous-versions-max";
    set_table_property(&db.root_uri(), &directory, cap_property, "100").unwrap();
    db.datastore.refresh_from_store().unwrap();

    for value in 0..105 {
        db.insert_rows("t", vec![int_batch(&[value])]);
    }

    assert_eq!(
        logged_metadata_versions(&db.root_uri(), &directory).unwrap(),
        (1..=106).collect::<Vec<u64>>()
    );
}

#[test]
fn an_expiry_whose_outcome_is_unknown_hands_back_no_metadata_file_to_delete() {
    let (_directory, faults, mut db) = open_faulty_database();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (2)");
    let table_directory = db.table_directory("t");
    faults.fail_metadata_puts.store(true, Ordering::SeqCst);

    let error = Vacuumer::new(DEFAULT_VACUUM_POLL, db.datastore.clone())
        .vacuum_all(now_ms() + FIVE_HOURS_MS)
        .expect_err("the expiry's outcome is unknown");

    assert!(error.may_have_committed());
    assert_eq!(
        metadata_versions(&db.root_uri(), &table_directory).unwrap(),
        [1, 2, 3]
    );
}

#[test]
fn a_writer_whose_base_outlived_the_window_loses_after_its_successor_is_deleted() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let (root, directory) = (db.root_uri(), db.table_directory("t"));
    let window_property = "history.expire.max-snapshot-age-ms";
    set_table_property(&root, &directory, window_property, "1").unwrap();
    // The datastore holds version 2 from here on: a stale writer's base.
    db.datastore.refresh_from_store().unwrap();
    append_version(&db, &directory, "a.parquet");
    append_version(&db, &directory, "b.parquet");
    std::thread::sleep(Duration::from_millis(5));
    vacuum_from_another_process(&db, now_ms());
    assert_eq!(metadata_versions(&root, &directory).unwrap(), [4, 5]);

    db.insert_rows("t", vec![int_batch(&[3])]);

    // Version 3, the name the writer's commit takes, was free: the commit
    // withdrew it and landed on top of the current version.
    assert_eq!(metadata_versions(&root, &directory).unwrap(), [4, 5, 6]);
    assert_eq!(
        live_data_file_names(&root, &directory).unwrap().len(),
        3,
        "the two files committed meanwhile and the writer's own"
    );
}

#[test]
fn the_hint_is_written_by_the_commit_of_version_one_hundred_and_by_none_around_it() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let (root, directory) = (db.root_uri(), db.table_directory("t"));
    let hint_path = db.table_path("t").join("metadata/pivot-version-hint.text");
    copy_newest_version_up_to(&db, "t", 98);

    append_version(&db, &directory, "a.parquet");
    let hinted_at_99 = hinted_version(&root, &directory).unwrap();
    append_version(&db, &directory, "b.parquet");
    let written_at_100 = std::fs::read_to_string(&hint_path).unwrap();
    append_version(&db, &directory, "c.parquet");
    let hinted_at_101 = hinted_version(&root, &directory).unwrap();

    assert_eq!(hinted_at_99, None);
    assert_eq!(written_at_100, "00000000000000000100");
    assert_eq!(hinted_at_101, Some(100));
}

#[test]
fn a_table_with_no_hint_is_opened_at_its_newest_version() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    for file in ["a.parquet", "b.parquet", "c.parquet"] {
        append_version(&db, &directory, file);
    }
    let hinted = hinted_version(&db.root_uri(), &directory).unwrap();

    let reopened = Database::reopen(db.close());

    assert_eq!(hinted, None);
    assert_eq!(hold_table(&reopened, "t").version(), 4);
}

#[test]
fn a_table_whose_hint_is_several_versions_behind_is_opened_at_its_newest_version() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    copy_newest_version_up_to(&db, "t", 99);
    for file in ["a.parquet", "b.parquet", "c.parquet", "d.parquet"] {
        append_version(&db, &directory, file);
    }
    let hinted = hinted_version(&db.root_uri(), &directory).unwrap();

    let reopened = Database::reopen(db.close());

    assert_eq!(hinted, Some(100));
    assert_eq!(hold_table(&reopened, "t").version(), 103);
}

/// A store lists from the name the hint holds, not from a file: the hinted
/// version and the fifty after it are gone, as an expiry leaves them, and
/// the listing still runs on to the newest.
#[test]
fn a_table_whose_hinted_version_was_deleted_is_opened_at_its_newest_version() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let metadata_dir = db.table_path("t").join("metadata");
    copy_newest_version_up_to(&db, "t", 99);
    append_version(&db, &directory, "a.parquet");
    let hinted = hinted_version(&db.root_uri(), &directory).unwrap();
    copy_newest_version_up_to(&db, "t", 199);
    for version in 100..=150 {
        let deleted = metadata_dir.join(format!("v{version:020}.metadata.json"));
        std::fs::remove_file(deleted).unwrap();
    }

    let reopened = Database::reopen(db.close());

    assert_eq!(hinted, Some(100));
    assert_eq!(hold_table(&reopened, "t").version(), 199);
}

#[test]
fn a_table_that_wrote_its_hint_holds_no_file_named_version_hint_text() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let metadata_dir = db.table_path("t").join("metadata");
    copy_newest_version_up_to(&db, "t", 99);

    append_version(&db, &directory, "a.parquet");

    let names = names_in(&metadata_dir);
    assert!(names.contains(&"pivot-version-hint.text".to_string()));
    assert!(!names.contains(&"version-hint.text".to_string()));
}
