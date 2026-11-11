//! Routine cleanup expires a managed table's snapshots and previous metadata
//! files once they are past the retention window and deletes exactly those
//! files and what only those snapshots used, with no listing of the table's
//! directory: it never touches a live file, a file some retained snapshot
//! still names, a metadata file replaced inside the window, or a file no
//! metadata names. Age is judged against a `now_ms` threaded in, so the
//! window is not waited out on the wall clock.

mod iceberg_support;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use datastore_pivotlake::test_support::{
    commit_external_append, commit_external_replace, live_data_file_names,
    logged_metadata_versions, metadata_versions, retained_snapshots, set_table_property,
    tag_current_snapshot,
};
use datastore_pivotlake::{DEFAULT_VACUUM_POLL, Vacuumer};
use iceberg_support::{
    Database, FIVE_HOURS_MS, backdate_by, int_parquet_bytes, names_in, now_ms, single_i64,
    write_int_parquet,
};

/// A database with `events` adopting `live.parquet` from the `events`
/// directory under the root, and the table's own directory.
fn adopting_database() -> (Database, PathBuf, PathBuf) {
    let mut db = Database::new();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_int_parquet(&adopted_dir.join("live.parquet"), &[1, 2, 3]);
    db.execute("CREATE TABLE events (value BIGINT) WITH (with_pre_existing_parquets = 'events')");
    let table_dir = db.table_path("events");
    (db, adopted_dir, table_dir)
}

/// One vacuum round: what the background maintenance timer and the SQL
/// `VACUUM` command both run.
fn vacuum(db: &Database, now_ms: u64) {
    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, db.datastore.clone()))
        .vacuum_all(now_ms)
        .unwrap();
}

const THIRTY_DAYS: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// The retention window of a table that sets none.
const WINDOW_MS: u64 = 4 * 60 * 60 * 1000;

/// The wall clock at a moment no commit is dated at: every commit before
/// the call is dated earlier and every commit after it later.
fn pause_between_commits() -> u64 {
    std::thread::sleep(Duration::from_millis(5));
    let paused_at = now_ms();
    std::thread::sleep(Duration::from_millis(5));
    paused_at
}

/// Create the table `t` and return its directory: a table with its first
/// version and no file.
fn create_empty_table(db: &mut Database) -> String {
    db.execute("CREATE TABLE t (value BIGINT)");
    db.table_directory("t")
}

/// Commit one more version of the table at `directory`, adding the file
/// `file_name`, as a writer in another process does.
fn append_version(db: &Database, directory: &str, file_name: &str) {
    commit_external_append(
        &db.root_uri(),
        directory,
        file_name,
        &int_parquet_bytes(&[1]),
        1,
    )
    .unwrap();
}

/// A table whose first file `a` was retired by a rewrite into `b`; returns
/// the table directory and the two files' names.
fn table_with_a_retired_file(db: &mut Database) -> (String, String, String) {
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    let directory = db.table_directory("t");
    let first = live_data_file_names(&db.root_uri(), &directory).unwrap()[0].clone();
    let bytes = std::fs::read(db.table_path("t").join(&first)).unwrap();
    commit_external_replace(
        &db.root_uri(),
        &directory,
        &[&first],
        "b.parquet",
        &bytes,
        1,
    )
    .unwrap();
    (directory, first, "b.parquet".to_string())
}

#[test]
fn a_file_in_the_adopted_directory_is_never_deleted() {
    let (db, adopted_dir, _table_dir) = adopting_database();
    write_int_parquet(&adopted_dir.join("theirs.parquet"), &[9]);
    backdate_by(&adopted_dir.join("theirs.parquet"), THIRTY_DAYS);

    vacuum(&db, now_ms());

    assert!(adopted_dir.join("theirs.parquet").exists());
}

#[test]
fn a_retired_file_is_kept_while_its_snapshot_is_inside_the_window() {
    let mut db = Database::new();
    let (_directory, retired, _live) = table_with_a_retired_file(&mut db);
    backdate_by(&db.table_path("t").join(&retired), THIRTY_DAYS);

    vacuum(&db, now_ms());

    assert!(
        db.table_path("t").join(&retired).exists(),
        "a file a retained snapshot names is kept whatever its mtime says"
    );
}

#[test]
fn routine_cleanup_deletes_a_retired_file_and_its_snapshots_objects_once_the_snapshot_expires() {
    let mut db = Database::new();
    let (directory, retired, live) = table_with_a_retired_file(&mut db);
    let metadata_dir = db.table_path("t").join("metadata");
    let objects_before = names_in(&metadata_dir);

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(!db.table_path("t").join(&retired).exists());
    assert!(db.table_path("t").join(&live).exists());
    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 1);
    // The expiry was committed as a version of its own, and the two
    // versions replaced before the window went with the snapshot.
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [3, 4]
    );
    let objects_after = names_in(&metadata_dir);
    let swept: Vec<_> = objects_before
        .iter()
        .filter(|name| !objects_after.contains(name))
        .collect();
    assert_eq!(
        swept
            .iter()
            .filter(|name| name.starts_with("snap-"))
            .count(),
        1,
        "the expired snapshot's manifest list is gone: {swept:?}"
    );
    assert!(
        swept.iter().any(|name| name.ends_with("-m0.avro")),
        "the manifest only the expired snapshot named is gone: {swept:?}"
    );
    let count = db.execute("SELECT count(*) FROM t");
    assert_eq!(single_i64(&count), 1);
}

#[test]
fn routine_cleanup_keeps_a_retired_file_while_its_snapshot_is_inside_the_window() {
    let mut db = Database::new();
    let (directory, retired, _live) = table_with_a_retired_file(&mut db);

    vacuum(&db, now_ms());

    assert!(db.table_path("t").join(&retired).exists());
    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 2);
}

#[test]
fn routine_cleanup_keeps_a_file_a_tagged_snapshot_uses_when_the_snapshot_that_retired_it_expires() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    let directory = db.table_directory("t");
    let tagged_file = live_data_file_names(&db.root_uri(), &directory).unwrap()[0].clone();
    let bytes = std::fs::read(db.table_path("t").join(&tagged_file)).unwrap();
    tag_current_snapshot(&db.root_uri(), &directory, "release").unwrap();
    let (root, replaced) = (db.root_uri(), [tagged_file.as_str()]);
    commit_external_replace(&root, &directory, &replaced, "b.parquet", &bytes, 1).unwrap();
    commit_external_append(&root, &directory, "c.parquet", &bytes, 1).unwrap();

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert_eq!(
        retained_snapshots(&root, &directory).unwrap(),
        2,
        "the replace expired; the tagged snapshot and the current one stay"
    );
    assert!(db.table_path("t").join(&tagged_file).exists());
}

#[test]
fn routine_cleanup_never_deletes_a_file_no_metadata_names() {
    let mut db = Database::new();
    let (_directory, _retired, _live) = table_with_a_retired_file(&mut db);
    let table_dir = db.table_path("t");
    std::fs::create_dir_all(table_dir.join("data/nested")).unwrap();
    let strays = [
        table_dir.join("orphan.parquet"),
        table_dir.join("notes.bin"),
        table_dir.join("data/nested/orphan.parquet"),
        table_dir.join("metadata/00000000-0000-0000-0000-000000000000-m0.avro"),
        table_dir.join("metadata/snap-1-1-00000000-0000-0000-0000-000000000000.avro"),
    ];
    for stray in &strays {
        std::fs::write(stray, b"named by no metadata").unwrap();
        backdate_by(stray, THIRTY_DAYS);
    }

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    for stray in &strays {
        assert!(stray.exists(), "{}", stray.display());
    }
}

#[test]
fn a_table_with_gc_disabled_expires_no_snapshot_and_deletes_no_file() {
    let mut db = Database::new();
    let (directory, retired, _live) = table_with_a_retired_file(&mut db);
    set_table_property(&db.root_uri(), &directory, "gc.enabled", "false").unwrap();
    let orphan = db.table_path("t").join("orphan.parquet");
    write_int_parquet(&orphan, &[9]);
    backdate_by(&orphan, THIRTY_DAYS);

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 2);
    assert!(db.table_path("t").join(&retired).exists());
    assert!(orphan.exists());
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3, 4]
    );
    assert_eq!(
        logged_metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3]
    );
}

#[test]
fn an_insert_expires_no_snapshot_however_far_past_the_window() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let window_property = "history.expire.max-snapshot-age-ms";
    set_table_property(&db.root_uri(), &directory, window_property, "1").unwrap();
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (2)");
    std::thread::sleep(Duration::from_millis(5));

    db.execute("INSERT INTO t VALUES (3)");

    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 3);
}

#[test]
fn a_reopened_datastore_expires_by_the_history_it_loads() {
    let mut db = Database::new();
    let (directory, retired, _live) = table_with_a_retired_file(&mut db);
    let table_path = db.table_path("t");
    let reopened = Database::reopen(db.close());

    vacuum(&reopened, now_ms());
    assert!(table_path.join(&retired).exists());
    vacuum(&reopened, now_ms() + FIVE_HOURS_MS);

    assert!(!table_path.join(&retired).exists());
    assert_eq!(
        retained_snapshots(&reopened.root_uri(), &directory).unwrap(),
        1
    );
}

#[test]
fn old_metadata_versions_are_kept_while_the_metadata_log_names_them() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let directory = db.table_directory("t");
    let bytes = {
        db.execute("INSERT INTO t VALUES (1)");
        let first = live_data_file_names(&db.root_uri(), &directory).unwrap()[0].clone();
        std::fs::read(db.table_path("t").join(first)).unwrap()
    };
    for version in 0..3 {
        commit_external_append(
            &db.root_uri(),
            &directory,
            &format!("x{version}.parquet"),
            &bytes,
            1,
        )
        .unwrap();
    }
    for name in names_in(&db.table_path("t").join("metadata")) {
        backdate_by(&db.table_path("t").join("metadata").join(name), THIRTY_DAYS);
    }

    vacuum(&db, now_ms());

    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3, 4, 5],
        "every version the metadata log names is kept, however old its file"
    );
}

#[test]
fn routine_cleanup_deletes_the_metadata_files_replaced_before_the_window_and_no_other() {
    let mut db = Database::new();
    let directory = create_empty_table(&mut db);
    append_version(&db, &directory, "a.parquet");
    append_version(&db, &directory, "b.parquet");
    let window_start = pause_between_commits();
    append_version(&db, &directory, "c.parquet");

    vacuum(&db, window_start + WINDOW_MS);

    // Versions 2 and 3 replaced 1 and 2 before the window; 4 replaced 3
    // inside it, and 5 is the expiry itself.
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [3, 4, 5]
    );
    assert_eq!(
        logged_metadata_versions(&db.root_uri(), &directory).unwrap(),
        [3, 4]
    );
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory).unwrap(),
        ["a.parquet", "b.parquet", "c.parquet"]
    );
}

#[test]
fn routine_cleanup_with_nothing_past_the_window_commits_nothing_and_deletes_nothing() {
    let mut db = Database::new();
    let directory = create_empty_table(&mut db);
    append_version(&db, &directory, "a.parquet");
    append_version(&db, &directory, "b.parquet");
    let metadata_dir = db.table_path("t").join("metadata");
    let objects_before = names_in(&metadata_dir);

    vacuum(&db, now_ms() + WINDOW_MS - 60 * 1000);

    assert_eq!(names_in(&metadata_dir), objects_before);
}

#[test]
fn a_reopened_datastore_deletes_the_metadata_files_its_log_names() {
    let mut db = Database::new();
    let directory = create_empty_table(&mut db);
    append_version(&db, &directory, "a.parquet");
    append_version(&db, &directory, "b.parquet");
    let reopened = Database::reopen(db.close());

    vacuum(&reopened, now_ms() + FIVE_HOURS_MS);

    assert_eq!(
        metadata_versions(&reopened.root_uri(), &directory).unwrap(),
        [3, 4]
    );
    assert_eq!(
        logged_metadata_versions(&reopened.root_uri(), &directory).unwrap(),
        [3]
    );
}
