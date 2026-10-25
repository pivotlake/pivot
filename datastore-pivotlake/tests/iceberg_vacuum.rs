//! Vacuum deletes what a managed table no longer references once it has
//! been unreferenced for the retention window, and never touches a live
//! file, a file some retained snapshot still names, or an object it does not
//! own. Age is judged against a `now_ms` threaded in, so the window is
//! exercised without waiting on the wall clock.

mod iceberg_support;

use std::path::PathBuf;
use std::sync::Arc;

use datastore_pivotlake::test_support::{
    commit_external_append, commit_external_replace, hinted_version, live_data_file_names,
    metadata_versions, retained_snapshots,
};
use datastore_pivotlake::{DEFAULT_VACUUM_POLL, Vacuumer};
use iceberg_support::{
    Database, FIVE_HOURS_MS, backdate_past_retention, names_in, now_ms, single_i64,
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

fn vacuum(db: &Database, now_ms: u64) {
    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, db.datastore.clone()))
        .vacuum_all(now_ms)
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
fn an_orphan_past_the_window_is_deleted_and_the_live_file_is_kept() {
    let (db, adopted_dir, table_dir) = adopting_database();
    write_int_parquet(&table_dir.join("orphan.parquet"), &[9]);

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(!table_dir.join("orphan.parquet").exists());
    assert!(adopted_dir.join("live.parquet").exists());
}

#[test]
fn an_orphan_inside_the_window_is_kept() {
    let (db, _adopted_dir, table_dir) = adopting_database();
    write_int_parquet(&table_dir.join("orphan.parquet"), &[9]);

    vacuum(&db, now_ms());

    assert!(table_dir.join("orphan.parquet").exists());
}

#[test]
fn a_file_in_the_adopted_directory_is_never_deleted() {
    let (db, adopted_dir, _table_dir) = adopting_database();
    write_int_parquet(&adopted_dir.join("theirs.parquet"), &[9]);

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(adopted_dir.join("theirs.parquet").exists());
}

#[test]
fn a_retired_file_is_kept_while_its_snapshot_is_inside_the_window() {
    let mut db = Database::new();
    let (_directory, retired, _live) = table_with_a_retired_file(&mut db);
    backdate_past_retention(&db.table_path("t").join(&retired));

    vacuum(&db, now_ms());

    assert!(
        db.table_path("t").join(&retired).exists(),
        "a file a retained snapshot names is kept whatever its mtime says"
    );
}

#[test]
fn a_retired_file_and_its_snapshots_objects_go_once_the_successor_is_past_the_window() {
    let mut db = Database::new();
    let (directory, retired, live) = table_with_a_retired_file(&mut db);
    let metadata_dir = db.table_path("t").join("metadata");
    let objects_before = names_in(&metadata_dir);

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(!db.table_path("t").join(&retired).exists());
    assert!(db.table_path("t").join(&live).exists());
    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 1);
    // The expiry was committed as a version of its own.
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3, 4]
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
fn vacuum_keeps_the_hint_and_objects_it_does_not_own() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    let metadata_dir = db.table_path("t").join("metadata");
    std::fs::write(metadata_dir.join("notes.txt"), b"theirs").unwrap();
    backdate_past_retention(&metadata_dir.join("notes.txt"));
    backdate_past_retention(&metadata_dir.join("version-hint.text"));

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(metadata_dir.join("notes.txt").exists());
    assert!(metadata_dir.join("version-hint.text").exists());
}

#[test]
fn vacuum_advances_a_lagging_hint_before_sweeping() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (2)");
    let directory = db.table_directory("t");
    // A hint falls behind when its write fails or another tool commits
    // without one: vacuum brings it up before it deletes anything.
    std::fs::write(
        db.table_path("t").join("metadata/version-hint.text"),
        format!("{:020}", 1),
    )
    .unwrap();
    assert_eq!(hinted_version(&db.root_uri(), &directory).unwrap(), Some(1));

    vacuum(&db, now_ms());

    assert_eq!(hinted_version(&db.root_uri(), &directory).unwrap(), Some(3));
}

#[test]
fn an_orphan_manifest_from_a_lost_commit_is_deleted_past_retention() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    let metadata_dir = db.table_path("t").join("metadata");
    let orphan = metadata_dir.join("00000000-0000-0000-0000-000000000000-m0.avro");
    std::fs::write(&orphan, b"a manifest no snapshot names").unwrap();

    vacuum(&db, now_ms());
    assert!(orphan.exists(), "inside the window the orphan is kept");
    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert!(!orphan.exists());
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
    for (name, _) in names_in(&db.table_path("t").join("metadata"))
        .iter()
        .map(|name| (name.clone(), ()))
    {
        backdate_past_retention(&db.table_path("t").join("metadata").join(name));
    }

    vacuum(&db, now_ms() + FIVE_HOURS_MS);

    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2, 3, 4, 5, 6],
        "every version the metadata log names is kept"
    );
}
