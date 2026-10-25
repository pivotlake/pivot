//! The managed mode against every store backend: the conditional create a
//! commit lands on, the cursor listing a refresh uses, and a whole table
//! written and read through MinIO and `fake-gcs-server` (Docker; the remote
//! cases skip when it is absent) exactly as through a local directory.

mod iceberg_support;

use std::sync::Arc;

use datastore_pivotlake::test_support::{
    self, Backend, commit_external_append, live_data_file_names, metadata_versions,
};
use iceberg_support::{Database, collect_i64s, int_batch, int_parquet_bytes, single_i64};
use object_storage::{ObjectPath, ObjectStore, StoreError};

/// Every backend the harness can provide, each rooted at a prefix of its own
/// for `test_name`.
fn backends(test_name: &str) -> Vec<(&'static str, Backend)> {
    let (dir, local) = test_support::local();
    // The directory outlives the test binary; the local case needs no
    // cleanup beyond what the process's temporary directory gives.
    std::mem::forget(dir);
    let mut backends = vec![("local", local)];
    if let Some(s3) = test_support::s3(test_name) {
        backends.push(("s3", s3));
    } else {
        eprintln!("[{test_name}] skipping s3: MinIO unavailable");
    }
    if let Some(gcs) = test_support::gcs(test_name) {
        backends.push(("gcs", gcs));
    } else {
        eprintln!("[{test_name}] skipping gcs: fake-gcs-server unavailable");
    }
    backends
}

#[test]
fn put_if_absent_admits_exactly_one_of_racing_writers() {
    for (name, backend) in backends("managed-put-if-absent") {
        let store: Arc<dyn ObjectStore> = Arc::from(backend.store);
        let key = ObjectPath::new("t/metadata/v00000000000000000007.metadata.json");

        let outcomes: Vec<_> = (0..8u8)
            .map(|writer| {
                let store = Arc::clone(&store);
                let key = key.clone();
                std::thread::spawn(move || store.put_if_absent(&key, &[writer]))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        let winners = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
        assert_eq!(winners, 1, "{name}: {outcomes:?}");
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome, Ok(()) | Err(StoreError::VersionConflict { .. }))),
            "{name}: {outcomes:?}"
        );
        assert_eq!(store.get(&key).unwrap().unwrap().len(), 1, "{name}");
    }
}

#[test]
fn list_from_returns_only_names_after_the_cursor() {
    for (name, backend) in backends("managed-list-from") {
        let store = backend.store;
        for version in [
            "v00000000000000000009",
            "v00000000000000000010",
            "v00000000000000000011",
        ] {
            store
                .put(
                    &ObjectPath::new(format!("t/metadata/{version}.metadata.json")),
                    b"{}",
                )
                .unwrap();
        }
        store
            .put(&ObjectPath::new("t/metadata/version-hint.text"), b"9")
            .unwrap();

        let listed = store
            .list_from(
                &ObjectPath::new("t/metadata"),
                "v",
                "v00000000000000000009.metadata.json",
            )
            .unwrap();

        // The hint sorts after every padded version name and shares the
        // prefix, so it rides along; the reader parses names strictly and
        // ignores it.
        let mut names: Vec<_> = listed
            .objects
            .iter()
            .map(|object| object.file.path.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "v00000000000000000010.metadata.json",
                "v00000000000000000011.metadata.json",
                "version-hint.text",
            ],
            "{name}"
        );
    }
}

#[test]
fn a_table_is_written_read_and_appended_to_from_outside_on_every_backend() {
    for (name, backend) in backends("managed-table") {
        let store: Arc<dyn ObjectStore> = Arc::from(backend.store);
        let mut db = Database::over_store(backend.root.clone(), store);
        db.execute("CREATE TABLE events (value BIGINT)");
        db.insert_rows("events", vec![int_batch(&[1, 2, 3])]);
        let directory = db.table_directory("events");

        let version = commit_external_append(
            &backend.root,
            &directory,
            "external.parquet",
            &int_parquet_bytes(&[4]),
            1,
        )
        .unwrap();
        assert!(db.datastore.refresh_from_store().unwrap(), "{name}");
        let total = db.execute("SELECT CAST(sum(value) AS BIGINT) FROM events");
        let rows = db.execute("SELECT value FROM events ORDER BY value");

        assert_eq!(version, 3, "{name}");
        assert_eq!(single_i64(&total), 10, "{name}");
        assert_eq!(collect_i64s(&rows, 0), [1, 2, 3, 4], "{name}");
        assert_eq!(
            metadata_versions(&backend.root, &directory).unwrap(),
            [1, 2, 3],
            "{name}"
        );
        assert_eq!(
            live_data_file_names(&backend.root, &directory)
                .unwrap()
                .len(),
            2,
            "{name}"
        );
    }
}

#[test]
fn an_insert_racing_an_external_commit_lands_on_top_of_it() {
    for (name, backend) in backends("managed-race") {
        let store: Arc<dyn ObjectStore> = Arc::from(backend.store);
        let db = Database::over_store(backend.root.clone(), store);
        let mut db = db;
        db.execute("CREATE TABLE events (value BIGINT)");
        let directory = db.table_directory("events");
        let root = backend.root.clone();

        // The external writer loads version 1 and commits version 2 while
        // the in-process insert, also from version 1, commits concurrently:
        // one of them loses the conditional create and lands on top.
        let external = std::thread::spawn({
            let directory = directory.clone();
            move || {
                commit_external_append(
                    &root,
                    &directory,
                    "external.parquet",
                    &int_parquet_bytes(&[9]),
                    1,
                )
                .unwrap()
            }
        });
        db.insert_rows("events", vec![int_batch(&[1])]);
        external.join().unwrap();
        db.datastore.refresh_from_store().unwrap();

        let rows = db.execute("SELECT value FROM events ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 9], "{name}");
        assert_eq!(
            metadata_versions(&backend.root, &directory).unwrap(),
            [1, 2, 3],
            "{name}: both commits landed, one above the other"
        );
    }
}
