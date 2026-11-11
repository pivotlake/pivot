//! A local store's conditional create, its overwriting put under concurrent
//! writers, and its listing from a cursor.

use object_storage::{LocalStore, ObjectPath, ObjectStore, StoreError};

fn p(key: &str) -> ObjectPath {
    ObjectPath::new(key)
}

#[test]
fn put_if_absent_creates_once_then_conflicts_and_keeps_the_first_object() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalStore::new(dir.path()).unwrap();

    store.put_if_absent(&p("t/v1"), b"first").unwrap();
    let second = store.put_if_absent(&p("t/v1"), b"second");

    assert!(matches!(second, Err(StoreError::VersionConflict { .. })));
    assert_eq!(store.get(&p("t/v1")).unwrap().unwrap(), b"first");
}

#[test]
fn put_if_absent_leaves_no_staging_file_behind() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalStore::new(dir.path()).unwrap();
    store.put_if_absent(&p("k"), b"1").unwrap();

    let _ = store.put_if_absent(&p("k"), b"2");

    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, vec!["k"]);
}

#[test]
fn racing_creates_of_one_key_admit_exactly_one_writer() {
    let dir = tempfile::tempdir().unwrap();
    let store = std::sync::Arc::new(LocalStore::new(dir.path()).unwrap());

    let outcomes: Vec<_> = (0..16u8)
        .map(|writer| {
            let store = store.clone();
            std::thread::spawn(move || store.put_if_absent(&p("v"), &[writer]))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    let winners = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
    assert_eq!(winners, 1);
    assert!(
        outcomes
            .iter()
            .all(|outcome| matches!(outcome, Ok(()) | Err(StoreError::VersionConflict { .. })))
    );
}

#[test]
fn concurrent_puts_of_one_key_all_succeed_and_leave_one_whole_object() {
    let dir = tempfile::tempdir().unwrap();
    let store = std::sync::Arc::new(LocalStore::new(dir.path()).unwrap());

    let handles: Vec<_> = (0..8u8)
        .map(|writer| {
            let store = store.clone();
            std::thread::spawn(move || store.put(&p("k"), &[writer; 64]))
        })
        .collect();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    let bytes = store.get(&p("k")).unwrap().unwrap();
    assert_eq!(bytes.len(), 64);
    assert!(bytes.iter().all(|byte| *byte == bytes[0]));
}

#[test]
fn a_listing_after_a_cursor_returns_only_the_names_past_it() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalStore::new(dir.path()).unwrap();
    for name in ["v01", "v02", "v03", "w01"] {
        store.put(&p(&format!("d/{name}")), b"x").unwrap();
    }

    let listed = store.list_filtered(&p("d"), "v", Some("v02")).unwrap();

    let mut names: Vec<_> = listed
        .objects
        .iter()
        .map(|object| object.file.path.to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["v03"]);
}
