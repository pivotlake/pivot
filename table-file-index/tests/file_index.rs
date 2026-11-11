//! The order a partition's files stand in, and the index that keeps it.

use std::ops::ControlFlow;
use std::sync::Arc;

use object_storage::{FileRef, ObjectPath};
use table_file_index::{FileFacts, FileIndex, IndexedRecord, PartitionFiles};

struct Record(FileFacts);

impl IndexedRecord for Record {
    fn facts(&self) -> &FileFacts {
        &self.0
    }

    fn estimate_heap_bytes(&self) -> u64 {
        self.0.estimate_heap_bytes()
    }
}

fn record(path: &str, sort_minimum: &[u8], commit_sequence: u64) -> Arc<Record> {
    let file = FileRef {
        path: ObjectPath::new(path),
        size: 10,
    };
    Arc::new(Record(FileFacts::new(
        file,
        Some(1),
        Arc::from(sort_minimum),
        None,
        commit_sequence,
    )))
}

fn list_paths(index: &FileIndex<Record>) -> Vec<&str> {
    index
        .files()
        .map(|record| record.facts().file().path.as_str())
        .collect()
}

#[test]
fn files_stand_by_where_they_start_on_the_sort_key() {
    let mut index = FileIndex::default();

    index.insert(record("a", &[9], 1));
    index.insert(record("b", &[2], 2));

    assert_eq!(list_paths(&index), ["b", "a"]);
}

#[test]
fn files_starting_at_one_place_stand_in_commit_order_then_by_path() {
    let mut index = FileIndex::default();

    index.insert(record("z", &[], 1));
    index.insert(record("b", &[], 2));
    index.insert(record("a", &[], 2));

    assert_eq!(list_paths(&index), ["z", "a", "b"]);
}

#[test]
fn a_file_inserted_at_a_held_path_replaces_it() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[1], 1));

    index.insert(record("a", &[5], 3));

    let held = index.find(&ObjectPath::new("a")).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(held.facts().commit_sequence(), 3);
}

#[test]
fn removing_a_file_takes_its_bytes_out_of_the_estimate() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[1], 1));
    let one_file_bytes = index.estimate_heap_bytes();
    index.insert(record("b", &[2], 1));

    let removed = index.remove(&ObjectPath::new("b"));

    assert!(removed.is_some());
    assert_eq!(index.estimate_heap_bytes(), one_file_bytes);
    assert_eq!(list_paths(&index), ["a"]);
}

#[test]
fn an_edit_leaves_a_clone_as_it_was() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[1], 1));
    let earlier = index.clone();

    index.insert(record("b", &[2], 1));

    assert_eq!(list_paths(&earlier), ["a"]);
}

#[test]
fn a_walk_over_the_facts_stops_where_the_visit_breaks() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[1], 1));
    index.insert(record("b", &[2], 1));
    let mut visited = Vec::new();

    let outcome = index.visit_files(&mut |facts| {
        visited.push(facts.file().path.as_str().to_string());
        ControlFlow::Break(())
    });

    assert!(outcome.is_break());
    assert_eq!(visited, ["a"]);
}
