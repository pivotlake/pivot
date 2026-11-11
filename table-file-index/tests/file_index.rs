//! The order a partition's files stand in, and the index that keeps it.

mod support;

use std::ops::ControlFlow;

use object_storage::ObjectPath;
use support::{index_of, list_paths, record};
use table_file_index::SortValue::{Float, Integer, Missing, String as Text};
use table_file_index::{FileIndex, IndexedRecord, PartitionFiles};

#[test]
fn files_stand_by_where_they_start_on_the_sort_key() {
    let mut index = FileIndex::default();

    index.insert(record("a", &[Integer(9)], 1));
    index.insert(record("b", &[Integer(2)], 2));

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
fn a_file_starting_at_a_prefix_stands_before_the_file_starting_at_what_extends_it() {
    let files = [
        record("extended", &[Text("ab\u{e9}")], 1),
        record("prefix", &[Text("ab")], 2),
    ];

    let index = index_of(files);

    assert_eq!(list_paths(&index), ["prefix", "extended"]);
}

#[test]
fn files_tied_on_the_first_sort_column_stand_by_the_second() {
    let files = [
        record("seven-late", &[Integer(7), Text("b")], 1),
        record("seven-early", &[Integer(7), Text("a")], 2),
        record("three", &[Integer(3), Text("z")], 3),
    ];

    let index = index_of(files);

    assert_eq!(list_paths(&index), ["three", "seven-early", "seven-late"]);
}

#[test]
fn a_file_without_a_bound_stands_before_every_file_with_one() {
    let files = [
        record("lowest", &[Integer(i128::MIN)], 1),
        record("unbounded", &[Missing], 2),
    ];

    let index = index_of(files);

    assert_eq!(list_paths(&index), ["unbounded", "lowest"]);
}

#[test]
fn files_starting_at_negative_floats_stand_before_those_starting_at_positive_ones() {
    let files = [
        record("small", &[Float(0.5)], 1),
        record("zero", &[Float(0.0)], 2),
        record("negative-zero", &[Float(-0.0)], 3),
        record("infinite", &[Float(f64::NEG_INFINITY)], 4),
        record("negative", &[Float(-2.5)], 5),
    ];

    let index = index_of(files);

    let by_value = ["infinite", "negative", "negative-zero", "zero", "small"];
    assert_eq!(list_paths(&index), by_value);
}

#[test]
fn a_file_inserted_at_a_held_path_replaces_it() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[Integer(1)], 1));

    index.insert(record("a", &[Integer(5)], 3));

    let held = index.find(&ObjectPath::new("a")).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(held.facts().commit_sequence(), 3);
}

#[test]
fn removing_one_of_many_files_starting_at_one_place_removes_exactly_it() {
    let names: Vec<String> = (0..500).map(|number| format!("file-{number:03}")).collect();
    let files = names
        .iter()
        .map(|name| record(name, &[Integer(4), Text("same")], 7));
    let mut index = index_of(files);

    let removed = index.remove(&ObjectPath::new("file-250")).unwrap();

    let left: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|name| *name != "file-250")
        .collect();
    assert_eq!(removed.facts().file().path.as_str(), "file-250");
    assert_eq!(list_paths(&index), left);
}

#[test]
fn removing_a_file_takes_its_bytes_out_of_the_estimate() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[Integer(1)], 1));
    let one_file_bytes = index.estimate_heap_bytes();
    index.insert(record("b", &[Integer(2)], 1));

    let removed = index.remove(&ObjectPath::new("b"));

    assert!(removed.is_some());
    assert_eq!(index.estimate_heap_bytes(), one_file_bytes);
    assert_eq!(list_paths(&index), ["a"]);
}

#[test]
fn an_edit_leaves_a_clone_as_it_was() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[Integer(1)], 1));
    let earlier = index.clone();

    index.insert(record("b", &[Integer(2)], 1));

    assert_eq!(list_paths(&earlier), ["a"]);
}

#[test]
fn a_walk_over_the_facts_stops_where_the_visit_breaks() {
    let mut index = FileIndex::default();
    index.insert(record("a", &[Integer(1)], 1));
    index.insert(record("b", &[Integer(2)], 1));
    let mut visited = Vec::new();

    let outcome = index.visit_files(&mut |facts| {
        visited.push(facts.file().path.as_str().to_string());
        ControlFlow::Break(())
    });

    assert!(outcome.is_break());
    assert_eq!(visited, ["a"]);
}

#[test]
fn a_walk_over_the_records_visits_them_in_key_order() {
    let files = [
        record("late", &[Integer(8)], 1),
        record("early", &[Integer(1)], 2),
    ];
    let index = index_of(files);
    let mut visited = Vec::new();

    let outcome = index.visit_records(&mut |record| {
        visited.push(record.facts().file().path.as_str().to_string());
        ControlFlow::Continue(())
    });

    assert!(outcome.is_continue());
    assert_eq!(visited, ["early", "late"]);
}
