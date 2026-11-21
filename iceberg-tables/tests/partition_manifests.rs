//! Which manifests a plan finds a partition's files in: a scan, a
//! description and a lookup of one partition's files take each partition
//! from every manifest that holds a file of it, whatever else the table's
//! manifests hold.

mod support;

use std::sync::Arc;

use iceberg_tables::Table;
use iceberg_tables::test_support::Faults;
use object_storage::ObjectPath;
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use support::{
    Fixture, append, commit_replace, create_partitioned_table, describe, find_partition_holding,
    int_scalar, load, load_over_faulty_store, selected, selected_sorted, v_equals,
};

/// A table partitioned by `v` of three manifests, listed newest first. The
/// newest holds files of the partitions 1 and 3, so its manifest-list
/// entry admits the partition 2, of which it holds none; the one before it
/// holds files of the partitions 2 and 3; the oldest holds one file, of the
/// partition 2, which its entry names.
fn create_table_of_three_manifests() -> (Fixture, Table) {
    let (fixture, created) = create_partitioned_table();
    let oldest = append(&created, vec![fixture.write_file("two-a", 2)]);
    let middle = append(
        &oldest,
        vec![
            fixture.write_file("two-b", 2),
            fixture.write_file("three-b", 3),
        ],
    );
    let newest = append(
        &middle,
        vec![
            fixture.write_file("one-c", 1),
            fixture.write_file("three-c", 3),
        ],
    );
    (fixture, newest)
}

/// The filter `v >= value`, which leaves the partitions to the manifests.
fn v_at_least(value: i64) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(int_scalar(value))),
        compare_type: CompareType::GreaterEqual,
        return_type: Type::Boolean,
    })]
}

#[test]
fn a_scan_of_the_partition_its_filter_names_selects_its_files_from_every_manifest_that_holds_one() {
    let (fixture, _written) = create_table_of_three_manifests();
    let opened = load(&fixture);

    let files = selected(&opened, &v_equals(2));

    assert_eq!(files, ["two-b.parquet", "two-a.parquet"]);
}

#[test]
fn a_scan_of_the_partitions_its_manifests_hold_selects_each_in_turn_manifest_by_manifest() {
    let (fixture, _written) = create_table_of_three_manifests();
    let opened = load(&fixture);

    let files = selected(&opened, &v_at_least(2));

    let expected = [
        "two-b.parquet",
        "two-a.parquet",
        "three-c.parquet",
        "three-b.parquet",
    ];
    assert_eq!(files, expected);
}

#[test]
fn a_scan_of_the_same_partitions_selects_the_same_files_once_their_manifests_were_read() {
    let (fixture, _written) = create_table_of_three_manifests();
    let opened = load(&fixture);
    let first = selected(&opened, &v_at_least(2));

    let named = selected(&opened, &v_equals(2));
    let again = selected(&opened, &v_at_least(2));

    assert_eq!(named, ["two-b.parquet", "two-a.parquet"]);
    assert_eq!(again, first);
}

#[test]
fn a_description_lists_the_files_manifest_by_manifest_and_a_manifests_partition_by_partition() {
    let (fixture, _written) = create_table_of_three_manifests();
    let opened = load(&fixture);

    let description = describe(&opened);

    let files = description.files.iter();
    let paths: Vec<&str> = files.map(|file| file.path.as_str()).collect();
    let expected = [
        "one-c.parquet",
        "three-c.parquet",
        "two-b.parquet",
        "three-b.parquet",
        "two-a.parquet",
    ];
    assert_eq!(paths, expected);
}

#[test]
fn a_description_reads_each_manifest_once_when_nothing_is_cached() {
    let (fixture, _written) = create_table_of_three_manifests();
    let faults = Arc::new(Faults::default());
    let opened = load_over_faulty_store(&fixture, &faults);
    fixture
        .cache_slot
        .cache
        .remove_table(fixture.cache_slot.table_id);
    faults.take_requests();

    let description = describe(&opened);
    let requests = faults.take_requests();

    assert_eq!(description.files.len(), 5);
    assert_eq!(requests.manifests_sourced, 3);
}

#[test]
fn a_lookup_finds_a_partitions_files_in_every_manifest_that_holds_one() {
    let (fixture, written) = create_table_of_three_manifests();
    let partition = find_partition_holding(&written, "two-a.parquet");
    let opened = load(&fixture);
    let paths = ["two-a.parquet", "two-b.parquet", "three-b.parquet"].map(ObjectPath::new);

    let found = opened.find_datastore_entries(&partition, &paths).unwrap();

    let is_found: Vec<bool> = found.iter().map(Option::is_some).collect();
    assert_eq!(is_found, [true, true, false]);
}

#[test]
fn a_merge_replaces_a_partitions_files_in_every_manifest_that_holds_one() {
    let (fixture, written) = create_table_of_three_manifests();
    let merged_file = vec![fixture.write_file("two-merged", 2)];

    let merged = commit_replace(&written, &["two-a.parquet", "two-b.parquet"], merged_file)
        .unwrap()
        .expect("no other writer");

    let expected = [
        "one-c.parquet",
        "three-b.parquet",
        "three-c.parquet",
        "two-merged.parquet",
    ];
    assert_eq!(selected_sorted(&merged, &[]), expected);
}
