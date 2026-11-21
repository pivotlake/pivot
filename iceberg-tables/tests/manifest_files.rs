//! A version's live files as maintenance takes them, a data manifest at a
//! time: each file with its bounds on the table's sort key, read through
//! the parts of the manifests the cache holds.

mod support;

use iceberg_tables::{BoundSide, SortBound};
use iceberg_tables::{Error, Table};
use support::{
    append, create_partitioned_table, create_table, create_table_laid_out_by, entry, load,
    partition_entry, replace,
};

/// The names of the files each data manifest of `table` lists, manifest by
/// manifest in manifest-list order, each manifest's sorted.
fn list_files_per_manifest(table: &Table) -> Vec<Vec<String>> {
    let manifest_paths = table.list_data_manifest_paths();
    let manifests = table.load_manifest_files(&manifest_paths).unwrap();
    manifests
        .iter()
        .map(|partitions| {
            let files = partitions.iter().flat_map(|partition| &partition.files);
            let mut names: Vec<String> = files.map(|file| file.file().path.to_string()).collect();
            names.sort_unstable();
            names
        })
        .collect()
}

#[test]
fn every_manifest_lists_the_files_its_commit_added() {
    let (_fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("early", 10, 1, 2)]);
    let second = append(
        &first,
        vec![entry("late", 10, 3, 4), entry("later", 10, 5, 6)],
    );

    let files = list_files_per_manifest(&second);

    // A commit lists its own manifest before the ones it carries over.
    assert_eq!(
        files,
        [vec!["late.parquet", "later.parquet"], vec!["early.parquet"]]
    );
}

#[test]
fn a_manifests_files_come_by_partition_with_their_bounds_on_the_sort_key() {
    let by_v = ["v".to_string()];
    let (_fixture, created) = create_table_laid_out_by(&by_v, &by_v);
    let written = append(
        &created,
        vec![
            partition_entry("nine", 9, 0),
            partition_entry("seven", 7, 0),
        ],
    );

    let manifests = written
        .load_manifest_files(&written.list_data_manifest_paths())
        .unwrap();

    let read_range = |file: &iceberg_tables::SortBoundedFile| {
        let read = |side| match file.read_sort_bound(0, side) {
            SortBound::Integer(value) => value,
            other => panic!("the sort column is bounded by an integer, not {other:?}"),
        };
        (read(BoundSide::Lower), read(BoundSide::Upper))
    };
    let ranges_per_partition: Vec<Vec<(i128, i128)>> = manifests[0]
        .iter()
        .map(|partition| partition.files.iter().map(read_range).collect())
        .collect();
    assert_eq!(ranges_per_partition, [vec![(7, 7)], vec![(9, 9)]]);
}

#[test]
fn a_rewritten_manifest_lists_the_files_it_carried_over_under_another_path() {
    let (_fixture, created) = create_table(&[]);
    let appended = append(
        &created,
        vec![entry("kept", 10, 1, 1), entry("replaced", 10, 2, 2)],
    );
    let rewritten = replace(&appended, &["replaced.parquet"], "merged");

    let paths_before = appended.list_data_manifest_paths();
    let paths_after = rewritten.list_data_manifest_paths();
    let files = list_files_per_manifest(&rewritten);

    assert!(paths_after.iter().all(|path| !paths_before.contains(path)));
    assert_eq!(files, [vec!["merged.parquet"], vec!["kept.parquet"]]);
}

#[test]
fn the_files_of_a_manifest_a_scan_read_are_listed_with_no_manifest_read() {
    let (fixture, created) = create_partitioned_table();
    append(
        &created,
        vec![
            fixture.write_file("seven", 7),
            fixture.write_file("nine", 9),
        ],
    );
    let opened = load(&fixture);
    opened.select_scan(&[]).unwrap();
    fixture.delete_data_manifests(&opened);

    let files = list_files_per_manifest(&opened);

    assert_eq!(files, [vec!["nine.parquet", "seven.parquet"]]);
}

#[test]
fn the_files_of_a_manifest_a_version_does_not_list_are_refused() {
    let (_fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("early", 10, 1, 2)]);
    let rewritten = replace(&appended, &["early.parquet"], "merged");

    let files = rewritten.load_manifest_files(&appended.list_data_manifest_paths());

    assert!(matches!(files, Err(Error::UnlistedManifest { .. })));
}
