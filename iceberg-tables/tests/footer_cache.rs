//! The footers of a table's data files: read once, kept for every later
//! plan of the versions that read the file alike, shared as they are,
//! forgotten on request, and neither read nor kept by a description.

mod support;

use std::sync::Arc;

use parquet_engine::ScanSelection;
use support::{append, create_partitioned_table, describe, load, v_equals};

/// The rows of each row group `selection` scans, in scan order.
fn row_group_rows(selection: &ScanSelection) -> Vec<i64> {
    selection
        .row_groups
        .row_groups()
        .iter()
        .map(|row_group| row_group.num_rows)
        .collect()
}

#[test]
fn a_second_plan_over_a_file_with_another_filter_reads_no_footer() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = load(&fixture);
    opened.select_scan(&v_equals(7)).unwrap();
    fixture.replace_with_garbage("seven");

    let selection = opened.select_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&selection), [10]);
}

#[test]
fn a_version_committed_on_top_reads_only_the_footer_of_the_file_it_added() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = load(&fixture);
    opened.select_scan(&[]).unwrap();
    fixture.replace_with_garbage("seven");

    let committed = append(&opened, vec![fixture.write_file("nine", 9)]);
    let selection = committed.select_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&selection), [10, 10]);
}

#[test]
fn footers_loaded_for_named_files_are_kept_for_the_next_plan() {
    let (fixture, created) = create_partitioned_table();
    let seven = fixture.write_file("seven", 7);
    append(&created, vec![seven.clone()]);
    let opened = load(&fixture);

    opened.load_footers(&[seven.file]).unwrap();
    fixture.replace_with_garbage("seven");
    let selection = opened.select_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&selection), [10]);
}

#[test]
fn a_second_plan_shares_the_row_groups_of_the_first() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = load(&fixture);
    let first = opened.select_scan(&[]).unwrap();

    let second = opened.select_scan(&[]).unwrap();

    assert!(Arc::ptr_eq(
        &first.row_groups.row_groups()[0],
        &second.row_groups.row_groups()[0]
    ));
}

#[test]
fn a_forgotten_footer_is_read_again_and_its_file_no_longer_held_open() {
    let (fixture, created) = create_partitioned_table();
    let seven = fixture.write_file("seven", 7);
    append(&created, vec![seven.clone()]);
    let opened = load(&fixture);
    let held_open = Arc::downgrade(&opened.select_scan(&[]).unwrap().row_groups.row_groups()[0]);

    opened.forget_footers(&[seven.file.path]).unwrap();
    fixture.replace_with_garbage("seven");
    let read_without_its_footer = opened.select_scan(&[]);

    assert!(held_open.upgrade().is_none());
    assert!(read_without_its_footer.is_err());
}

#[test]
fn describing_a_table_caches_none_of_the_footers_it_reads() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    append(&first, vec![fixture.write_file("nine", 9)]);
    let opened = load(&fixture);

    let description = describe(&opened);
    fixture.replace_with_garbage("seven");
    let read_without_its_footer = opened.select_scan(&[]);

    assert_eq!(description.files.len(), 2);
    assert!(read_without_its_footer.is_err());
}

#[test]
fn describing_a_table_reads_no_footer_the_cache_holds() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = load(&fixture);
    opened.select_scan(&[]).unwrap();
    fixture.replace_with_garbage("seven");

    let description = describe(&opened);

    assert_eq!(description.total_rows, 10);
}
