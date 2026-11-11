//! The footers of a table's data files: read once, kept for every later
//! plan of the versions that read the file alike, shared as they are,
//! forgotten on request, and neither read nor kept by a description.

mod support;

use std::sync::Arc;

use iceberg::spec::{NestedField, PrimitiveType, Schema, TableMetadata, Type};
use iceberg_tables::{Table, forget_cached_table};
use parquet_engine::ScanSelection;
use support::{
    Fixture, append, create_partitioned_table, describe, load, publish_changed, v_equals,
};

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

/// The table of `fixture` once another tool has added the column `added`
/// to the schema of `table`, its current version.
fn add_a_column(fixture: &Fixture, table: &Table) -> Table {
    let schema = table.metadata().current_schema();
    let added = NestedField::optional(
        schema.highest_field_id() + 1,
        "added",
        Type::Primitive(PrimitiveType::Long),
    );
    let fields = schema.as_struct().fields().iter().cloned();
    let widened = Schema::builder()
        .with_fields(fields.chain([Arc::new(added)]))
        .build()
        .unwrap();
    publish_changed(fixture, |builder| {
        builder.add_current_schema(widened).unwrap()
    })
}

#[test]
fn a_version_under_another_schema_reads_a_files_footer_again() {
    let (fixture, created) = create_partitioned_table();
    let written = append(&created, vec![fixture.write_file("seven", 7)]);
    written.select_scan(&[]).unwrap();
    let altered = add_a_column(&fixture, &written);
    fixture.replace_with_garbage("seven");

    let read_without_its_footer = altered.select_scan(&[]);

    assert!(read_without_its_footer.is_err());
}

#[test]
fn a_table_created_again_in_the_directory_of_another_shares_nothing_the_other_cached() {
    let (fixture, created) = create_partitioned_table();
    let dropped = append(&created, vec![fixture.write_file("seven", 7)]);
    let cached = dropped.select_scan(&[]).unwrap();
    let created_again = fixture.create_table_again();
    let rewritten = append(&created_again, vec![fixture.write_file("seven", 7)]);

    let read_again = rewritten.select_scan(&[]).unwrap();

    assert!(!Arc::ptr_eq(
        &cached.row_groups.row_groups()[0],
        &read_again.row_groups.row_groups()[0]
    ));
}

#[test]
fn a_tables_metadata_copied_to_another_directory_shares_nothing_the_table_cached() {
    let (fixture, created) = create_partitioned_table();
    let written = append(&created, vec![fixture.write_file("seven", 7)]);
    let cached = written.select_scan(&[]).unwrap();
    let elsewhere = Fixture::new();
    let copied = elsewhere.build_table_at(TableMetadata::clone(written.metadata()));

    let read_by_the_copy = copied.select_scan(&[]).unwrap();

    assert!(!Arc::ptr_eq(
        &cached.row_groups.row_groups()[0],
        &read_by_the_copy.row_groups.row_groups()[0]
    ));
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
fn a_table_the_cache_forgot_has_no_file_held_open_and_is_indexed_from_its_manifests_again() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = load(&fixture);
    let held_open = Arc::downgrade(&opened.select_scan(&[]).unwrap().row_groups.row_groups()[0]);

    forget_cached_table(&fixture.location);
    fixture.delete_data_manifests(&opened);
    let scan_without_its_manifests = opened.select_scan(&[]);

    assert!(held_open.upgrade().is_none());
    assert!(scan_without_its_manifests.is_err());
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
