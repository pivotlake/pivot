//! The order the cache of footers and manifest parts lets go of its entries
//! in past its budget: what is spared, what is stored again or as a part
//! nobody asked for, and what the order keeps of itself as entries leave.

use std::sync::Arc;

use iceberg::spec::FormatVersion;
use iceberg_tables::table_metadata_cache::{CachedPartKey, FooterSchema, TableMetadataCache};
use iceberg_tables::test_support::{
    build_text_partition, find_part, store_empty_part, store_unasked_empty_part,
};
use uuid::Uuid;

/// A table as a test names it to the cache: its id, what its footers are
/// parsed under, and the format version its manifests are read under.
#[derive(Clone)]
struct NamedTable {
    table_id: Uuid,
    footer_schema: FooterSchema,
    format_version: FormatVersion,
}

/// The id of the table a test calls `name`: one id per name.
fn build_table_id(name: &str) -> Uuid {
    let mut id_bytes = [0u8; 16];
    id_bytes[..name.len()].copy_from_slice(name.as_bytes());
    Uuid::from_bytes(id_bytes)
}

/// The table a test calls `name`, its footers parsed under the schema 0
/// and its manifests read under format version 2.
fn name_table(name: &str) -> NamedTable {
    NamedTable {
        table_id: build_table_id(name),
        footer_schema: FooterSchema {
            schema_id: 0,
            logical_types: None,
        },
        format_version: FormatVersion::V2,
    }
}

/// Where the files of the one partition of the manifest at `manifest` are.
fn locate_files(manifest: &str) -> CachedPartKey {
    CachedPartKey::of_files(&Arc::from(manifest), &build_text_partition("a"))
}

/// What a cache counts for what `fill` stores in it.
fn measure_held(fill: impl FnOnce(&TableMetadataCache)) -> u64 {
    let cache = TableMetadataCache::new(u64::MAX);
    fill(&cache);
    u64::MAX - cache.free_bytes()
}

/// Hold in `table` the files of the one partition of each of the manifests
/// at `manifests`, each as used now, in the order given.
fn store_parts(cache: &TableMetadataCache, table: &NamedTable, manifests: &[&str]) {
    for manifest in manifests {
        store_empty_part(
            cache,
            table.table_id,
            table.format_version,
            &locate_files(manifest),
        );
    }
}

/// Whether `table` holds the files of the one partition of each of the
/// manifests at `manifests`, each found as a use of it, in the order
/// given.
fn hold_parts(cache: &TableMetadataCache, table: &NamedTable, manifests: &[&str]) -> Vec<bool> {
    let manifests = manifests.iter();
    manifests
        .map(|manifest| {
            find_part(
                cache,
                table.table_id,
                table.format_version,
                &locate_files(manifest),
            )
        })
        .collect()
}

/// Hold in `table` a footer of no row group for each of the files at
/// `paths`, in the order given, sparing the parts of `table` at `spared`.
fn store_footers(
    cache: &TableMetadataCache,
    table: &NamedTable,
    paths: &[&str],
    spared: &[CachedPartKey],
) {
    let footers = paths.iter().map(|path| (Arc::from(*path), Vec::new()));
    cache.store_footers(
        table.table_id,
        &table.footer_schema,
        footers.collect(),
        table.format_version,
        spared,
    );
}

/// Whether `table` holds a footer for each of the files at `paths`, none
/// counted as used.
fn hold_footers(cache: &TableMetadataCache, table: &NamedTable, paths: &[&str]) -> Vec<bool> {
    let found = cache.peek_footers(table.table_id, &table.footer_schema, paths.iter().copied());
    found.iter().map(Option::is_some).collect()
}

#[test]
fn a_part_spared_when_it_is_the_oldest_stays_and_the_entry_used_after_it_leaves() {
    let table = name_table("t");
    let every_entry = measure_held(|cache| {
        store_parts(cache, &table, &["spared", "left"]);
        store_footers(cache, &table, &["f"], &[]);
    });
    let cache = TableMetadataCache::new(every_entry - 1);
    store_parts(&cache, &table, &["spared", "left"]);

    store_footers(&cache, &table, &["f"], &[locate_files("spared")]);

    assert_eq!(
        hold_parts(&cache, &table, &["spared", "left"]),
        [true, false]
    );
    assert_eq!(hold_footers(&cache, &table, &["f"]), [true]);
}

#[test]
fn a_part_is_spared_once_and_then_leaves_like_any_other_entry() {
    let table = name_table("t");
    let part_and_footer = measure_held(|cache| {
        store_parts(cache, &table, &["p"]);
        store_footers(cache, &table, &["f1"], &[]);
    });
    let cache = TableMetadataCache::new(part_and_footer - 1);
    store_parts(&cache, &table, &["p"]);

    store_footers(&cache, &table, &["f1", "f2"], &[locate_files("p")]);

    assert_eq!(hold_footers(&cache, &table, &["f1", "f2"]), [false, true]);
    assert_eq!(hold_parts(&cache, &table, &["p"]), [false]);
}

#[test]
fn a_part_read_under_another_format_version_than_the_one_named_is_not_spared() {
    let table = name_table("t");
    let every_entry = measure_held(|cache| {
        store_parts(cache, &table, &["p"]);
        store_footers(cache, &table, &["f"], &[]);
    });
    let cache = TableMetadataCache::new(every_entry - 1);
    store_parts(&cache, &table, &["p"]);
    let footers = vec![(Arc::from("f"), Vec::new())];

    cache.store_footers(
        table.table_id,
        &table.footer_schema,
        footers,
        FormatVersion::V3,
        &[locate_files("p")],
    );

    assert_eq!(hold_parts(&cache, &table, &["p"]), [false]);
    assert_eq!(hold_footers(&cache, &table, &["f"]), [true]);
}

#[test]
fn storing_a_part_again_takes_the_room_of_the_one_it_replaces() {
    let table = name_table("t");
    let two_parts = measure_held(|cache| store_parts(cache, &table, &["a", "b"]));
    let cache = TableMetadataCache::new(two_parts);
    store_parts(&cache, &table, &["a", "b"]);

    store_parts(&cache, &table, &["a"]);

    assert_eq!(cache.free_bytes(), 0);
    assert_eq!(hold_parts(&cache, &table, &["a", "b"]), [true, true]);
}

#[test]
fn storing_a_footer_again_takes_the_room_of_the_one_it_replaces() {
    let table = name_table("t");
    let two_footers = measure_held(|cache| store_footers(cache, &table, &["a", "b"], &[]));
    let cache = TableMetadataCache::new(two_footers);
    store_footers(&cache, &table, &["a", "b"], &[]);

    store_footers(&cache, &table, &["a"], &[]);

    assert_eq!(cache.free_bytes(), 0);
    assert_eq!(hold_footers(&cache, &table, &["a", "b"]), [true, true]);
}

#[test]
fn a_removed_footer_leaves_its_room() {
    let table = name_table("t");
    let two_footers = measure_held(|cache| store_footers(cache, &table, &["f1", "f2"], &[]));
    let cache = TableMetadataCache::new(two_footers);
    store_footers(&cache, &table, &["f1", "f2"], &[]);

    cache.remove_footers(table.table_id, &table.footer_schema, ["f1"].into_iter());
    store_footers(&cache, &table, &["f3"], &[]);

    assert_eq!(
        hold_footers(&cache, &table, &["f1", "f2", "f3"]),
        [false, true, true]
    );
}

#[test]
fn the_entry_used_longest_ago_goes_whichever_table_holds_it() {
    let (first, second) = (name_table("t"), name_table("u"));
    let every_part = measure_held(|cache| {
        store_parts(cache, &first, &["old"]);
        store_parts(cache, &second, &["kept", "new"]);
    });
    let cache = TableMetadataCache::new(every_part - 1);
    store_parts(&cache, &first, &["old"]);
    store_parts(&cache, &second, &["kept"]);

    store_parts(&cache, &second, &["new"]);

    assert_eq!(hold_parts(&cache, &first, &["old"]), [false]);
    assert_eq!(hold_parts(&cache, &second, &["kept", "new"]), [true, true]);
}

#[test]
fn a_part_stored_unasked_leaves_the_one_already_held_where_it_stood() {
    let table = name_table("t");
    let three_parts = measure_held(|cache| store_parts(cache, &table, &["old", "kept", "new"]));
    let cache = TableMetadataCache::new(three_parts - 1);
    store_parts(&cache, &table, &["old", "kept"]);

    let is_held = store_unasked_empty_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_files("kept"),
    );

    store_parts(&cache, &table, &["new"]);
    assert!(is_held);
    assert_eq!(hold_parts(&cache, &table, &["old", "kept"]), [false, true]);
}

#[test]
fn a_part_stored_unasked_and_then_found_outlives_the_ones_used_before_it() {
    let table = name_table("t");
    let three_parts = measure_held(|cache| store_parts(cache, &table, &["left", "unasked", "new"]));
    let cache = TableMetadataCache::new(three_parts - 1);
    store_parts(&cache, &table, &["left"]);
    store_unasked_empty_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_files("unasked"),
    );
    hold_parts(&cache, &table, &["unasked"]);

    store_parts(&cache, &table, &["new"]);

    assert_eq!(
        hold_parts(&cache, &table, &["left", "unasked"]),
        [false, true]
    );
}

/// The location of the footer `number` of those a test stores under
/// `prefix`: every one as long as the others.
fn name_numbered(prefix: &str, number: u32) -> String {
    format!("{prefix}-{number:04}")
}

/// Hold in `table` a footer for each of `numbers` under `prefix`, in order.
fn store_numbered(
    cache: &TableMetadataCache,
    table: &NamedTable,
    prefix: &str,
    numbers: std::ops::Range<u32>,
) {
    for number in numbers {
        store_footers(cache, table, &[&name_numbered(prefix, number)], &[]);
    }
}

/// Remove from `table` the footer of each of `numbers` under `file`.
fn remove_numbered(cache: &TableMetadataCache, table: &NamedTable, numbers: std::ops::Range<u32>) {
    let paths: Vec<String> = numbers
        .map(|number| name_numbered("file", number))
        .collect();
    cache.remove_footers(
        table.table_id,
        &table.footer_schema,
        paths.iter().map(String::as_str),
    );
}

#[test]
fn entries_keep_their_order_of_last_use_when_most_of_them_have_left() {
    let table = name_table("t");
    let cache = TableMetadataCache::new(measure_held(|cache| {
        store_numbered(cache, &table, "file", 0..5000);
    }));
    store_numbered(&cache, &table, "file", 0..5000);
    cache.find_footers(
        table.table_id,
        &table.footer_schema,
        ["file-0010"].into_iter(),
    );
    remove_numbered(&cache, &table, 0..10);
    remove_numbered(&cache, &table, 11..4997);

    store_numbered(&cache, &table, "more", 0..4998);

    assert_eq!(
        hold_footers(
            &cache,
            &table,
            &["file-4997", "file-4998", "file-4999", "file-0010"]
        ),
        [false, false, true, true]
    );
}

#[test]
fn a_footer_removed_with_most_of_the_others_is_found_when_stored_again() {
    let (cache, table) = (TableMetadataCache::new(u64::MAX), name_table("t"));
    store_numbered(&cache, &table, "file", 0..5000);
    remove_numbered(&cache, &table, 0..4990);

    store_numbered(&cache, &table, "file", 7..8);

    assert_eq!(
        hold_footers(&cache, &table, &["file-0007", "file-4995", "file-0008"]),
        [true, true, false]
    );
}

#[test]
fn a_cache_emptied_and_refilled_finds_what_it_holds() {
    let (cache, table) = (TableMetadataCache::new(u64::MAX), name_table("t"));
    store_numbered(&cache, &table, "file", 0..5000);
    remove_numbered(&cache, &table, 0..5000);

    store_numbered(&cache, &table, "more", 0..5000);

    assert_eq!(
        hold_footers(&cache, &table, &["more-0000", "more-4999", "file-0000"]),
        [true, true, false]
    );
}

#[test]
fn parts_keep_their_order_of_last_use_when_most_of_the_entries_have_left() {
    let table = name_table("t");
    let cache = TableMetadataCache::new(measure_held(|cache| {
        store_parts(cache, &table, &["older", "newer"]);
        store_numbered(cache, &table, "file", 0..4998);
    }));
    store_parts(&cache, &table, &["older", "newer"]);
    store_numbered(&cache, &table, "file", 0..4998);
    remove_numbered(&cache, &table, 0..4998);

    store_numbered(&cache, &table, "more", 0..4999);

    assert_eq!(
        hold_parts(&cache, &table, &["older", "newer"]),
        [false, true]
    );
}
