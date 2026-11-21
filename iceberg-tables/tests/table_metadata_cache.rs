//! What the cache of footers and manifest parts holds, what it lets go of
//! past its budget, which entries a table's id finds, and what one
//! manifest's parts are among those held.

use std::sync::Arc;

use iceberg::spec::FormatVersion;
use iceberg_tables::table_metadata_cache::{CachedPartKey, FooterSchema, TableMetadataCache};
use iceberg_tables::test_support::{
    build_text_partition, find_part, find_partitions_parts, store_empty_part,
    store_unasked_empty_part,
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

/// The table a test calls `name`, its footers parsed under the schema
/// `schema_id` and its manifests read under format version 2.
fn name_table(name: &str, schema_id: i32) -> NamedTable {
    NamedTable {
        table_id: build_table_id(name),
        footer_schema: FooterSchema {
            schema_id,
            logical_types: None,
        },
        format_version: FormatVersion::V2,
    }
}

/// Where the partitions of the manifest at `manifest` are.
fn locate_partitions(manifest: &str) -> CachedPartKey {
    CachedPartKey::of_partitions(&Arc::from(manifest))
}

/// Where the files of the partition `partition` in the manifest at
/// `manifest` are.
fn locate_files(manifest: &str, partition: &str) -> CachedPartKey {
    CachedPartKey::of_files(&Arc::from(manifest), &build_text_partition(partition))
}

/// Where what the manifest at `manifest` records of the field `field_id` in
/// the partition `partition` is.
fn locate_bounds(manifest: &str, partition: &str, field_id: i32) -> CachedPartKey {
    let partition = build_text_partition(partition);
    CachedPartKey::of_bounds(&Arc::from(manifest), &partition, field_id)
}

/// What a cache counts for what `fill` stores in it.
fn measure_held(fill: impl FnOnce(&TableMetadataCache)) -> u64 {
    let cache = TableMetadataCache::new(u64::MAX);
    fill(&cache);
    u64::MAX - cache.free_bytes()
}

/// Hold in `table` the partitions of each of the manifests at `manifests`,
/// each as used now, in the order given.
fn store_parts(cache: &TableMetadataCache, table: &NamedTable, manifests: &[&str]) {
    for manifest in manifests {
        store_empty_part(
            cache,
            table.table_id,
            table.format_version,
            &locate_partitions(manifest),
        );
    }
}

/// Whether `table` holds the partitions of each of the manifests at
/// `manifests`, asked of the cache at once, each found as a use of it.
fn hold_parts(cache: &TableMetadataCache, table: &NamedTable, manifests: &[&str]) -> Vec<bool> {
    let manifests: Vec<Arc<str>> = manifests.iter().map(|path| Arc::from(*path)).collect();
    find_partitions_parts(
        cache,
        table.table_id,
        table.format_version,
        &manifests.iter().collect::<Vec<_>>(),
    )
}

/// Hold in `table` a footer of no row group for the file at `path`, with
/// no part spared for it.
fn store_footer(cache: &TableMetadataCache, table: &NamedTable, path: &str) {
    cache.store_footers(
        table.table_id,
        &table.footer_schema,
        vec![(Arc::from(path), Vec::new())],
        table.format_version,
        &[],
    );
}

/// Whether `table` holds a footer for each of the files at `paths`, none
/// counted as used.
fn hold_footers(cache: &TableMetadataCache, table: &NamedTable, paths: &[&str]) -> Vec<bool> {
    let found = cache.peek_footers(table.table_id, &table.footer_schema, paths.iter().copied());
    found.iter().map(Option::is_some).collect()
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_goes() {
    let table = name_table("t", 0);
    let three_parts = measure_held(|cache| store_parts(cache, &table, &["old", "kept", "new"]));
    let cache = TableMetadataCache::new(three_parts - 1);
    store_parts(&cache, &table, &["old", "kept"]);
    hold_parts(&cache, &table, &["old"]);

    store_parts(&cache, &table, &["new"]);

    assert_eq!(
        hold_parts(&cache, &table, &["kept", "old", "new"]),
        [false, true, true]
    );
}

#[test]
fn a_footer_found_unrecorded_is_not_counted_as_used() {
    let table = name_table("t", 0);
    let every_entry = measure_held(|cache| {
        store_footer(cache, &table, "f");
        store_parts(cache, &table, &["kept", "new"]);
    });
    let cache = TableMetadataCache::new(every_entry - 1);
    store_footer(&cache, &table, "f");
    store_parts(&cache, &table, &["kept"]);
    let found = cache.peek_footers(table.table_id, &table.footer_schema, ["f"].into_iter());

    store_parts(&cache, &table, &["new"]);

    assert!(found[0].is_some());
    assert!(
        cache.find_footers(table.table_id, &table.footer_schema, ["f"].into_iter())[0].is_none()
    );
    assert_eq!(hold_parts(&cache, &table, &["kept"]), [true]);
}

#[test]
fn parts_found_together_are_each_counted_as_used() {
    let table = name_table("t", 0);
    let three_parts = measure_held(|cache| store_parts(cache, &table, &["first", "second", "new"]));
    let cache = TableMetadataCache::new(three_parts - 1);
    store_parts(&cache, &table, &["first", "second"]);
    let found = hold_parts(&cache, &table, &["second", "absent", "first"]);

    store_parts(&cache, &table, &["new"]);

    assert_eq!(found, [true, false, true]);
    assert_eq!(
        hold_parts(&cache, &table, &["second", "first"]),
        [false, true]
    );
}

#[test]
fn the_entry_just_stored_stays_whatever_it_takes() {
    let table = name_table("t", 0);
    let cache = TableMetadataCache::new(5);
    store_parts(&cache, &table, &["old"]);

    store_parts(&cache, &table, &["new"]);

    assert_eq!(hold_parts(&cache, &table, &["old", "new"]), [false, true]);
}

#[test]
fn footers_stored_past_the_budget_leave_before_the_part_they_were_selected_through() {
    let table = name_table("t", 0);
    let every_entry = measure_held(|cache| {
        store_parts(cache, &table, &["p"]);
        store_footer(cache, &table, "f1");
        store_footer(cache, &table, "f2");
    });
    let cache = TableMetadataCache::new(every_entry - 1);
    store_parts(&cache, &table, &["p"]);
    let footers = vec![(Arc::from("f1"), Vec::new()), (Arc::from("f2"), Vec::new())];

    cache.store_footers(
        table.table_id,
        &table.footer_schema,
        footers,
        table.format_version,
        &[locate_partitions("p")],
    );

    assert_eq!(hold_footers(&cache, &table, &["f1", "f2"]), [false, true]);
    assert_eq!(hold_parts(&cache, &table, &["p"]), [true]);
}

#[test]
fn a_part_marked_used_outlives_the_footers_taken_before_it() {
    let table = name_table("t", 0);
    let every_entry = measure_held(|cache| {
        store_parts(cache, &table, &["p", "q"]);
        store_footer(cache, &table, "f");
    });
    let cache = TableMetadataCache::new(every_entry - 1);
    store_parts(&cache, &table, &["p"]);
    store_footer(&cache, &table, "f");
    cache.mark_manifest_parts_used(
        table.table_id,
        table.format_version,
        &[locate_partitions("p")],
    );

    store_parts(&cache, &table, &["q"]);

    assert_eq!(hold_footers(&cache, &table, &["f"]), [false]);
    assert_eq!(hold_parts(&cache, &table, &["p"]), [true]);
}

#[test]
fn a_removed_footer_is_no_longer_found() {
    let table = name_table("t", 0);
    let cache = TableMetadataCache::new(u64::MAX);
    store_footer(&cache, &table, "f");

    cache.remove_footers(table.table_id, &table.footer_schema, ["f"].into_iter());

    assert!(
        cache.find_footers(table.table_id, &table.footer_schema, ["f"].into_iter())[0].is_none()
    );
}

#[test]
fn equal_names_built_apart_find_each_others_entries() {
    let cache = TableMetadataCache::new(u64::MAX);
    store_parts(&cache, &name_table("t", 1), &["p"]);
    store_footer(&cache, &name_table("t", 1), "f");

    let parts = hold_parts(&cache, &name_table("t", 1), &["p"]);
    let footers = cache.find_footers(
        name_table("t", 1).table_id,
        &name_table("t", 1).footer_schema,
        ["f"].into_iter(),
    );

    assert!(parts[0] && footers[0].is_some());
}

#[test]
fn names_that_differ_in_any_field_do_not_find_each_others_entries() {
    let cache = TableMetadataCache::new(u64::MAX);
    store_parts(&cache, &name_table("t", 1), &["p"]);
    store_footer(&cache, &name_table("t", 1), "f");
    let with_another_version = NamedTable {
        format_version: FormatVersion::V3,
        ..name_table("t", 1)
    };
    let with_logical_types = NamedTable {
        footer_schema: FooterSchema {
            schema_id: 1,
            logical_types: Some(Arc::from("{}")),
        },
        ..name_table("t", 1)
    };

    let read_another_way = hold_parts(&cache, &with_another_version, &["p"]);
    let of_another_table = hold_parts(&cache, &name_table("u", 1), &["p"]);
    let footers_read_another_way = hold_footers(&cache, &name_table("t", 2), &["f"]);
    let footers_of_another_table = hold_footers(&cache, &name_table("u", 1), &["f"]);
    let read_as_other_types = hold_footers(&cache, &with_logical_types, &["f"]);

    assert_eq!(
        [read_another_way, of_another_table, footers_read_another_way],
        [[false], [false], [false]]
    );
    assert_eq!(
        [footers_of_another_table, read_as_other_types],
        [[false], [false]]
    );
}

#[test]
fn a_table_whose_last_entry_is_evicted_is_forgotten() {
    let (evicted, kept) = (name_table("t", 0), name_table("u", 0));
    let one_table = measure_held(|cache| store_parts(cache, &evicted, &["p"]));
    let cache = TableMetadataCache::new(one_table);
    store_parts(&cache, &evicted, &["p"]);

    store_parts(&cache, &kept, &["p"]);
    cache.remove_table(kept.table_id);

    assert_eq!(cache.free_bytes(), one_table);
}

#[test]
fn a_manifest_whose_last_part_is_removed_is_forgotten() {
    let table = name_table("t", 0);
    let cache = TableMetadataCache::new(u64::MAX);
    let manifest: Arc<str> = Arc::from("m");
    let partition = build_text_partition("a");
    store_empty_part(
        &cache,
        table.table_id,
        table.format_version,
        &CachedPartKey::of_files(&manifest, &partition),
    );
    let is_kept_with_a_part = Arc::strong_count(&manifest) > 1;

    cache.remove_manifests(table.table_id, table.format_version, ["m"].into_iter());

    assert!(is_kept_with_a_part);
    assert_eq!(Arc::strong_count(&manifest), 1);
    assert_eq!(cache.free_bytes(), u64::MAX);
}

#[test]
fn removing_a_table_removes_it_however_it_was_read_and_leaves_the_others() {
    let cache = TableMetadataCache::new(u64::MAX);
    let first = name_table("t", 1);
    let second = NamedTable {
        format_version: FormatVersion::V3,
        ..name_table("t", 2)
    };
    let other = name_table("u", 1);
    for table in [&first, &second, &other] {
        store_parts(&cache, table, &["p"]);
        store_footer(&cache, table, "f");
    }

    cache.remove_table(first.table_id);

    assert_eq!(hold_parts(&cache, &first, &["p"]), [false]);
    assert_eq!(hold_parts(&cache, &second, &["p"]), [false]);
    assert_eq!(hold_footers(&cache, &first, &["f"]), [false]);
    assert_eq!(hold_footers(&cache, &second, &["f"]), [false]);
    assert_eq!(hold_parts(&cache, &other, &["p"]), [true]);
}

#[test]
fn a_removed_table_leaves_its_room_to_the_others() {
    let (removed, kept) = (name_table("t", 0), name_table("u", 0));
    let both_tables = measure_held(|cache| {
        store_parts(cache, &kept, &["a"]);
        store_parts(cache, &removed, &["p"]);
        store_footer(cache, &removed, "f");
    });
    let cache = TableMetadataCache::new(both_tables);
    store_parts(&cache, &kept, &["a"]);
    store_parts(&cache, &removed, &["p"]);
    store_footer(&cache, &removed, "f");

    cache.remove_table(removed.table_id);
    let free_bytes = cache.free_bytes();
    store_parts(&cache, &kept, &["b"]);

    assert_eq!(
        free_bytes,
        both_tables - measure_held(|cache| store_parts(cache, &kept, &["a"]))
    );
    assert_eq!(hold_parts(&cache, &kept, &["a", "b"]), [true, true]);
}

#[test]
fn the_cache_stays_within_its_budget_as_tables_come_and_go() {
    let name_numbered = |number: u32| name_table(&format!("t{number}"), 0);
    let two_tables = measure_held(|cache| {
        store_parts(cache, &name_numbered(0), &["p"]);
        store_parts(cache, &name_numbered(1), &["p"]);
    });
    let cache = TableMetadataCache::new(two_tables);

    for number in 0..100 {
        store_parts(&cache, &name_numbered(number), &["p"]);
    }

    let still_held: Vec<u32> = (0..100)
        .filter(|number| hold_parts(&cache, &name_numbered(*number), &["p"])[0])
        .collect();
    assert_eq!(still_held, [98, 99]);
}

#[test]
fn a_stored_part_is_found_where_it_was_stored_and_nowhere_else() {
    let (cache, table) = (TableMetadataCache::new(u64::MAX), name_table("t", 0));

    store_empty_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_bounds("m", "a", 1),
    );

    assert!(find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_bounds("m", "a", 1)
    ));
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_bounds("m", "a", 2)
    ));
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_bounds("m", "b", 1)
    ));
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_bounds("n", "a", 1)
    ));
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_files("m", "a")
    ));
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &locate_partitions("m")
    ));
}

#[test]
fn a_part_nobody_asked_for_goes_before_a_part_stored_earlier_as_used() {
    let table = name_table("t", 0);
    let unasked = locate_files("unasked", "a");
    let every_part = measure_held(|cache| {
        store_parts(cache, &table, &["used", "new"]);
        store_unasked_empty_part(cache, table.table_id, table.format_version, &unasked);
    });
    let cache = TableMetadataCache::new(every_part - 1);
    store_parts(&cache, &table, &["used"]);
    let is_unasked_held =
        store_unasked_empty_part(&cache, table.table_id, table.format_version, &unasked);

    store_parts(&cache, &table, &["new"]);

    assert!(is_unasked_held);
    assert!(!find_part(
        &cache,
        table.table_id,
        table.format_version,
        &unasked
    ));
    assert_eq!(hold_parts(&cache, &table, &["used", "new"]), [true, true]);
}

#[test]
fn parts_nobody_asked_for_are_held_only_while_the_budget_has_room() {
    let table = name_table("t", 0);
    let (fits, over) = (locate_files("m", "a"), locate_bounds("m", "a", 1));
    let two_parts = measure_held(|cache| {
        store_parts(cache, &table, &["used"]);
        store_unasked_empty_part(cache, table.table_id, table.format_version, &fits);
    });
    let cache = TableMetadataCache::new(two_parts);
    store_parts(&cache, &table, &["used"]);

    let is_fitting_part_held =
        store_unasked_empty_part(&cache, table.table_id, table.format_version, &fits);
    let is_part_over_held =
        store_unasked_empty_part(&cache, table.table_id, table.format_version, &over);

    assert_eq!([is_fitting_part_held, is_part_over_held], [true, false]);
    assert_eq!(
        cache.list_manifest_parts(table.table_id, table.format_version, "m"),
        [fits]
    );
    assert_eq!(hold_parts(&cache, &table, &["used"]), [true]);
}

#[test]
fn a_part_marked_used_outlives_the_ones_used_before_it() {
    let table = name_table("t", 0);
    let three_parts = measure_held(|cache| store_parts(cache, &table, &["marked", "left", "new"]));
    let cache = TableMetadataCache::new(three_parts - 1);
    store_parts(&cache, &table, &["marked", "left"]);
    cache.mark_manifest_parts_used(
        table.table_id,
        table.format_version,
        &[locate_partitions("marked")],
    );

    store_parts(&cache, &table, &["new"]);

    assert_eq!(
        hold_parts(&cache, &table, &["marked", "left", "new"]),
        [true, false, true]
    );
}

#[test]
fn the_budget_left_free_is_what_the_entries_held_do_not_count_for() {
    let table = name_table("t", 0);
    let one_part = measure_held(|cache| store_parts(cache, &table, &["a"]));
    let cache = TableMetadataCache::new(one_part + 7);

    store_parts(&cache, &table, &["a"]);

    assert_eq!(cache.free_bytes(), 7);
}

#[test]
fn removing_a_table_lets_go_of_its_parts() {
    let cache = TableMetadataCache::new(u64::MAX);
    let table = name_table("t", 0);
    store_parts(&cache, &table, &["a"]);

    cache.remove_table(table.table_id);

    assert_eq!(hold_parts(&cache, &table, &["a"]), [false]);
}

/// Hold in `table` a part at each of `keys`, as used now, in the
/// order given.
fn store_at(cache: &TableMetadataCache, table: &NamedTable, keys: &[CachedPartKey]) {
    for key in keys {
        store_empty_part(cache, table.table_id, table.format_version, key);
    }
}

#[test]
fn the_parts_of_one_manifest_are_listed_together_a_partition_at_a_time() {
    let (cache, table) = (TableMetadataCache::new(u64::MAX), name_table("t", 0));
    let of_the_manifest = [
        locate_partitions("m"),
        locate_files("m", "a"),
        locate_bounds("m", "a", -1),
        locate_bounds("m", "a", 2),
        locate_files("m", "b"),
        locate_bounds("m", "b", 1),
    ];
    let mut stored = vec![locate_partitions("l")];
    stored.extend(of_the_manifest.iter().rev().cloned());
    stored.push(locate_bounds("n", "a", 1));
    store_at(&cache, &name_table("s", 0), &of_the_manifest[..2]);
    store_at(&cache, &table, &stored);
    store_at(&cache, &name_table("u", 0), &of_the_manifest[2..]);

    let listed = cache.list_manifest_parts(table.table_id, table.format_version, "m");

    assert_eq!(listed, of_the_manifest);
}

#[test]
fn removing_a_manifest_removes_every_part_of_it_and_leaves_the_others() {
    let (table, other_table) = (name_table("t", 0), name_table("u", 0));
    let cache = TableMetadataCache::new(u64::MAX);
    let of_the_manifest = [
        locate_partitions("m"),
        locate_files("m", "a"),
        locate_bounds("m", "a", 1),
    ];
    let of_others = [locate_files("l", "a"), locate_bounds("n", "a", 1)];
    store_at(&cache, &table, &of_the_manifest);
    store_at(&cache, &table, &of_others);
    store_at(&cache, &other_table, &of_the_manifest);
    store_footer(&cache, &table, "m");

    cache.remove_manifests(table.table_id, table.format_version, ["m"].into_iter());

    assert!(
        cache
            .list_manifest_parts(table.table_id, table.format_version, "m")
            .is_empty()
    );
    assert_eq!(
        cache.list_manifest_parts(table.table_id, table.format_version, "l"),
        of_others[..1]
    );
    assert_eq!(
        cache.list_manifest_parts(table.table_id, table.format_version, "n"),
        of_others[1..]
    );
    assert_eq!(
        cache.list_manifest_parts(other_table.table_id, other_table.format_version, "m"),
        of_the_manifest
    );
    assert_eq!(hold_footers(&cache, &table, &["m"]), [true]);
}

#[test]
fn a_removed_manifest_leaves_its_room() {
    let table = name_table("t", 0);
    let cache = TableMetadataCache::new(u64::MAX);
    store_at(&cache, &table, &[locate_partitions("kept")]);
    let free_bytes_before = cache.free_bytes();
    store_at(
        &cache,
        &table,
        &[
            locate_partitions("m"),
            locate_files("m", "a"),
            locate_bounds("m", "a", 1),
        ],
    );

    cache.remove_manifests(table.table_id, table.format_version, ["m"].into_iter());

    assert_eq!(cache.free_bytes(), free_bytes_before);
}

#[test]
fn removing_a_path_no_part_is_held_of_removes_nothing() {
    let (cache, table) = (TableMetadataCache::new(u64::MAX), name_table("t", 0));
    store_at(&cache, &table, &[locate_files("m", "a")]);
    store_footer(&cache, &table, "f");

    cache.remove_manifests(
        table.table_id,
        table.format_version,
        ["f", "absent"].into_iter(),
    );
    cache.remove_manifests(
        name_table("u", 0).table_id,
        name_table("u", 0).format_version,
        ["m"].into_iter(),
    );

    assert_eq!(
        cache.list_manifest_parts(table.table_id, table.format_version, "m"),
        [locate_files("m", "a")]
    );
    assert_eq!(hold_footers(&cache, &table, &["f"]), [true]);
}

#[test]
fn removing_every_table_leaves_the_whole_budget_free() {
    let (cache, table) = (TableMetadataCache::new(1 << 20), name_table("t", 0));
    store_parts(&cache, &table, &["p"]);
    store_footer(&cache, &table, "f");

    cache.remove_all_tables();

    assert_eq!(cache.free_bytes(), 1 << 20);
    assert_eq!(hold_parts(&cache, &table, &["p"]), [false]);
    assert_eq!(hold_footers(&cache, &table, &["f"]), [false]);
}
