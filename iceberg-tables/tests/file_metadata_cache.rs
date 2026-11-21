//! What the cache of footers and metadata parts holds, what it lets go of
//! past its budget, and which items a table's key finds.

use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::Duration;

use iceberg_tables::file_metadata_cache::{
    FileMetadataCache, FooterUse, MetadataPart, TableKey, measure_footer_entry, measure_part_entry,
};

/// A part of a table's metadata of a size the test names.
struct SizedPart(u64);

impl MetadataPart for SizedPart {
    fn estimate_heap_bytes(&self) -> u64 {
        self.0
    }
}

fn key(name: &str) -> Arc<[u8]> {
    Arc::from(name.as_bytes())
}

/// What a cache counts parts of the sizes given for between them, each held
/// under the name it comes with.
fn measure_parts(parts: &[(&str, u64)]) -> u64 {
    parts
        .iter()
        .map(|(name, heap_bytes)| measure_part_entry(name.as_bytes(), &SizedPart(*heap_bytes)))
        .sum()
}

/// The key of the table `table` whose files are read under `reading`.
fn name_table(table: &str, reading: &str) -> TableKey {
    TableKey {
        table: Arc::from(table),
        reading: Arc::from(reading),
    }
}

/// A cache of `capacity_bytes`, and the key of one table to hold items of
/// in it.
fn open_cache(capacity_bytes: u64) -> (Arc<FileMetadataCache>, TableKey) {
    (FileMetadataCache::new(capacity_bytes), name_table("t", ""))
}

/// Hold in `table` a part of `heap_bytes` under `name`, as used now.
fn store_part(cache: &FileMetadataCache, table: &TableKey, name: &str, heap_bytes: u64) {
    cache.store_metadata_parts(table, build_parts(&[(name, heap_bytes)]));
}

/// The part `table` holds under `name`, found as a use of it.
fn find_part(cache: &FileMetadataCache, table: &TableKey, name: &str) -> Option<Arc<SizedPart>> {
    cache
        .find_metadata_parts(table, [name.as_bytes()].into_iter())
        .remove(0)
}

/// Parts of the sizes given, each under the name it comes with, as the
/// cache is handed them.
fn build_parts(parts: &[(&str, u64)]) -> Vec<(Arc<[u8]>, Arc<SizedPart>)> {
    parts
        .iter()
        .map(|(name, heap_bytes)| (key(name), Arc::new(SizedPart(*heap_bytes))))
        .collect()
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_goes() {
    let (cache, table) = open_cache(measure_parts(&[("old", 10), ("kept", 10), ("new", 10)]) - 1);
    store_part(&cache, &table, "old", 10);
    store_part(&cache, &table, "kept", 10);
    find_part(&cache, &table, "old");

    store_part(&cache, &table, "new", 10);

    assert!(find_part(&cache, &table, "kept").is_none());
    assert!(find_part(&cache, &table, "old").is_some());
    assert!(find_part(&cache, &table, "new").is_some());
}

#[test]
fn a_footer_found_unrecorded_is_not_counted_as_used() {
    let footer_bytes = measure_footer_entry("f", &[]);
    let (cache, table) = open_cache(measure_parts(&[("kept", 10), ("new", 10)]) + footer_bytes - 1);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);
    store_part(&cache, &table, "kept", 10);
    let found = cache.find_footers(&table, ["f"].into_iter(), FooterUse::Unrecorded);

    store_part(&cache, &table, "new", 10);

    assert!(found[0].is_some());
    assert!(cache.find_footers(&table, ["f"].into_iter(), FooterUse::Kept)[0].is_none());
    assert!(find_part(&cache, &table, "kept").is_some());
}

#[test]
fn parts_found_together_are_each_counted_as_used() {
    let (cache, table) =
        open_cache(measure_parts(&[("first", 10), ("second", 20), ("new", 10)]) - 1);
    store_part(&cache, &table, "first", 10);
    store_part(&cache, &table, "second", 20);
    let found = cache.find_metadata_parts::<SizedPart>(
        &table,
        [b"second".as_slice(), b"first".as_slice()].into_iter(),
    );

    store_part(&cache, &table, "new", 10);

    assert_eq!(found[0].as_ref().unwrap().0, 20);
    assert!(find_part(&cache, &table, "second").is_none());
    assert!(find_part(&cache, &table, "first").is_some());
}

#[test]
fn the_entry_just_stored_stays_whatever_it_takes() {
    let (cache, table) = open_cache(5);

    store_part(&cache, &table, "p", 100);

    assert!(find_part(&cache, &table, "p").is_some());
}

#[test]
fn a_part_marked_used_outlives_the_footers_taken_before_it() {
    let footer_bytes = measure_footer_entry("f", &[]);
    let (cache, table) = open_cache(measure_parts(&[("p", 10), ("q", 1)]) + footer_bytes - 1);
    store_part(&cache, &table, "p", 10);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);
    cache.mark_metadata_parts_used(&table, &[key("p")]);

    store_part(&cache, &table, "q", 1);

    assert!(find_part(&cache, &table, "p").is_some());
    assert!(cache.find_footers(&table, ["f"].into_iter(), FooterUse::Unrecorded)[0].is_none());
}

#[test]
fn a_removed_footer_is_no_longer_found() {
    let (cache, table) = open_cache(u64::MAX);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);

    cache.remove_footers(&table, ["f"].into_iter());

    assert!(cache.find_footers(&table, ["f"].into_iter(), FooterUse::Kept)[0].is_none());
}

#[test]
fn equal_keys_computed_apart_find_each_others_items() {
    let cache = FileMetadataCache::new(u64::MAX);
    store_part(&cache, &name_table("t", "1"), "p", 10);
    cache.store_footers(&name_table("t", "1"), vec![(Arc::from("f"), Vec::new())]);

    let index = find_part(&cache, &name_table("t", "1"), "p");
    let footers = cache.find_footers(&name_table("t", "1"), ["f"].into_iter(), FooterUse::Kept);

    assert!(index.is_some() && footers[0].is_some());
}

#[test]
fn keys_that_differ_in_either_part_do_not_find_each_others_items() {
    let cache = FileMetadataCache::new(u64::MAX);
    store_part(&cache, &name_table("t", "1"), "p", 10);

    let read_another_way = find_part(&cache, &name_table("t", "2"), "p");
    let of_another_table = find_part(&cache, &name_table("u", "1"), "p");

    assert!(read_another_way.is_none() && of_another_table.is_none());
}

#[test]
fn a_table_whose_last_item_is_evicted_is_forgotten() {
    let cache = FileMetadataCache::new(measure_parts(&[("p", 10)]));
    let (evicted, kept) = (name_table("t", ""), name_table("u", ""));
    store_part(&cache, &evicted, "p", 10);
    let is_kept_with_an_item = Arc::strong_count(&evicted.table) > 1;

    store_part(&cache, &kept, "p", 10);

    assert!(is_kept_with_an_item);
    assert_eq!(Arc::strong_count(&evicted.table), 1);
}

#[test]
fn removing_a_table_removes_it_under_every_key_and_leaves_the_others() {
    let cache = FileMetadataCache::new(u64::MAX);
    let (first, second) = (name_table("t", "1"), name_table("t", "2"));
    let other = name_table("u", "1");
    for table in [&first, &second, &other] {
        store_part(&cache, table, "p", 10);
    }
    cache.store_footers(&first, vec![(Arc::from("f"), Vec::new())]);

    cache.remove_tables("t");

    assert!(find_part(&cache, &first, "p").is_none());
    assert!(find_part(&cache, &second, "p").is_none());
    assert!(cache.find_footers(&first, ["f"].into_iter(), FooterUse::Kept)[0].is_none());
    assert!(find_part(&cache, &other, "p").is_some());
}

#[test]
fn a_removed_table_leaves_its_room_to_the_others() {
    let cache = FileMetadataCache::new(measure_parts(&[("a", 5), ("b", 10)]));
    let (removed, kept) = (name_table("t", ""), name_table("u", ""));
    store_part(&cache, &kept, "a", 5);
    store_part(&cache, &removed, "p", 10);

    cache.remove_tables("t");
    store_part(&cache, &kept, "b", 10);

    assert!(find_part(&cache, &kept, "a").is_some());
    assert!(find_part(&cache, &kept, "b").is_some());
}

#[test]
fn the_cache_stays_within_its_budget_as_tables_come_and_go() {
    let cache = FileMetadataCache::new(measure_parts(&[("p", 10), ("p", 10)]));
    let name_numbered = |number: u32| name_table(&format!("t{number}"), "");

    for number in 0..100 {
        store_part(&cache, &name_numbered(number), "p", 10);
    }

    let still_held: Vec<u32> = (0..100)
        .filter(|number| find_part(&cache, &name_numbered(*number), "p").is_some())
        .collect();
    assert_eq!(still_held, [98, 99]);
}

#[test]
fn takers_of_one_tables_build_lock_hold_it_one_at_a_time() {
    let (cache, table) = open_cache(u64::MAX);
    let (report_locked, locked) = channel();
    let held = cache.lock_table(&table);

    let waiter = std::thread::spawn({
        let (cache, table) = (Arc::clone(&cache), table.clone());
        move || {
            let _held = cache.lock_table(&table);
            report_locked.send(()).unwrap();
        }
    });
    let locked_together = locked.recv_timeout(Duration::from_millis(100)).is_ok();
    drop(held);
    waiter.join().unwrap();

    assert!(!locked_together);
    assert!(locked.try_recv().is_ok());
}

#[test]
fn the_build_locks_of_two_tables_are_held_together() {
    let cache = FileMetadataCache::new(u64::MAX);
    let (report_locked, locked) = channel();
    let _held = cache.lock_table(&name_table("t", "1"));

    std::thread::spawn({
        let cache = Arc::clone(&cache);
        move || {
            let _held_together = cache.lock_table(&name_table("t", "2"));
            report_locked.send(()).unwrap();
        }
    });

    assert!(locked.recv_timeout(Duration::from_secs(10)).is_ok());
}

#[test]
fn a_released_build_lock_leaves_nothing_of_its_table_in_the_cache() {
    let (cache, table) = open_cache(u64::MAX);

    drop(cache.lock_table(&table));

    assert_eq!(Arc::strong_count(&table.table), 1);
}

/// Whether `table` holds a part under each of `names`.
fn hold_parts(cache: &FileMetadataCache, table: &TableKey, names: &[&str]) -> Vec<bool> {
    cache
        .find_metadata_parts::<SizedPart>(table, names.iter().map(|name| name.as_bytes()))
        .iter()
        .map(Option::is_some)
        .collect()
}

#[test]
fn a_stored_part_is_found_under_its_name() {
    let (cache, table) = open_cache(u64::MAX);

    cache.store_metadata_parts(&table, build_parts(&[("a", 10)]));

    let found = cache
        .find_metadata_parts::<SizedPart>(&table, [b"a".as_slice(), b"b".as_slice()].into_iter());
    assert_eq!(found[0].as_ref().map(|part| part.0), Some(10));
    assert!(found[1].is_none());
}

#[test]
fn a_part_nobody_asked_for_goes_before_a_part_stored_earlier_as_used() {
    let (cache, table) =
        open_cache(measure_parts(&[("used", 10), ("unasked", 10), ("new", 10)]) - 1);
    cache.store_metadata_parts(&table, build_parts(&[("used", 10)]));
    cache.store_unasked_metadata_parts(&table, build_parts(&[("unasked", 10)]));

    cache.store_metadata_parts(&table, build_parts(&[("new", 10)]));

    assert_eq!(
        hold_parts(&cache, &table, &["used", "unasked", "new"]),
        [true, false, true]
    );
}

#[test]
fn parts_nobody_asked_for_are_held_only_while_the_budget_has_room() {
    let (cache, table) = open_cache(measure_parts(&[("used", 10), ("fits", 10)]));
    cache.store_metadata_parts(&table, build_parts(&[("used", 10)]));

    let is_every_part_held =
        cache.store_unasked_metadata_parts(&table, build_parts(&[("fits", 10), ("over", 10)]));

    assert!(!is_every_part_held);
    assert_eq!(
        hold_parts(&cache, &table, &["used", "fits", "over"]),
        [true, true, false]
    );
}

#[test]
fn a_part_marked_used_outlives_the_ones_used_before_it() {
    let (cache, table) =
        open_cache(measure_parts(&[("marked", 10), ("left", 10), ("new", 10)]) - 1);
    cache.store_metadata_parts(&table, build_parts(&[("marked", 10), ("left", 10)]));
    cache.mark_metadata_parts_used(&table, &[key("marked")]);

    cache.store_metadata_parts(&table, build_parts(&[("new", 10)]));

    assert_eq!(
        hold_parts(&cache, &table, &["marked", "left", "new"]),
        [true, false, true]
    );
}

#[test]
fn the_budget_left_free_is_what_the_entries_held_do_not_count_for() {
    let (cache, table) = open_cache(measure_parts(&[("a", 10)]) + 7);

    cache.store_metadata_parts(&table, build_parts(&[("a", 10)]));

    assert_eq!(cache.free_bytes(), 7);
}

#[test]
fn removing_a_table_lets_go_of_its_parts() {
    let cache = FileMetadataCache::new(u64::MAX);
    let table = name_table("t", "manifests");
    cache.store_metadata_parts(&table, build_parts(&[("a", 10)]));

    cache.remove_tables("t");

    assert_eq!(hold_parts(&cache, &table, &["a"]), [false]);
}
