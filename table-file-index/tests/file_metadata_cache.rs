//! What the cache of partition indexes and footers holds, what it lets go
//! of past its budget, and which items a table's key finds.

use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::Duration;

use table_file_index::{
    CacheableIndex, FileMetadataCache, LookupUse, TableKey, VersionOrder, measure_footer_entry,
    measure_index_entry,
};

/// An index of a size the test names.
struct SizedIndex(u64);

impl CacheableIndex for SizedIndex {
    fn estimate_heap_bytes(&self) -> u64 {
        self.0
    }
}

fn key(name: &str) -> Arc<[u8]> {
    Arc::from(name.as_bytes())
}

fn at(sequence: i64) -> VersionOrder {
    VersionOrder {
        sequence,
        updated_ms: 0,
    }
}

/// What a cache counts indexes of the sizes given for between them, each
/// held for the partition it is named with.
fn measure_indexes(indexes: &[(&str, u64)]) -> u64 {
    indexes
        .iter()
        .map(|(name, heap_bytes)| measure_index_entry(name.as_bytes(), &SizedIndex(*heap_bytes)))
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

/// Hold in `table` an index of `heap_bytes` for the partition `name`, as
/// the version at `sequence` built it.
fn store_index(
    cache: &FileMetadataCache,
    table: &TableKey,
    name: &str,
    heap_bytes: u64,
    sequence: i64,
) {
    cache.store_index(
        table,
        &key(name),
        &Arc::new(SizedIndex(heap_bytes)),
        at(sequence),
    );
}

/// The index `table` holds for the partition `name`, found as a use of it.
fn find_index(
    cache: &FileMetadataCache,
    table: &TableKey,
    name: &str,
) -> Option<(Arc<SizedIndex>, VersionOrder)> {
    cache
        .find_indexes(table, [name.as_bytes()].into_iter(), LookupUse::Counted)
        .remove(0)
}

/// The index `table` holds for the partition `name`, not counted as used.
fn peek_index(
    cache: &FileMetadataCache,
    table: &TableKey,
    name: &str,
) -> Option<(Arc<SizedIndex>, VersionOrder)> {
    cache
        .find_indexes(table, [name.as_bytes()].into_iter(), LookupUse::Uncounted)
        .remove(0)
}

#[test]
fn a_stored_index_is_found_with_the_version_that_built_it() {
    let (cache, table) = open_cache(u64::MAX);

    store_index(&cache, &table, "p", 10, 4);

    let (index, built_at) = find_index(&cache, &table, "p").unwrap();
    assert_eq!((index.0, built_at), (10, at(4)));
}

#[test]
fn an_earlier_versions_index_does_not_replace_a_later_ones() {
    let (cache, table) = open_cache(u64::MAX);
    store_index(&cache, &table, "p", 10, 4);

    store_index(&cache, &table, "p", 20, 3);

    assert_eq!(find_index(&cache, &table, "p").unwrap().0.0, 10);
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_goes() {
    let (cache, table) = open_cache(measure_indexes(&[("old", 10), ("kept", 10), ("new", 10)]) - 1);
    store_index(&cache, &table, "old", 10, 1);
    store_index(&cache, &table, "kept", 10, 1);
    find_index(&cache, &table, "old");

    store_index(&cache, &table, "new", 10, 1);

    assert!(find_index(&cache, &table, "kept").is_none());
    assert!(find_index(&cache, &table, "old").is_some());
    assert!(find_index(&cache, &table, "new").is_some());
}

#[test]
fn a_peeked_index_is_not_counted_as_used() {
    let (cache, table) =
        open_cache(measure_indexes(&[("peeked", 10), ("kept", 10), ("new", 10)]) - 1);
    store_index(&cache, &table, "peeked", 10, 1);
    store_index(&cache, &table, "kept", 10, 1);
    let peeked = peek_index(&cache, &table, "peeked");

    store_index(&cache, &table, "new", 10, 1);

    assert!(peeked.is_some());
    assert!(find_index(&cache, &table, "peeked").is_none());
    assert!(find_index(&cache, &table, "kept").is_some());
}

#[test]
fn indexes_found_together_are_each_counted_as_used() {
    let (cache, table) =
        open_cache(measure_indexes(&[("first", 10), ("second", 10), ("new", 10)]) - 1);
    store_index(&cache, &table, "first", 10, 1);
    store_index(&cache, &table, "second", 10, 2);
    let found = cache.find_indexes::<SizedIndex>(
        &table,
        [b"second".as_slice(), b"first".as_slice()].into_iter(),
        LookupUse::Counted,
    );

    store_index(&cache, &table, "new", 10, 1);

    assert_eq!(found[0].as_ref().unwrap().1, at(2));
    assert!(find_index(&cache, &table, "second").is_none());
    assert!(find_index(&cache, &table, "first").is_some());
}

#[test]
fn the_entry_just_stored_stays_whatever_it_takes() {
    let (cache, table) = open_cache(5);

    store_index(&cache, &table, "p", 100, 1);

    assert!(find_index(&cache, &table, "p").is_some());
}

#[test]
fn an_index_marked_used_outlives_the_footers_taken_before_it() {
    let footer_bytes = measure_footer_entry("f", &[]);
    let (cache, table) = open_cache(measure_indexes(&[("p", 10), ("q", 1)]) + footer_bytes - 1);
    store_index(&cache, &table, "p", 10, 1);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);
    cache.mark_indexes_used(&table, &[key("p")]);

    store_index(&cache, &table, "q", 1, 1);

    assert!(find_index(&cache, &table, "p").is_some());
    assert!(cache.find_footers(&table, ["f"].into_iter(), LookupUse::Uncounted)[0].is_none());
}

#[test]
fn a_removed_footer_is_no_longer_found() {
    let (cache, table) = open_cache(u64::MAX);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);

    cache.remove_footers(&table, ["f"].into_iter());

    assert!(cache.find_footers(&table, ["f"].into_iter(), LookupUse::Counted)[0].is_none());
}

#[test]
fn equal_keys_computed_apart_find_each_others_items() {
    let cache = FileMetadataCache::new(u64::MAX);
    store_index(&cache, &name_table("t", "1"), "p", 10, 1);
    cache.store_footers(&name_table("t", "1"), vec![(Arc::from("f"), Vec::new())]);

    let index = find_index(&cache, &name_table("t", "1"), "p");
    let footers = cache.find_footers(&name_table("t", "1"), ["f"].into_iter(), LookupUse::Counted);

    assert!(index.is_some() && footers[0].is_some());
}

#[test]
fn keys_that_differ_in_either_part_do_not_find_each_others_items() {
    let cache = FileMetadataCache::new(u64::MAX);
    store_index(&cache, &name_table("t", "1"), "p", 10, 1);

    let read_another_way = find_index(&cache, &name_table("t", "2"), "p");
    let of_another_table = find_index(&cache, &name_table("u", "1"), "p");

    assert!(read_another_way.is_none() && of_another_table.is_none());
}

#[test]
fn a_tables_indexed_partitions_are_listed_without_its_footers() {
    let (cache, table) = open_cache(u64::MAX);
    store_index(&cache, &table, "q", 10, 1);
    store_index(&cache, &table, "p", 10, 1);
    cache.store_footers(&table, vec![(Arc::from("f"), Vec::new())]);

    let partitions = cache.list_indexed_partitions(&table);

    assert_eq!(partitions, [key("p"), key("q")]);
}

#[test]
fn an_evicted_index_leaves_its_tables_listing() {
    let (cache, table) = open_cache(measure_indexes(&[("a", 10), ("b", 10)]));
    store_index(&cache, &table, "a", 10, 1);
    store_index(&cache, &table, "b", 10, 1);

    store_index(&cache, &table, "c", 10, 1);

    assert_eq!(cache.list_indexed_partitions(&table), [key("b"), key("c")]);
}

#[test]
fn a_table_whose_last_item_is_evicted_is_forgotten() {
    let cache = FileMetadataCache::new(measure_indexes(&[("p", 10)]));
    let (evicted, kept) = (name_table("t", ""), name_table("u", ""));
    store_index(&cache, &evicted, "p", 10, 1);
    let is_kept_with_an_item = Arc::strong_count(&evicted.table) > 1;

    store_index(&cache, &kept, "p", 10, 1);

    assert!(is_kept_with_an_item);
    assert_eq!(Arc::strong_count(&evicted.table), 1);
}

#[test]
fn removing_a_table_removes_it_under_every_key_and_leaves_the_others() {
    let cache = FileMetadataCache::new(u64::MAX);
    let (first, second) = (name_table("t", "1"), name_table("t", "2"));
    let other = name_table("u", "1");
    for table in [&first, &second, &other] {
        store_index(&cache, table, "p", 10, 1);
    }
    cache.store_footers(&first, vec![(Arc::from("f"), Vec::new())]);

    cache.remove_tables("t");

    assert!(find_index(&cache, &first, "p").is_none());
    assert!(find_index(&cache, &second, "p").is_none());
    assert!(cache.find_footers(&first, ["f"].into_iter(), LookupUse::Counted)[0].is_none());
    assert!(find_index(&cache, &other, "p").is_some());
}

#[test]
fn a_removed_table_leaves_its_room_to_the_others() {
    let cache = FileMetadataCache::new(measure_indexes(&[("a", 5), ("b", 10)]));
    let (removed, kept) = (name_table("t", ""), name_table("u", ""));
    store_index(&cache, &kept, "a", 5, 1);
    store_index(&cache, &removed, "p", 10, 1);

    cache.remove_tables("t");
    store_index(&cache, &kept, "b", 10, 1);

    assert!(find_index(&cache, &kept, "a").is_some());
    assert!(find_index(&cache, &kept, "b").is_some());
}

#[test]
fn the_cache_stays_within_its_budget_as_tables_come_and_go() {
    let cache = FileMetadataCache::new(measure_indexes(&[("p", 10), ("p", 10)]));
    let name_numbered = |number: u32| name_table(&format!("t{number}"), "");

    for number in 0..100 {
        store_index(&cache, &name_numbered(number), "p", 10, 1);
    }

    let still_held: Vec<u32> = (0..100)
        .filter(|number| peek_index(&cache, &name_numbered(*number), "p").is_some())
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
