//! What the cache of partition indexes and footers holds, and what it lets
//! go of past its budget.

use std::sync::Arc;

use table_file_index::{
    CacheableIndex, FileMetadataCache, LookupUse, TableCacheScope, VersionOrder,
    measure_footer_entry, measure_index_entry,
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

fn open_scope(capacity_bytes: u64) -> TableCacheScope<SizedIndex> {
    TableCacheScope::new(FileMetadataCache::new(capacity_bytes))
}

/// The index `scope` holds for the partition `name`, found as a use of it.
fn find_index(
    scope: &TableCacheScope<SizedIndex>,
    name: &str,
) -> Option<(Arc<SizedIndex>, VersionOrder)> {
    scope
        .find_indexes([name.as_bytes()].into_iter(), LookupUse::Counted)
        .remove(0)
}

/// The index `scope` holds for the partition `name`, not counted as used.
fn peek_index(
    scope: &TableCacheScope<SizedIndex>,
    name: &str,
) -> Option<(Arc<SizedIndex>, VersionOrder)> {
    scope
        .find_indexes([name.as_bytes()].into_iter(), LookupUse::Uncounted)
        .remove(0)
}

#[test]
fn a_stored_index_is_found_with_the_version_that_built_it() {
    let scope = open_scope(u64::MAX);

    scope.store_index(&key("p"), &Arc::new(SizedIndex(10)), at(4));

    let (index, built_at) = find_index(&scope, "p").unwrap();
    assert_eq!((index.0, built_at), (10, at(4)));
}

#[test]
fn an_earlier_versions_index_does_not_replace_a_later_ones() {
    let scope = open_scope(u64::MAX);
    scope.store_index(&key("p"), &Arc::new(SizedIndex(10)), at(4));

    scope.store_index(&key("p"), &Arc::new(SizedIndex(20)), at(3));

    assert_eq!(find_index(&scope, "p").unwrap().0.0, 10);
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_goes() {
    let scope = open_scope(measure_indexes(&[("old", 10), ("kept", 10), ("new", 10)]) - 1);
    scope.store_index(&key("old"), &Arc::new(SizedIndex(10)), at(1));
    scope.store_index(&key("kept"), &Arc::new(SizedIndex(10)), at(1));
    find_index(&scope, "old");

    scope.store_index(&key("new"), &Arc::new(SizedIndex(10)), at(1));

    assert!(find_index(&scope, "kept").is_none());
    assert!(find_index(&scope, "old").is_some() && find_index(&scope, "new").is_some());
}

#[test]
fn a_peeked_index_is_not_counted_as_used() {
    let scope = open_scope(measure_indexes(&[("peeked", 10), ("kept", 10), ("new", 10)]) - 1);
    scope.store_index(&key("peeked"), &Arc::new(SizedIndex(10)), at(1));
    scope.store_index(&key("kept"), &Arc::new(SizedIndex(10)), at(1));
    let peeked = peek_index(&scope, "peeked");

    scope.store_index(&key("new"), &Arc::new(SizedIndex(10)), at(1));

    assert!(peeked.is_some());
    assert!(find_index(&scope, "peeked").is_none());
    assert!(find_index(&scope, "kept").is_some());
}

#[test]
fn indexes_found_together_are_each_counted_as_used() {
    let scope = open_scope(measure_indexes(&[("first", 10), ("second", 10), ("new", 10)]) - 1);
    scope.store_index(&key("first"), &Arc::new(SizedIndex(10)), at(1));
    scope.store_index(&key("second"), &Arc::new(SizedIndex(10)), at(2));
    let found = scope.find_indexes(
        [b"second".as_slice(), b"first".as_slice()].into_iter(),
        LookupUse::Counted,
    );

    scope.store_index(&key("new"), &Arc::new(SizedIndex(10)), at(1));

    assert_eq!(found[0].as_ref().unwrap().1, at(2));
    assert!(find_index(&scope, "second").is_none());
    assert!(find_index(&scope, "first").is_some());
}

#[test]
fn the_entry_just_stored_stays_whatever_it_takes() {
    let scope = open_scope(5);

    scope.store_index(&key("p"), &Arc::new(SizedIndex(100)), at(1));

    assert!(find_index(&scope, "p").is_some());
}

#[test]
fn an_index_marked_used_outlives_the_footers_taken_before_it() {
    let footer_bytes = measure_footer_entry("f", &[]);
    let scope = open_scope(measure_indexes(&[("p", 10), ("q", 1)]) + footer_bytes - 1);
    scope.store_index(&key("p"), &Arc::new(SizedIndex(10)), at(1));
    scope.store_footers(vec![(Arc::from("f"), Vec::new())]);
    scope.mark_indexes_used(&[key("p")]);

    scope.store_index(&key("q"), &Arc::new(SizedIndex(1)), at(1));

    assert!(find_index(&scope, "p").is_some());
    assert!(scope.find_footers(["f"].into_iter(), LookupUse::Uncounted)[0].is_none());
}

#[test]
fn a_removed_footer_is_no_longer_found() {
    let scope = open_scope(u64::MAX);
    scope.store_footers(vec![(Arc::from("f"), Vec::new())]);

    scope.remove_footers(["f"].into_iter());

    assert!(scope.find_footers(["f"].into_iter(), LookupUse::Counted)[0].is_none());
}

#[test]
fn a_dropped_scope_leaves_its_room_to_the_others() {
    let cache = FileMetadataCache::new(measure_indexes(&[("a", 5), ("b", 10)]));
    let dropped: TableCacheScope<SizedIndex> = TableCacheScope::new(Arc::clone(&cache));
    let kept: TableCacheScope<SizedIndex> = TableCacheScope::new(cache);
    kept.store_index(&key("a"), &Arc::new(SizedIndex(5)), at(1));
    dropped.store_index(&key("p"), &Arc::new(SizedIndex(10)), at(1));

    drop(dropped);
    kept.store_index(&key("b"), &Arc::new(SizedIndex(10)), at(1));

    assert!(find_index(&kept, "a").is_some() && find_index(&kept, "b").is_some());
}

#[test]
fn scopes_do_not_see_each_others_partitions() {
    let cache = FileMetadataCache::new(u64::MAX);
    let first: TableCacheScope<SizedIndex> = TableCacheScope::new(Arc::clone(&cache));
    let second: TableCacheScope<SizedIndex> = TableCacheScope::new(cache);

    first.store_index(&key("p"), &Arc::new(SizedIndex(10)), at(1));

    assert!(find_index(&second, "p").is_none());
    assert_eq!(first.list_indexed_partitions(), [key("p")]);
}
