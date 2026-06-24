//! A cross-query cache of which rows survive a filter, so re-running the same
//! query replays the surviving row indices instead of decoding and filtering
//! the data again.
//!
//! ```text
//!   first run:   scan -> filter -> [populate]   captures survivors per row group
//!   repeat run:  scan reads only the cached surviving rows  (no re-filtering)
//! ```
//!
//! # Why this is sound
//!
//! A "global row group" identifies a row group across the whole table, and the
//! Parquet files behind it are immutable (content-addressed in the catalog log).
//! So once we record the rows that pass a filter over a row group, that answer
//! can never go stale: the only way the rows change is a new file, which is a
//! new global row group with a different key. There is no invalidation.
//!
//! # Granularity and completeness
//!
//! Entries are keyed per `(filter, global row group)`. A row group is cached
//! independently of every other: a present entry is *complete* (it lists every
//! surviving row of that group), and an absent entry simply means "not cached,
//! scan it normally". This is what lets the populate side publish whatever
//! groups it managed to fully observe without needing a table-wide barrier.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Identifies one cached answer: the rows that survived a particular filter over
/// one table-wide row group.
///
/// `filter` is a stable hash of the *full* filter (every `WHERE` condition), not
/// just the subset pushed into the scan. Keying on a subset would be unsound:
/// two queries that push the same predicate but differ in their remaining
/// conditions would collide and replay each other's (wrong) survivors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ConditionKey {
    filter: u64,
    global_row_group: u32,
}

/// Per-`(filter, row group)` cache of surviving row indices.
///
/// Read-mostly: an entry is written once, the first time a filter is seen over a
/// row group, and read on every repeat of that query. A single `RwLock` over a
/// `HashMap` fits the access pattern (many concurrent readers, rare writers) and
/// keeps the structure obvious; the values are cheap-to-clone `Arc<[u32]>`, so a
/// lookup clones a pointer, not the indices.
#[derive(Default)]
pub struct QueryConditionCache {
    entries: RwLock<HashMap<ConditionKey, Arc<[u32]>>>,
    /// Filters that have been fully populated (every row group observed). Kept
    /// separately from `entries` because a filter can be complete yet have no
    /// surviving rows (it matched nothing), which would store no entries. Lets
    /// the populate side skip a filter that is already cached.
    populated: RwLock<HashSet<u64>>,
    /// Filters deliberately *not* cached because too many rows survive them.
    /// Replaying a non-selective filter is no faster than a plain scan (every
    /// data page still has surviving rows, so none can be skipped) and risks
    /// exhausting the page-cache ring, so we record the decision once and take
    /// the plain path on every future run instead of re-populating each time.
    skipped: RwLock<HashSet<u64>>,
}

impl QueryConditionCache {
    /// The surviving rows of `global_row_group` under `filter`, if cached. The
    /// returned slice is sorted ascending (the scan consumes it as sorted global
    /// indices). `None` means "not cached" — scan the group normally.
    pub fn lookup(&self, filter: u64, global_row_group: u32) -> Option<Arc<[u32]>> {
        let key = ConditionKey {
            filter,
            global_row_group,
        };
        self.entries.read().unwrap().get(&key).cloned()
    }

    /// Whether `filter` has already been fully populated, so the populate side can
    /// avoid scanning in metadata mode and re-recording it.
    pub fn is_populated(&self, filter: u64) -> bool {
        self.populated.read().unwrap().contains(&filter)
    }

    /// Whether `filter` was judged not worth caching (too many survivors). The
    /// compiler then takes the plain scan path and never re-populates it.
    pub fn is_skipped(&self, filter: u64) -> bool {
        self.skipped.read().unwrap().contains(&filter)
    }

    /// Record that `filter` is not worth caching. Idempotent.
    pub fn mark_skipped(&self, filter: u64) {
        self.skipped.write().unwrap().insert(filter);
    }

    /// Number of filters fully populated so far. For tests/introspection.
    pub fn populated_count(&self) -> usize {
        self.populated.read().unwrap().len()
    }

    /// Publish the complete per-row-group survivor sets for one filter and mark it
    /// populated.
    ///
    /// Every `(group, indices)` pair must be the group's *complete* survivor set:
    /// callers accumulate across all workers and publish once, at the end, so a
    /// partial set is never exposed (which would later drop matching rows). Each
    /// set is sorted before it is stored. Already-present keys are left as-is — a
    /// group's answer can't differ between runs of the same filter over the same
    /// table version, so the first writer wins and concurrent re-populations are
    /// harmless. An empty `survivors` still marks the filter populated (it matched
    /// no rows, which is itself the cached answer).
    pub fn publish(&self, filter: u64, survivors: HashMap<u32, Vec<u32>>) {
        let mut entries = self.entries.write().unwrap();
        for (global_row_group, mut indices) in survivors {
            let key = ConditionKey {
                filter,
                global_row_group,
            };
            entries.entry(key).or_insert_with(|| {
                indices.sort_unstable();
                Arc::from(indices)
            });
        }
        self.populated.write().unwrap().insert(filter);
    }

    /// Drop every cached answer. Called when a table's committed file set changes:
    /// the "global row group" is a positional index over the current files, so a
    /// change can shift what an index refers to, and stale survivors would select
    /// the wrong rows. Cheaper and simpler than version-scoping each key, and
    /// table changes are rare relative to queries.
    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
        self.populated.write().unwrap().clear();
        self.skipped.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_misses_before_anything_is_published() {
        let cache = QueryConditionCache::default();

        assert!(cache.lookup(7, 0).is_none());
    }

    #[test]
    fn published_survivors_come_back_sorted_per_group() {
        let cache = QueryConditionCache::default();

        cache.publish(7, HashMap::from([(0, vec![5, 1, 3]), (2, vec![9])]));

        assert_eq!(cache.lookup(7, 0).unwrap().as_ref(), &[1, 3, 5]);
        assert_eq!(cache.lookup(7, 2).unwrap().as_ref(), &[9]);
        assert!(cache.lookup(7, 1).is_none());
    }

    #[test]
    fn entries_are_keyed_by_both_filter_and_row_group() {
        let cache = QueryConditionCache::default();

        cache.publish(1, HashMap::from([(0, vec![10])]));
        cache.publish(2, HashMap::from([(0, vec![20])]));

        assert_eq!(cache.lookup(1, 0).unwrap().as_ref(), &[10]);
        assert_eq!(cache.lookup(2, 0).unwrap().as_ref(), &[20]);
    }

    #[test]
    fn first_writer_wins_for_an_already_cached_group() {
        let cache = QueryConditionCache::default();

        cache.publish(1, HashMap::from([(0, vec![1, 2])]));
        cache.publish(1, HashMap::from([(0, vec![3, 4])]));

        assert_eq!(cache.lookup(1, 0).unwrap().as_ref(), &[1, 2]);
    }

    #[test]
    fn a_filter_matching_no_rows_is_still_marked_populated() {
        let cache = QueryConditionCache::default();

        assert!(!cache.is_populated(1));
        cache.publish(1, HashMap::new());

        assert!(cache.is_populated(1));
        assert!(cache.lookup(1, 0).is_none());
    }

    #[test]
    fn clear_drops_entries_and_populated_marks() {
        let cache = QueryConditionCache::default();
        cache.publish(1, HashMap::from([(0, vec![1, 2])]));

        cache.clear();

        assert!(!cache.is_populated(1));
        assert!(cache.lookup(1, 0).is_none());
    }

    #[test]
    fn skip_marker_is_recorded_and_cleared() {
        let cache = QueryConditionCache::default();

        assert!(!cache.is_skipped(1));
        cache.mark_skipped(1);
        assert!(cache.is_skipped(1));

        cache.clear();
        assert!(!cache.is_skipped(1));
    }
}
