//! Per-row-group memory of pushed-down filter outcomes.
//!
//! When a scan actually evaluates a pushed equality predicate against a row
//! group's data (i.e. min/max statistics could not rule the row group out), the
//! decoder remembers whether any row matched. The next scan consults that
//! memory at claim time and skips a row group a remembered predicate — or a
//! remembered *conjunction* of predicates (`a = 5 AND b = 8`) — proved empty,
//! before any of the row group's pages are fetched. Statistics never learn
//! this: the constant lies inside the min/max range, so only running the filter
//! can prove the value absent.
//!
//! The cache lives on the shared [`RowGroupMetadata`], so it persists for as
//! long as the row group itself is held in memory and is dropped with it (a
//! rewritten file gets fresh metadata and an empty cache). Each row group keeps
//! its own small LRU of recent outcomes; an entry is bumped to most recently
//! used whenever it is used to skip the row group or re-recorded by a scan.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, Datum, Scalar};

use crate::parquet::RowGroupFilter;
use crate::parquet::reading::ScanEqualityPredicate;
use crate::parquet::types::metadata::RowGroupMetadata;

/// How many outcomes one row group remembers; older entries fall off LRU-style.
const FILTER_CACHE_CAPACITY: usize = 10;

/// One remembered outcome: whether any single row of the row group satisfied
/// every predicate in `predicates` at once. A single pushed filter is a set of
/// one; a conjunction (`a = 5 AND b = 8`) carries every predicate it ANDed.
struct FilterOutcome {
    predicates: Vec<ScanEqualityPredicate>,
    matched: bool,
}

impl FilterOutcome {
    /// Whether this entry records exactly the given predicate set, in any order.
    fn is_for(&self, predicates: &[&ScanEqualityPredicate]) -> bool {
        self.predicates.len() == predicates.len() && self.is_subset_of_slice(predicates)
    }

    /// Whether every predicate of this entry appears in `predicates`.
    fn is_subset_of(&self, predicates: &[ScanEqualityPredicate]) -> bool {
        self.predicates
            .iter()
            .all(|own| predicates.iter().any(|p| predicate_eq(own, p)))
    }

    fn is_subset_of_slice(&self, predicates: &[&ScanEqualityPredicate]) -> bool {
        self.predicates
            .iter()
            .all(|own| predicates.iter().any(|p| predicate_eq(own, p)))
    }
}

/// Remembered outcomes of pushed-down equality predicates evaluated against one
/// row group's data. See the module docs for the full contract.
#[derive(Default)]
pub struct FilterResultCache {
    /// Most recently used first.
    entries: Mutex<VecDeque<FilterOutcome>>,
}

impl FilterResultCache {
    /// Remember that a scan evaluated the conjunction of `predicates` over the
    /// *entire* row group and whether any row satisfied all of them, inserting
    /// or refreshing the entry as most recently used. A predicate set with a
    /// NULL constant is not remembered: SQL NULL never equals anything, so the
    /// entry could never be matched again and would only crowd out live ones.
    pub fn record(&self, predicates: &[&ScanEqualityPredicate], matched: bool) {
        if predicates.is_empty() || predicates.iter().any(|p| scalar_is_null(&p.value)) {
            return;
        }
        let mut entries = self.entries.lock().unwrap();
        if let Some(pos) = entries.iter().position(|e| e.is_for(predicates)) {
            let mut entry = entries.remove(pos).unwrap();
            entry.matched = matched;
            entries.push_front(entry);
            return;
        }
        entries.push_front(FilterOutcome {
            predicates: predicates.iter().map(|&p| p.clone()).collect(),
            matched,
        });
        entries.truncate(FILTER_CACHE_CAPACITY);
    }

    /// The remembered outcome for exactly this predicate set, bumping the entry
    /// to most recently used on a hit.
    pub fn lookup(&self, predicates: &[&ScanEqualityPredicate]) -> Option<bool> {
        let mut entries = self.entries.lock().unwrap();
        let pos = entries.iter().position(|e| e.is_for(predicates))?;
        let entry = entries.remove(pos).unwrap();
        let matched = entry.matched;
        entries.push_front(entry);
        Some(matched)
    }

    /// Whether a remembered outcome proves no row can pass a query that
    /// requires all of `predicates`: some entry whose predicate set is a subset
    /// of them matched no row. The proving entry is bumped to most recently
    /// used, since it just pruned the row group.
    pub fn proves_no_match(&self, predicates: &[ScanEqualityPredicate]) -> bool {
        let mut entries = self.entries.lock().unwrap();
        let Some(pos) = entries
            .iter()
            .position(|e| !e.matched && e.is_subset_of(predicates))
        else {
            return false;
        };
        let entry = entries.remove(pos).unwrap();
        entries.push_front(entry);
        true
    }
}

/// Wrap a claim-time row-group `filter` with each row group's remembered filter
/// outcomes: a row group some cached outcome proves empty for this query's
/// pushed predicates is skipped before any of its pages are fetched. Applied at
/// claim time exactly like the dynamic filter, so skipping never renumbers the
/// surviving row groups and a late materialize still addresses the same global
/// indices.
pub(crate) fn filter_with_cached_outcomes(
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    filter: Option<RowGroupFilter>,
) -> Option<RowGroupFilter> {
    if eq_predicates.is_empty() {
        return filter;
    }
    Some(Arc::new(move |row_group: &RowGroupMetadata| -> bool {
        if row_group.filter_cache.proves_no_match(&eq_predicates) {
            return false;
        }
        filter.as_ref().is_none_or(|f| f(row_group))
    }))
}

/// Whether two predicates compare the same leaf against the same constant.
fn predicate_eq(a: &ScanEqualityPredicate, b: &ScanEqualityPredicate) -> bool {
    a.column_idx == b.column_idx && a.path == b.path && scalar_eq(&a.value, &b.value)
}

/// `a == b` over two single-value scalars: same type and equal per the arrow
/// comparison kernel. A NULL on either side is never equal.
fn scalar_eq(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    a.get().0.data_type() == b.get().0.data_type()
        && arrow_ord::cmp::eq(a as &dyn Datum, b as &dyn Datum)
            .is_ok_and(|r| r.len() == 1 && r.is_valid(0) && r.value(0))
}

fn scalar_is_null(value: &Scalar<ArrayRef>) -> bool {
    let (array, _) = value.get();
    array.len() != 1 || array.is_null(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use std::sync::Arc;

    fn eq_predicate(column_idx: usize, value: i64) -> ScanEqualityPredicate {
        ScanEqualityPredicate {
            column_idx,
            path: Vec::new(),
            value: Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
        }
    }

    #[test]
    fn lookup_returns_the_recorded_outcome_and_misses_unknown_values() {
        let cache = FilterResultCache::default();
        let present = eq_predicate(0, 30);
        let absent = eq_predicate(0, 83);

        cache.record(&[&present], true);
        cache.record(&[&absent], false);

        assert_eq!(cache.lookup(&[&present]), Some(true));
        assert_eq!(cache.lookup(&[&absent]), Some(false));
        assert_eq!(cache.lookup(&[&eq_predicate(0, 1337)]), None);
        assert_eq!(cache.lookup(&[&eq_predicate(1, 30)]), None);
    }

    #[test]
    fn recording_past_capacity_evicts_the_least_recently_used_entry() {
        let cache = FilterResultCache::default();
        for value in 0..10 {
            cache.record(&[&eq_predicate(0, value)], false);
        }

        // Bump the oldest entry, then push one more: the eviction must take
        // the now-oldest value 1, not the bumped value 0.
        assert!(cache.proves_no_match(&[eq_predicate(0, 0)]));
        cache.record(&[&eq_predicate(0, 10)], false);

        assert_eq!(cache.lookup(&[&eq_predicate(0, 0)]), Some(false));
        assert_eq!(cache.lookup(&[&eq_predicate(0, 1)]), None);
        assert_eq!(cache.lookup(&[&eq_predicate(0, 10)]), Some(false));
    }

    #[test]
    fn re_recording_updates_the_entry_in_place() {
        let cache = FilterResultCache::default();
        let predicate = eq_predicate(0, 5);

        cache.record(&[&predicate], false);
        cache.record(&[&predicate], true);

        assert_eq!(cache.lookup(&[&predicate]), Some(true));
        assert!(!cache.proves_no_match(&[predicate]));
    }

    #[test]
    fn a_false_conjunction_proves_only_queries_carrying_every_predicate() {
        let cache = FilterResultCache::default();
        let a = eq_predicate(0, 5);
        let b = eq_predicate(1, 8);
        cache.record(&[&a, &b], false);

        // Both predicates present (in either order, with extras) → proven empty.
        assert!(cache.proves_no_match(&[a.clone(), b.clone()]));
        assert!(cache.proves_no_match(&[b.clone(), a.clone(), eq_predicate(2, 1)]));
        // Either predicate alone could still match rows on its own.
        assert!(!cache.proves_no_match(&[a.clone()]));
        assert!(!cache.proves_no_match(&[b]));
        // The conjunction and the single predicate are distinct entries.
        assert_eq!(cache.lookup(&[&a]), None);
    }

    #[test]
    fn a_false_single_prunes_any_query_that_includes_it() {
        let cache = FilterResultCache::default();
        let a = eq_predicate(0, 5);
        cache.record(&[&a], false);

        assert!(cache.proves_no_match(&[a.clone(), eq_predicate(1, 8)]));
        assert!(cache.proves_no_match(&[a]));
    }

    #[test]
    fn null_constants_are_not_remembered() {
        let cache = FilterResultCache::default();
        let null = ScanEqualityPredicate {
            column_idx: 0,
            path: Vec::new(),
            value: Scalar::new(Arc::new(Int64Array::from(vec![None::<i64>])) as ArrayRef),
        };

        cache.record(&[&null], false);

        assert_eq!(cache.lookup(&[&null]), None);
        assert!(!cache.proves_no_match(&[null]));
    }
}
