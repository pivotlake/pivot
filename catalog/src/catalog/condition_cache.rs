//! [`QueryConditionCache`] — memoized filter results per row group.
//!
//! Maps `(file path, row group, canonical filter conjunction)` to the sorted
//! row positions the conjunction keeps in that row group. A scan that has the
//! positions for its conjunction reads only those rows (skipping the fetch,
//! decompress, and decode of everything else, and skipping empty row groups
//! outright); a scan that does not observes its filter and publishes the
//! positions here for the next run.
//!
//! Keys use the file's store-relative resolved path plus the row group's
//! file-local index: stable across queries and restarts, and self-invalidating
//! under compaction (rewritten data gets new paths, so stale entries are
//! simply never asked for again; [`invalidate_files`] reclaims their memory
//! when files leave the manifest).
//!
//! [`invalidate_files`]: QueryConditionCache::invalidate_files

use ahash::{HashMap, HashSet};
use dispatch::env::get_env_var_with_default;
use planner::condition_key::ConditionKey;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed per-entry overhead charged on top of the positions themselves (map
/// keys, entry bookkeeping). An estimate; the budget only needs to be roughly
/// honored.
const ENTRY_OVERHEAD_BYTES: usize = 128;

/// A condition keeping more than this fraction of a row group's rows is
/// recorded as [`CachedPositions::Dense`] instead of storing the positions.
/// Masked reads only beat a plain scan when survivors are rare: at even a few
/// percent kept (uniformly spread), no page is entirely skippable and the
/// decoder's skip/keep run cursor alternates tiny runs, costing more than
/// decoding straight through — while the positions cost 4 bytes per surviving
/// row to keep. A dense row group simply scans plain; the marker still counts
/// as covered, ending observation for that row group.
const DENSE_POSITIONS_FRACTION: f64 = 0.05;

/// On overflow, evict down to this fraction of capacity rather than stopping
/// at the cap: hysteresis amortizes the eviction sweep over many inserts.
const EVICTION_WATERMARK_FRACTION: f64 = 0.75;

/// A row group's stable identity: the file's resolved path + its file-local
/// row-group index.
type RowGroupId = (Arc<str>, usize);

/// A cached filter result for one row group.
#[derive(Clone)]
pub enum CachedPositions {
    /// The sorted surviving row positions; the scan reads only these.
    Sparse(Arc<Vec<u32>>),
    /// The condition keeps most of the row group (see
    /// [`DENSE_POSITIONS_FRACTION`]): scan plain, but skip re-observation.
    Dense,
}

/// Cross-query cache of filter results, owned by the catalog and shared with
/// every query. Bounded by a byte budget with least-recently-used eviction.
pub struct QueryConditionCache {
    enabled: bool,
    capacity_bytes: usize,
    state: Mutex<CacheState>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
}

#[derive(Default)]
struct CacheState {
    /// Per condition, the row groups whose surviving positions are known.
    conditions: HashMap<ConditionKey, HashMap<RowGroupId, CacheEntry>>,
    /// Total bytes charged across every entry.
    bytes: usize,
    /// Monotonic use counter stamping entries for least-recently-used eviction.
    tick: u64,
}

struct CacheEntry {
    positions: CachedPositions,
    bytes: usize,
    last_used: u64,
}

/// Counters exposed for tests and introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub entries: usize,
    pub bytes: usize,
}

impl QueryConditionCache {
    pub fn new(enabled: bool, capacity_bytes: usize) -> Self {
        Self {
            enabled,
            capacity_bytes,
            state: Mutex::new(CacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }

    /// Build from the environment: `PIVOT_CONDITION_CACHE` (true/false, default
    /// true) and `PIVOT_CONDITION_CACHE_MB` (default 512). The default is sized
    /// so a handful of moderately selective conditions over a large table fit
    /// without eviction churn (positions cost 4 bytes per surviving row; dense
    /// conditions cost a marker only).
    pub fn from_env() -> Self {
        Self::new(
            get_env_var_with_default("PIVOT_CONDITION_CACHE", true),
            get_env_var_with_default("PIVOT_CONDITION_CACHE_MB", 512usize) * 1024 * 1024,
        )
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The cached filter result of `condition` for one row group, if known.
    pub fn get(
        &self,
        file_path: &Arc<str>,
        file_row_group_idx: usize,
        condition: &ConditionKey,
    ) -> Option<CachedPositions> {
        if !self.enabled {
            return None;
        }
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        let entry = state
            .conditions
            .get_mut(condition)
            .and_then(|row_groups| row_groups.get_mut(&(file_path.clone(), file_row_group_idx)));
        match entry {
            Some(entry) => {
                entry.last_used = tick;
                let positions = entry.positions.clone();
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(positions)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Record `condition`'s surviving positions for one row group. A result
    /// keeping more than [`DENSE_POSITIONS_FRACTION`] of the rows is stored as
    /// a dense marker (scan plain, stop observing) instead of the positions.
    /// Idempotent: an already-present entry is kept (concurrent identical
    /// queries publish the same positions).
    pub fn insert(
        &self,
        file_path: Arc<str>,
        file_row_group_idx: usize,
        condition: &ConditionKey,
        positions: Arc<Vec<u32>>,
        row_group_num_rows: u64,
    ) {
        if !self.enabled {
            return;
        }
        let dense = positions.len() as f64 > row_group_num_rows as f64 * DENSE_POSITIONS_FRACTION;
        let (positions, entry_bytes) = if dense {
            (CachedPositions::Dense, ENTRY_OVERHEAD_BYTES)
        } else {
            let bytes = positions.len() * size_of::<u32>() + ENTRY_OVERHEAD_BYTES;
            (CachedPositions::Sparse(positions), bytes)
        };
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        let row_groups = match state.conditions.get_mut(condition) {
            Some(row_groups) => row_groups,
            None => state.conditions.entry(condition.clone()).or_default(),
        };
        if row_groups.contains_key(&(file_path.clone(), file_row_group_idx)) {
            return;
        }
        row_groups.insert(
            (file_path, file_row_group_idx),
            CacheEntry {
                positions,
                bytes: entry_bytes,
                last_used: tick,
            },
        );
        state.bytes += entry_bytes;
        self.inserts.fetch_add(1, Ordering::Relaxed);
        if state.bytes > self.capacity_bytes {
            let watermark = (self.capacity_bytes as f64 * EVICTION_WATERMARK_FRACTION) as usize;
            evict_to_watermark(&mut state, watermark);
        }
    }

    /// Drop every entry for the given file paths — called when files leave a
    /// table's manifest (compaction, deletion). Purely memory hygiene:
    /// correctness never depends on it because removed paths are never asked
    /// for again.
    pub fn invalidate_files<S: AsRef<str>>(&self, paths: impl IntoIterator<Item = S>) {
        let paths: HashSet<String> = paths.into_iter().map(|p| p.as_ref().to_string()).collect();
        if paths.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let mut freed = 0usize;
        for row_groups in state.conditions.values_mut() {
            row_groups.retain(|(path, _), entry| {
                let keep = !paths.contains(&**path);
                if !keep {
                    freed += entry.bytes;
                }
                keep
            });
        }
        state
            .conditions
            .retain(|_, row_groups| !row_groups.is_empty());
        state.bytes -= freed;
    }

    pub fn stats(&self) -> ConditionCacheStats {
        let state = self.state.lock().unwrap();
        ConditionCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            entries: state.conditions.values().map(HashMap::len).sum(),
            bytes: state.bytes,
        }
    }
}

/// Evict least-recently-used entries until the charged bytes drop to
/// `watermark`. One sorted sweep per overflow (not one scan per evicted
/// entry): the watermark's hysteresis makes overflows rare, so the sweep
/// amortizes over many inserts instead of serializing every insert on an
/// O(entries) scan.
fn evict_to_watermark(state: &mut CacheState, watermark: usize) {
    let mut entries: Vec<(u64, ConditionKey, RowGroupId, usize)> = state
        .conditions
        .iter()
        .flat_map(|(condition, row_groups)| {
            row_groups.iter().map(move |(id, entry)| {
                (entry.last_used, condition.clone(), id.clone(), entry.bytes)
            })
        })
        .collect();
    entries.sort_unstable_by_key(|(last_used, _, _, _)| *last_used);
    for (_, condition, id, bytes) in entries {
        if state.bytes <= watermark {
            break;
        }
        let row_groups = state.conditions.get_mut(&condition).unwrap();
        row_groups.remove(&id);
        state.bytes -= bytes;
        if row_groups.is_empty() {
            state.conditions.remove(&condition);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, Scalar};
    use planner::condition_key::{ConditionKeyOutcome, build_condition_key};
    use planner::expression::{Compare, CompareType, Expression, Ref};
    use planner::types::Type;

    fn condition(column: &str, value: i32) -> ConditionKey {
        let expression = Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Int32,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(
                Arc::new(Int32Array::from(vec![value])) as ArrayRef,
            ))),
            compare_type: CompareType::Equal,
            return_type: Type::Int8,
        });
        match build_condition_key(&[&expression], &[column.to_string()]) {
            ConditionKeyOutcome::Cacheable(key) => key,
            ConditionKeyOutcome::NotCacheable => unreachable!("a plain compare is cacheable"),
        }
    }

    fn path(name: &str) -> Arc<str> {
        Arc::from(name)
    }

    /// The sparse positions of a cached entry, or `None` for absent/dense.
    fn sparse(cached: Option<CachedPositions>) -> Option<Arc<Vec<u32>>> {
        match cached {
            Some(CachedPositions::Sparse(positions)) => Some(positions),
            Some(CachedPositions::Dense) | None => None,
        }
    }

    #[test]
    fn insert_then_get_round_trips() {
        let cache = QueryConditionCache::new(true, 1 << 20);
        let key = condition("a", 7);

        cache.insert(path("t/f1.parquet"), 0, &key, Arc::new(vec![1, 5, 9]), 100);

        assert_eq!(
            sparse(cache.get(&path("t/f1.parquet"), 0, &key)).as_deref(),
            Some(&vec![1, 5, 9])
        );
        assert!(cache.get(&path("t/f1.parquet"), 1, &key).is_none());
        assert!(
            cache
                .get(&path("t/f1.parquet"), 0, &condition("a", 8))
                .is_none()
        );
    }

    #[test]
    fn duplicate_insert_keeps_the_first_entry() {
        let cache = QueryConditionCache::new(true, 1 << 20);
        let key = condition("a", 7);

        cache.insert(path("f"), 0, &key, Arc::new(vec![1]), 100);
        cache.insert(path("f"), 0, &key, Arc::new(vec![2]), 100);

        assert_eq!(
            sparse(cache.get(&path("f"), 0, &key)).as_deref(),
            Some(&vec![1])
        );
        assert_eq!(cache.stats().inserts, 1);
    }

    #[test]
    fn eviction_drops_the_least_recently_used_entries() {
        let one_entry = 4 + ENTRY_OVERHEAD_BYTES;
        let cache = QueryConditionCache::new(true, 3 * one_entry);
        let key = condition("a", 7);
        cache.insert(path("f"), 0, &key, Arc::new(vec![1]), 100);
        cache.insert(path("f"), 1, &key, Arc::new(vec![2]), 100);
        cache.insert(path("f"), 2, &key, Arc::new(vec![3]), 100);
        cache.get(&path("f"), 0, &key);

        cache.insert(path("f"), 3, &key, Arc::new(vec![4]), 100);

        // Overflow evicts down to the watermark, oldest first: the untouched
        // early entries go, the recently-used and the newest stay.
        assert!(cache.get(&path("f"), 1, &key).is_none());
        assert!(cache.get(&path("f"), 3, &key).is_some());
        assert!(cache.stats().bytes <= 3 * one_entry);
    }

    #[test]
    fn a_dense_result_is_marked_not_stored() {
        let cache = QueryConditionCache::new(true, 1 << 20);
        let key = condition("a", 7);

        cache.insert(path("f"), 0, &key, Arc::new((0..10).collect()), 100);

        assert!(matches!(
            cache.get(&path("f"), 0, &key),
            Some(CachedPositions::Dense)
        ));
        assert!(cache.stats().bytes <= ENTRY_OVERHEAD_BYTES);
    }

    #[test]
    fn invalidating_a_file_removes_only_its_entries() {
        let cache = QueryConditionCache::new(true, 1 << 20);
        let key = condition("a", 7);
        cache.insert(path("old"), 0, &key, Arc::new(vec![1]), 100);
        cache.insert(path("new"), 0, &key, Arc::new(vec![2]), 100);

        cache.invalidate_files(["old"]);

        assert!(cache.get(&path("old"), 0, &key).is_none());
        assert!(cache.get(&path("new"), 0, &key).is_some());
    }

    #[test]
    fn disabled_cache_stores_and_returns_nothing() {
        let cache = QueryConditionCache::new(false, 1 << 20);
        let key = condition("a", 7);

        cache.insert(path("f"), 0, &key, Arc::new(vec![1]), 100);

        assert!(cache.get(&path("f"), 0, &key).is_none());
        assert_eq!(cache.stats().entries, 0);
    }
}
