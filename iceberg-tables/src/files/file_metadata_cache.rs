//! What this process holds of its tables' files between uses: one cache for
//! every table, within one budget of memory.
//!
//! Two things are held, and both cost reads and a decode to get again:
//!
//! - A partition's index ([`FileIndex`]), built from manifests. The cache
//!   keeps the index a version built for a partition so the next scan,
//!   commit or maintenance pass over that partition finds it, and the next
//!   version of the table builds its own on it by the manifests that
//!   changed.
//! - A data file's footer: the file's row groups ([`RowGroupMetadata`]) as
//!   the footer read produced them, each reading through the open of the
//!   file that read made, with no row deleted. A data file never changes,
//!   so its row groups serve every version that lists the file, shared as
//!   they are: a scan's plan takes them with nothing opened, read or
//!   copied. Row-level deletes belong to a version, so a plan records them
//!   on copies of its own and the row groups held here never carry any.
//!   A local file held here is held open, so the footer of a file the
//!   table deletes is dropped with it ([`CacheScope::remove_footers`]).
//!
//! A table's versions share one [`CacheScope`] in the cache. A scope holds
//! one index per partition, the one the newest version that asked built,
//! and one footer per data file, by the file's location. The cache drops
//! what a scope holds when the last version of the table is dropped.
//!
//! The budget is of the entries' estimated memory
//! ([`FileIndex::heap_bytes`], [`measure_footer_entry`]). Indexes and
//! footers stand in one order of last use, and past the budget the cache
//! lets go of the entries used longest ago, of whichever kind and whichever
//! table, keeping the one just stored. A scan marks the indexes it selected
//! from as used once it has taken its files' footers, so a partition's
//! footers go before its index. The budget bounds what the cache holds, not
//! what the process does: a plan holds the indexes it is selecting from and
//! the row groups it took for as long as it needs them, cached or not, and
//! the cache only drops its own hold on an entry.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use parquet_engine::{RowGroupMetadata, estimate_footer_heap_bytes};

use crate::files::file_index::FileIndex;

/// The budget of the cache every table uses unless it is given another.
pub(crate) const DEFAULT_FILE_METADATA_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

/// What the cache's own maps spend on one footer besides its path.
const FOOTER_ENTRY_BYTES: u64 = 128;

/// What the footer of the file at `path`, read as `row_groups`, counts for
/// in the budget: the row groups' estimated memory, the path, shared with
/// its two reference counts, and the cache's own record of the entry.
pub(crate) fn measure_footer_entry(path: &str, row_groups: &[Arc<RowGroupMetadata>]) -> u64 {
    let path_bytes = (path.len() + size_of::<[usize; 2]>()).next_multiple_of(16) as u64;
    estimate_footer_heap_bytes(row_groups) + path_bytes + FOOTER_ENTRY_BYTES
}

/// Where a table version stands among the versions of its table: a later
/// version's is not below an earlier one's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VersionOrder {
    pub last_sequence_number: i64,
    pub last_updated_ms: i64,
}

/// The partition indexes and footers of every table of the process,
/// within a budget.
pub(crate) struct FileMetadataCache {
    capacity_bytes: u64,
    state: Mutex<CacheState>,
    next_scope_id: AtomicU64,
}

#[derive(Default)]
struct CacheState {
    /// What each scope holds.
    scopes: HashMap<u64, ScopeEntries>,
    /// The scope and key of every entry held, by when it was last used: the
    /// first is the one used longest ago.
    by_last_use: BTreeMap<u64, (u64, EntryKey)>,
    /// What the last use of an entry was numbered.
    last_use: u64,
    /// The sum of the held entries' estimated memory.
    held_bytes: u64,
}

/// What one scope holds: its indexes by partition key, and its footers by
/// file location.
#[derive(Default)]
struct ScopeEntries {
    indexes: HashMap<Arc<[u8]>, CachedIndex>,
    footers: HashMap<Arc<str>, CachedFooter>,
}

/// Which entry of a scope: a partition's index, or a file's footer.
enum EntryKey {
    Index(Arc<[u8]>),
    Footer(Arc<str>),
}

/// An [`EntryKey`] to look an entry up by.
#[derive(Clone, Copy)]
enum EntryName<'a> {
    Index(&'a [u8]),
    Footer(&'a str),
}

/// An entry's place in the budget and in the order of last use.
struct Placement {
    /// The entry's estimated memory, as it was counted into the total.
    heap_bytes: u64,
    last_use: u64,
}

struct CachedIndex {
    index: Arc<FileIndex>,
    /// The version that built the index.
    built_at: VersionOrder,
    placement: Placement,
}

struct CachedFooter {
    /// The file's row groups, in file order, none with a deleted row.
    row_groups: Vec<Arc<RowGroupMetadata>>,
    placement: Placement,
}

impl ScopeEntries {
    fn find_placement(&mut self, name: EntryName<'_>) -> Option<&mut Placement> {
        match name {
            EntryName::Index(partition) => self
                .indexes
                .get_mut(partition)
                .map(|cached| &mut cached.placement),
            EntryName::Footer(path) => self
                .footers
                .get_mut(path)
                .map(|cached| &mut cached.placement),
        }
    }

    fn remove(&mut self, name: EntryName<'_>) -> Option<Placement> {
        match name {
            EntryName::Index(partition) => self
                .indexes
                .remove(partition)
                .map(|cached| cached.placement),
            EntryName::Footer(path) => self.footers.remove(path).map(|cached| cached.placement),
        }
    }
}

impl EntryKey {
    fn name(&self) -> EntryName<'_> {
        match self {
            EntryKey::Index(partition) => EntryName::Index(partition),
            EntryKey::Footer(path) => EntryName::Footer(path),
        }
    }
}

impl CacheState {
    /// Number a use of an entry.
    fn next_use(&mut self) -> u64 {
        self.last_use += 1;
        self.last_use
    }

    /// Count the entry `name` of the scope `scope_id` as used now, if it is
    /// held.
    fn mark_used(&mut self, scope_id: u64, name: EntryName<'_>) {
        let used = self.next_use();
        let Some(placement) = self
            .scopes
            .get_mut(&scope_id)
            .and_then(|entries| entries.find_placement(name))
        else {
            return;
        };
        let earlier_use = std::mem::replace(&mut placement.last_use, used);
        let place = self
            .by_last_use
            .remove(&earlier_use)
            .expect("every entry held has a place among the uses");
        self.by_last_use.insert(used, place);
    }

    /// Let go of the entry `name` of the scope `scope_id`, if it is held.
    fn remove(&mut self, scope_id: u64, name: EntryName<'_>) {
        let removed = self
            .scopes
            .get_mut(&scope_id)
            .and_then(|entries| entries.remove(name));
        if let Some(placement) = removed {
            self.by_last_use.remove(&placement.last_use);
            self.held_bytes -= placement.heap_bytes;
        }
    }

    /// Make room in the budget and in the order of last use for an entry
    /// of `heap_bytes` about to be held under `key` by the scope
    /// `scope_id`, used now.
    fn place(&mut self, scope_id: u64, key: EntryKey, heap_bytes: u64) -> Placement {
        let used = self.next_use();
        self.held_bytes += heap_bytes;
        self.by_last_use.insert(used, (scope_id, key));
        Placement {
            heap_bytes,
            last_use: used,
        }
    }

    /// While more than `capacity_bytes` are held, let go of the entry used
    /// longest ago, but never of the one used last.
    fn evict_past(&mut self, capacity_bytes: u64) {
        while self.held_bytes > capacity_bytes && self.by_last_use.len() > 1 {
            let (_, (scope_id, key)) = self
                .by_last_use
                .pop_first()
                .expect("more than one entry is held");
            let placement = self
                .scopes
                .get_mut(&scope_id)
                .and_then(|entries| entries.remove(key.name()))
                .expect("an entry with a place among the uses is held");
            self.held_bytes -= placement.heap_bytes;
        }
    }
}

impl FileMetadataCache {
    /// A cache that holds entries of up to `capacity_bytes` of estimated
    /// memory between them.
    pub(crate) fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            capacity_bytes,
            state: Mutex::default(),
            next_scope_id: AtomicU64::new(0),
        })
    }

    /// The cache every table uses unless it is given another, of
    /// [`DEFAULT_FILE_METADATA_CACHE_BYTES`].
    pub(crate) fn process_wide() -> Arc<Self> {
        static CACHE: OnceLock<Arc<FileMetadataCache>> = OnceLock::new();
        Arc::clone(CACHE.get_or_init(|| Self::new(DEFAULT_FILE_METADATA_CACHE_BYTES)))
    }
}

/// One table's share of a cache: the indexes of its partitions, each as
/// the newest version that asked for the partition built it, and the
/// footers of its data files.
pub(crate) struct CacheScope {
    id: u64,
    cache: Arc<FileMetadataCache>,
}

impl CacheScope {
    pub(crate) fn new(cache: Arc<FileMetadataCache>) -> Self {
        let id = cache.next_scope_id.fetch_add(1, Ordering::Relaxed);
        Self { id, cache }
    }

    pub(crate) fn cache(&self) -> &Arc<FileMetadataCache> {
        &self.cache
    }

    /// The index held for the partition `key`, as whichever version built
    /// it. Finding it is a use of it.
    pub(crate) fn find_index(&self, key: &[u8]) -> Option<Arc<FileIndex>> {
        let mut state = self.cache.state.lock().unwrap();
        let index = Arc::clone(&state.scopes.get(&self.id)?.indexes.get(key)?.index);
        state.mark_used(self.id, EntryName::Index(key));
        Some(index)
    }

    /// Hold `index` for the partition `key`, as the version at `built_at`
    /// built it, unless a later version's is held. Then, while the cache
    /// holds more than its budget, let go of the entry used longest ago,
    /// but never of this one.
    pub(crate) fn store_index(
        &self,
        key: &Arc<[u8]>,
        index: &Arc<FileIndex>,
        built_at: VersionOrder,
    ) {
        let mut state = self.cache.state.lock().unwrap();
        let is_superseded = state
            .scopes
            .get(&self.id)
            .and_then(|entries| entries.indexes.get(key))
            .is_some_and(|held| held.built_at > built_at);
        if is_superseded {
            return;
        }
        state.remove(self.id, EntryName::Index(key));
        let placement = state.place(
            self.id,
            EntryKey::Index(Arc::clone(key)),
            index.heap_bytes(),
        );
        state.scopes.entry(self.id).or_default().indexes.insert(
            Arc::clone(key),
            CachedIndex {
                index: Arc::clone(index),
                built_at,
                placement,
            },
        );
        state.evict_past(self.cache.capacity_bytes);
    }

    /// Count the indexes held for the partitions `keys` as used now. One
    /// the cache let go of stays let go of.
    pub(crate) fn mark_indexes_used(&self, keys: &[Arc<[u8]>]) {
        let mut state = self.cache.state.lock().unwrap();
        for key in keys {
            state.mark_used(self.id, EntryName::Index(key));
        }
    }

    /// The keys of the partitions an index is held for.
    pub(crate) fn list_indexed_partitions(&self) -> Vec<Arc<[u8]>> {
        let state = self.cache.state.lock().unwrap();
        state
            .scopes
            .get(&self.id)
            .map(|entries| entries.indexes.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// The row groups held for each of the files at `paths`, in the order
    /// given, shared with the cache. Finding a file's is a use of them.
    pub(crate) fn find_footers<'p>(
        &self,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let mut state = self.cache.state.lock().unwrap();
        let held = Self::peek_footers_in(&state, self.id, paths);
        for (path, footer) in &held {
            if footer.is_some() {
                state.mark_used(self.id, EntryName::Footer(path));
            }
        }
        held.into_iter().map(|(_, footer)| footer).collect()
    }

    /// The row groups held for each of the files at `paths`, in the order
    /// given, shared with the cache, none of them counted as used.
    pub(crate) fn peek_footers<'p>(
        &self,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let state = self.cache.state.lock().unwrap();
        Self::peek_footers_in(&state, self.id, paths)
            .into_iter()
            .map(|(_, footer)| footer)
            .collect()
    }

    /// Each of `paths` with the row groups the scope `scope_id` holds for
    /// it in `state`.
    fn peek_footers_in<'p>(
        state: &CacheState,
        scope_id: u64,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<(&'p str, Option<Vec<Arc<RowGroupMetadata>>>)> {
        let footers = state.scopes.get(&scope_id).map(|entries| &entries.footers);
        paths
            .map(|path| {
                let held = footers
                    .and_then(|footers| footers.get(path))
                    .map(|cached| cached.row_groups.clone());
                (path, held)
            })
            .collect()
    }

    /// Hold each of `footers`, a file's row groups as its footer read
    /// produced them, with no row deleted, for the file at the path they
    /// come with, replacing those already held for it. After each, while
    /// the cache holds more than its budget, let go of the entry used
    /// longest ago, but never of the footer just stored.
    pub(crate) fn store_footers(&self, footers: Vec<(Arc<str>, Vec<Arc<RowGroupMetadata>>)>) {
        let mut state = self.cache.state.lock().unwrap();
        for (path, row_groups) in footers {
            assert!(
                row_groups
                    .iter()
                    .all(|row_group| row_group.deleted_rows.is_none()),
                "a footer is held without any version's deleted rows"
            );
            state.remove(self.id, EntryName::Footer(&path));
            let heap_bytes = measure_footer_entry(&path, &row_groups);
            let placement = state.place(self.id, EntryKey::Footer(Arc::clone(&path)), heap_bytes);
            state.scopes.entry(self.id).or_default().footers.insert(
                path,
                CachedFooter {
                    row_groups,
                    placement,
                },
            );
            state.evict_past(self.cache.capacity_bytes);
        }
    }

    /// Let go of the footers held for the files at `paths`, and with each
    /// of the cache's hold on the open of its file.
    pub(crate) fn remove_footers<'p>(&self, paths: impl Iterator<Item = &'p str>) {
        let mut state = self.cache.state.lock().unwrap();
        for path in paths {
            state.remove(self.id, EntryName::Footer(path));
        }
    }
}

impl Drop for CacheScope {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap();
        let Some(entries) = state.scopes.remove(&self.id) else {
            return;
        };
        let placements = entries
            .indexes
            .into_values()
            .map(|cached| cached.placement)
            .chain(entries.footers.into_values().map(|cached| cached.placement));
        for placement in placements {
            state.by_last_use.remove(&placement.last_use);
            state.held_bytes -= placement.heap_bytes;
        }
    }
}
