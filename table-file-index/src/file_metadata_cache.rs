//! What this process holds of its tables' files between uses: one cache for
//! every table, of whichever format, within one budget of memory.
//!
//! Two things are held, and both cost reads and a decode to get again:
//!
//! - A partition's index, built from the table's metadata. The cache keeps
//!   the index a version built for a partition so the next scan, commit or
//!   maintenance pass over that partition finds it, and the next version of
//!   the table builds its own on it by what changed. The cache does not
//!   read an index: a format hands it whatever it indexes a partition as
//!   ([`CacheableIndex`]).
//! - A data file's footer: the file's row groups ([`RowGroupMetadata`]) as
//!   the footer read produced them, each reading through the open of the
//!   file that read made, with no row deleted. A data file never changes,
//!   so its row groups serve every version that lists the file, shared as
//!   they are: a scan's selection takes them with nothing opened, read or
//!   copied. Row-level deletes belong to a version, so a scan records them
//!   on copies of its own and the row groups held here never carry any.
//!   A local file held here is held open, so the footer of a file the
//!   table deletes is dropped with it ([`TableCacheScope::remove_footers`]).
//!
//! A table's versions share one [`TableCacheScope`] in the cache. A scope
//! holds one index per partition, the one the newest version that asked
//! built, and one footer per data file, by the file's location. The cache
//! drops what a scope holds when the scope is dropped.
//!
//! The budget is of the entries' estimated memory
//! ([`measure_index_entry`], [`measure_footer_entry`]), each with the
//! cache's own record of it.
//! Indexes and footers stand in one order of last use, and past the budget
//! the cache lets go of the entries used longest ago, of whichever kind and
//! whichever table, keeping the one just stored. A scan marks the indexes it
//! selected from as used once it has taken its files' footers, so a
//! partition's footers go before its index. The budget bounds what the cache
//! holds, not what the process does: a scan holds the indexes it is
//! selecting from and the row groups it took for as long as it needs them,
//! cached or not, and the cache only drops its own hold on an entry.

use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use hashbrown::Equivalent;
use parquet_engine::{RowGroupMetadata, estimate_footer_heap_bytes, measure_shared_allocation};

use crate::file_index::{FileIndex, IndexedRecord};
use crate::weighted_lru::WeightedLru;

/// What a format indexes a partition as, as the cache can hold it.
pub trait CacheableIndex: Send + Sync + 'static {
    /// An estimate of the memory the index holds.
    fn estimate_heap_bytes(&self) -> u64;
}

impl<Record: IndexedRecord> CacheableIndex for FileIndex<Record> {
    fn estimate_heap_bytes(&self) -> u64 {
        FileIndex::estimate_heap_bytes(self)
    }
}

/// Whether a lookup counts as a use of the entries it finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupUse {
    /// Every entry found is used now, so it is the last the cache lets go
    /// of.
    Counted,
    /// Every entry found stays where it stood in the order of last use:
    /// for a caller that counts the use itself, or reads the entry for
    /// something that should not keep it held.
    Uncounted,
}

/// The budget of the cache every table uses unless it is given another.
pub const DEFAULT_FILE_METADATA_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

/// What the cache's own record of one entry takes besides the bytes of the
/// entry's name: the entry's place in the order of last use
/// ([`WeightedLru::ENTRY_OVERHEAD_BYTES`]) and its key among its scope's.
const ENTRY_RECORD_BYTES: usize =
    WeightedLru::<HeldKey, HeldValue>::ENTRY_OVERHEAD_BYTES + size_of::<HeldKey>();

/// What the cache's record of an entry named by `name_len` bytes takes: the
/// record itself and the name, one shared allocation that every key of the
/// entry points to.
fn measure_entry_record(name_len: usize) -> u64 {
    (ENTRY_RECORD_BYTES + measure_shared_allocation(name_len)) as u64
}

/// What the footer of the file at `path`, read as `row_groups`, counts for
/// in the budget: the row groups' estimated memory and the cache's own
/// record of the entry, the path with it.
pub fn measure_footer_entry(path: &str, row_groups: &[Arc<RowGroupMetadata>]) -> u64 {
    estimate_footer_heap_bytes(row_groups) + measure_entry_record(path.len())
}

/// What `index`, held for the partition `key`, counts for in the budget:
/// the index's estimated memory and the cache's own record of the entry,
/// the key with it.
pub fn measure_index_entry<Index: CacheableIndex>(key: &[u8], index: &Index) -> u64 {
    index.estimate_heap_bytes() + measure_entry_record(key.len())
}

/// Where a table version stands among the versions of its table: a later
/// version's is not below an earlier one's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VersionOrder {
    /// The number the table's format gives the version's commit.
    pub sequence: i64,
    /// When the version was committed, where two versions may share a
    /// number.
    pub updated_ms: i64,
}

/// The partition indexes and footers of every table of the process,
/// within a budget.
pub struct FileMetadataCache {
    state: Mutex<CacheState>,
    next_scope_id: AtomicU64,
}

/// What an entry holds: a partition's index, or a data file's footer.
/// Indexes order before footers, so a scope's indexes stand first among its
/// keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntryKind {
    PartitionIndex,
    Footer,
}

/// Which entry of the cache: of which scope, of which kind, and by what
/// name, a partition's key or a file's location. `Name` is the bytes of the
/// name as the key keeps them: shared for a key the cache holds
/// ([`HeldKey`]), borrowed for one it is asked by.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EntryKey<Name> {
    scope_id: u64,
    kind: EntryKind,
    name: Name,
}

type HeldKey = EntryKey<Arc<[u8]>>;

/// A key with a borrowed name asks for the entry held under the same scope,
/// kind and name. Both hash alike: a shared slice hashes as its bytes do.
impl Equivalent<HeldKey> for EntryKey<&[u8]> {
    fn equivalent(&self, held: &HeldKey) -> bool {
        self.scope_id == held.scope_id && self.kind == held.kind && *self.name == *held.name
    }
}

/// What an entry holds, as whatever its scope stored: a [`HeldIndex`] of
/// the scope's index type, or a [`HeldFooter`].
type HeldValue = Box<dyn Any + Send + Sync>;

/// One partition's index as a scope holds it.
struct HeldIndex<Index> {
    index: Arc<Index>,
    /// The version that built the index.
    built_at: VersionOrder,
}

/// A file's row groups, in file order, none with a deleted row.
type HeldFooter = Vec<Arc<RowGroupMetadata>>;

struct CacheState {
    /// Every entry held, of every scope, in one order of last use.
    entries: WeightedLru<HeldKey, HeldValue>,
    /// The keys of the entries each scope holds, so a scope lists its
    /// partitions and lets go of everything it holds without the other
    /// scopes' entries being read.
    keys_by_scope: HashMap<u64, BTreeSet<HeldKey>>,
}

impl CacheState {
    /// The value held under `key`; whether finding it is a use of it is
    /// `lookup_use`.
    fn find(&mut self, key: &EntryKey<&[u8]>, lookup_use: LookupUse) -> Option<&HeldValue> {
        match lookup_use {
            LookupUse::Counted => self.entries.get(key),
            LookupUse::Uncounted => self.entries.peek(key),
        }
    }

    /// Hold `value` under `key` for `heap_bytes` of the budget, used now,
    /// in the place of any value held under it. Then, while the cache holds
    /// more than its budget, let go of the entry used longest ago, but
    /// never of this one.
    fn insert(&mut self, key: HeldKey, value: HeldValue, heap_bytes: u64) {
        self.keys_by_scope
            .entry(key.scope_id)
            .or_default()
            .insert(key.clone());
        let evicted = self.entries.insert(key, value, heap_bytes);
        self.forget_evicted(evicted);
    }

    /// Let go of the entry held under `key`, if one is.
    fn remove(&mut self, key: &EntryKey<&[u8]>) {
        if let Some((held_key, _)) = self.entries.remove(key) {
            self.forget_key(&held_key);
        }
    }

    /// Drop the entries the cache let go of, and their keys from their
    /// scopes'.
    fn forget_evicted(&mut self, evicted: Vec<(HeldKey, HeldValue)>) {
        for (key, _) in evicted {
            self.forget_key(&key);
        }
    }

    fn forget_key(&mut self, key: &HeldKey) {
        self.keys_by_scope
            .get_mut(&key.scope_id)
            .expect("the scope of an entry held keeps its key")
            .remove(key);
    }
}

impl FileMetadataCache {
    /// A cache that holds entries of up to `capacity_bytes` of estimated
    /// memory between them.
    pub fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CacheState {
                entries: WeightedLru::new(capacity_bytes),
                keys_by_scope: HashMap::new(),
            }),
            next_scope_id: AtomicU64::new(0),
        })
    }

    /// The cache every table uses unless it is given another, of
    /// [`DEFAULT_FILE_METADATA_CACHE_BYTES`].
    pub fn process_wide() -> Arc<Self> {
        static CACHE: OnceLock<Arc<FileMetadataCache>> = OnceLock::new();
        Arc::clone(CACHE.get_or_init(|| Self::new(DEFAULT_FILE_METADATA_CACHE_BYTES)))
    }
}

/// One table's share of a cache: the indexes of its partitions, each an
/// `Index` as the newest version that asked for the partition built it,
/// and the footers of its data files.
pub struct TableCacheScope<Index> {
    id: u64,
    cache: Arc<FileMetadataCache>,
    index: PhantomData<fn() -> Index>,
}

impl<Index: CacheableIndex> TableCacheScope<Index> {
    pub fn new(cache: Arc<FileMetadataCache>) -> Self {
        let id = cache.next_scope_id.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            cache,
            index: PhantomData,
        }
    }

    pub fn cache(&self) -> &Arc<FileMetadataCache> {
        &self.cache
    }

    /// The key of this scope's entry of `kind` named `name`.
    fn name_entry<Name>(&self, kind: EntryKind, name: Name) -> EntryKey<Name> {
        EntryKey {
            scope_id: self.id,
            kind,
            name,
        }
    }

    /// The index held for each of the partitions `keys`, in the order
    /// given, with the version that built it. Whether finding one is a use
    /// of it is `lookup_use`: a caller that counts the indexes itself once
    /// it has taken what it reads through them
    /// ([`mark_indexes_used`](Self::mark_indexes_used)) looks them up
    /// [`LookupUse::Uncounted`].
    pub fn find_indexes<'k>(
        &self,
        keys: impl Iterator<Item = &'k [u8]>,
        lookup_use: LookupUse,
    ) -> Vec<Option<(Arc<Index>, VersionOrder)>> {
        let mut state = self.cache.state.lock().unwrap();
        keys.map(|key| {
            let key = self.name_entry(EntryKind::PartitionIndex, key);
            let held = read_index::<Index>(state.find(&key, lookup_use)?);
            Some((Arc::clone(&held.index), held.built_at))
        })
        .collect()
    }

    /// Hold `index` for the partition `key`, as the version at `built_at`
    /// built it, unless a later version's is held. Then, while the cache
    /// holds more than its budget, let go of the entry used longest ago,
    /// but never of this one.
    pub fn store_index(&self, key: &Arc<[u8]>, index: &Arc<Index>, built_at: VersionOrder) {
        let heap_bytes = measure_index_entry(key, &**index);
        let mut state = self.cache.state.lock().unwrap();
        let held = state
            .entries
            .peek(&self.name_entry(EntryKind::PartitionIndex, &**key));
        if held.is_some_and(|held| read_index::<Index>(held).built_at > built_at) {
            return;
        }
        state.insert(
            self.name_entry(EntryKind::PartitionIndex, Arc::clone(key)),
            hold_index(index, built_at),
            heap_bytes,
        );
    }

    /// Count the indexes held for the partitions `keys` as used now. One
    /// the cache let go of stays let go of.
    pub fn mark_indexes_used(&self, keys: &[Arc<[u8]>]) {
        let mut state = self.cache.state.lock().unwrap();
        for key in keys {
            state
                .entries
                .mark_used(&self.name_entry(EntryKind::PartitionIndex, &**key));
        }
    }

    /// The keys of the partitions an index is held for.
    pub fn list_indexed_partitions(&self) -> Vec<Arc<[u8]>> {
        let state = self.cache.state.lock().unwrap();
        state
            .keys_by_scope
            .get(&self.id)
            .into_iter()
            .flatten()
            .take_while(|key| key.kind == EntryKind::PartitionIndex)
            .map(|key| Arc::clone(&key.name))
            .collect()
    }

    /// The row groups held for each of the files at `paths`, in the order
    /// given, shared with the cache. Whether finding a file's is a use of
    /// them is `lookup_use`.
    pub fn find_footers<'p>(
        &self,
        paths: impl Iterator<Item = &'p str>,
        lookup_use: LookupUse,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let mut state = self.cache.state.lock().unwrap();
        paths
            .map(|path| {
                let key = self.name_entry(EntryKind::Footer, path.as_bytes());
                let row_groups = state
                    .find(&key, lookup_use)?
                    .downcast_ref::<HeldFooter>()
                    .expect("a footer entry holds a file's row groups");
                Some(row_groups.clone())
            })
            .collect()
    }

    /// Hold each of `footers`, a file's row groups as its footer read
    /// produced them, with no row deleted, for the file at the path they
    /// come with, replacing those already held for it. After each, while
    /// the cache holds more than its budget, let go of the entry used
    /// longest ago, but never of the footer just stored.
    ///
    /// Every footer is checked and measured before the cache is locked, so
    /// the lock is held for the cache's own bookkeeping alone.
    ///
    /// # Panics
    ///
    /// When a row group of any of `footers` carries deleted rows; none of
    /// them is stored then.
    pub fn store_footers(&self, footers: Vec<(Arc<str>, Vec<Arc<RowGroupMetadata>>)>) {
        let measured: Vec<(HeldKey, HeldValue, u64)> = footers
            .into_iter()
            .map(|(path, row_groups)| {
                assert!(
                    row_groups
                        .iter()
                        .all(|row_group| row_group.deleted_rows.is_none()),
                    "a footer is held without any version's deleted rows"
                );
                let heap_bytes = measure_footer_entry(&path, &row_groups);
                let key = self.name_entry(EntryKind::Footer, Arc::<[u8]>::from(path));
                (key, Box::new(row_groups) as HeldValue, heap_bytes)
            })
            .collect();
        let mut state = self.cache.state.lock().unwrap();
        for (key, row_groups, heap_bytes) in measured {
            state.insert(key, row_groups, heap_bytes);
        }
    }

    /// Let go of the footers held for the files at `paths`, and with each
    /// of the cache's hold on the open of its file.
    pub fn remove_footers<'p>(&self, paths: impl Iterator<Item = &'p str>) {
        let mut state = self.cache.state.lock().unwrap();
        for path in paths {
            state.remove(&self.name_entry(EntryKind::Footer, path.as_bytes()));
        }
    }
}

/// `index`, as the version at `built_at` built it, as the cache holds it.
fn hold_index<Index: CacheableIndex>(index: &Arc<Index>, built_at: VersionOrder) -> HeldValue {
    Box::new(HeldIndex {
        index: Arc::clone(index),
        built_at,
    })
}

/// The index a partition's entry holds, of the type its scope stores.
fn read_index<Index: CacheableIndex>(held: &HeldValue) -> &HeldIndex<Index> {
    held.downcast_ref()
        .expect("a scope holds indexes of its one type")
}

impl<Index> Drop for TableCacheScope<Index> {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap();
        let Some(keys) = state.keys_by_scope.remove(&self.id) else {
            return;
        };
        for key in &keys {
            state.entries.remove(key);
        }
    }
}
