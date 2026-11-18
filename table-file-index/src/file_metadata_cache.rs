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
//!   table deletes is dropped with it
//!   ([`FileMetadataCache::remove_footers`]).
//!
//! An item is found in two steps: by its table ([`TableKey`]), a value the
//! table's format computes, and then by what it is, a partition's index
//! under the partition's key or a footer under its file's location. A table
//! holds one index per partition, the one the newest version that asked
//! built, and one footer per data file. Whatever computes an equal key
//! finds the same items, and nothing leaves the cache because a table
//! object went away: an item stays until the budget lets go of it, or
//! until whoever holds its table removes it
//! ([`FileMetadataCache::remove_tables`]). The cache knows a table only
//! while it holds an item of it or the table's build lock is held
//! ([`FileMetadataCache::lock_table`]).
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
use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

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
pub const DEFAULT_FILE_METADATA_CACHE_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// What the cache's own record of one entry takes besides the bytes of the
/// entry's name: the entry's place in its table and in the order of last
/// use ([`WeightedLru::ENTRY_OVERHEAD_BYTES`]).
const ENTRY_RECORD_BYTES: usize = WeightedLru::<TableKey, HeldKey, HeldValue>::ENTRY_OVERHEAD_BYTES;

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

/// Which table an item of the cache belongs to, as the table's format
/// computes it: two table objects find each other's items exactly when
/// their keys are equal. Both parts are compared whole, so two keys that
/// differ never share an item, and a key is cloned by sharing its parts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TableKey {
    /// What names the table to whoever removes it from the cache
    /// ([`FileMetadataCache::remove_tables`]): where it is, or a name of
    /// the format's own.
    pub table: Arc<str>,
    /// Whatever else an item is built under, as the format writes it down:
    /// items built under another are not this key's.
    pub reading: Arc<str>,
}

/// The partition indexes and footers of every table of the process,
/// within a budget.
pub struct FileMetadataCache {
    state: Mutex<CacheState>,
    /// Signalled whenever a table's build lock is released.
    table_unlocked: Condvar,
}

/// What an entry holds: a partition's index, or a data file's footer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EntryKind {
    PartitionIndex,
    Footer,
}

/// Which item of a table: of which kind, and by what name, a partition's
/// key or a file's location. `Name` is the bytes of the name as the key
/// keeps them: shared for a key the cache holds ([`HeldKey`]), borrowed for
/// one it is asked by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ItemKey<Name> {
    kind: EntryKind,
    name: Name,
}

type HeldKey = ItemKey<Arc<[u8]>>;

/// A key with a borrowed name asks for the item held under the same kind
/// and name. Both hash alike: a shared slice hashes as its bytes do.
impl Equivalent<HeldKey> for ItemKey<&[u8]> {
    fn equivalent(&self, held: &HeldKey) -> bool {
        self.kind == held.kind && *self.name == *held.name
    }
}

/// What an entry holds, as whatever its table stored: a [`HeldIndex`] of
/// the table's index type, or a [`HeldFooter`].
type HeldValue = Box<dyn Any + Send + Sync>;

/// One partition's index as a table holds it.
struct HeldIndex<Index> {
    index: Arc<Index>,
    /// The version that built the index.
    built_at: VersionOrder,
}

/// A file's row groups, in file order, none with a deleted row.
type HeldFooter = Vec<Arc<RowGroupMetadata>>;

struct CacheState {
    /// Every item held, by its table and then by its kind and name, in one
    /// order of last use. A table's items are listed off this map alone.
    items: WeightedLru<TableKey, HeldKey, HeldValue>,
    /// The tables whose build lock is held.
    locked_tables: HashSet<TableKey>,
}

impl CacheState {
    /// The value `table` holds under `key`; whether finding it is a use of
    /// it is `lookup_use`.
    fn find(
        &mut self,
        table: &TableKey,
        key: &ItemKey<&[u8]>,
        lookup_use: LookupUse,
    ) -> Option<&HeldValue> {
        match lookup_use {
            LookupUse::Counted => self.items.get(table, key),
            LookupUse::Uncounted => self.items.peek(table, key),
        }
    }
}

impl FileMetadataCache {
    /// A cache that holds entries of up to `capacity_bytes` of estimated
    /// memory between them.
    pub fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CacheState {
                items: WeightedLru::new(capacity_bytes),
                locked_tables: HashSet::new(),
            }),
            table_unlocked: Condvar::new(),
        })
    }

    /// The cache every table uses unless it is given another, of
    /// [`DEFAULT_FILE_METADATA_CACHE_BYTES`].
    pub fn process_wide() -> Arc<Self> {
        static CACHE: OnceLock<Arc<FileMetadataCache>> = OnceLock::new();
        Arc::clone(CACHE.get_or_init(|| Self::new(DEFAULT_FILE_METADATA_CACHE_BYTES)))
    }

    /// The index `table` holds for each of the partitions `keys`, in the
    /// order given, with the version that built it. Whether finding one is
    /// a use of it is `lookup_use`: a caller that counts the indexes itself
    /// once it has taken what it reads through them
    /// ([`mark_indexes_used`](Self::mark_indexes_used)) looks them up
    /// [`LookupUse::Uncounted`]. Everything that stores indexes under one
    /// table key stores them as one `Index`.
    pub fn find_indexes<'k, Index: CacheableIndex>(
        &self,
        table: &TableKey,
        keys: impl Iterator<Item = &'k [u8]>,
        lookup_use: LookupUse,
    ) -> Vec<Option<(Arc<Index>, VersionOrder)>> {
        let mut state = self.state.lock().unwrap();
        keys.map(|key| {
            let key = name_item(EntryKind::PartitionIndex, key);
            let held = read_index::<Index>(state.find(table, &key, lookup_use)?);
            Some((Arc::clone(&held.index), held.built_at))
        })
        .collect()
    }

    /// Hold `index` in `table` for the partition `key`, as the version at
    /// `built_at` built it, unless a later version's is held. Then, while
    /// the cache holds more than its budget, let go of the entry used
    /// longest ago, but never of this one.
    pub fn store_index<Index: CacheableIndex>(
        &self,
        table: &TableKey,
        key: &Arc<[u8]>,
        index: &Arc<Index>,
        built_at: VersionOrder,
    ) {
        let heap_bytes = measure_index_entry(key, &**index);
        let mut state = self.state.lock().unwrap();
        let held = state
            .items
            .peek(table, &name_item(EntryKind::PartitionIndex, &**key));
        if held.is_some_and(|held| read_index::<Index>(held).built_at > built_at) {
            return;
        }
        state.items.insert(
            table.clone(),
            name_item(EntryKind::PartitionIndex, Arc::clone(key)),
            hold_index(index, built_at),
            heap_bytes,
        );
    }

    /// Count the indexes `table` holds for the partitions `keys` as used
    /// now. One the cache let go of stays let go of.
    pub fn mark_indexes_used(&self, table: &TableKey, keys: &[Arc<[u8]>]) {
        let mut state = self.state.lock().unwrap();
        for key in keys {
            state
                .items
                .mark_used(table, &name_item(EntryKind::PartitionIndex, &**key));
        }
    }

    /// The keys of the partitions `table` holds an index for, in key order.
    pub fn list_indexed_partitions(&self, table: &TableKey) -> Vec<Arc<[u8]>> {
        let state = self.state.lock().unwrap();
        let mut partitions: Vec<Arc<[u8]>> = state
            .items
            .list_keys(table)
            .filter(|key| key.kind == EntryKind::PartitionIndex)
            .map(|key| Arc::clone(&key.name))
            .collect();
        partitions.sort_unstable();
        partitions
    }

    /// The row groups `table` holds for each of the files at `paths`, in
    /// the order given, shared with the cache. Whether finding a file's is
    /// a use of them is `lookup_use`.
    pub fn find_footers<'p>(
        &self,
        table: &TableKey,
        paths: impl Iterator<Item = &'p str>,
        lookup_use: LookupUse,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let mut state = self.state.lock().unwrap();
        paths
            .map(|path| {
                let key = name_item(EntryKind::Footer, path.as_bytes());
                let row_groups = state
                    .find(table, &key, lookup_use)?
                    .downcast_ref::<HeldFooter>()
                    .expect("a footer entry holds a file's row groups");
                Some(row_groups.clone())
            })
            .collect()
    }

    /// Hold in `table` each of `footers`, a file's row groups as its footer
    /// read produced them, with no row deleted, for the file at the path
    /// they come with, replacing those already held for it. After each,
    /// while the cache holds more than its budget, let go of the entry used
    /// longest ago, but never of the footer just stored.
    ///
    /// Every footer is checked and measured before the cache is locked, so
    /// the lock is held for the cache's own bookkeeping alone.
    ///
    /// # Panics
    ///
    /// When a row group of any of `footers` carries deleted rows; none of
    /// them is stored then.
    pub fn store_footers(
        &self,
        table: &TableKey,
        footers: Vec<(Arc<str>, Vec<Arc<RowGroupMetadata>>)>,
    ) {
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
                let key = name_item(EntryKind::Footer, Arc::<[u8]>::from(path));
                (key, Box::new(row_groups) as HeldValue, heap_bytes)
            })
            .collect();
        let mut state = self.state.lock().unwrap();
        for (key, row_groups, heap_bytes) in measured {
            state
                .items
                .insert(table.clone(), key, row_groups, heap_bytes);
        }
    }

    /// Let go of the footers `table` holds for the files at `paths`, and
    /// with each of the cache's hold on the open of its file.
    pub fn remove_footers<'p>(&self, table: &TableKey, paths: impl Iterator<Item = &'p str>) {
        let mut state = self.state.lock().unwrap();
        for path in paths {
            state
                .items
                .remove(table, &name_item(EntryKind::Footer, path.as_bytes()));
        }
    }

    /// Let go of every item of every table named `table`, whatever else
    /// its key says: what whoever deletes a table's files, or is the last
    /// to use a table of its own, calls. A footer held holds its file
    /// open, and a deleted local file keeps its blocks for as long as it
    /// is open.
    pub fn remove_tables(&self, table: &str) {
        let mut state = self.state.lock().unwrap();
        let removed: Vec<TableKey> = state
            .items
            .list_groups()
            .filter(|key| &*key.table == table)
            .cloned()
            .collect();
        for key in &removed {
            state.items.remove_group(key);
        }
    }

    /// Wait until nothing else holds the build lock of `table`, and hold it
    /// until what is handed back is dropped: what wants one partition of
    /// the table at once then waits for one build of its index and finds
    /// it held.
    pub fn lock_table(self: &Arc<Self>, table: &TableKey) -> TableLock {
        let mut state = self.state.lock().unwrap();
        while state.locked_tables.contains(table) {
            state = self.table_unlocked.wait(state).unwrap();
        }
        state.locked_tables.insert(table.clone());
        TableLock {
            cache: Arc::clone(self),
            table: table.clone(),
        }
    }
}

/// A table's build lock, released when dropped.
pub struct TableLock {
    cache: Arc<FileMetadataCache>,
    table: TableKey,
}

impl Drop for TableLock {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap();
        state.locked_tables.remove(&self.table);
        self.cache.table_unlocked.notify_all();
    }
}

/// The key of a table's item of `kind` named `name`.
fn name_item<Name>(kind: EntryKind, name: Name) -> ItemKey<Name> {
    ItemKey { kind, name }
}

/// `index`, as the version at `built_at` built it, as the cache holds it.
fn hold_index<Index: CacheableIndex>(index: &Arc<Index>, built_at: VersionOrder) -> HeldValue {
    Box::new(HeldIndex {
        index: Arc::clone(index),
        built_at,
    })
}

/// The index a partition's entry holds, of the type its table stores.
fn read_index<Index: CacheableIndex>(held: &HeldValue) -> &HeldIndex<Index> {
    held.downcast_ref()
        .expect("a table holds indexes of its one type")
}
