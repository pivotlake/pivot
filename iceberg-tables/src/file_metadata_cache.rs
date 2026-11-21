//! What this process holds of its tables' files between uses: one cache for
//! every table, within one budget of memory.
//!
//! Two things are held, and each costs reads and a decode to get again:
//!
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
//! - A part of a table's metadata, as the table's format decoded it and cut
//!   it up ([`MetadataPart`]): whatever the format reads off a metadata
//!   object that never changes, under a name of the format's own. A decode
//!   of a metadata object yields parts nobody asked for beside those that
//!   were: the first are stored as used now, the others as the first the
//!   cache lets go of, and only while it has room for them
//!   ([`FileMetadataCache::store_unasked_metadata_parts`]), so what a decode
//!   happens to yield never costs the cache a part that is in use.
//!
//! An item is found in two steps: by its table ([`TableKey`]), a value the
//! table's format computes, and then by its name, a footer's being its
//! file's location and a part's a name of the format's own. A table key
//! holds one item under a name, so a format that keeps footers and parts
//! under one key names no part as a file is located. A table holds one
//! footer per data file. Whatever computes an equal key
//! finds the same items, and nothing leaves the cache because a table
//! object went away: an item stays until the budget lets go of it, or
//! until whoever holds its table removes it
//! ([`FileMetadataCache::remove_tables`]). The cache knows a table only
//! while it holds an item of it or the table's build lock is held
//! ([`FileMetadataCache::lock_table`]).
//!
//! The budget is of the entries' estimated memory
//! ([`measure_footer_entry`], [`measure_part_entry`]), each with the
//! cache's own record of it. Footers and parts stand in one order of last
//! use, and past the budget the cache lets go of the entries used longest
//! ago, of whichever kind and whichever table, keeping the one just stored.
//! A scan marks the parts it selected through as used once it has taken its
//! files' footers, so the footers go before the parts. The budget bounds
//! what the cache holds, not what the process does: a scan holds the parts
//! it is selecting through and the row groups it took for as long as it
//! needs them, cached or not, and the cache only drops its own hold on an
//! entry.

use std::any::Any;
use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use parquet_engine::{RowGroupMetadata, estimate_footer_heap_bytes, measure_shared_allocation};

use crate::weighted_lru::WeightedLru;

/// A part of a table's metadata as the table's format decoded it and cut it
/// up, as the cache can hold it.
pub trait MetadataPart: Send + Sync + 'static {
    /// An estimate of the memory the part holds, itself included.
    fn estimate_heap_bytes(&self) -> u64;
}

/// What a taker of footers leaves of itself in the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FooterUse {
    /// Every footer found is used now, so it is the last the cache lets go
    /// of, and the taker stores those it reads.
    Kept,
    /// The cache is left as it was: every footer found stays where it
    /// stood in the order of last use, and the taker stores none it reads.
    Unrecorded,
}

/// The budget of the cache every table uses unless it is given another.
pub const DEFAULT_FILE_METADATA_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

/// What the cache's own record of one entry takes besides the bytes of the
/// entry's name: the entry's place in its table and in the order of last
/// use ([`WeightedLru::ENTRY_OVERHEAD_BYTES`]).
const ENTRY_RECORD_BYTES: usize =
    WeightedLru::<TableKey, HeldName, HeldValue>::ENTRY_OVERHEAD_BYTES;

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

/// What `part`, held under `name`, counts for in the budget: the part's
/// estimated memory and the cache's own record of the entry, the name and
/// the entry's hold on the part with it.
pub fn measure_part_entry<Part: MetadataPart>(name: &[u8], part: &Part) -> u64 {
    part.estimate_heap_bytes() + measure_entry_record(name.len()) + size_of::<Arc<Part>>() as u64
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

/// The footers and the metadata parts of every table of the process,
/// within a budget.
pub struct FileMetadataCache {
    state: Mutex<CacheState>,
    /// Signalled whenever a table's build lock is released.
    table_unlocked: Condvar,
}

/// The name an item of a table is held under, a file's location or a part's
/// own name: shared, and asked for by its bytes.
type HeldName = Arc<[u8]>;

/// What an entry holds, as whatever its table stored: a [`HeldFooter`], or
/// a shared [`MetadataPart`] of the type it was stored as.
type HeldValue = Box<dyn Any + Send + Sync>;

/// A file's row groups, in file order, none with a deleted row.
type HeldFooter = Vec<Arc<RowGroupMetadata>>;

struct CacheState {
    /// Every item held, by its table and then by its name, in one order of
    /// last use. A table's items are listed off this map alone.
    items: WeightedLru<TableKey, HeldName, HeldValue>,
    /// The tables whose build lock is held.
    locked_tables: HashSet<TableKey>,
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

    /// The row groups `table` holds for each of the files at `paths`, in
    /// the order given, shared with the cache. Whether finding a file's is
    /// a use of them is `footer_use`.
    pub fn find_footers<'p>(
        &self,
        table: &TableKey,
        paths: impl Iterator<Item = &'p str>,
        footer_use: FooterUse,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let mut state = self.state.lock().unwrap();
        paths
            .map(|path| {
                let held = match footer_use {
                    FooterUse::Kept => state.items.get(table, path.as_bytes()),
                    FooterUse::Unrecorded => state.items.peek(table, path.as_bytes()),
                };
                let row_groups = held?
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
    /// Every footer is checked and measured before the cache is locked, and
    /// what leaves for the budget is dropped once it is unlocked, so the
    /// lock is held for the cache's own bookkeeping alone.
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
        let measured: Vec<(HeldName, HeldValue, u64)> = footers
            .into_iter()
            .map(|(path, row_groups)| {
                assert!(
                    row_groups
                        .iter()
                        .all(|row_group| row_group.deleted_rows.is_none()),
                    "a footer is held without any version's deleted rows"
                );
                let heap_bytes = measure_footer_entry(&path, &row_groups);
                let name = HeldName::from(path);
                (name, Box::new(row_groups) as HeldValue, heap_bytes)
            })
            .collect();
        self.store_items(table, measured);
    }

    /// Hold in `table` each of `items`, measured, under its name, as used
    /// now. What leaves the cache for the budget is dropped once the cache
    /// is unlocked: a footer that goes closes its file.
    fn store_items(&self, table: &TableKey, items: Vec<(HeldName, HeldValue, u64)>) {
        let mut evicted = Vec::new();
        let mut state = self.state.lock().unwrap();
        for (name, value, heap_bytes) in items {
            let left = state.items.insert(table.clone(), name, value, heap_bytes);
            evicted.extend(left);
        }
        drop(state);
        drop(evicted);
    }

    /// The part `table` holds under each of `names`, in the order given,
    /// shared with the cache, each found counted as used now. Whatever
    /// stores a part under a name stores it as one `Part`.
    pub fn find_metadata_parts<'n, Part: MetadataPart>(
        &self,
        table: &TableKey,
        names: impl Iterator<Item = &'n [u8]>,
    ) -> Vec<Option<Arc<Part>>> {
        let mut state = self.state.lock().unwrap();
        names
            .map(|name| {
                let held = state.items.get(table, name)?;
                Some(Arc::clone(read_part::<Part>(held)))
            })
            .collect()
    }

    /// Hold in `table` each of `parts` under the name it comes with, as
    /// used now, replacing one already held under the name. After each,
    /// while the cache holds more than its budget, let go of the entry used
    /// longest ago, but never of the part just stored.
    pub fn store_metadata_parts<Part: MetadataPart>(
        &self,
        table: &TableKey,
        parts: Vec<(Arc<[u8]>, Arc<Part>)>,
    ) {
        self.store_items(table, measure_parts(parts));
    }

    /// Hold in `table` each of `parts`, which nobody asked for, under the
    /// name it comes with, as the first entries the cache lets go of, and
    /// only those the budget has room for: nothing leaves the cache for
    /// them, and a part already held under a name stays as it is. Whether
    /// every one of them is held afterwards.
    pub fn store_unasked_metadata_parts<Part: MetadataPart>(
        &self,
        table: &TableKey,
        parts: Vec<(Arc<[u8]>, Arc<Part>)>,
    ) -> bool {
        let measured = measure_parts(parts);
        let mut state = self.state.lock().unwrap();
        let mut is_every_part_held = true;
        for (name, part, heap_bytes) in measured {
            is_every_part_held &=
                state
                    .items
                    .insert_as_oldest(table.clone(), name, part, heap_bytes);
        }
        is_every_part_held
    }

    /// Count the parts `table` holds under `names` as used now. One the
    /// cache let go of stays let go of.
    pub fn mark_metadata_parts_used(&self, table: &TableKey, names: &[Arc<[u8]>]) {
        let mut state = self.state.lock().unwrap();
        for name in names {
            state.items.mark_used(table, &**name);
        }
    }

    /// How much of its budget the cache has left: what it can still hold
    /// without letting go of an entry.
    pub fn free_bytes(&self) -> u64 {
        self.state.lock().unwrap().items.free_bytes()
    }

    /// Let go of the footers `table` holds for the files at `paths`, and
    /// with each of the cache's hold on the open of its file, which is
    /// dropped once the cache is unlocked.
    pub fn remove_footers<'p>(&self, table: &TableKey, paths: impl Iterator<Item = &'p str>) {
        let mut state = self.state.lock().unwrap();
        let removed: Vec<_> = paths
            .filter_map(|path| state.items.remove(table, path.as_bytes()))
            .collect();
        drop(state);
        drop(removed);
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
    /// until what is handed back is dropped: what wants one part of the
    /// table at once then waits for one build of it and finds it held.
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

/// Each of `parts` under its name, as the cache holds it, with what it
/// counts for in the budget: measured before the cache is locked.
fn measure_parts<Part: MetadataPart>(
    parts: Vec<(Arc<[u8]>, Arc<Part>)>,
) -> Vec<(HeldName, HeldValue, u64)> {
    parts
        .into_iter()
        .map(|(name, part)| {
            let heap_bytes = measure_part_entry(&name, &*part);
            (name, Box::new(part) as HeldValue, heap_bytes)
        })
        .collect()
}

/// The part an entry holds, of the type it was stored as.
fn read_part<Part: MetadataPart>(held: &HeldValue) -> &Arc<Part> {
    held.downcast_ref()
        .expect("a part is found as the type it was stored as")
}
