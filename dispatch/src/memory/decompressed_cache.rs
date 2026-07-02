//! A cache of decompressed blocks, sharing the ring's CLOCK
//! ([`Clock`](super::clock::Clock)) with the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache), and modelled on
//! it: a cached block sits in its ring slot(s) at `used == 0`, owned via the clock
//! (no held pin). A [`get`](DecompressedCache::get) reconstructs a zero-copy view
//! with `try_read`; eviction acquires the slots with `try_write` first and only
//! then removes the block - never the drop-into-pool-then-grab dance.
//!
//! ## Maps
//!
//! `files` maps a [`FileLocation`] to its own `RwLock`'d table of `offset -> block`
//! (like [`CompressedCache`](super::compressed_cache::CompressedCache)'s `file_maps`),
//! so a `get` only contends on one file's lock. `reverse` is a per-slot index - one
//! entry per ring slot, no lock, like the compressed cache's per-slot tenant list -
//! naming the block that owns each
//! slot, so the evictor (which only has a slot index) can find the block. A slot's
//! reverse entry is written and read only while the slot is held exclusively (the
//! inserter's `WRITING`, or the evictor's `try_write`), which is what lets it skip a
//! lock.
//!
//! ## Eviction
//!
//! The cache claims each block's slots in the shared clock as
//! [`Owner::Decompressed`], whose lower tier max makes them age out before
//! compressed pages. The clock picks a slot and calls
//! [`reclaim`](DecompressedCache::reclaim): it `try_write`s the victim, reads which
//! block owns it, then `try_write`s the block's other slots (non-blocking, so any
//! order is deadlock-free). If any slot is still pinned by a reader it releases the
//! rest and bails, so the block stays cached. The compressed source pages age out
//! more slowly, so an evicted decompressed block falls back to an in-memory
//! re-decompress rather than a disk read.

use crate::env::get_env_var_with_default;
use crate::io::FileLocation;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Valid byte length of slot `i` of a block whose decompressed bytes total
/// `total_len`: every slot is full (`BUFFER_SIZE`) except the last, which holds the
/// remainder. Lets a block store one length instead of a per-slot `Vec`.
fn slot_len(total_len: usize, i: usize) -> usize {
    (total_len - i * BUFFER_SIZE).min(BUFFER_SIZE)
}

/// Whether `stored` (a block's slots, in insert order) names the same set as
/// `acquired` (the slots an evictor `try_write`-locked, sorted) - i.e. the forward
/// entry at an offset is still the block being reclaimed and not a re-inserted one.
/// Since the evictor holds every `acquired` slot `WRITING`, no re-insert could have
/// reused them, so equal sets prove identity.
fn slots_match(stored: &[usize], acquired_sorted: &[usize]) -> bool {
    stored.len() == acquired_sorted.len()
        && stored
            .iter()
            .all(|s| acquired_sorted.binary_search(s).is_ok())
}

/// Identity of a cached decompressed block: the file, the byte offset of its
/// source bytes, and their length. Unique per block. The length is part of the
/// key so a lookup only hits when both offset and size match, never serving bytes
/// decompressed from a differently-sized region at the same offset.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub location: FileLocation,
    pub offset: usize,
    pub len: usize,
}

impl BlockKey {
    /// This block's identity within its file: `(source offset, source length)`. The
    /// forward map is keyed by it, so a lookup misses unless both match.
    fn within_file(&self) -> (usize, usize) {
        (self.offset, self.len)
    }
}

/// A cached decompressed block: the ring slots holding its bytes (one per 2 MB) and
/// the decompressed total length (per-slot lengths come from [`slot_len`]). Its
/// identity - file, source offset, source length - is the key it is stored under.
struct DecompressedBlock {
    slots: Vec<usize>,
    total_len: usize,
}

/// One file's cached blocks, keyed by their source `(offset, length)`.
type FileBlocks = RwLock<HashMap<(usize, usize), DecompressedBlock>>;

/// A cache of decompressed blocks, shared across workers behind an `Arc` like the
/// [`CompressedCache`](super::compressed_cache::CompressedCache).
pub struct DecompressedCache {
    /// Per-file: file -> (source `(offset, length)` -> block). `get` only locks one
    /// file's table. Length is part of the key, so a lookup misses unless both offset
    /// and length match - never serving bytes from a differently-sized region at the
    /// same offset.
    files: RwLock<HashMap<FileLocation, FileBlocks>>,
    /// Per-slot reverse index, indexed by ring slot: the key of the block that owns
    /// the slot, or `None` if unowned. `UnsafeCell` (like the compressed cache's
    /// per-slot `tenants`): an entry is written/read only while its slot is held
    /// exclusively (the inserter's `WRITING`, or the evictor's `try_write`), so it
    /// needs no lock. Lets the evictor, holding only a slot index, find the block.
    reverse: Box<[UnsafeCell<Option<BlockKey>>]>,
    /// Number of ring slots currently bound `Decompressed` (mapped or orphaned by a
    /// pending invalidate/clear). Drives [`is_empty`](Self::is_empty); a slot is
    /// only subtracted when the clock actually reclaims it.
    count: AtomicUsize,
    /// Kill switch (`PIVOT_DECOMPRESSED_CACHE`, default on). When off, `get` is a
    /// no-op and `insert` returns the bytes uncached.
    enabled: bool,
}

// SAFETY: `reverse`'s `UnsafeCell`s are only accessed while the corresponding ring
// slot is held exclusively (`WRITING`), so there is never concurrent access to one
// entry. Same discipline as [`CompressedCache`].
unsafe impl Send for DecompressedCache {}
unsafe impl Sync for DecompressedCache {}

impl DecompressedCache {
    /// `capacity` is the ring's slot count: one reverse entry per slot.
    pub fn new(capacity: usize) -> Self {
        Self {
            files: RwLock::new(HashMap::default()),
            reverse: (0..capacity).map(|_| UnsafeCell::new(None)).collect(),
            count: AtomicUsize::new(0),
            enabled: get_env_var_with_default("PIVOT_DECOMPRESSED_CACHE", true),
        }
    }

    /// Mutable access to slot `idx`'s reverse entry. Sound only while the slot is
    /// held exclusively (the inserter's `WRITING`, or the evictor's `try_write`) -
    /// the one `unsafe` for the reverse invariant.
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell; see above
    fn reverse_mut(&self, idx: usize) -> &mut Option<BlockKey> {
        unsafe { &mut *self.reverse[idx].get() }
    }

    /// Look up a block. On a hit, pins each of its slots with `try_read`, refreshes
    /// the clock, and returns one zero-copy `Bytes` view per slot. Misses (returning
    /// `None`, releasing any pins taken) if the block is absent, its size doesn't
    /// match `key.len`, or a slot is mid-reclaim (`WRITING`).
    pub fn get(&self, key: &BlockKey) -> Option<Vec<Bytes>> {
        if !self.enabled {
            return None;
        }
        let files = self.files.read().unwrap();
        let table = files.get(&key.location)?.read().unwrap();
        let block = table.get(&key.within_file())?;
        let mut views = Vec::with_capacity(block.slots.len());
        for (i, &slot) in block.slots.iter().enumerate() {
            let read = memory_ctx().ring().try_read(slot)?;
            memory_ctx().clock().touch(slot);
            views.push(Bytes::from_owner(read).slice(..slot_len(block.total_len, i)));
        }
        Some(views)
    }

    /// Cache the page just decompressed into `write_buffers` (one per 2 MB slot,
    /// `total_len` decompressed bytes across them) and return the bytes to hand the
    /// decoder. If it caches, each buffer is converted to a read pin so its slot
    /// becomes resident at `used == 0` once the decoder drops the returned `Bytes`;
    /// otherwise (cache off, empty page, or the block already cached) the
    /// `WriteBuffer`-backed bytes are returned and their slots return to the pool on
    /// drop.
    pub fn insert(
        &self,
        key: BlockKey,
        write_buffers: Vec<WriteBuffer>,
        total_len: usize,
    ) -> Vec<Bytes> {
        if !self.enabled || write_buffers.is_empty() {
            return transient(write_buffers, total_len);
        }
        let slots: Vec<usize> = write_buffers.iter().map(|b| b.slot_idx).collect();

        // Claim `offset` under the file's write lock (so two workers decompressing
        // the same page don't both cache it); on a dup, leave this copy transient.
        if !self.claim(&key, &slots, total_len) {
            return transient(write_buffers, total_len);
        }

        // Record the reverse index and clock ownership while the slots are still
        // `WRITING` (held by `write_buffers`), so the evictor can't touch them yet -
        // which is also what makes writing each slot's reverse entry sound.
        for &slot in &slots {
            *self.reverse_mut(slot) = Some(key.clone());
            memory_ctx().clock().bind(slot, Owner::Decompressed);
        }
        self.count.fetch_add(slots.len(), Ordering::Relaxed);

        // Convert each buffer to a read pin (`used == 1`); when the decoder drops
        // the returned views the slot drops to `used == 0` - resident, not pooled.
        write_buffers
            .into_iter()
            .enumerate()
            .map(|(i, buffer)| {
                Bytes::from_owner(ReadBuffer::from(buffer)).slice(..slot_len(total_len, i))
            })
            .collect()
    }

    /// Insert the block's forward entry under the file's write lock, creating the
    /// file's table if needed. Returns `false` if `offset` is already cached.
    fn claim(&self, key: &BlockKey, slots: &[usize], total_len: usize) -> bool {
        loop {
            {
                let files = self.files.read().unwrap();
                if let Some(table) = files.get(&key.location) {
                    let mut table = table.write().unwrap();
                    if table.contains_key(&key.within_file()) {
                        return false;
                    }
                    table.insert(
                        key.within_file(),
                        DecompressedBlock {
                            slots: slots.to_vec(),
                            total_len,
                        },
                    );
                    return true;
                }
            }
            // File absent: create its (empty) table, then retry the claim above.
            self.files
                .write()
                .unwrap()
                .entry(key.location.clone())
                .or_default();
        }
    }

    /// The clock picked `slot` as a victim. Take it exclusively (`try_write`), read
    /// which block owns it, and evict that whole block: `try_write` every one of its
    /// slots and only then remove it and recycle them. If any slot is still pinned by
    /// a reader, release the ones taken without pooling them (the block stays cached)
    /// and return `None`. Returns the requested slot writable; the block's other
    /// slots return to the pool.
    pub fn reclaim(&self, slot: usize) -> Option<WriteBuffer> {
        // Take the victim first; only with it held `WRITING` is its reverse entry
        // sound to read (the same discipline `CompressedCache` uses for `tenants`).
        let victim = memory_ctx().ring().try_write(slot)?;
        if memory_ctx().clock().owner(slot) != Owner::Decompressed {
            // Raced to the free pool or another cache between the sweep and try_write.
            victim.release_in_place();
            return None;
        }
        // Borrow (don't clone) the owning block's key: we hold `slot` `WRITING` for
        // the rest of this call, so its reverse entry can't change under us. Sound to
        // read now that the owner is `Decompressed` - `insert` sets the entry before
        // binding the owner and the evict paths clear it after releasing, both under
        // `WRITING`, so a `Decompressed` slot always has one.
        let key = self
            .reverse_mut(slot)
            .as_ref()
            .expect("a Decompressed slot always has a reverse entry");

        // The block's full slot set. If its forward entry is gone (invalidate/clear
        // dropped it), this slot is an orphan: recycle just this one - no other
        // reader can reach an unmapped block, so no atomic group is needed.
        let block_slots = {
            let files = self.files.read().unwrap();
            files.get(&key.location).and_then(|table| {
                table
                    .read()
                    .unwrap()
                    .get(&key.within_file())
                    .map(|b| b.slots.clone())
            })
        };
        let Some(mut block_slots) = block_slots else {
            return Some(self.recycle_orphan(slot, victim));
        };
        block_slots.sort_unstable(); // `slots_match` binary-searches this set

        // Acquire the block's other slots; the victim is already held, and
        // `try_write` is non-blocking so any order is deadlock-free. Any miss (a
        // reader still pinning a slot) releases all taken and bails, leaving the
        // block cached.
        let mut buffers = vec![victim];
        for &s in block_slots.iter().filter(|&&s| s != slot) {
            match memory_ctx().ring().try_write(s) {
                Some(buffer) => buffers.push(buffer),
                None => {
                    for buffer in buffers {
                        buffer.release_in_place();
                    }
                    return None;
                }
            }
        }

        // Hold all the block's slots exclusively: drop the block and recycle them.
        // Each slot's reverse entry is cleared under its own `WRITING` hold.
        self.remove_block_if_current(key, &block_slots);
        for &s in &block_slots {
            *self.reverse_mut(s) = None;
            memory_ctx().clock().release(s);
        }
        self.count.fetch_sub(block_slots.len(), Ordering::Relaxed);

        // Hand `slot` back; the rest return to the pool when their buffers drop.
        let pos = buffers.iter().position(|b| b.slot_idx == slot)?;
        Some(buffers.swap_remove(pos))
    }

    /// Remove the block at `key`, then prune the file's table if that empties it.
    ///
    /// `acquired` is the set of slots the evictor `try_write`-locked (sorted). The
    /// removal only happens if the block now at `key` still owns exactly those slots,
    /// because of a race: the evictor read the block's slots, dropped the table lock
    /// to lock those slots, and is only now back to remove the entry. In that gap an
    /// `invalidate`/`clear` plus a re-insert could have put a *different*, still-live
    /// block at the same `key`, which must survive. Equal slots prove it is the same
    /// block: the evictor holds every `acquired` slot `WRITING`, so a re-insert could
    /// not have reused any of them.
    fn remove_block_if_current(&self, key: &BlockKey, acquired: &[usize]) {
        let mut emptied_file = false;
        if let Some(table) = self.files.read().unwrap().get(&key.location) {
            let mut table = table.write().unwrap();
            if table
                .get(&key.within_file())
                .is_some_and(|block| slots_match(&block.slots, acquired))
            {
                table.remove(&key.within_file());
            }
            emptied_file = table.is_empty();
        }
        if emptied_file {
            self.prune_empty_file(&key.location);
        }
    }

    /// Remove `location`'s table if it is now empty, so its never-reused
    /// `FileLocation` (an `Arc<File>`/`Arc<RemoteFile>` pinning the file/connection
    /// it holds) isn't kept alive for every file ever opened. Re-checks emptiness
    /// under the outer write lock so a concurrent `claim` that just re-created the
    /// table isn't dropped.
    fn prune_empty_file(&self, location: &FileLocation) {
        let mut files = self.files.write().unwrap();
        if let Some(table) = files.get(location)
            && table.read().unwrap().is_empty()
        {
            files.remove(location);
        }
    }

    /// Recycle a single orphan slot already held `WRITING` (`buffer`): its block was
    /// dropped from the forward map by [`invalidate`](Self::invalidate)/
    /// [`clear`](Self::clear), leaving the slot bound `Decompressed`. Clears its
    /// reverse entry (sound: held `WRITING`) and releases it from the clock.
    fn recycle_orphan(&self, slot: usize, buffer: WriteBuffer) -> WriteBuffer {
        *self.reverse_mut(slot) = None;
        memory_ctx().clock().release(slot);
        self.count.fetch_sub(1, Ordering::Relaxed);
        buffer
    }

    /// Whether the cache currently owns no ring slots. `get_write_buffer` checks
    /// this to decide whether eviction would have to fall through to the compressed
    /// cache.
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    /// Drop every block belonging to `location` (e.g. when its fd is reopened) so a
    /// stale block can't serve a later read. The slots stay `Decompressed` and are
    /// recycled lazily by the clock (via [`recycle_orphan`](Self::recycle_orphan)).
    pub fn invalidate(&self, location: &FileLocation) {
        if !self.enabled {
            return;
        }
        // Drop the file's forward entries only. The slots stay bound `Decompressed`
        // with their reverse entries set; the clock recycles each as an orphan (its
        // forward entry now gone) under `try_write` - the only point a slot's reverse
        // entry is sound to clear.
        self.files.write().unwrap().remove(location);
    }

    /// Drop every cached block, returning how many were dropped. Frees each free
    /// slot to the pool immediately; a slot still pinned by a reader stays
    /// `Decompressed` and is recycled lazily by the clock. Backs `drop_cache()`.
    pub fn clear(&self) -> usize {
        let files = std::mem::take(&mut *self.files.write().unwrap());
        let mut dropped = 0;
        for table in files.into_values() {
            for block in table.into_inner().unwrap().into_values() {
                dropped += 1;
                for slot in block.slots {
                    // Clear + recycle only slots we can take exclusively; a
                    // reader-pinned slot stays an orphan (its forward entry is now
                    // gone) and the clock recycles it later under `try_write`.
                    if let Some(buffer) = memory_ctx().ring().try_write(slot) {
                        *self.reverse_mut(slot) = None;
                        memory_ctx().clock().release(slot);
                        self.count.fetch_sub(1, Ordering::Relaxed);
                        drop(buffer); // returns the slot to the pool
                    }
                }
            }
        }
        dropped
    }
}

/// Build `WriteBuffer`-backed (uncached, pooled-on-drop) views from the buffers.
fn transient(write_buffers: Vec<WriteBuffer>, total_len: usize) -> Vec<Bytes> {
    write_buffers
        .into_iter()
        .enumerate()
        .map(|(i, buffer)| Bytes::from_owner(buffer).slice(..slot_len(total_len, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};
    use std::sync::{Arc, OnceLock};

    /// A fresh write buffer over one ring slot (the shape the decompressor feeds
    /// `insert`).
    fn ring_write() -> WriteBuffer {
        memory_ctx().get_write_buffer(false)
    }

    /// A fresh, independent local location (distinct cache key bucket).
    fn new_file() -> FileLocation {
        FileLocation::Local(Arc::new(std::fs::File::open("/dev/null").unwrap()))
    }

    fn key_for(location: &FileLocation) -> BlockKey {
        BlockKey {
            location: location.clone(),
            offset: 0,
            len: 1,
        }
    }

    /// Keys over one shared file, distinguished by `offset`.
    fn test_key(offset: usize) -> BlockKey {
        static FILE: OnceLock<Arc<std::fs::File>> = OnceLock::new();
        let location = FileLocation::Local(
            FILE.get_or_init(|| Arc::new(std::fs::File::open("/dev/null").unwrap()))
                .clone(),
        );
        BlockKey {
            location,
            offset,
            len: 1,
        }
    }

    fn for_test(enabled: bool) -> DecompressedCache {
        // Matches the 128-slot ring `init_test_free_pool` installs. Built without
        // touching `memory_ctx` so `get_misses_for_an_absent_key` (which never inits
        // a pool) keeps working.
        DecompressedCache {
            files: RwLock::new(HashMap::default()),
            reverse: (0..128).map(|_| UnsafeCell::new(None)).collect(),
            count: AtomicUsize::new(0),
            enabled,
        }
    }

    #[test]
    fn get_misses_for_an_absent_key() {
        let cache = for_test(true);

        assert!(cache.get(&test_key(0)).is_none());
    }

    #[test]
    fn a_hit_is_ring_backed_and_copy_free() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let buffer = ring_write();
        let ptr = buffer.as_slice().as_ptr();
        cache.insert(test_key(0), vec![buffer], 1);

        let hit = cache.get(&test_key(0)).unwrap();

        assert_eq!(hit[0].as_ptr(), ptr);
    }

    #[test]
    fn a_different_length_at_the_same_offset_misses() {
        init_test_free_pool(2);
        let cache = for_test(true);
        cache.insert(
            BlockKey {
                len: 100,
                ..test_key(0)
            },
            vec![ring_write()],
            1,
        );

        let other = BlockKey {
            len: 200,
            ..test_key(0)
        };

        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn insert_is_a_noop_when_the_offset_is_already_cached() {
        init_test_free_pool(2);
        let cache = for_test(true);
        cache.insert(test_key(0), vec![ring_write()], 1);
        let dup = ring_write();
        let dup_slot = dup.slot_idx;

        // The dup is returned transient (WriteBuffer-backed): dropping it pools the
        // slot, which the next allocation hands back.
        drop(cache.insert(test_key(0), vec![dup], 1));

        assert_eq!(memory_ctx().get_write_buffer(false).slot_idx, dup_slot);
    }

    #[test]
    fn reclaim_evicts_the_owning_block_and_returns_the_slot() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        drop(cache.insert(test_key(0), vec![buffer], 1)); // drop the view → resident

        let reclaimed = cache.reclaim(slot);

        assert_eq!(
            reclaimed.unwrap().slot_idx,
            slot,
            "the freed slot is handed back"
        );
        assert!(cache.is_empty(), "its block was evicted");
    }

    #[test]
    fn a_multi_slot_block_is_evicted_atomically_from_any_slot() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let a = ring_write();
        let b = ring_write();
        let hi = a.slot_idx.max(b.slot_idx);
        drop(cache.insert(test_key(0), vec![a, b], BUFFER_SIZE + 1)); // spans 2 slots

        let reclaimed = cache.reclaim(hi);

        assert_eq!(
            reclaimed.unwrap().slot_idx,
            hi,
            "the victim slot is handed back"
        );
        assert!(
            cache.is_empty(),
            "the whole block, both slots, was reclaimed"
        );
    }

    #[test]
    fn evicting_a_files_last_block_releases_its_file_handle() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let location = new_file();
        let FileLocation::Local(file) = &location else {
            unreachable!()
        };
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        drop(cache.insert(key_for(&location), vec![buffer], 1));

        cache.reclaim(slot);

        assert_eq!(
            Arc::strong_count(file),
            1,
            "evicting a file's last block drops the cache's clone of its fd, not just the block"
        );
    }

    #[test]
    fn reclaim_is_none_for_an_unowned_slot() {
        init_test_free_pool(1);
        let cache = for_test(true);

        assert!(cache.reclaim(0).is_none());
    }

    #[test]
    fn invalidate_drops_only_the_matching_location() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let a = new_file();
        let b = new_file();
        cache.insert(key_for(&a), vec![ring_write()], 1);
        cache.insert(key_for(&b), vec![ring_write()], 1);

        cache.invalidate(&a);

        assert!(cache.get(&key_for(&a)).is_none());
        assert!(cache.get(&key_for(&b)).is_some());
    }

    #[test]
    fn clear_drops_everything() {
        init_test_free_pool(4);
        let cache = for_test(true);
        drop(cache.insert(test_key(0), vec![ring_write()], 1));
        drop(cache.insert(test_key(1), vec![ring_write()], 1));

        let dropped = cache.clear();

        assert_eq!(dropped, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        init_test_free_pool(1);
        let cache = for_test(false);

        // insert returns the bytes transient; nothing is cached.
        drop(cache.insert(test_key(0), vec![ring_write()], 1));

        assert!(cache.get(&test_key(0)).is_none());
        assert!(cache.is_empty());
    }
}
