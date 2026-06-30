//! A cache of decompressed blocks, sharing the ring's CLOCK
//! ([`Clock`](super::clock::Clock)) with the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache), and modelled on
//! it: a cached block sits in its ring slot(s) at `used == 0`, owned via the clock
//! (no held pin). A [`get`](DecompressedCache::get) reconstructs a zero-copy view
//! with `try_read`; eviction acquires the slots with `try_write` first and only
//! then removes the block - never the drop-into-pool-then-grab dance.
//!
//! ## Per-file maps
//!
//! `files` maps a [`FileLocation`] to its own `RwLock`'d table of `offset -> block`
//! (like [`CompressedCache`]'s `file_maps`), so a `get` only contends on one file's
//! lock. `reverse` maps each ring slot to the block key that owns it, so the
//! evictor - which only has a slot index - can find the block.
//!
//! ## Eviction
//!
//! The cache claims each block's slots in the shared clock as
//! [`Owner::Decompressed`], whose lower tier max makes them age out before
//! compressed pages. The clock picks a slot and calls
//! [`reclaim`](DecompressedCache::reclaim), which `try_write`s every slot of that
//! slot's block (in sorted order, deadlock-free); if any slot is still pinned by a
//! reader it releases the rest and bails, so the block stays cached. The compressed
//! source pages age out more slowly, so an evicted decompressed block falls back to
//! an in-memory re-decompress rather than a disk read.

use crate::env::get_env_var_with_default;
use crate::io::FileLocation;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
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

/// A cached decompressed block: the ring slots holding its bytes (one per 2 MB),
/// the decompressed total length (per-slot lengths come from [`slot_len`]), and the
/// source `len` from its [`BlockKey`] (so a `get` can reject a same-offset key of a
/// different size).
struct DecompBlock {
    slots: Vec<usize>,
    total_len: usize,
    len: usize,
}

/// A cache of decompressed blocks, shared across workers behind an `Arc` like the
/// [`CompressedCache`](super::compressed_cache::CompressedCache).
pub struct DecompressedCache {
    /// Per-file: file -> (source offset -> block). `get` only locks one file's table.
    files: RwLock<HashMap<FileLocation, RwLock<HashMap<usize, DecompBlock>>>>,
    /// slot -> the key whose block owns it, so `reclaim` (which only has a slot
    /// index) can find and remove the block.
    reverse: RwLock<HashMap<usize, BlockKey>>,
    /// Number of ring slots currently bound `Decompressed` (mapped or orphaned by a
    /// pending invalidate/clear). Drives [`is_empty`](Self::is_empty); a slot is
    /// only subtracted when the clock actually reclaims it.
    count: AtomicUsize,
    /// Kill switch (`PIVOT_DECOMPRESSED_CACHE`, default on). When off, `get` is a
    /// no-op and `insert` returns the bytes uncached.
    enabled: bool,
}

impl Default for DecompressedCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DecompressedCache {
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::default()),
            reverse: RwLock::new(HashMap::default()),
            count: AtomicUsize::new(0),
            enabled: get_env_var_with_default("PIVOT_DECOMPRESSED_CACHE", true),
        }
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
        let block = table.get(&key.offset)?;
        if block.len != key.len {
            return None;
        }
        let mut views = Vec::with_capacity(block.slots.len());
        for (i, &slot) in block.slots.iter().enumerate() {
            // `?` on a failed `try_read` drops `views`, releasing the pins already
            // taken, and the whole lookup misses.
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
        // `WRITING` (held by `write_buffers`), so the evictor can't touch them yet.
        {
            let mut reverse = self.reverse.write().unwrap();
            for &slot in &slots {
                reverse.insert(slot, key.clone());
                memory_ctx().clock().bind(slot, Owner::Decompressed);
            }
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
                    if table.contains_key(&key.offset) {
                        return false;
                    }
                    table.insert(
                        key.offset,
                        DecompBlock {
                            slots: slots.to_vec(),
                            total_len,
                            len: key.len,
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

    /// The clock picked `slot` as a victim. Evict the whole block it belongs to:
    /// `try_write` every one of the block's slots (sorted, so concurrent evictors
    /// can't deadlock), and only on acquiring them all remove the block and recycle
    /// the slots. If any slot is still pinned by a reader, release the ones taken
    /// without pooling them (the block stays cached) and return `None`. Returns the
    /// requested slot writable; the block's other slots return to the pool.
    pub fn reclaim(&self, slot: usize) -> Option<WriteBuffer> {
        let key = self.reverse.read().unwrap().get(&slot).cloned();
        let Some(key) = key else {
            return self.reclaim_orphan(slot);
        };

        let mut block_slots = {
            let files = self.files.read().unwrap();
            let entry = files.get(&key.location).and_then(|table| {
                table
                    .read()
                    .unwrap()
                    .get(&key.offset)
                    .map(|b| b.slots.clone())
            });
            match entry {
                Some(slots) => slots,
                None => return self.reclaim_orphan(slot),
            }
        };
        block_slots.sort_unstable();

        // Acquire every slot of the block, in order. Any miss → release and bail.
        let mut buffers = Vec::with_capacity(block_slots.len());
        for &s in &block_slots {
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
        self.remove_block_if_current(&key.location, key.offset, &block_slots);
        {
            let mut reverse = self.reverse.write().unwrap();
            for &s in &block_slots {
                reverse.remove(&s);
            }
        }
        for &s in &block_slots {
            memory_ctx().clock().release(s);
        }
        self.count.fetch_sub(block_slots.len(), Ordering::Relaxed);

        // Hand `slot` back; the rest return to the pool when their buffers drop.
        let pos = buffers.iter().position(|b| b.slot_idx == slot)?;
        Some(buffers.swap_remove(pos))
    }

    /// Remove the block at `offset` only if it still names `acquired` (the slots an
    /// evictor `try_write`-locked, sorted). An invalidate/clear + re-insert in the
    /// acquisition window can replace it with a different, live block at the same
    /// offset, which must survive; the evictor holds every `acquired` slot `WRITING`,
    /// so a re-insert can't have reused them and matching slots prove it is the same
    /// block. Prunes the file's table if it goes empty.
    fn remove_block_if_current(&self, location: &FileLocation, offset: usize, acquired: &[usize]) {
        let mut emptied_file = false;
        if let Some(table) = self.files.read().unwrap().get(location) {
            let mut table = table.write().unwrap();
            if table
                .get(&offset)
                .is_some_and(|block| slots_match(&block.slots, acquired))
            {
                table.remove(&offset);
            }
            emptied_file = table.is_empty();
        }
        if emptied_file {
            self.prune_empty_file(location);
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

    /// Reclaim a slot the reverse map doesn't know about - an orphan left
    /// `Decompressed` by [`invalidate`](Self::invalidate)/[`clear`](Self::clear).
    /// Only touches a slot the clock still marks `Decompressed` (so a stray call for
    /// an unowned slot is a no-op).
    fn reclaim_orphan(&self, slot: usize) -> Option<WriteBuffer> {
        if memory_ctx().clock().owner(slot) != Owner::Decompressed {
            return None;
        }
        let buffer = memory_ctx().ring().try_write(slot)?;
        if memory_ctx().clock().owner(slot) != Owner::Decompressed {
            buffer.release_in_place();
            return None;
        }
        self.reverse.write().unwrap().remove(&slot);
        memory_ctx().clock().release(slot);
        self.count.fetch_sub(1, Ordering::Relaxed);
        Some(buffer)
    }

    /// Whether the cache currently owns no ring slots. `get_write_buffer` checks
    /// this to decide whether eviction would have to fall through to the compressed
    /// cache.
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    /// Drop every block belonging to `location` (e.g. when its fd is reopened) so a
    /// stale block can't serve a later read. The slots stay `Decompressed` and are
    /// recycled lazily by the clock (via [`reclaim_orphan`](Self::reclaim_orphan)).
    pub fn invalidate(&self, location: &FileLocation) {
        if !self.enabled {
            return;
        }
        let table = self.files.write().unwrap().remove(location);
        if let Some(table) = table {
            let mut reverse = self.reverse.write().unwrap();
            for block in table.into_inner().unwrap().into_values() {
                for slot in block.slots {
                    reverse.remove(&slot);
                }
            }
        }
    }

    /// Drop every cached block, returning how many were dropped. Frees each free
    /// slot to the pool immediately; a slot still pinned by a reader stays
    /// `Decompressed` and is recycled lazily by the clock. Backs `drop_cache()`.
    pub fn clear(&self) -> usize {
        let files = std::mem::take(&mut *self.files.write().unwrap());
        let mut reverse = self.reverse.write().unwrap();
        let mut dropped = 0;
        for table in files.into_values() {
            for block in table.into_inner().unwrap().into_values() {
                dropped += 1;
                for slot in block.slots {
                    reverse.remove(&slot);
                    if let Some(buffer) = memory_ctx().ring().try_write(slot) {
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
        DecompressedCache {
            files: RwLock::new(HashMap::default()),
            reverse: RwLock::new(HashMap::default()),
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
