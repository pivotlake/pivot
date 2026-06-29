//! A cache of decompressed blocks, sharing the ring's CLOCK
//! ([`Clock`](super::clock::Clock)) with the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache).
//!
//! A block is bytes that were decompressed from a contiguous source range of a
//! file (a [`FileLocation`] plus a byte offset and length). The cache retains the
//! ring-backed `Bytes` keyed by that source range, so a later read of the same
//! range skips decompression and reuses the bytes. It carries no knowledge of
//! what produced the bytes.
//!
//! ## Everything stays on the ring
//!
//! The cached payload is the exact same ring-backed `Bytes` the consumer uses; a
//! hit clones them (an `Arc` bump, no byte copy) and releasing memory just drops
//! them, returning the slot to the free pool. Nothing is ever migrated between
//! heap and ring.
//!
//! ## Eviction
//!
//! The cache claims each block's slots in the shared clock as
//! [`Owner::Decompressed`], whose lower tier max (1 vs 2) makes them age out
//! first. When the sweep picks a decompressed victim it calls
//! [`reclaim`](DecompressedCache::reclaim), dropping the block that owns
//! that slot. The compressed source pages age out more slowly (tier max 2), so an
//! evicted decompressed block falls back to an in-memory re-decompress rather than
//! a disk read.

use crate::env::get_env_var_with_default;
use crate::io::FileLocation;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::sync::Mutex;

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

/// A retained decompressed block.
struct CachedBlock {
    /// The decompressed bytes, ring-backed (one `Bytes` per 2 MB slot).
    data: Vec<Bytes>,
    /// The ring slots `data` lives in (one per `Bytes`).
    slots: Vec<usize>,
}

#[derive(Default)]
struct Inner {
    /// Block lookup, keyed by source range.
    blocks: HashMap<BlockKey, CachedBlock>,
    /// Reverse index, so the shared clock's evictor can find the block that owns a
    /// victim slot.
    slot_to_block: HashMap<usize, BlockKey>,
}

/// A cache of decompressed blocks, shared across workers behind an `Arc` like the
/// [`CompressedCache`](super::compressed_cache::CompressedCache).
pub struct DecompressedCache {
    inner: Mutex<Inner>,
    /// Kill switch (`PIVOT_DECOMPRESSED_CACHE`, default on). When off, `get`/
    /// `insert` are no-ops, so behaviour matches a build without the cache.
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
            inner: Mutex::new(Inner::default()),
            enabled: get_env_var_with_default("PIVOT_DECOMPRESSED_CACHE", true),
        }
    }

    /// Look up a block. On a hit, marks its slots used in the shared clock and
    /// returns a clone of its ring-backed bytes (an `Arc` bump per buffer, no byte
    /// copy).
    pub fn get(&self, key: &BlockKey) -> Option<Vec<Bytes>> {
        if !self.enabled {
            return None;
        }
        let inner = self.inner.lock().unwrap();
        let block = inner.blocks.get(key)?;
        for &slot in &block.slots {
            memory_ctx().clock().touch(slot);
        }
        Some(block.data.clone())
    }

    /// Retain `data` (occupying ring `slots` at `key`) claiming those slots in the
    /// shared clock as decompressed. A block already present is left as is (the new
    /// `data` is dropped, freeing its slots).
    pub fn insert(&self, key: BlockKey, data: Vec<Bytes>, slots: Vec<usize>) {
        if !self.enabled {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.blocks.contains_key(&key) {
            return;
        }
        for &slot in &slots {
            memory_ctx().clock().bind(slot, Owner::Decompressed);
            inner.slot_to_block.insert(slot, key.clone());
        }
        inner.blocks.insert(key, CachedBlock { data, slots });
    }

    /// Drop the block that owns ring slot `slot_idx` - the shared clock picked it
    /// as a victim - releasing all of its slots. Returns `slot_idx` writable when
    /// it frees immediately; `None` if a live query still pins the block (its
    /// slots return to the pool when that query ends) or the slot was already
    /// reclaimed. The block's bytes drop after the lock is released.
    pub fn reclaim(&self, slot_idx: usize) -> Option<WriteBuffer> {
        let block = {
            let mut inner = self.inner.lock().unwrap();
            let key = inner.slot_to_block.get(&slot_idx)?.clone();
            let block = inner.blocks.remove(&key)?;
            for &slot in &block.slots {
                inner.slot_to_block.remove(&slot);
                memory_ctx().clock().release(slot);
            }
            block
        };
        drop(block.data); // releases the slots' write buffers to the free pool
        // Hand the requested slot straight back if it is now free (skipping a
        // pool round-trip); otherwise it returns to the pool on its own.
        memory_ctx().ring().try_write(slot_idx)
    }

    /// Whether the cache holds no blocks. `get_write_buffer` checks this to decide
    /// whether eviction would have to fall through to the compressed cache.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().blocks.is_empty()
    }

    /// Drop every block belonging to `location` (e.g. when its file descriptor is
    /// reopened), releasing their slots, so a stale block can't serve a later read.
    pub fn invalidate(&self, location: &FileLocation) {
        if !self.enabled {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let stale: Vec<BlockKey> = inner
            .blocks
            .keys()
            .filter(|key| &key.location == location)
            .cloned()
            .collect();
        for key in stale {
            self.remove_block(&mut inner, &key);
        }
    }

    /// Drop every cached block, returning how many were dropped. Backs
    /// `drop_cache()`.
    pub fn clear(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        for &slot in inner.slot_to_block.keys() {
            memory_ctx().clock().release(slot);
        }
        inner.slot_to_block.clear();
        let dropped = inner.blocks.len();
        inner.blocks.clear();
        dropped
    }

    /// Remove one block and release its slots from the clock and reverse index.
    fn remove_block(&self, inner: &mut Inner, key: &BlockKey) {
        if let Some(block) = inner.blocks.remove(key) {
            for &slot in &block.slots {
                inner.slot_to_block.remove(&slot);
                memory_ctx().clock().release(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};
    use std::sync::{Arc, OnceLock};

    /// A ring-backed one-slot block (the same shape the decompressor produces):
    /// its bytes plus the slot they occupy.
    fn ring_block() -> (Vec<Bytes>, Vec<usize>) {
        let buffer = memory_ctx().get_write_buffer(false);
        let slot = buffer.slot_idx;
        (vec![Bytes::from_owner(buffer)], vec![slot])
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
            inner: Mutex::new(Inner::default()),
            enabled,
        }
    }

    #[test]
    fn get_misses_for_an_absent_key() {
        let cache = for_test(true);

        assert!(cache.get(&test_key(0)).is_none());
    }

    #[test]
    fn a_different_length_at_the_same_offset_misses() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let (data, slots) = ring_block();
        let stored = BlockKey {
            len: 100,
            ..test_key(0)
        };
        cache.insert(stored, data, slots);

        let same_offset_other_len = BlockKey {
            len: 200,
            ..test_key(0)
        };

        assert!(cache.get(&same_offset_other_len).is_none());
    }

    #[test]
    fn a_hit_is_ring_backed_and_copy_free() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let (data, slots) = ring_block();
        let ptr = data[0].as_ptr();
        cache.insert(test_key(0), data, slots);

        let hit = cache.get(&test_key(0)).unwrap();

        assert_eq!(hit[0].as_ptr(), ptr);
    }

    #[test]
    fn reclaim_drops_the_owning_block_and_returns_the_slot() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let (data, slots) = ring_block();
        let slot = slots[0];
        cache.insert(test_key(0), data, slots);

        let reclaimed = cache.reclaim(slot);

        assert!(
            reclaimed.is_some(),
            "the freed slot is handed straight back"
        );
        assert!(cache.is_empty(), "its block was evicted");
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
        let (da, sa) = ring_block();
        let (db, sb) = ring_block();
        cache.insert(key_for(&a), da, sa);
        cache.insert(key_for(&b), db, sb);

        cache.invalidate(&a);

        assert!(cache.get(&key_for(&a)).is_none());
        assert!(cache.get(&key_for(&b)).is_some());
    }

    #[test]
    fn clear_drops_everything() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let (d0, s0) = ring_block();
        let (d1, s1) = ring_block();
        cache.insert(test_key(0), d0, s0);
        cache.insert(test_key(1), d1, s1);

        let dropped = cache.clear();

        assert_eq!(dropped, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        init_test_free_pool(1);
        let cache = for_test(false);
        let (data, slots) = ring_block();

        cache.insert(test_key(0), data, slots);

        assert!(cache.get(&test_key(0)).is_none());
        assert!(cache.is_empty());
    }
}
