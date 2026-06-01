//! A CLOCK-based page cache that maps [`IOLocation`]s to ring buffer slots.
//!
//! After an I/O read completes, the resulting [`WriteBuffer`] is inserted into the cache
//! (converting it to a [`ReadBuffer`]) so subsequent reads to the same file offset can be
//! served from memory. Each cached entry has a *reference bit* that is set on access. When
//! the system needs to reclaim a buffer (eviction), the CLOCK hand sweeps through entries,
//! clearing reference bits until it finds an unreferenced slot it can acquire for writing.
//!
//! The cache is keyed two levels deep: `RawFd → offset → ring slot index`.  A file must be
//! registered with [`FileCache::open_file_entry`] before its pages can be cached.
//!
//! Concurrency: lookups in [`FileCache::get`] use the ring's reader-count protocol to pin a slot before
//! verifying the location still matches, preventing TOCTOU races with concurrent evictions.

use crate::io::IOLocation;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use std::cell::UnsafeCell;
use std::os::fd::RawFd;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct FileCacheEntry {
    ring_idx: usize,
}

/// A CLOCK-eviction page cache over the shared [`Ring`].
///
/// * `ring` – the global ring of mmap'd buffer slots.
/// * `file_maps` – two-level map: `fd → (offset → FileCacheEntry)`.
/// * `entries` – per-slot metadata (reference bit + cached location), indexed by ring slot.
/// * `hand` – CLOCK sweep cursor for eviction.
pub struct FileCache {
    file_maps: RwLock<HashMap<RawFd, RwLock<HashMap<usize, FileCacheEntry>>>>,
    entries: Box<[UnsafeCell<Entry>]>,
    hand: AtomicUsize,
}

unsafe impl Send for FileCache {}
unsafe impl Sync for FileCache {}

/// Per-slot CLOCK metadata.
///
/// * `ref_bit` – set on access, cleared by the CLOCK sweep. A slot with a cleared
///   ref bit is eligible for eviction.
/// * `location` – the file location currently cached in this slot, or `None` if unused.
struct Entry {
    ref_bit: AtomicBool,
    location: Option<IOLocation>,
}

impl FileCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            file_maps: Default::default(),
            entries: (0..capacity)
                .map(|_| {
                    UnsafeCell::new(Entry {
                        ref_bit: Default::default(),
                        location: None,
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hand: Default::default(),
        }
    }

    /// Evict a slot using the CLOCK algorithm and return it as a writable buffer.
    ///
    /// Sweeps from the current `hand` position, clearing reference bits. The first
    /// unreferenced slot that can be exclusively acquired for writing is evicted:
    /// its file-map entry is removed and the slot is returned as a [`WriteBuffer`].
    pub fn evict(&self) -> WriteBuffer {
        loop {
            let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % memory_ctx().ring().len();
            let cell = &self.entries[slot_idx];
            let entry = unsafe { &*cell.get() };
            if entry.ref_bit.swap(false, Ordering::Relaxed) {
                continue;
            }

            if let Some(r) = memory_ctx().ring().try_write(slot_idx) {
                // We now know no one is touching entry, since ring guarantees us there can be only
                // one writer per buffer. We therefore take mut of entry
                let entry = unsafe { &mut *cell.get() };
                if let Some(l) = entry.location.take() {
                    // Remove the old entry
                    self.file_maps
                        .read()
                        .unwrap()
                        .get(&l.raw_fd)
                        .unwrap()
                        .write()
                        .unwrap()
                        .remove(&l.offset);
                }
                return r;
            }
        }
    }

    /// Register a file descriptor so its pages can be cached.
    /// Must be called before any [`Self::insert`] or [`Self::get`] for this fd.
    ///
    /// If the fd was previously registered, all its cached entries are cleared.
    /// This handles fd reuse: after a file is closed the OS may assign the same
    /// fd number to a new file, so stale entries must not survive.
    pub fn open_file_entry(&self, raw_fd: RawFd) {
        let mut file_maps = self.file_maps.write().unwrap();
        if let Some(old_map) = file_maps.remove(&raw_fd) {
            for (_, entry) in old_map.into_inner().unwrap() {
                let cell = &self.entries[entry.ring_idx];
                let slot = unsafe { &mut *cell.get() };
                slot.location = None;
                // Return the slot to the free pool if no readers hold it.
                // try_write returns Some(WriteBuffer), whose Drop pushes
                // the index back to the free pool. If readers are outstanding
                // it returns None and the CLOCK sweep reclaims the slot later.
                drop(memory_ctx().ring().try_write(entry.ring_idx));
            }
        }
        file_maps.insert(raw_fd, Default::default());
    }

    /// Insert a completed I/O buffer into the cache, converting it to a [`ReadBuffer`].
    ///
    /// Records the mapping `(fd, offset) → ring slot`, sets the reference bit, and
    /// transitions the buffer from write to read mode.
    pub fn insert(&self, location: IOLocation, buffer: WriteBuffer) -> ReadBuffer {
        let read_file_maps = self.file_maps.read().unwrap();
        let entry = FileCacheEntry {
            ring_idx: buffer.slot_idx,
        };
        let map = read_file_maps
            .get(&location.raw_fd)
            .expect("Missing file in file cache!");
        map.write().unwrap().insert(location.offset, entry);

        let cell = &self.entries[buffer.slot_idx];
        let slot = unsafe { &mut *cell.get() };
        slot.location = Some(location);
        slot.ref_bit.store(true, Ordering::Relaxed);

        ReadBuffer::from(buffer)
    }

    /// Look up a cached page by location.
    ///
    /// Pins the slot for reading via the ring's reader-count, then re-checks that the
    /// location still matches (guards against a concurrent eviction that recycled the slot
    /// between the map lookup and the pin).
    pub fn get(&self, location: &IOLocation) -> Option<ReadBuffer> {
        let read_file_maps = self.file_maps.read().unwrap();
        let inner_map = read_file_maps
            .get(&location.raw_fd)
            .unwrap()
            .read()
            .unwrap();
        let entry = inner_map.get(&location.offset)?;

        // First, try to pin it for reading - after this, we know we're necessarily looking at the "same" slot (ie it
        // can't be written to)
        let guard = memory_ctx().ring().try_read(entry.ring_idx)?;
        // Now we want to check if the location is the same as the one we're trying to get - we know
        // the location cannot change while we're doing this operation since we have a read guard
        let cell = &self.entries[guard.ring_idx()];
        let entry = unsafe { &*cell.get() };
        if entry.location.as_ref()? == location {
            entry.ref_bit.store(true, Ordering::Relaxed);
            return Some(guard);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    fn get_write_buffer(prefer_zeroed: bool) -> WriteBuffer {
        memory_ctx().get_write_buffer(prefer_zeroed)
    }

    fn loc(fd: RawFd, offset: usize) -> IOLocation {
        IOLocation { raw_fd: fd, offset }
    }

    fn file_cache() -> FileCache {
        FileCache::new(1_000)
    }

    #[test]
    fn get_returns_none_for_uncached_location() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);

        let result = cache.get(&loc(42, 0));

        assert!(result.is_none());
    }

    #[test]
    fn insert_then_get_returns_data() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let mut wb = get_write_buffer(false);
        wb[0..5].copy_from_slice(b"hello");
        let _rb = cache.insert(loc(42, 0), wb);

        let rb = cache.get(&loc(42, 0)).unwrap();

        assert_eq!(&rb[0..5], b"hello");
    }

    #[test]
    fn get_returns_none_for_different_offset() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let wb = get_write_buffer(false);
        let _rb = cache.insert(loc(42, 0), wb);

        let result = cache.get(&loc(42, 4096));

        assert!(result.is_none());
    }

    #[test]
    fn multiple_offsets_same_fd() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let mut wb1 = get_write_buffer(false);
        wb1[0] = 0xAA;
        let mut wb2 = get_write_buffer(false);
        wb2[0] = 0xBB;
        let _r1 = cache.insert(loc(42, 0), wb1);
        let _r2 = cache.insert(loc(42, 4096), wb2);

        let rb1 = cache.get(&loc(42, 0)).unwrap();
        let rb2 = cache.get(&loc(42, 4096)).unwrap();

        assert_eq!(rb1[0], 0xAA);
        assert_eq!(rb2[0], 0xBB);
    }

    #[test]
    fn insert_returns_readable_buffer_with_same_data() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let mut wb = get_write_buffer(false);
        wb[0..4].copy_from_slice(&[1, 2, 3, 4]);

        let rb = cache.insert(loc(42, 0), wb);

        assert_eq!(&rb[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn open_file_entry_is_idempotent() {
        init_test_free_pool(0);
        let cache = file_cache();

        cache.open_file_entry(42);
        cache.open_file_entry(42);

        assert!(cache.get(&loc(42, 0)).is_none());
    }

    #[test]
    fn reopen_fd_clears_stale_entries() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let mut wb = get_write_buffer(false);
        wb[0..5].copy_from_slice(b"stale");
        let _rb = cache.insert(loc(42, 0), wb);
        assert!(cache.get(&loc(42, 0)).is_some());

        cache.open_file_entry(42);

        assert!(cache.get(&loc(42, 0)).is_none());
    }

    #[test]
    fn evict_returns_writable_buffer() {
        init_test_free_pool(4);
        let cache = file_cache();

        let mut wb = cache.evict();
        wb[0] = 0xFF;

        assert_eq!(wb[0], 0xFF);
    }

    #[test]
    fn evict_removes_cached_entry() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let wb = get_write_buffer(false);
        let slot = wb.slot_idx;
        let rb = cache.insert(loc(42, 0), wb);
        drop(rb);
        // First sweep: ref_bit is true from insert - cleared, some other slot evicted
        cache.hand.store(slot, Ordering::Relaxed);
        let _other = cache.evict();
        // Second sweep: ref_bit now false → our cached slot is evicted
        cache.hand.store(slot, Ordering::Relaxed);
        let _evicted = cache.evict();

        assert!(cache.get(&loc(42, 0)).is_none());
    }

    #[test]
    fn evict_does_not_evict_referenced_slot() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let wb = get_write_buffer(false);
        let slot = wb.slot_idx;
        let rb = cache.insert(loc(42, 0), wb);
        drop(rb);
        // insert set ref_bit=true — evict should skip our slot (clearing the bit)
        // and evict some other empty slot instead
        cache.hand.store(slot, Ordering::Relaxed);

        let evicted = cache.evict();
        assert_ne!(evicted.slot_idx, slot);
        assert!(cache.get(&loc(42, 0)).is_some());
    }

    #[test]
    fn get_refreshes_ref_bit_protecting_from_eviction() {
        init_test_free_pool(4);
        let cache = file_cache();
        cache.open_file_entry(42);
        let wb = get_write_buffer(false);
        let slot = wb.slot_idx;
        let rb = cache.insert(loc(42, 0), wb);
        drop(rb);

        // First sweep: ref_bit is true from insert — cleared, slot skipped
        cache.hand.store(slot, Ordering::Relaxed);
        let _other = cache.evict();

        // Access the page — get() should re-set ref_bit
        let rb = cache.get(&loc(42, 0)).unwrap();
        drop(rb);

        // Second sweep: ref_bit should be true again from the get() — slot skipped
        cache.hand.store(slot, Ordering::Relaxed);
        let evicted = cache.evict();
        assert_ne!(
            evicted.slot_idx, slot,
            "recently accessed slot should not be evicted"
        );
        assert!(
            cache.get(&loc(42, 0)).is_some(),
            "page should still be cached after access"
        );
    }
}
