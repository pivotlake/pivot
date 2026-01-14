use std::os::fd::RawFd;
use std::sync::atomic::Ordering;
use std::sync::{LazyLock, RwLock};
use ahash::HashMap;
use crate::io::IOLocation;
use crate::memory::buffer_cache::{BufferCache, ReadBufferGuard, WriteBufferGuard, BUFFER_CACHE, WRITING};

static FILE_CACHE: LazyLock<FileCache> = LazyLock::new(|| FileCache {
    file_maps: Default::default(),
    buffer_cache: &BUFFER_CACHE,
});

struct FileCacheEntry {
    slot_idx: usize,
    generation: usize
}

pub struct FileCache {
    file_maps: RwLock<HashMap<RawFd, RwLock<HashMap<usize, FileCacheEntry>>>>,
    buffer_cache: &'static BufferCache
}

impl FileCache {
    pub fn insert(&self, location: IOLocation, write_guard: WriteBufferGuard) -> ReadBufferGuard {
        let read_file_maps = self.file_maps.read().unwrap();
        let entry = FileCacheEntry {
            slot_idx: write_guard.slot_idx,
            generation: write_guard.buffer.generation,
        };
        let map = read_file_maps.get(&location.raw_fd).expect("Missing file in file cache!");
        map.write().unwrap().insert(location.offset, entry);
        ReadBufferGuard::from(write_guard)
    }

    pub fn open_file_entry(&self, raw_fd: RawFd) {
        let mut file_maps = self.file_maps.write().unwrap();
        file_maps.entry(raw_fd).or_insert_with(Default::default);
    }

    pub fn get(&self, location: IOLocation) -> Option<ReadBufferGuard> {
        let read_file_maps = self.file_maps.read().unwrap();
        let inner_map = read_file_maps.get(&location.raw_fd)?.read().unwrap();
        let entry = inner_map.get(&location.offset)?;

        // We found a slot- but now we need to check if it's valid
        let slot = &self.buffer_cache.slots[entry.slot_idx];

        // First, pin it - after this, we know we're necessarily looking at the "same" slot (ie it
        // cant be written to)
        let prev = slot.pin_count.fetch_add(1, Ordering::Acquire);

        if prev & WRITING != 0 {
            // Being written to right now - we don't need to remove our pin count since it'll reset
            // to 0 anyway when it's dropped, but we do if logic changes
            slot.pin_count.fetch_sub(1, Ordering::Relaxed);

            drop(inner_map);
            read_file_maps.get(&location.raw_fd)?.write().unwrap().remove(&location.offset);
            return None;
        }

        // Now we *know* it's not possible it's being evicted - since we added one (which means no
        // one new can evict) and the previous res didn't have an eviction on it
        let buffer = unsafe { *slot.buffer.get() };
        // Check generation in case hashmap was stale
        if buffer.generation != entry.generation {
            slot.pin_count.fetch_sub(1, Ordering::Relaxed);

            drop(inner_map);
            read_file_maps.get(&location.raw_fd)?.write().unwrap().remove(&location.offset);
            return None
        }

        slot.ref_bit.store(true, Ordering::Relaxed);

        // It's valid!
        Some(ReadBufferGuard {
            buffer,
            slot_idx: entry.slot_idx,
            cache: self.buffer_cache,
        })
    }
}
