//! A CLOCK-based page cache that maps 2 MB file *regions* to ring buffer slots,
//! with per-slot validity tracked at 4 KB sub-block granularity.
//!
//! Each slot owns a fixed, 2 MB-aligned window of a file (`region_base`), so
//! every file byte belongs to exactly one slot — cache entries can never
//! overlap. A slot's [`Entry::valid`] bitmap records which of its 512 × 4 KB
//! sub-blocks have actually been read from disk; a slot can therefore be
//! *partially* present and get filled incrementally as different columns touch
//! different parts of the same window.
//!
//! ## Lifecycle
//!
//! A region slot is allocated once (a free ring slot, its bitmap zeroed while
//! held exclusively) and registered in the map; it then lives as a *readable*
//! slot (reader-count tracked, never re-taking the ring's `WRITING` bit) until
//! CLOCK eviction reclaims it. Reads land directly into the slot's holes — the
//! "mutate a `ReadBuffer` in place" case — which is sound because a fill only
//! targets sub-blocks that are currently invalid and a reader only reads
//! sub-blocks it has seen as valid, so they never touch the same bytes. The
//! `valid` bit is flipped `Release` *after* the bytes land and read `Acquire`
//! before use.
//!
//! ## Lookups
//!
//! [`FileCache::get`] pins the region slot (allocating it on a miss) and
//! inspects the bitmap over the requested intra-slot range:
//! - all sub-blocks valid → [`CacheLookup::Hit`] with the data as [`Bytes`];
//! - some missing → [`CacheLookup::Miss`] carrying the slot pin plus the
//!   [`MissingBlock`]s (coalesced contiguous missing runs) the caller must read.
//!   Each block covers *whole* sub-blocks and is read straight into the slot at
//!   [`MissingBlock::dest`]; once its bytes land the caller marks them valid via
//!   [`MissingBlock::commit`].

use crate::io::IOLocation;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::os::fd::RawFd;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Sub-block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range
/// is rounded out to whole sub-blocks.
pub const SUB_BLOCK_SIZE: usize = 4096;
/// Number of 4 KB sub-blocks per 2 MB slot (512).
pub const SUB_BLOCKS_PER_SLOT: usize = BUFFER_SIZE / SUB_BLOCK_SIZE;
/// Number of `u64` words in a slot's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = SUB_BLOCKS_PER_SLOT / 64;

/// Mask the byte `offset` down to its 2 MB region base.
#[inline]
pub fn region_base_of(offset: usize) -> usize {
    offset & !(BUFFER_SIZE - 1)
}

struct FileCacheEntry {
    ring_idx: usize,
}

/// The result of a [`FileCache::get`]: either the requested bytes are all
/// resident, or they are not and the caller is handed the reads needed to fill
/// the holes (plus the pinned slot to keep it alive meanwhile).
pub enum CacheLookup {
    /// Every sub-block in the requested range is valid; here is the data.
    Hit(Bytes),
    /// Some sub-blocks are missing — read & [`MissingBlock::fill`] them, then
    /// take the bytes.
    Miss(MissRegion),
}

/// One contiguous run of missing sub-blocks that must be read from disk to
/// satisfy a lookup. Covers whole sub-blocks, so the read targets [`dest`] (a
/// region inside the pinned slot) directly — no intermediate buffer — and once
/// it lands the run is marked valid wholesale via [`commit`].
///
/// [`dest`]: MissingBlock::dest
/// [`commit`]: MissingBlock::commit
#[derive(Clone)]
pub struct MissingBlock {
    /// File offset to read from (`region_base + first_sub * SUB_BLOCK_SIZE`).
    pub file_offset: usize,
    /// Destination address inside the pinned ring slot (`ptr as usize`).
    dest: usize,
    /// Bytes to read — a multiple of `SUB_BLOCK_SIZE`.
    pub len: usize,
    /// Ring slot index whose validity bitmap this run belongs to.
    slot_idx: usize,
    /// First sub-block index (within the slot) covered by this run.
    first_sub: usize,
    /// How many sub-blocks this run covers.
    sub_count: usize,
}

impl MissingBlock {
    /// The destination to read this block's `len` bytes into: a 4 KB-aligned
    /// region `[dest, dest+len)` inside the pinned slot — a valid O_DIRECT
    /// target. The slot stays alive for the read because the owning `MissRegion`
    /// pins it, and only this block's (currently invalid) sub-blocks live here,
    /// so the read never races a reader of the slot's valid bytes.
    pub fn dest(&self) -> *mut u8 {
        self.dest as *mut u8
    }

    /// Mark this block's sub-blocks valid — call once its bytes have been read
    /// into the slot (`Release` so a reader that observes a valid bit `Acquire`
    /// is guaranteed to see the bytes the read landed).
    pub fn commit(&self) {
        let entry = unsafe { &*memory_ctx().file_cache().entries[self.slot_idx].get() };
        for s in self.first_sub..self.first_sub + self.sub_count {
            entry.valid[s / 64].fetch_or(1u64 << (s % 64), Ordering::Release);
        }
    }
}

/// A pinned region slot with an outstanding set of [`MissingBlock`]s. The
/// caller reads each block into its [`dest`](MissingBlock::dest) and
/// [`commit`](MissingBlock::commit)s it on completion, then takes the assembled
/// [`Bytes`] for its `[start, end)` slice once the whole range is resident.
pub struct MissRegion {
    /// Keeps the slot alive (and un-evictable) until the data is taken.
    pin: ReadBuffer,
    start: usize,
    end: usize,
    missing: Vec<MissingBlock>,
}

impl MissRegion {
    /// The runs that still need to be read & filled to make the range resident.
    pub fn missing_blocks(&self) -> &[MissingBlock] {
        &self.missing
    }

    /// Take the requested `[start, end)` slice. Only valid to call once every
    /// missing block has been filled; the bytes are a zero-copy view into the
    /// slot, kept alive by the owned pin.
    pub fn into_bytes(self) -> Bytes {
        Bytes::from_owner(self.pin).slice(self.start..self.end)
    }
}

/// A CLOCK-eviction region cache over the shared [`Ring`].
pub struct FileCache {
    file_maps: RwLock<HashMap<RawFd, RwLock<HashMap<usize, FileCacheEntry>>>>,
    entries: Box<[UnsafeCell<Entry>]>,
    hand: AtomicUsize,
}

unsafe impl Send for FileCache {}
unsafe impl Sync for FileCache {}

/// Per-slot CLOCK metadata + validity.
///
/// * `ref_bit` – set on access, cleared by the CLOCK sweep.
/// * `region` – the 2 MB region currently cached here, or `None` if unused.
/// * `valid` – which of the slot's 4 KB sub-blocks are resident.
struct Entry {
    ref_bit: AtomicBool,
    region: Option<IOLocation>,
    valid: [AtomicU64; BITMAP_WORDS],
}

impl FileCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            file_maps: Default::default(),
            entries: (0..capacity)
                .map(|_| {
                    UnsafeCell::new(Entry {
                        ref_bit: Default::default(),
                        region: None,
                        valid: Default::default(),
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hand: Default::default(),
        }
    }

    /// Look up `[start, end)` (intra-slot byte offsets) of the 2 MB `region` of
    /// file `fd`, allocating the slot on a miss. See [`CacheLookup`].
    pub fn get(&self, fd: RawFd, region: usize, start: usize, end: usize) -> CacheLookup {
        debug_assert!(start < end && end <= BUFFER_SIZE);
        debug_assert_eq!(region_base_of(region), region);
        let location = IOLocation {
            raw_fd: fd,
            offset: region,
        };
        loop {
            // Fast path: the region is already cached — pin it and inspect.
            if let Some(pin) = self.pin_existing(&location) {
                return self.lookup_pinned(pin, region, start, end);
            }

            // Miss: get a free slot *before* taking the map lock, since
            // get_write_buffer may evict (which locks the cache itself).
            let wb = memory_ctx().get_write_buffer(false);
            let slot_idx = wb.slot_idx;

            // Register region → slot, unless another worker beat us to it.
            let maps = self.file_maps.read().unwrap();
            let inner = maps.get(&fd).expect("Missing file in file cache!");
            let mut inner = inner.write().unwrap();
            if inner.contains_key(&region) {
                // Lost the race: release our (untouched) slot and retry the fast
                // path. We must not have bound `entry.region` yet — otherwise a
                // later eviction of this slot would remove the winner's mapping.
                drop(inner);
                drop(maps);
                drop(wb); // returns the slot to the free pool, entry untouched
                continue;
            }
            // We won. Initialize the slot while it's still exclusively held
            // (WRITING) — bitmap zeroed and region bound before any reader can
            // pin it. `entry.region` is set only now, atomically with the map
            // insert, so a dropped loser never leaves a dangling region behind.
            {
                let entry = unsafe { &mut *self.entries[slot_idx].get() };
                for w in &entry.valid {
                    w.store(0, Ordering::Relaxed);
                }
                entry.region = Some(location.clone());
                entry.ref_bit.store(true, Ordering::Relaxed);
            }
            inner.insert(region, FileCacheEntry { ring_idx: slot_idx });
            // Transition WRITING → readable (1 reader = our pin) while still
            // holding the map lock, so a concurrent `pin_existing` either blocks
            // on the lock or sees a pinnable slot — never a WRITING one.
            let pin = ReadBuffer::from(wb);
            drop(inner);
            drop(maps);
            return self.lookup_pinned(pin, region, start, end);
        }
    }

    /// Pin the slot caching `location`, or `None` if it isn't cached (or a
    /// concurrent eviction recycled it between the map lookup and the pin).
    fn pin_existing(&self, location: &IOLocation) -> Option<ReadBuffer> {
        let maps = self.file_maps.read().unwrap();
        let inner = maps.get(&location.raw_fd).unwrap().read().unwrap();
        let ring_idx = inner.get(&location.offset)?.ring_idx;
        let guard = memory_ctx().ring().try_read(ring_idx)?;
        let entry = unsafe { &*self.entries[guard.ring_idx()].get() };
        if entry.region.as_ref() == Some(location) {
            entry.ref_bit.store(true, Ordering::Relaxed);
            Some(guard)
        } else {
            None
        }
    }

    /// Inspect the bitmap of a pinned slot over `[start, end)` and produce a
    /// `Hit` (all valid) or a `Miss` with the coalesced missing runs.
    fn lookup_pinned(
        &self,
        pin: ReadBuffer,
        region: usize,
        start: usize,
        end: usize,
    ) -> CacheLookup {
        let slot_idx = pin.slot_idx;
        let base = pin.ptr as usize;
        let entry = unsafe { &*self.entries[slot_idx].get() };

        let first = start / SUB_BLOCK_SIZE;
        let last = (end - 1) / SUB_BLOCK_SIZE;

        let mut missing = Vec::new();
        let mut run: Option<usize> = None; // start sub-block of the current missing run
        for s in first..=last {
            let present = entry.valid[s / 64].load(Ordering::Acquire) & (1u64 << (s % 64)) != 0;
            if present {
                if let Some(a) = run.take() {
                    missing.push(make_missing_block(region, base, slot_idx, a, s - 1));
                }
            } else if run.is_none() {
                run = Some(s);
            }
        }
        if let Some(a) = run.take() {
            missing.push(make_missing_block(region, base, slot_idx, a, last));
        }

        if missing.is_empty() {
            CacheLookup::Hit(Bytes::from_owner(pin).slice(start..end))
        } else {
            CacheLookup::Miss(MissRegion {
                pin,
                start,
                end,
                missing,
            })
        }
    }

    /// Evict a slot using the CLOCK algorithm and return it as a writable buffer.
    pub fn evict(&self) -> crate::memory::write_buffer::WriteBuffer {
        loop {
            let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % memory_ctx().ring().len();
            let cell = &self.entries[slot_idx];
            let entry = unsafe { &*cell.get() };
            if entry.ref_bit.swap(false, Ordering::Relaxed) {
                continue;
            }

            if let Some(r) = memory_ctx().ring().try_write(slot_idx) {
                let entry = unsafe { &mut *cell.get() };
                if let Some(l) = entry.region.take() {
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

    /// Register a file descriptor so its regions can be cached. Clears any stale
    /// entries from a previous use of the same fd number.
    pub fn open_file_entry(&self, raw_fd: RawFd) {
        let mut file_maps = self.file_maps.write().unwrap();
        if let Some(old_map) = file_maps.remove(&raw_fd) {
            for (_, entry) in old_map.into_inner().unwrap() {
                let cell = &self.entries[entry.ring_idx];
                let slot = unsafe { &mut *cell.get() };
                slot.region = None;
                drop(memory_ctx().ring().try_write(entry.ring_idx));
            }
        }
        file_maps.insert(raw_fd, Default::default());
    }
}

/// Build a [`MissingBlock`] covering sub-blocks `[a, b]` (inclusive) of a slot.
#[inline]
fn make_missing_block(region: usize, base: usize, slot_idx: usize, a: usize, b: usize) -> MissingBlock {
    MissingBlock {
        file_offset: region + a * SUB_BLOCK_SIZE,
        dest: base + a * SUB_BLOCK_SIZE,
        len: (b - a + 1) * SUB_BLOCK_SIZE,
        slot_idx,
        first_sub: a,
        sub_count: b - a + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    const SB: usize = SUB_BLOCK_SIZE;
    const FD: RawFd = 7;

    fn cache() -> &'static FileCache {
        memory_ctx().file_cache()
    }

    fn miss(lookup: CacheLookup) -> MissRegion {
        match lookup {
            CacheLookup::Miss(m) => m,
            CacheLookup::Hit(_) => panic!("expected a miss"),
        }
    }

    fn hit(lookup: CacheLookup) -> Bytes {
        match lookup {
            CacheLookup::Hit(b) => b,
            CacheLookup::Miss(_) => panic!("expected a hit"),
        }
    }

    /// Simulate a completed read: write `byte` into the block's slot region,
    /// then mark it valid — what the IO path does (read into `dest`, `commit`).
    fn read_and_commit(block: &MissingBlock, byte: u8) {
        unsafe { std::ptr::write_bytes(block.dest(), byte, block.len) };
        block.commit();
    }

    #[test]
    fn region_base_masks_to_two_megabytes() {
        assert_eq!(region_base_of(0), 0);
        assert_eq!(region_base_of(BUFFER_SIZE - 1), 0);
        assert_eq!(region_base_of(BUFFER_SIZE), BUFFER_SIZE);
        assert_eq!(region_base_of(BUFFER_SIZE + 5), BUFFER_SIZE);
    }

    #[test]
    fn miss_then_fill_then_hit() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        // A fresh region: looking up the first sub-block misses with one block.
        let region = miss(cache().get(FD, 0, 0, 10));
        assert_eq!(region.missing_blocks().len(), 1);
        let block = region.missing_blocks()[0].clone();
        assert_eq!((block.first_sub, block.sub_count, block.len), (0, 1, SB));

        // Read a pattern into that sub-block, drop the pin, and look up again.
        read_and_commit(&block, 0xAB);
        drop(region);

        let bytes = hit(cache().get(FD, 0, 0, 10));
        assert_eq!(&bytes[..], &[0xAB; 10]);
    }

    #[test]
    fn coalesces_contiguous_missing_sub_blocks_into_one_block() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        // Range spanning three sub-blocks, none present → a single coalesced run.
        let region = miss(cache().get(FD, 0, 0, 2 * SB + 1));
        assert_eq!(region.missing_blocks().len(), 1);
        let block = &region.missing_blocks()[0];
        assert_eq!((block.first_sub, block.sub_count, block.len), (0, 3, 3 * SB));
    }

    #[test]
    fn partial_validity_only_reports_the_holes() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        // Fill sub-block 0 only.
        let region = miss(cache().get(FD, 0, 0, 10));
        read_and_commit(&region.missing_blocks()[0].clone(), 1);
        drop(region);

        // Now ask for sub-blocks 0..=2; only 1 and 2 are missing, coalesced.
        let region = miss(cache().get(FD, 0, 0, 2 * SB + 1));
        assert_eq!(region.missing_blocks().len(), 1);
        let block = &region.missing_blocks()[0];
        assert_eq!((block.first_sub, block.sub_count), (1, 2));
    }

    #[test]
    fn distinct_regions_use_distinct_slots() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        let r0 = miss(cache().get(FD, 0, 0, 10));
        let r1 = miss(cache().get(FD, BUFFER_SIZE, 0, 10));
        assert_ne!(
            r0.missing_blocks()[0].slot_idx,
            r1.missing_blocks()[0].slot_idx
        );
    }

    #[test]
    fn second_column_in_same_region_shares_the_slot() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        // Two disjoint ranges within the same 2 MB region resolve to one slot.
        let a = miss(cache().get(FD, 0, 0, 10));
        let b = miss(cache().get(FD, 0, 5 * SB, 5 * SB + 10));
        assert_eq!(
            a.missing_blocks()[0].slot_idx,
            b.missing_blocks()[0].slot_idx
        );
        // ...and a's fill doesn't satisfy b's distant sub-block.
        assert_eq!(b.missing_blocks()[0].first_sub, 5);
    }
}
