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

    /// Pin the destination slot for the lifetime of this block's read, so it
    /// can't be evicted out from under an in-flight read — e.g. if the owning
    /// query is cancelled and drops its [`MissRegion`] (and thus its pin) before
    /// the read lands. The slot is already pinned by that `MissRegion` when this
    /// is called, so `try_read` cannot fail.
    pub fn pin(&self) -> ReadBuffer {
        memory_ctx()
            .ring()
            .try_read(self.slot_idx)
            .expect("missing-block slot is pinned by its MissRegion")
    }

    /// Mark this block's sub-blocks valid — call once its bytes have been read
    /// into the slot.
    pub fn commit(&self) {
        memory_ctx()
            .file_cache()
            .entry(self.slot_idx)
            .valid
            .set(self.first_sub, self.sub_count);
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

/// A slot's validity bitmap: bit `sub` set means sub-block `sub` (a 4 KB chunk)
/// has been read into the slot. Keeps all the word/bit indexing — and the
/// memory ordering that makes the in-place fill sound — in one place.
#[derive(Default)]
struct ValidBitmap([AtomicU64; BITMAP_WORDS]);

impl ValidBitmap {
    /// Is sub-block `sub` present? `Acquire` pairs with `set`'s `Release`, so a
    /// reader that observes the bit is guaranteed to see the sub-block's bytes.
    fn is_set(&self, sub: usize) -> bool {
        self.0[sub / 64].load(Ordering::Acquire) & (1 << (sub % 64)) != 0
    }

    /// Mark sub-blocks `[first, first + count)` present. `Release` so it only
    /// becomes visible after the bytes have landed in the slot.
    fn set(&self, first: usize, count: usize) {
        for sub in first..first + count {
            self.0[sub / 64].fetch_or(1 << (sub % 64), Ordering::Release);
        }
    }

    /// Reset every sub-block to absent. Sound only while the slot is held
    /// exclusively (no concurrent readers).
    fn clear(&self) {
        for word in &self.0 {
            word.store(0, Ordering::Relaxed);
        }
    }
}

/// Per-slot CLOCK metadata + validity.
///
/// * `ref_bit` – set on access, cleared by the CLOCK sweep.
/// * `region` – the 2 MB region currently cached here, or `None` if unused.
/// * `valid` – which of the slot's 4 KB sub-blocks are resident.
struct Entry {
    ref_bit: AtomicBool,
    region: Option<IOLocation>,
    valid: ValidBitmap,
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

            // Miss: grab a free slot *before* taking the map lock, since
            // get_write_buffer may evict (which locks the cache itself).
            let slot = memory_ctx().get_write_buffer(false);
            let slot_idx = slot.slot_idx;

            // Register region → slot, unless another worker beat us to it.
            let file_maps = self.file_maps.read().unwrap();
            let fd_regions = file_maps.get(&fd).expect("Missing file in file cache!");
            let mut fd_regions = fd_regions.write().unwrap();
            if fd_regions.contains_key(&region) {
                // Lost the race: drop our untouched slot and retry the fast path.
                // We must not have bound the slot's region yet — otherwise a
                // later eviction of it would remove the winner's mapping.
                drop(fd_regions);
                drop(file_maps);
                drop(slot); // returns the slot to the free pool, entry untouched
                continue;
            }
            // We won. While the slot is still exclusively held (WRITING, so no
            // reader can pin it) reset its bitmap and bind its region — set only
            // now, atomically with the map insert, so a dropped loser never
            // leaves a dangling region behind.
            let entry = unsafe { &mut *self.entries[slot_idx].get() };
            entry.valid.clear();
            entry.region = Some(location.clone());
            entry.ref_bit.store(true, Ordering::Relaxed);

            fd_regions.insert(region, FileCacheEntry { ring_idx: slot_idx });
            // Publish as readable (1 reader = our pin) while still holding the
            // map lock, so a racing pin_existing waits on the lock rather than
            // ever seeing a WRITING slot.
            let pin = ReadBuffer::from(slot);
            drop(fd_regions);
            drop(file_maps);
            return self.lookup_pinned(pin, region, start, end);
        }
    }

    /// Borrow ring slot `idx`'s cache metadata. The atomic `valid`/`ref_bit` are
    /// always safe to touch; `region` is only mutated while the slot is held
    /// exclusively (WRITING), which can't coexist with a read pin, so reading it
    /// under a pin is sound.
    fn entry(&self, idx: usize) -> &Entry {
        unsafe { &*self.entries[idx].get() }
    }

    /// Pin the slot caching `location`, or `None` if it isn't cached (or a
    /// concurrent eviction recycled it between the map lookup and the pin).
    fn pin_existing(&self, location: &IOLocation) -> Option<ReadBuffer> {
        let file_maps = self.file_maps.read().unwrap();
        let fd_regions = file_maps.get(&location.raw_fd).unwrap().read().unwrap();
        let ring_idx = fd_regions.get(&location.offset)?.ring_idx;
        let pin = memory_ctx().ring().try_read(ring_idx)?;
        // Re-check under the pin: eviction may have recycled the slot for a
        // different region between the map lookup and the pin.
        let entry = self.entry(pin.ring_idx());
        if entry.region.as_ref() == Some(location) {
            entry.ref_bit.store(true, Ordering::Relaxed);
            Some(pin)
        } else {
            None
        }
    }

    /// Inspect the bitmap of a pinned slot over `[start, end)` and produce a
    /// `Hit` (all valid) or a `Miss` listing the missing runs to read.
    fn lookup_pinned(
        &self,
        pin: ReadBuffer,
        region: usize,
        start: usize,
        end: usize,
    ) -> CacheLookup {
        let slot_idx = pin.slot_idx;
        let slot_ptr = pin.ptr as usize;
        let valid = &self.entry(slot_idx).valid;

        let first_sub = start / SUB_BLOCK_SIZE;
        let last_sub = (end - 1) / SUB_BLOCK_SIZE;

        // Walk the requested sub-blocks, gathering each maximal run of missing
        // ones into a single block to read.
        let mut missing = Vec::new();
        let mut run_start: Option<usize> = None;
        for sub in first_sub..=last_sub {
            if valid.is_set(sub) {
                if let Some(run_first) = run_start.take() {
                    missing.push(MissingBlock::new(region, slot_ptr, slot_idx, run_first, sub - 1));
                }
            } else {
                run_start.get_or_insert(sub);
            }
        }
        if let Some(run_first) = run_start.take() {
            missing.push(MissingBlock::new(region, slot_ptr, slot_idx, run_first, last_sub));
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

    /// Evict a slot using the (second-chance) CLOCK algorithm and return it as a
    /// writable buffer.
    pub fn evict(&self) -> crate::memory::write_buffer::WriteBuffer {
        loop {
            let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % memory_ctx().ring().len();
            // Give recently-used slots a second chance: clear the ref bit and
            // skip; an unreferenced slot we can write-lock gets evicted.
            if self.entry(slot_idx).ref_bit.swap(false, Ordering::Relaxed) {
                continue;
            }
            if let Some(write_buffer) = memory_ctx().ring().try_write(slot_idx) {
                // Exclusive now: unbind the region from the map before reusing.
                let entry = unsafe { &mut *self.entries[slot_idx].get() };
                if let Some(region) = entry.region.take() {
                    let file_maps = self.file_maps.read().unwrap();
                    file_maps.get(&region.raw_fd).unwrap().write().unwrap().remove(&region.offset);
                }
                return write_buffer;
            }
        }
    }

    /// Register a file descriptor so its regions can be cached. Clears any stale
    /// entries from a previous use of the same fd number.
    pub fn open_file_entry(&self, raw_fd: RawFd) {
        let mut file_maps = self.file_maps.write().unwrap();
        if let Some(stale) = file_maps.remove(&raw_fd) {
            for (_, FileCacheEntry { ring_idx }) in stale.into_inner().unwrap() {
                unsafe { &mut *self.entries[ring_idx].get() }.region = None;
                drop(memory_ctx().ring().try_write(ring_idx));
            }
        }
        file_maps.insert(raw_fd, Default::default());
    }
}

impl MissingBlock {
    /// A block covering sub-blocks `[first_sub, last_sub]` (inclusive) of the
    /// slot at `slot_ptr` caching `region`.
    fn new(
        region: usize,
        slot_ptr: usize,
        slot_idx: usize,
        first_sub: usize,
        last_sub: usize,
    ) -> Self {
        let sub_count = last_sub - first_sub + 1;
        MissingBlock {
            file_offset: region + first_sub * SUB_BLOCK_SIZE,
            dest: slot_ptr + first_sub * SUB_BLOCK_SIZE,
            len: sub_count * SUB_BLOCK_SIZE,
            slot_idx,
            first_sub,
            sub_count,
        }
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

    /// A deterministic byte for intra-slot offset `off`, so a filled slot's
    /// contents are predictable for any slice.
    fn pattern_at(off: usize) -> u8 {
        off.wrapping_mul(7).wrapping_add(1) as u8
    }

    /// Read the pattern into every missing block of `miss` and commit it,
    /// deriving each block's intra-slot offset from its public `file_offset`.
    fn fill_pattern(region_base: usize, miss: &MissRegion) {
        for block in miss.missing_blocks() {
            let slot_off = block.file_offset - region_base;
            for i in 0..block.len {
                unsafe { *block.dest().add(i) = pattern_at(slot_off + i) };
            }
            block.commit();
        }
    }

    fn assert_pattern(bytes: &[u8], slot_start: usize) {
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, pattern_at(slot_start + i), "byte {i}");
        }
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

    /// A hit returns the exact byte contents that were read in, reassembled
    /// across several sub-blocks.
    #[test]
    fn hit_reassembles_bytes_across_sub_blocks() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);
        let region = miss(cache().get(FD, 0, SB - 5, 2 * SB + 5));
        fill_pattern(0, &region);
        drop(region);

        let bytes = hit(cache().get(FD, 0, SB - 5, 2 * SB + 5));

        assert_pattern(&bytes, SB - 5);
        assert_eq!(bytes.len(), SB + 10);
    }

    /// A lookup returns exactly its requested `[start, end)` byte sub-slice, not
    /// the whole (4 KB-rounded) sub-block it lives in.
    #[test]
    fn hit_returns_the_exact_requested_byte_range() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);
        fill_pattern(0, &miss(cache().get(FD, 0, 0, SB)));

        let bytes = hit(cache().get(FD, 0, 3, 9));

        assert_eq!(bytes.len(), 6);
        assert_pattern(&bytes, 3);
    }

    /// Bytes committed by one lookup serve a later, differently-aligned lookup
    /// that overlaps them.
    #[test]
    fn committed_bytes_serve_an_overlapping_later_lookup() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);
        let first = miss(cache().get(FD, 0, 0, 2 * SB));
        fill_pattern(0, &first);
        drop(first);

        let bytes = hit(cache().get(FD, 0, SB / 2, SB + SB / 2));

        assert_pattern(&bytes, SB / 2);
    }

    /// The real IO path: take the bytes straight off the `MissRegion` after
    /// filling its own blocks (no second lookup).
    #[test]
    fn into_bytes_yields_the_freshly_filled_range() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);

        let region = miss(cache().get(FD, 0, 10, SB + 10));
        fill_pattern(0, &region);
        let bytes = region.into_bytes();

        assert_eq!(bytes.len(), SB);
        assert_pattern(&bytes, 10);
    }

    /// Validity gaps split into separate runs, and filling them all makes the
    /// whole range hit with the right bytes.
    #[test]
    fn interleaved_gaps_split_then_fill_to_a_full_hit() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);
        fill_pattern(0, &miss(cache().get(FD, 0, 0, 1))); // sub-block 0
        fill_pattern(0, &miss(cache().get(FD, 0, 2 * SB, 2 * SB + 1))); // sub-block 2

        let gaps = miss(cache().get(FD, 0, 0, 4 * SB));
        fill_pattern(0, &gaps);
        drop(gaps);
        let bytes = hit(cache().get(FD, 0, 0, 4 * SB));

        assert_pattern(&bytes, 0);
    }

    /// Re-registering a file descriptor forgets everything previously cached for
    /// it, so the next lookup re-reads.
    #[test]
    fn reopening_a_file_forgets_its_cached_regions() {
        init_test_free_pool(8);
        cache().open_file_entry(FD);
        fill_pattern(0, &miss(cache().get(FD, 0, 0, SB)));

        cache().open_file_entry(FD);

        assert!(matches!(cache().get(FD, 0, 0, SB), CacheLookup::Miss(_)));
    }
}
