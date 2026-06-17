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
//! [`FileCache::get`] takes a file byte range `[offset, offset + len)` and
//! returns one [`CacheLookup`] per 2 MB window the range spans (callers never
//! deal in windows themselves). Each lookup carries the window's `data` (a
//! zero-copy view into the cache slot) plus its `missing` runs — the
//! [`MissingBlock`]s the caller must still read. `missing` is empty on a full
//! hit; otherwise each block covers *whole* sub-blocks and is read straight into
//! the slot at [`MissingBlock::dest`], then marked valid via
//! [`MissingBlock::commit`], after which `data` is valid to read.

use crate::io::FileLocation;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Sub-block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range
/// is rounded out to whole sub-blocks.
const SUB_BLOCK_SIZE: usize = 4096;
/// Number of 4 KB sub-blocks per 2 MB slot (512).
const SUB_BLOCKS_PER_SLOT: usize = BUFFER_SIZE / SUB_BLOCK_SIZE;
/// Number of `u64` words in a slot's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = SUB_BLOCKS_PER_SLOT / 64;

/// Mask the byte `offset` down to its 2 MB region base.
#[inline]
fn region_base_of(offset: usize) -> usize {
    offset & !(BUFFER_SIZE - 1)
}

/// Assemble a [`CacheLookup`] for `[start, end)` of a pinned slot caching
/// `region`, given the slot's already-computed missing sub-block `runs`
/// (each `[first, last]` inclusive; empty on a full hit). One shared pin keeps
/// the slot alive for the lookup's `data` *and* every in-flight read targeting
/// it.
fn create_cache_lookup_from_missing(
    buffer: ReadBuffer,
    region: usize,
    start: usize,
    end: usize,
    runs: Vec<(usize, usize)>,
) -> CacheLookup {
    let slot_idx = buffer.slot_idx;
    let slot_ptr = buffer.ptr as usize;

    let buffer = Arc::new(buffer);
    let missing = runs
        .into_iter()
        .map(|(first, last)| {
            MissingBlock::new(region, slot_ptr, slot_idx, first, last, buffer.clone())
        })
        .collect();
    let data = Bytes::from_owner(SlotPin(buffer)).slice(start..end);
    CacheLookup { data, missing }
}

struct FileCacheEntry {
    ring_idx: usize,
}

/// The result of a [`FileCache::get`] over one window: the looked-up bytes
/// (a zero-copy view into the cache slot) plus the runs still missing from the
/// slot. `missing` is empty on a full hit; otherwise `data` only becomes valid
/// once every [`MissingBlock`] has been read & committed.
pub struct CacheLookup {
    data: Bytes,
    missing: Vec<MissingBlock>,
}

impl CacheLookup {
    /// Runs that must be read & committed before [`data`](Self::into_data) is
    /// valid. Empty on a full hit.
    pub fn missing(&self) -> &[MissingBlock] {
        &self.missing
    }

    /// Take the looked-up bytes — valid to read once every [`missing`](Self::missing)
    /// block has been filled. A zero-copy view into the cache slot, kept alive by
    /// the returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// One contiguous run of missing sub-blocks that must be read from disk to
/// satisfy a lookup. Covers whole sub-blocks, so the read targets [`dest`] (a
/// region inside the pinned slot) directly — no intermediate buffer — and once
/// it lands the run is marked valid wholesale via [`commit`].
///
/// Holds an [`Arc`] on the slot pin shared with the lookup's `data`, so the slot
/// can't be evicted while a read targeting it is in flight — even if the owning
/// query is cancelled and drops its [`CacheLookup`] first.
///
/// [`dest`]: MissingBlock::dest
/// [`commit`]: MissingBlock::commit
#[derive(Clone)]
pub struct MissingBlock {
    /// 2 MB region base of the file this run lives in. Combined with
    /// `first_sub_block` it gives the [`file_offset`](Self::file_offset) to read from.
    region: usize,
    /// Destination address inside the pinned ring slot (`ptr as usize`).
    dest: usize,
    /// Ring slot index whose validity bitmap this run belongs to.
    slot_idx: usize,
    /// First sub-block index (within the slot) covered by this run.
    first_sub_block: usize,
    /// How many sub-blocks this run covers.
    sub_block_count: usize,
    /// Keeps the destination slot pinned for the read's whole lifetime.
    _pin: Arc<ReadBuffer>,
}

impl MissingBlock {
    /// A block covering sub-blocks `[first_sub_block, last_sub_block]` (inclusive) of the
    /// slot at `slot_ptr` caching `region`, sharing the region's slot pin.
    fn new(
        region: usize,
        slot_ptr: usize,
        slot_idx: usize,
        first_sub_block: usize,
        last_sub_block: usize,
        pin: Arc<ReadBuffer>,
    ) -> Self {
        MissingBlock {
            region,
            dest: slot_ptr + first_sub_block * SUB_BLOCK_SIZE,
            slot_idx,
            first_sub_block,
            sub_block_count: last_sub_block - first_sub_block + 1,
            _pin: pin,
        }
    }

    /// File offset this block's bytes are read from (always a multiple of
    /// [`SUB_BLOCK_SIZE`], the direct-I/O alignment).
    pub fn file_offset(&self) -> usize {
        self.region + self.first_sub_block * SUB_BLOCK_SIZE
    }

    /// Number of bytes to read — a multiple of `SUB_BLOCK_SIZE`.
    pub fn len(&self) -> usize {
        self.sub_block_count * SUB_BLOCK_SIZE
    }

    /// Whether this block covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.sub_block_count == 0
    }

    /// The destination to read this block's [`len`](Self::len) bytes into: a
    /// 4 KB-aligned region `[dest, dest+len)` inside the pinned slot — a valid
    /// O_DIRECT target. The slot stays alive for the read because this block
    /// holds a pin (`_pin`), and only this block's (currently invalid) sub-blocks
    /// live here, so the read never races a reader of the slot's valid bytes.
    pub fn dest(&self) -> *mut u8 {
        self.dest as *mut u8
    }

    /// Mark this block's sub-blocks valid — call once its bytes have been read
    /// into the slot.
    pub fn commit(&self) {
        memory_ctx()
            .file_cache()
            .entry(self.slot_idx)
            .valid
            .set(self.first_sub_block, self.sub_block_count);
    }
}

/// Owns a slot pin and exposes its 2 MB contents, so a [`Bytes`] can borrow a
/// slice of the slot zero-copy while keeping the slot pinned.
struct SlotPin(Arc<ReadBuffer>);

impl AsRef<[u8]> for SlotPin {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// A slot's validity bitmap: bit `sub_block` set means sub-block `sub_block`
/// (a 4 KB span of the slot) has been read into it. Keeps all the word/bit
/// indexing — and the memory ordering that makes the in-place fill sound — in one place.
#[derive(Default)]
struct ValidBitmap([AtomicU64; BITMAP_WORDS]);

impl ValidBitmap {
    /// Is sub-block `sub_block` present? `Acquire` pairs with `set`'s `Release`,
    /// so a reader that observes the bit is guaranteed to see the sub-block's bytes.
    fn is_set(&self, sub_block: usize) -> bool {
        self.0[sub_block / 64].load(Ordering::Acquire) & (1 << (sub_block % 64)) != 0
    }

    /// Mark sub-blocks `[first, first + count)` present. `Release` so it only
    /// becomes visible after the bytes have landed in the slot.
    fn set(&self, first: usize, count: usize) {
        for sub_block in first..first + count {
            self.0[sub_block / 64].fetch_or(1 << (sub_block % 64), Ordering::Release);
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
/// * `region` – the `(location, region_base)` currently cached here, or `None` if unused.
/// * `valid` – which of the slot's 4 KB sub-blocks are resident.
struct Entry {
    ref_bit: AtomicBool,
    region: Option<(FileLocation, usize)>,
    valid: ValidBitmap,
}

/// A CLOCK-eviction region cache over the shared [`Ring`](super::Ring).
///
/// Regions are bucketed by [`FileLocation`], so the same cache serves both local
/// files and remote HTTP objects — only the transport that fills a missing block
/// differs.
pub struct FileCache {
    file_maps: RwLock<HashMap<FileLocation, RwLock<HashMap<usize, FileCacheEntry>>>>,
    entries: Box<[UnsafeCell<Entry>]>,
    hand: AtomicUsize,
}

unsafe impl Send for FileCache {}
unsafe impl Sync for FileCache {}

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

    /// Look up the file byte range `[offset, offset + len)` of file `fd`. The
    /// range is split across the 2 MB regions it spans, yielding one
    /// [`CacheLookup`] per region in file order — callers never deal in regions
    /// themselves: read & fill every lookup's [`missing`](CacheLookup::missing)
    /// blocks, then concatenate the parts' bytes.
    pub fn get(&self, location: &FileLocation, offset: usize, len: usize) -> Vec<CacheLookup> {
        let end = offset + len;
        let mut lookups = Vec::new();
        let mut at = offset;
        while at < end {
            let region = region_base_of(at);
            let region_end = (region + BUFFER_SIZE).min(end);
            lookups.push(self.get_region(location, region, at - region, region_end - region));
            at = region_end;
        }
        lookups
    }

    /// Look up `[start, end)` (intra-region byte offsets) of one 2 MB `region`
    /// of `location`, allocating the slot on a miss. See [`CacheLookup`].
    fn get_region(
        &self,
        location: &FileLocation,
        region: usize,
        start: usize,
        end: usize,
    ) -> CacheLookup {
        debug_assert!(start < end && end <= BUFFER_SIZE);
        debug_assert_eq!(region_base_of(region), region);
        loop {
            // Fast path: the region is already cached — pin it and inspect.
            if let Some(buffer) = self.get_buffer(location, region) {
                return self.get_lookup_within_existing_region(buffer, region, start, end);
            }

            // Miss: grab a free write buffer *before* taking the map lock, since
            // get_write_buffer may evict (which locks the cache itself).
            let write_buffer = memory_ctx().get_write_buffer(false);
            let slot_idx = write_buffer.slot_idx;

            // Register region → slot, unless another worker beat us to it.
            let file_maps = self.file_maps.read().unwrap();
            let fd_regions = file_maps
                .get(location)
                .expect("Missing file in file cache!");
            let mut fd_regions = fd_regions.write().unwrap();
            if fd_regions.contains_key(&region) {
                // Another worker already cached this region; retry the fast path
                // to pin theirs. Our write buffer was never bound, so it just
                // goes back to the free pool unused.
                continue;
            }
            // We won. While the slot is still exclusively held (WRITING, so no
            // reader can pin it) reset its bitmap and bind its region — set only
            // now, atomically with the map insert, so a dropped loser never
            // leaves a dangling region behind.
            let entry = unsafe { &mut *self.entries[slot_idx].get() };
            entry.valid.clear();
            entry.region = Some((location.clone(), region));
            entry.ref_bit.store(true, Ordering::Relaxed);

            fd_regions.insert(region, FileCacheEntry { ring_idx: slot_idx });
            // Publish as readable (1 reader = our pin) while still holding the
            // map lock, so a racing pin_existing waits on the lock rather than
            // ever seeing a WRITING slot.
            let buffer = ReadBuffer::from(write_buffer);
            drop(fd_regions);
            drop(file_maps);
            // Freshly allocated: we just cleared the bitmap, so the whole
            // requested range is one missing run — no need to walk the bitmap.
            let run = (start / SUB_BLOCK_SIZE, (end - 1) / SUB_BLOCK_SIZE);
            return create_cache_lookup_from_missing(buffer, region, start, end, vec![run]);
        }
    }

    /// Borrow ring slot `idx`'s cache metadata. The atomic `valid`/`ref_bit` are
    /// always safe to touch; `region` is only mutated while the slot is held
    /// exclusively (WRITING), which can't coexist with a read pin, so reading it
    /// under a pin is sound.
    fn entry(&self, idx: usize) -> &Entry {
        unsafe { &*self.entries[idx].get() }
    }

    /// Get the cached buffer for `region` of `location`, or `None` if it isn't
    /// cached (or a concurrent eviction recycled it between the map lookup and
    /// the pin).
    fn get_buffer(&self, location: &FileLocation, region: usize) -> Option<ReadBuffer> {
        let file_maps = self.file_maps.read().unwrap();
        let fd_regions = file_maps.get(location).unwrap().read().unwrap();
        let ring_idx = fd_regions.get(&region)?.ring_idx;
        let buffer = memory_ctx().ring().try_read(ring_idx)?;
        // Re-check under the pin: eviction may have recycled the slot for a
        // different region between the map lookup and the pin. Comparing by
        // reference avoids cloning the (possibly Arc-backed) location.
        let entry = self.entry(buffer.ring_idx());
        if entry
            .region
            .as_ref()
            .is_some_and(|(loc, r)| loc == location && *r == region)
        {
            entry.ref_bit.store(true, Ordering::Relaxed);
            Some(buffer)
        } else {
            None
        }
    }

    /// Inspect the bitmap of a pinned slot over `[start, end)` and build its
    /// [`CacheLookup`]: the `[start, end)` bytes plus any missing runs to read
    /// (none when every sub-block is already valid).
    fn get_lookup_within_existing_region(
        &self,
        buffer: ReadBuffer,
        region: usize,
        start: usize,
        end: usize,
    ) -> CacheLookup {
        let valid = &self.entry(buffer.slot_idx).valid;

        let first_sub_block = start / SUB_BLOCK_SIZE;
        let last_sub_block = (end - 1) / SUB_BLOCK_SIZE;

        // Walk the requested sub-blocks, gathering each maximal run of missing
        // ones (`[run_first, run_last]`) into a single block to read.
        let mut runs = Vec::new();
        let mut run_start: Option<usize> = None;
        for sub_block in first_sub_block..=last_sub_block {
            if valid.is_set(sub_block) {
                if let Some(run_first) = run_start.take() {
                    runs.push((run_first, sub_block - 1));
                }
            } else {
                run_start.get_or_insert(sub_block);
            }
        }
        if let Some(run_first) = run_start.take() {
            runs.push((run_first, last_sub_block));
        }

        create_cache_lookup_from_missing(buffer, region, start, end, runs)
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
                if let Some((location, region)) = entry.region.take() {
                    let file_maps = self.file_maps.read().unwrap();
                    file_maps
                        .get(&location)
                        .unwrap()
                        .write()
                        .unwrap()
                        .remove(&region);
                }
                return write_buffer;
            }
        }
    }

    /// Register a [`FileLocation`] (a local fd or a remote object) so its regions
    /// can be cached. Clears any stale entries from a previous registration of an
    /// equal location (e.g. a reused fd number).
    pub fn open_entry(&self, location: FileLocation) {
        let mut file_maps = self.file_maps.write().unwrap();
        if let Some(stale) = file_maps.remove(&location) {
            for (_, FileCacheEntry { ring_idx }) in stale.into_inner().unwrap() {
                unsafe { &mut *self.entries[ring_idx].get() }.region = None;
                drop(memory_ctx().ring().try_write(ring_idx));
            }
        }
        file_maps.insert(location, Default::default());
    }

    /// Evict every cached region: unbind all `(fd, region) → slot` mappings,
    /// reset each slot's metadata, and return the ring slots to the free pool —
    /// so subsequent reads miss and re-read from disk. Registered file
    /// descriptors stay open (their region maps are just emptied). Returns the
    /// number of regions evicted.
    ///
    /// Intended for benchmarking true cold reads (`SELECT drop_cache()`). Sound
    /// only while no query is in flight: it write-locks and recycles every bound
    /// slot, which must not race a reader holding a pin.
    pub fn clear(&self) -> usize {
        let file_maps = self.file_maps.read().unwrap();
        let mut evicted = 0;
        for fd_regions in file_maps.values() {
            for (_region, FileCacheEntry { ring_idx }) in fd_regions.write().unwrap().drain() {
                let entry = unsafe { &mut *self.entries[ring_idx].get() };
                entry.region = None;
                entry.valid.clear();
                entry.ref_bit.store(false, Ordering::Relaxed);
                // Acquiring then dropping the write buffer returns the slot to
                // the free pool (same recycle path as `open_file_entry`).
                drop(memory_ctx().ring().try_write(ring_idx));
                evicted += 1;
            }
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    const SB: usize = SUB_BLOCK_SIZE;

    /// A local location used as the cache key throughout these tests. One
    /// shared open file, so every call keys the same cache bucket.
    #[allow(non_snake_case)]
    fn FD() -> FileLocation {
        use std::sync::OnceLock;
        static FILE: OnceLock<std::sync::Arc<std::fs::File>> = OnceLock::new();
        FileLocation::Local(
            FILE.get_or_init(|| std::sync::Arc::new(std::fs::File::open("/dev/null").unwrap()))
                .clone(),
        )
    }

    fn cache() -> &'static FileCache {
        memory_ctx().file_cache()
    }

    /// Assert the lookup has holes (some sub-blocks missing) and return it.
    fn miss(lookup: CacheLookup) -> CacheLookup {
        assert!(!lookup.missing().is_empty(), "expected a miss");
        lookup
    }

    /// Assert the lookup is a full hit (no holes) and take its bytes.
    fn hit(lookup: CacheLookup) -> Bytes {
        assert!(lookup.missing().is_empty(), "expected a hit");
        lookup.into_data()
    }

    /// Simulate a completed read: write `byte` into the block's slot region,
    /// then mark it valid — what the IO path does (read into `dest`, `commit`).
    fn read_and_commit(block: &MissingBlock, byte: u8) {
        unsafe { std::ptr::write_bytes(block.dest(), byte, block.len()) };
        block.commit();
    }

    /// A deterministic byte for intra-slot offset `off`, so a filled slot's
    /// contents are predictable for any slice.
    fn pattern_at(off: usize) -> u8 {
        off.wrapping_mul(7).wrapping_add(1) as u8
    }

    /// Read the pattern into every missing block of `miss` and commit it,
    /// deriving each block's intra-slot offset from its public `file_offset`.
    fn fill_pattern(region_base: usize, miss: &CacheLookup) {
        for block in miss.missing() {
            let slot_off = block.file_offset() - region_base;
            for i in 0..block.len() {
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
        cache().open_entry(FD());

        // A fresh region: looking up the first sub-block misses with one block.
        let region = miss(cache().get_region(&FD(), 0, 0, 10));
        assert_eq!(region.missing().len(), 1);
        let block = region.missing()[0].clone();
        assert_eq!(
            (block.first_sub_block, block.sub_block_count, block.len()),
            (0, 1, SB)
        );

        // Read a pattern into that sub-block, drop the pin, and look up again.
        read_and_commit(&block, 0xAB);
        drop(region);

        let bytes = hit(cache().get_region(&FD(), 0, 0, 10));
        assert_eq!(&bytes[..], &[0xAB; 10]);
    }

    #[test]
    fn coalesces_contiguous_missing_sub_blocks_into_one_block() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Range spanning three sub-blocks, none present → a single coalesced run.
        let region = miss(cache().get_region(&FD(), 0, 0, 2 * SB + 1));
        assert_eq!(region.missing().len(), 1);
        let block = &region.missing()[0];
        assert_eq!(
            (block.first_sub_block, block.sub_block_count, block.len()),
            (0, 3, 3 * SB)
        );
    }

    #[test]
    fn partial_validity_only_reports_the_holes() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Fill sub-block 0 only.
        let region = miss(cache().get_region(&FD(), 0, 0, 10));
        read_and_commit(&region.missing()[0].clone(), 1);
        drop(region);

        // Now ask for sub-blocks 0..=2; only 1 and 2 are missing, coalesced.
        let region = miss(cache().get_region(&FD(), 0, 0, 2 * SB + 1));
        assert_eq!(region.missing().len(), 1);
        let block = &region.missing()[0];
        assert_eq!((block.first_sub_block, block.sub_block_count), (1, 2));
    }

    #[test]
    fn distinct_regions_use_distinct_slots() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        let r0 = miss(cache().get_region(&FD(), 0, 0, 10));
        let r1 = miss(cache().get_region(&FD(), BUFFER_SIZE, 0, 10));
        assert_ne!(r0.missing()[0].slot_idx, r1.missing()[0].slot_idx);
    }

    #[test]
    fn second_column_in_same_region_shares_the_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Two disjoint ranges within the same 2 MB region resolve to one slot.
        let a = miss(cache().get_region(&FD(), 0, 0, 10));
        let b = miss(cache().get_region(&FD(), 0, 5 * SB, 5 * SB + 10));
        assert_eq!(a.missing()[0].slot_idx, b.missing()[0].slot_idx);
        // ...and a's fill doesn't satisfy b's distant sub-block.
        assert_eq!(b.missing()[0].first_sub_block, 5);
    }

    /// A hit returns the exact byte contents that were read in, reassembled
    /// across several sub-blocks.
    #[test]
    fn hit_reassembles_bytes_across_sub_blocks() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let region = miss(cache().get_region(&FD(), 0, SB - 5, 2 * SB + 5));
        fill_pattern(0, &region);
        drop(region);

        let bytes = hit(cache().get_region(&FD(), 0, SB - 5, 2 * SB + 5));

        assert_pattern(&bytes, SB - 5);
        assert_eq!(bytes.len(), SB + 10);
    }

    /// A lookup returns exactly its requested `[start, end)` byte sub-slice, not
    /// the whole (4 KB-rounded) sub-block it lives in.
    #[test]
    fn hit_returns_the_exact_requested_byte_range() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(0, &miss(cache().get_region(&FD(), 0, 0, SB)));

        let bytes = hit(cache().get_region(&FD(), 0, 3, 9));

        assert_eq!(bytes.len(), 6);
        assert_pattern(&bytes, 3);
    }

    /// Bytes committed by one lookup serve a later, differently-aligned lookup
    /// that overlaps them.
    #[test]
    fn committed_bytes_serve_an_overlapping_later_lookup() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let first = miss(cache().get_region(&FD(), 0, 0, 2 * SB));
        fill_pattern(0, &first);
        drop(first);

        let bytes = hit(cache().get_region(&FD(), 0, SB / 2, SB + SB / 2));

        assert_pattern(&bytes, SB / 2);
    }

    /// The real IO path: take the bytes straight off the `CacheLookup` after
    /// filling its own blocks (no second lookup).
    #[test]
    fn into_data_yields_the_freshly_filled_range() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        let region = miss(cache().get_region(&FD(), 0, 10, SB + 10));
        fill_pattern(0, &region);
        let bytes = region.into_data();

        assert_eq!(bytes.len(), SB);
        assert_pattern(&bytes, 10);
    }

    /// Validity gaps split into separate runs, and filling them all makes the
    /// whole range hit with the right bytes.
    #[test]
    fn interleaved_gaps_split_then_fill_to_a_full_hit() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(0, &miss(cache().get_region(&FD(), 0, 0, 1))); // sub-block 0
        fill_pattern(0, &miss(cache().get_region(&FD(), 0, 2 * SB, 2 * SB + 1))); // sub-block 2

        let gaps = miss(cache().get_region(&FD(), 0, 0, 4 * SB));
        fill_pattern(0, &gaps);
        drop(gaps);
        let bytes = hit(cache().get_region(&FD(), 0, 0, 4 * SB));

        assert_pattern(&bytes, 0);
    }

    /// Re-registering a file descriptor forgets everything previously cached for
    /// it, so the next lookup re-reads.
    #[test]
    fn reopening_a_file_forgets_its_cached_regions() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(0, &miss(cache().get_region(&FD(), 0, 0, SB)));

        cache().open_entry(FD());

        assert!(!cache().get_region(&FD(), 0, 0, SB).missing().is_empty());
    }

    /// The public range API hides bucketing: a range within one 2 MB window is a
    /// single lookup; one straddling a window boundary splits into two, in order.
    #[test]
    fn get_splits_a_range_across_buckets() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        let within_one = cache().get(&FD(), 100, SB);
        let across_two = cache().get(&FD(), BUFFER_SIZE - SB, 2 * SB);

        assert_eq!(within_one.len(), 1);
        assert_eq!(across_two.len(), 2);
    }

    /// A range straddling a *cached* window and an *uncached* one: the first part
    /// hits, the second misses, and assembling them yields the right bytes.
    #[test]
    fn get_spanning_cached_and_uncached_windows() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(
            0,
            &miss(cache().get_region(&FD(), 0, BUFFER_SIZE - SB, BUFFER_SIZE)),
        );

        let parts = cache().get(&FD(), BUFFER_SIZE - SB, 2 * SB);
        assert!(parts[0].missing().is_empty()); // cached window: full hit
        assert!(!parts[1].missing().is_empty()); // uncached window: has holes
        fill_pattern(BUFFER_SIZE, &parts[1]);

        let mut it = parts.into_iter();
        let hit = it.next().unwrap().into_data();
        let filled = it.next().unwrap().into_data();
        assert_pattern(&hit, BUFFER_SIZE - SB);
        assert_pattern(&filled, 0);
    }

    /// One window with *some sub-blocks cached and some not* is a single lookup
    /// that reports only the holes in `missing`; its `into_data` returns the
    /// cached bytes and the freshly-filled bytes together.
    #[test]
    fn a_partially_cached_window_is_one_miss_that_keeps_the_cached_bytes() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        read_and_commit(
            &miss(cache().get_region(&FD(), 0, 0, 1)).missing()[0].clone(),
            0xAA,
        );

        let m = miss(cache().get_region(&FD(), 0, 0, 2 * SB));
        assert_eq!(m.missing().len(), 1);
        assert_eq!(m.missing()[0].file_offset(), SB); // only the hole (sub-block 1)
        read_and_commit(&m.missing()[0].clone(), 0xBB);
        let bytes = m.into_data();

        assert!(bytes[..SB].iter().all(|&b| b == 0xAA)); // cached sub-block kept
        assert!(bytes[SB..].iter().all(|&b| b == 0xBB)); // hole freshly filled
    }
}
