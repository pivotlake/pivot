//! A CLOCK-based page cache that packs many small file reads into shared 2 MB
//! ring buffers, with validity tracked at 4 KB sub-block granularity.
//!
//! ## Why pack
//!
//! A cached read can be far smaller than a ring buffer (a dictionary-encoded
//! column chunk is a few KB). Giving each such read its own 2 MB buffer would let
//! a working set of thousands of skinny reads exhaust the ring while using almost
//! none of its bytes. Instead, a worker *bump-allocates* each read as a
//! contiguous run of 4 KB sub-blocks inside a shared buffer, so hundreds of
//! unrelated reads share one buffer and the working set is measured in bytes, not
//! buffers.
//!
//! ## Model
//!
//! A cached read is a [`CachedRun`]: a contiguous span of file sub-blocks (within
//! one 2 MB-aligned file region) mapped to a contiguous span of sub-blocks inside
//! one ring buffer. The cache index is keyed by [`FileLocation`] then file
//! region, and each region holds the runs cached for it.
//!
//! The **buffer is the unit of eviction**. The CLOCK hand sweeps buffers; when it
//! reclaims one it drops every run packed in it (its [`BufferTenant`]s) from the
//! index. A buffer carries a monotonic generation that is bumped on each
//! reclaim, so a reader holding a stale [`CachedRun`] copy detects the recycle
//! after it pins the buffer (see [`generation`](BufferMeta::generation)).
//!
//! ## Filling
//!
//! Each worker keeps a private *fill buffer* (a pinned [`ReadBuffer`], see
//! [`crate::memory::context::MemoryContext`]) and bump-allocates runs into it.
//! Because the worker pins it, the CLOCK hand cannot reclaim it mid-fill; other
//! workers can still read its already-committed sub-blocks. When it fills up the
//! worker drops the pin and the buffer becomes an ordinary evictable cache
//! buffer. A fill never spans two buffers: a region read is at most 2 MB (callers
//! split larger reads at region boundaries), so any run fits a fresh buffer.
//!
//! ## Lookups
//!
//! [`FileCache::get`] takes a file byte range and returns one [`CacheLookup`] per
//! 2 MB region the range spans. Each lookup carries the bytes (a zero-copy view
//! into the run's buffer) plus the [`MissingBlock`]s still to be read. `missing`
//! is empty on a full hit; otherwise each block is read straight into its buffer
//! at [`MissingBlock::dest`] and marked present via [`MissingBlock::commit`].

use crate::io::FileLocation;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Sub-block granularity for validity tracking and direct-I/O reads. A read never
/// pulls less than this, and every cached byte range is rounded out to whole
/// sub-blocks.
const SUB_BLOCK_SIZE: usize = 4096;
/// Number of 4 KB sub-blocks per 2 MB buffer (512).
pub(crate) const SUB_BLOCKS_PER_BUFFER: usize = BUFFER_SIZE / SUB_BLOCK_SIZE;
/// Number of `u64` words in a buffer's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = SUB_BLOCKS_PER_BUFFER / 64;

/// Mask `offset` down to its 2 MB region base.
#[inline]
fn region_base_of(offset: usize) -> usize {
    offset & !(BUFFER_SIZE - 1)
}

/// First sub-block index (within a region) of a region-relative byte `offset`.
#[inline]
fn sub_block_of(offset: usize) -> usize {
    offset / SUB_BLOCK_SIZE
}

/// A buffer's validity bitmap: bit `sub_block` set means that 4 KB sub-block of
/// the buffer has been read into it. Keeps all the word/bit indexing — and the
/// memory ordering that makes the in-place fill sound — in one place.
#[derive(Default)]
struct ValidBitmap([AtomicU64; BITMAP_WORDS]);

impl ValidBitmap {
    /// Is `sub_block` present? `Acquire` pairs with `set`'s `Release`, so a reader
    /// that observes the bit is guaranteed to see the sub-block's bytes.
    fn is_set(&self, sub_block: usize) -> bool {
        self.0[sub_block / 64].load(Ordering::Acquire) & (1 << (sub_block % 64)) != 0
    }

    /// Mark sub-blocks `[first, first + count)` present. `Release` so they only
    /// become visible after the bytes have landed in the buffer.
    fn set(&self, first: usize, count: usize) {
        for sub_block in first..first + count {
            self.0[sub_block / 64].fetch_or(1 << (sub_block % 64), Ordering::Release);
        }
    }

    /// Reset every sub-block to absent. Sound only while the buffer is held
    /// exclusively (no concurrent readers).
    fn clear(&self) {
        for word in &self.0 {
            word.store(0, Ordering::Relaxed);
        }
    }
}

/// Per-buffer cache metadata. Every field is atomic, so it is sound to read
/// through a shared reference while the buffer is pinned for reading.
///
/// * `reference_bit` — set on access, cleared by the CLOCK sweep (second chance).
/// * `generation` — bumped on each eviction; a [`CachedRun`] records the
///   generation it was cached under so a reader can detect a recycle that
///   happened after it copied the run out of the index but before it pinned the
///   buffer.
/// * `valid` — which of the buffer's 512 sub-blocks currently hold data.
struct BufferMeta {
    reference_bit: AtomicBool,
    generation: AtomicU64,
    valid: ValidBitmap,
}

impl Default for BufferMeta {
    fn default() -> Self {
        BufferMeta {
            reference_bit: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            valid: ValidBitmap::default(),
        }
    }
}

/// A cached read: a span of file sub-blocks (relative to its region base) mapped
/// to a span of buffer sub-blocks, tagged with the buffer generation it was
/// cached under.
#[derive(Clone)]
struct CachedRun {
    /// First file sub-block of the run, relative to the region base.
    file_first_sub_block: u32,
    /// Number of contiguous sub-blocks the run covers.
    sub_block_count: u32,
    /// Ring buffer holding the run's bytes.
    buffer_index: u32,
    /// First sub-block of the run within that buffer.
    buffer_first_sub_block: u32,
    /// Buffer generation when the run was cached (staleness check after pinning).
    generation: u64,
}

impl CachedRun {
    /// One past the run's last file sub-block.
    fn file_end_sub_block(&self) -> u32 {
        self.file_first_sub_block + self.sub_block_count
    }

    /// Does this run cover the whole file sub-block range `[first, last]`?
    fn covers(&self, first: u32, last: u32) -> bool {
        self.file_first_sub_block <= first && last < self.file_end_sub_block()
    }
}

/// A run packed into a buffer, recorded so eviction can drop the run's index
/// entry when it reclaims the buffer.
struct BufferTenant {
    location: FileLocation,
    region: usize,
    file_first_sub_block: u32,
}

/// The result of a [`FileCache::get`] over one region: the looked-up bytes (a
/// zero-copy view into the run's buffer) plus the runs still missing. `missing`
/// is empty on a full hit; otherwise `data` only becomes valid once every
/// [`MissingBlock`] has been read and committed.
pub struct CacheLookup {
    data: Bytes,
    missing: Vec<MissingBlock>,
}

impl CacheLookup {
    /// Runs that must be read and committed before [`data`](Self::into_data) is
    /// valid. Empty on a full hit.
    pub fn missing(&self) -> &[MissingBlock] {
        &self.missing
    }

    /// Take the looked-up bytes — valid to read once every
    /// [`missing`](Self::missing) block has been filled. A zero-copy view into the
    /// run's buffer, kept alive by the returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// One contiguous run of missing sub-blocks to read from storage. Covers whole
/// sub-blocks, so the read targets [`dest`](Self::dest) (a region inside the
/// pinned buffer) directly — no intermediate buffer — and once it lands the run
/// is marked valid wholesale via [`commit`](Self::commit).
///
/// Holds an [`Arc`] on the buffer pin shared with the lookup's `data`, so the
/// buffer cannot be evicted while a read targeting it is in flight, even if the
/// owning query is cancelled and drops its [`CacheLookup`] first.
#[derive(Clone)]
pub struct MissingBlock {
    /// File offset to read these bytes from (a multiple of [`SUB_BLOCK_SIZE`]).
    file_offset: usize,
    /// Destination address inside the pinned buffer (`ptr as usize`).
    dest: usize,
    /// Buffer holding the destination, whose validity bitmap this run belongs to.
    buffer_index: usize,
    /// First sub-block within the buffer covered by this run (for `commit`).
    buffer_first_sub_block: usize,
    /// How many sub-blocks this run covers.
    sub_block_count: usize,
    /// Keeps the destination buffer pinned for the read's whole lifetime.
    _pin: Arc<ReadBuffer>,
}

impl MissingBlock {
    /// File offset this block's bytes are read from (a multiple of
    /// [`SUB_BLOCK_SIZE`], the direct-I/O alignment).
    pub fn file_offset(&self) -> usize {
        self.file_offset
    }

    /// Number of bytes to read — a multiple of [`SUB_BLOCK_SIZE`].
    pub fn len(&self) -> usize {
        self.sub_block_count * SUB_BLOCK_SIZE
    }

    /// Whether this block covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.sub_block_count == 0
    }

    /// Carve out a sub-block covering `[relative_offset, relative_offset + len)`
    /// *within* this block — both relative to this block's start and both
    /// [`SUB_BLOCK_SIZE`]-aligned. The sub-block shares the buffer pin, reads into
    /// the matching slice of the same pinned buffer, and
    /// [`commit`](Self::commit)s only its own sub-blocks.
    ///
    /// Used to split one missing run across transports: bytes already in the local
    /// disk cache are read from there, the holes are fetched over HTTP, and each
    /// part fills its own slice of the buffer.
    pub fn carve_sub_block(&self, relative_offset: usize, len: usize) -> MissingBlock {
        debug_assert_eq!(relative_offset % SUB_BLOCK_SIZE, 0);
        debug_assert_eq!(len % SUB_BLOCK_SIZE, 0);
        debug_assert!(relative_offset + len <= self.len());
        MissingBlock {
            file_offset: self.file_offset + relative_offset,
            dest: self.dest + relative_offset,
            buffer_index: self.buffer_index,
            buffer_first_sub_block: self.buffer_first_sub_block + relative_offset / SUB_BLOCK_SIZE,
            sub_block_count: len / SUB_BLOCK_SIZE,
            _pin: self._pin.clone(),
        }
    }

    /// The destination to read this block's [`len`](Self::len) bytes into: a 4 KB
    /// aligned region `[dest, dest + len)` inside the pinned buffer, a valid
    /// O_DIRECT target. The buffer stays alive for the read because this block
    /// holds a pin (`_pin`), and only this block's (currently invalid) sub-blocks
    /// live here, so the read never races a reader of the buffer's valid bytes.
    pub fn dest(&self) -> *mut u8 {
        self.dest as *mut u8
    }

    /// Mark this block's sub-blocks valid — call once its bytes have been read
    /// into the buffer.
    pub fn commit(&self) {
        memory_ctx()
            .file_cache()
            .buffer_meta(self.buffer_index)
            .valid
            .set(self.buffer_first_sub_block, self.sub_block_count);
    }
}

/// Owns a buffer pin and exposes its 2 MB contents, so a [`Bytes`] can borrow a
/// slice of the buffer zero-copy while keeping the buffer pinned.
struct BufferPin(Arc<ReadBuffer>);

impl AsRef<[u8]> for BufferPin {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// A CLOCK-eviction page cache over the shared [`Ring`](super::Ring), packing
/// many file reads per 2 MB buffer.
pub struct FileCache {
    /// The cache index: file → region → the runs cached for that region.
    index: RwLock<HashMap<FileLocation, RwLock<HashMap<usize, Vec<CachedRun>>>>>,
    /// Per-buffer cache metadata (CLOCK bit, generation, validity).
    buffer_metas: Box<[BufferMeta]>,
    /// Per-buffer tenant lists (the runs packed into each buffer), used to drop
    /// those runs from the index when the buffer is reclaimed.
    ///
    /// Accessed only by a buffer's filling worker (while it holds the pin) and
    /// then by the evictor (after `try_write` succeeds, which can only happen once
    /// the fill pin is dropped). The buffer's `used` atomic orders the hand-off,
    /// so the two never touch the `Vec` concurrently and no lock is needed.
    buffer_tenants: Box<[UnsafeCell<Vec<BufferTenant>>]>,
    /// CLOCK hand.
    hand: AtomicUsize,
}

unsafe impl Send for FileCache {}
unsafe impl Sync for FileCache {}

impl FileCache {
    /// A cache over `buffer_count` ring buffers (one [`BufferMeta`] and tenant
    /// list per buffer).
    pub fn new(buffer_count: usize) -> Self {
        FileCache {
            index: RwLock::new(HashMap::default()),
            buffer_metas: (0..buffer_count).map(|_| BufferMeta::default()).collect(),
            buffer_tenants: (0..buffer_count)
                .map(|_| UnsafeCell::new(Vec::new()))
                .collect(),
            hand: AtomicUsize::new(0),
        }
    }

    /// Borrow buffer `index`'s metadata. The atomic fields are always safe to
    /// touch through a shared reference.
    fn buffer_meta(&self, index: usize) -> &BufferMeta {
        &self.buffer_metas[index]
    }

    /// Look up the file byte range `[offset, offset + len)` of `location`. The
    /// range is split across the 2 MB regions it spans, yielding one
    /// [`CacheLookup`] per region in file order — callers never deal in regions
    /// themselves: read and fill every lookup's [`missing`](CacheLookup::missing)
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

    /// Look up `[start, end)` (region-relative byte offsets) of one 2 MB `region`
    /// of `location`, caching it on a miss. See [`CacheLookup`].
    fn get_region(
        &self,
        location: &FileLocation,
        region: usize,
        start: usize,
        end: usize,
    ) -> CacheLookup {
        debug_assert!(start < end && end <= BUFFER_SIZE);
        debug_assert_eq!(region_base_of(region), region);

        let first_sub_block = sub_block_of(start) as u32;
        let last_sub_block = sub_block_of(end - 1) as u32;

        loop {
            // Fast path: a cached run already covers the whole requested range.
            if let Some(lookup) =
                self.resolve_cached_run(location, region, start, end, first_sub_block, last_sub_block)
            {
                return lookup;
            }

            // Miss: bump-allocate one run covering the whole requested sub-block
            // span in this worker's fill buffer, *before* taking the index lock
            // (acquiring a fill buffer may evict, which locks the cache itself).
            let sub_block_count = (last_sub_block - first_sub_block + 1) as usize;
            let run_placement = memory_ctx().acquire_fill_run(sub_block_count);

            let index = self.index.read().unwrap();
            let Some(file_regions) = index.get(location) else {
                // The file's entry was pruned (its last run was evicted) after it
                // was registered. Re-create it and retry; the bytes we reserved in
                // the fill buffer are simply left unused (reclaimed when the buffer
                // is evicted). The fresh entry is empty, so the evictor — which
                // only prunes on eviction — cannot race it away before we insert.
                drop(index);
                self.index
                    .write()
                    .unwrap()
                    .entry(location.clone())
                    .or_default();
                continue;
            };
            let mut file_regions = file_regions.write().unwrap();
            let region_runs = file_regions.entry(region).or_default();
            if region_runs
                .iter()
                .any(|run| run.covers(first_sub_block, last_sub_block))
            {
                // Another worker cached a covering run while we were allocating;
                // drop ours (the reserved bytes go unused) and retry the fast path.
                continue;
            }

            let cached_run = CachedRun {
                file_first_sub_block: first_sub_block,
                sub_block_count: sub_block_count as u32,
                buffer_index: run_placement.buffer_index as u32,
                buffer_first_sub_block: run_placement.buffer_first_sub_block as u32,
                generation: run_placement.generation,
            };
            region_runs.push(cached_run.clone());
            // Record the tenant so eviction can drop this run from the index. Only
            // this worker (the fill buffer's owner) touches the tenant list now.
            self.tenants_mut(run_placement.buffer_index).push(BufferTenant {
                location: location.clone(),
                region,
                file_first_sub_block: first_sub_block,
            });
            drop(file_regions);
            drop(index);

            // Freshly reserved: every requested sub-block is one missing run.
            let pin = memory_ctx()
                .ring()
                .try_read(run_placement.buffer_index)
                .expect("fill buffer is held in read mode by its owner");
            return self.build_lookup(pin, region, start, end, &cached_run, &[(
                first_sub_block,
                last_sub_block,
            )]);
        }
    }

    /// Fast path: find a cached run covering the whole requested range, pin its
    /// buffer, confirm it was not recycled, and assemble the [`CacheLookup`].
    /// Returns `None` (a miss) if no covering run exists or the pin lost a race
    /// with eviction.
    fn resolve_cached_run(
        &self,
        location: &FileLocation,
        region: usize,
        start: usize,
        end: usize,
        first_sub_block: u32,
        last_sub_block: u32,
    ) -> Option<CacheLookup> {
        // Copy the covering run out from under the index lock; we re-validate it
        // against the buffer generation after pinning.
        let cached_run = {
            let index = self.index.read().unwrap();
            let file_regions = index.get(location)?.read().unwrap();
            file_regions
                .get(&region)?
                .iter()
                .find(|run| run.covers(first_sub_block, last_sub_block))
                .cloned()?
        };

        let pin = memory_ctx().ring().try_read(cached_run.buffer_index as usize)?;
        // Re-check under the pin: eviction may have recycled the buffer between the
        // index lookup and the pin. The generation rises on every reclaim, so an
        // unequal generation means our run no longer lives here.
        if self
            .buffer_meta(cached_run.buffer_index as usize)
            .generation
            .load(Ordering::Acquire)
            != cached_run.generation
        {
            return None;
        }
        self.buffer_meta(cached_run.buffer_index as usize)
            .reference_bit
            .store(true, Ordering::Relaxed);

        // Walk the requested sub-blocks, gathering each maximal run of *missing*
        // ones into a single block to read.
        let valid = &self.buffer_meta(cached_run.buffer_index as usize).valid;
        let mut missing_file_runs = Vec::new();
        let mut run_start: Option<u32> = None;
        for file_sub_block in first_sub_block..=last_sub_block {
            let buffer_sub_block = self.buffer_sub_block_of(&cached_run, file_sub_block);
            if valid.is_set(buffer_sub_block) {
                if let Some(first) = run_start.take() {
                    missing_file_runs.push((first, file_sub_block - 1));
                }
            } else {
                run_start.get_or_insert(file_sub_block);
            }
        }
        if let Some(first) = run_start.take() {
            missing_file_runs.push((first, last_sub_block));
        }

        Some(self.build_lookup(pin, region, start, end, &cached_run, &missing_file_runs))
    }

    /// Buffer sub-block holding file sub-block `file_sub_block` of `run`.
    fn buffer_sub_block_of(&self, run: &CachedRun, file_sub_block: u32) -> usize {
        (run.buffer_first_sub_block + (file_sub_block - run.file_first_sub_block)) as usize
    }

    /// Assemble the [`CacheLookup`] for region byte range `[start, end)` served by
    /// `run` in the pinned buffer, given the file sub-block runs still missing
    /// (each `[first, last]` inclusive). One shared pin keeps the buffer alive for
    /// the lookup's `data` and every in-flight read targeting it.
    fn build_lookup(
        &self,
        pin: ReadBuffer,
        region: usize,
        start: usize,
        end: usize,
        run: &CachedRun,
        missing_file_runs: &[(u32, u32)],
    ) -> CacheLookup {
        let buffer_base = pin.as_slice().as_ptr() as usize;
        let pin = Arc::new(pin);

        let missing = missing_file_runs
            .iter()
            .map(|&(first_file_sub_block, last_file_sub_block)| {
                let buffer_first_sub_block = self.buffer_sub_block_of(run, first_file_sub_block);
                let sub_block_count = (last_file_sub_block - first_file_sub_block + 1) as usize;
                MissingBlock {
                    file_offset: region + first_file_sub_block as usize * SUB_BLOCK_SIZE,
                    dest: buffer_base + buffer_first_sub_block * SUB_BLOCK_SIZE,
                    buffer_index: run.buffer_index as usize,
                    buffer_first_sub_block,
                    sub_block_count,
                    _pin: pin.clone(),
                }
            })
            .collect();

        // The requested bytes sit at the run's buffer offset plus the intra-region
        // distance from the run's first sub-block.
        let run_byte_base = run.buffer_first_sub_block as usize * SUB_BLOCK_SIZE;
        let start_in_buffer = run_byte_base + (start - run.file_first_sub_block as usize * SUB_BLOCK_SIZE);
        let data = Bytes::from_owner(BufferPin(pin)).slice(start_in_buffer..start_in_buffer + (end - start));
        CacheLookup { data, missing }
    }

    /// Prepare a freshly acquired buffer to become a worker's fill buffer: drop
    /// any stale validity and tenants and return the buffer's current generation
    /// (the one runs allocated into it will carry). Called while the buffer is
    /// held exclusively (WRITING), so no reader can observe the reset.
    pub(crate) fn prepare_fill_buffer(&self, buffer_index: usize) {
        let meta = self.buffer_meta(buffer_index);
        meta.valid.clear();
        meta.reference_bit.store(true, Ordering::Relaxed);
        self.tenants_mut(buffer_index).clear();
    }

    /// The current generation of buffer `index` — the tag a run allocated into it
    /// must carry so a later read detects a recycle. Read fresh on each
    /// allocation: the buffer is pinned by its filling worker, but `clear` can
    /// still bump it, so a cached copy would go stale.
    pub(crate) fn buffer_generation(&self, index: usize) -> u64 {
        self.buffer_meta(index).generation.load(Ordering::Acquire)
    }

    /// Mutable access to buffer `index`'s tenant list. Sound only for the buffer's
    /// filling worker (while it holds the pin) or the evictor (after `try_write`),
    /// which the `used` atomic keeps from overlapping. See [`FileCache::buffer_tenants`].
    #[allow(clippy::mut_from_ref)]
    fn tenants_mut(&self, index: usize) -> &mut Vec<BufferTenant> {
        unsafe { &mut *self.buffer_tenants[index].get() }
    }

    /// Evict a buffer using the (second-chance) CLOCK algorithm and return it as a
    /// writable buffer. Drops every run the reclaimed buffer held from the index
    /// and bumps its generation so stale [`CachedRun`] copies are detected.
    pub fn evict(&self) -> crate::memory::write_buffer::WriteBuffer {
        loop {
            let buffer_index =
                self.hand.fetch_add(1, Ordering::Relaxed) % memory_ctx().ring().len();
            // Give recently-used buffers a second chance: clear the ref bit and
            // skip; an unreferenced buffer we can write-lock gets evicted.
            if self
                .buffer_meta(buffer_index)
                .reference_bit
                .swap(false, Ordering::Relaxed)
            {
                continue;
            }
            let Some(write_buffer) = memory_ctx().ring().try_write(buffer_index) else {
                continue;
            };

            // Exclusive now (no readers, no other writer). Only a buffer that
            // actually holds runs is a cache buffer we may reclaim. A buffer with
            // no tenants sitting at `used == 0` is a free slot parked in a worker's
            // pool; handing it out here would alias the pool's copy, leaving two
            // owners of one 2 MB slot. Release the write lock without pooling (its
            // pool listing stays valid) and keep scanning.
            let tenants = std::mem::take(self.tenants_mut(buffer_index));
            if tenants.is_empty() {
                std::mem::forget(write_buffer);
                memory_ctx()
                    .ring()
                    .set_slot_used(buffer_index, 0, Ordering::Release);
                continue;
            }

            // Drop every run this buffer held from the index, then bump the
            // generation so any reader still holding a stale run copy fails its
            // post-pin re-check.
            self.drop_tenant_runs(buffer_index, tenants);
            self.buffer_meta(buffer_index)
                .generation
                .fetch_add(1, Ordering::Release);
            return write_buffer;
        }
    }

    /// Drop the given `tenants`' runs (all packed into `buffer_index`) from the
    /// index, pruning regions and files that become empty. The buffer must be
    /// held exclusively.
    fn drop_tenant_runs(&self, buffer_index: usize, tenants: Vec<BufferTenant>) {
        for tenant in tenants {
            let index = self.index.read().unwrap();
            let Some(region_runs) = index.get(&tenant.location) else {
                continue;
            };
            let became_empty = {
                let mut region_runs = region_runs.write().unwrap();
                if let Some(runs) = region_runs.get_mut(&tenant.region) {
                    runs.retain(|run| {
                        run.buffer_index as usize != buffer_index
                            || run.file_first_sub_block != tenant.file_first_sub_block
                    });
                    if runs.is_empty() {
                        region_runs.remove(&tenant.region);
                    }
                }
                region_runs.is_empty()
            };
            drop(index);

            // Prune the per-file outer entry once its last region is gone. The map
            // is keyed by `FileLocation` (an `Arc<File>` / `Arc<RemoteFile>`) whose
            // key is never reused, so without this a dead entry — pinning its `Arc`
            // and the file/connection it holds — lingers for every file ever
            // opened, leaking unboundedly under steady ingest and compaction.
            // Re-check emptiness under the write lock so a concurrent insert is not
            // dropped.
            if became_empty {
                let mut index = self.index.write().unwrap();
                if let Some(region_runs) = index.get(&tenant.location) {
                    if region_runs.read().unwrap().is_empty() {
                        index.remove(&tenant.location);
                    }
                }
            }
        }
    }

    /// Register a [`FileLocation`] (a local fd or a remote object) so its regions
    /// can be cached. Drops any runs cached under an equal prior registration
    /// (e.g. a reused fd number) so future reads of the new file miss and re-read.
    ///
    /// We do not bump the holding buffers' generations: a buffer is now shared by
    /// many files, so that would spuriously invalidate unrelated co-tenants.
    /// Dropping the runs from the index is sufficient. A racing reader of an old
    /// run necessarily still holds the prior file's handle (so its fd/object is
    /// unchanged and the cached bytes are still correct), and a genuine buffer
    /// recycle still bumps the generation via [`evict`](Self::evict). The dropped
    /// runs become unreferenced bytes in their buffers, reclaimed on eviction.
    pub fn open_entry(&self, location: FileLocation) {
        let mut index = self.index.write().unwrap();
        index.remove(&location);
        index.entry(location).or_default();
    }

    /// Drop every cached run so subsequent reads miss and re-read from storage,
    /// returning the number of runs dropped. Bumps the generation of each holding
    /// buffer (invalidating stale run copies), clears its validity and tenants,
    /// and returns it to the free pool unless a worker still pins it as a fill
    /// buffer. Intended for benchmarking true cold reads (`SELECT drop_cache()`);
    /// sound only while no query is in flight.
    pub fn clear(&self) -> usize {
        let mut index = self.index.write().unwrap();
        let mut runs_dropped = 0;
        let mut holding_buffers = std::collections::HashSet::new();
        for (_location, file_regions) in index.drain() {
            for (_region, runs) in file_regions.into_inner().unwrap() {
                for run in runs {
                    runs_dropped += 1;
                    holding_buffers.insert(run.buffer_index as usize);
                }
            }
        }
        for buffer_index in holding_buffers {
            // The generation bump is atomic, so it is always safe. Reset the
            // non-atomic validity/tenants only under an exclusive write lock: a
            // worker's pinned fill buffer fails `try_write`, and touching its
            // `valid`/tenants would race the owner. Such a buffer keeps stale
            // bytes, but its runs are already gone from the drained index and its
            // owner clears it on the next fill, so leaving it is harmless.
            self.buffer_meta(buffer_index)
                .generation
                .fetch_add(1, Ordering::Release);
            if let Some(write_buffer) = memory_ctx().ring().try_write(buffer_index) {
                self.buffer_meta(buffer_index).valid.clear();
                self.tenants_mut(buffer_index).clear();
                drop(write_buffer);
            }
        }
        runs_dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::init_test_free_pool;
    use std::fs::File;

    /// A fresh local file location: a distinct `/dev/null` fd, so every call is a
    /// distinct cache key.
    fn local_file() -> FileLocation {
        FileLocation::Local(Arc::new(File::open("/dev/null").unwrap()))
    }

    fn cache() -> &'static FileCache {
        memory_ctx().file_cache()
    }

    /// Assert the lookup has missing runs and return it.
    fn miss(lookup: CacheLookup) -> CacheLookup {
        assert!(!lookup.missing().is_empty(), "expected a miss");
        lookup
    }

    /// Assert the lookup is a full hit and take its bytes.
    fn hit(lookup: CacheLookup) -> Bytes {
        assert!(lookup.missing().is_empty(), "expected a hit");
        lookup.into_data()
    }

    /// Read `byte` into every missing block of `lookup` and commit it (what the IO
    /// path does once a read lands).
    fn fill_with(lookup: &CacheLookup, byte: u8) {
        for block in lookup.missing() {
            unsafe { std::ptr::write_bytes(block.dest(), byte, block.len()) };
            block.commit();
        }
    }

    #[test]
    fn misses_then_fills_then_hits_the_same_bytes() {
        init_test_free_pool(8);
        let file = local_file();
        cache().open_entry(file.clone());

        let lookup = miss(cache().get_region(&file, 0, 0, 10));
        fill_with(&lookup, 0xAB);
        drop(lookup);

        assert_eq!(&hit(cache().get_region(&file, 0, 0, 10))[..], &[0xAB; 10]);
    }

    #[test]
    fn packs_distinct_files_into_shared_buffers_without_mixing_bytes() {
        init_test_free_pool(8);
        let file_one = local_file();
        let file_two = local_file();
        cache().open_entry(file_one.clone());
        cache().open_entry(file_two.clone());

        fill_with(&miss(cache().get_region(&file_one, 0, 0, 100)), 0xAA);
        fill_with(&miss(cache().get_region(&file_two, 0, 0, 100)), 0xBB);

        assert_eq!(&hit(cache().get_region(&file_one, 0, 0, 100))[..], &[0xAA; 100]);
        assert_eq!(&hit(cache().get_region(&file_two, 0, 0, 100))[..], &[0xBB; 100]);
    }

    #[test]
    fn dropping_a_runs_buffer_misses_on_reread_without_panicking() {
        init_test_free_pool(8);
        let file = local_file();
        cache().open_entry(file.clone());
        fill_with(&miss(cache().get_region(&file, 0, 0, 100)), 0x33);
        assert_eq!(&hit(cache().get_region(&file, 0, 0, 100))[..], &[0x33; 100]);

        // `clear` drops every run and bumps the holding buffers' generations, the
        // same index/generation effect a CLOCK eviction has on the runs it drops.
        let dropped = cache().clear();

        assert_eq!(dropped, 1, "the one cached run was dropped");
        assert!(
            !cache().get_region(&file, 0, 0, 100).missing().is_empty(),
            "the run is gone, so the re-read misses and re-creates the entry",
        );
    }

    #[test]
    fn re_registering_a_file_drops_its_cached_runs() {
        init_test_free_pool(8);
        let file = local_file();
        cache().open_entry(file.clone());
        fill_with(&miss(cache().get_region(&file, 0, 0, 100)), 0x44);

        // Re-registering (e.g. a reused fd) drops the file's runs from the index,
        // so the same read misses and re-reads rather than serving another file's
        // bytes.
        cache().open_entry(file.clone());

        assert!(!cache().get_region(&file, 0, 0, 100).missing().is_empty());
    }

    #[test]
    fn evict_does_not_steal_a_slot_from_the_free_pool() {
        init_test_free_pool(3);
        let file = local_file();
        cache().open_entry(file.clone());
        // A full-region read fills a buffer; the next region's read retires it, so
        // there is one evictable cache buffer (holds runs, unpinned) while a seeded
        // slot still sits free in the pool.
        fill_with(&miss(cache().get_region(&file, 0, 0, BUFFER_SIZE)), 0x11);
        fill_with(&miss(cache().get_region(&file, BUFFER_SIZE, 0, 100)), 0x22);

        let evicted = cache().evict();
        let still_pooled: Vec<usize> =
            std::iter::from_fn(|| memory_ctx().pop_free_idx(false)).collect();

        assert!(
            !still_pooled.contains(&evicted.slot_idx),
            "evict() handed out slot {} while the pool still lists it {:?}: two owners of one ring slot",
            evicted.slot_idx,
            still_pooled,
        );
    }
}
