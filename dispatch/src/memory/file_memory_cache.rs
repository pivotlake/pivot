//! A CLOCK page cache that *packs* many small file reads into shared ring buffer
//! slots, tracking validity at 4 KB sub-block granularity.
//!
//! Reads are not positional: a missed read is bump-allocated as a contiguous run
//! of 4 KB sub-blocks at a per-worker cursor inside whatever shared slot that
//! worker is currently filling, so hundreds of unrelated reads from many files
//! share one slot and the working set is measured in *bytes*, not slots. The
//! cache therefore records each cached run's *physical placement* - which slot and
//! which sub-block within it - in a per-file [`Extent`] map, decoupled from the
//! file offset.
//!
//! ## Extents
//!
//! Each file maps `first_block_in_file` (a file block index = `file_offset /
//! SUB_BLOCK_SIZE`) → [`Extent`] (`slot_idx`, `sub_block_in_slot`, `block_count`,
//! `generation`). Extents are non-overlapping by construction: allocation only
//! ever covers blocks no live extent already claims. A slot's [`Entry::valid`]
//! bitmap records which of its 512 × 4 KB sub-blocks have actually been read, so a
//! run can be present in the map yet still filling.
//!
//! ## Multi-tenant slots and eviction
//!
//! A slot holds runs from arbitrarily many files. CLOCK eviction reclaims a whole
//! slot, so it must invalidate every run living in it. Two lock-free mechanisms do
//! this: a per-slot `generation` (bumped on every recycle, snapshotted into each
//! extent, and re-checked after pinning - a reader that observes a bumped
//! generation treats the placement as a miss), and a per-slot tenant list (the
//! `(file, first_block_in_file)` of every run packed into the slot). The tenant list is
//! written only by the single worker filling the slot (while it holds the fill
//! pin) and read only by the evictor after `try_write` (which needs every pin
//! dropped) - so the ring's reader-pin atomics provide the happens-before with no
//! new lock. See [`ReadBuffer`]'s `Drop` for the publication edge.
//!
//! ## Lookups
//!
//! [`FileMemoryCache::get`] takes a file byte range and returns one [`CacheLookup`] per
//! contiguous run it resolves to - a *hit* run already resident in one slot, a
//! *refill* run mapped but not yet read, or a freshly *allocated* miss run. Each
//! lookup carries its `data` (a zero-copy view into the slot it lives in) plus its
//! `missing` blocks (empty on a hit). Concatenating every lookup's `data` in order
//! reproduces the requested bytes once every [`MissingBlock`] has been read into
//! its slot at [`MissingBlock::dest`] and marked valid via [`MissingBlock::commit`].
//! Callers feed the fragments into a scattered reader, so more (smaller) fragments
//! are fine.

use crate::io::FileLocation;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Sub-block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range is
/// rounded out to whole sub-blocks.
const SUB_BLOCK_SIZE: usize = 4096;
/// Number of 4 KB sub-blocks per 2 MB slot (512).
const SUB_BLOCKS_PER_SLOT: usize = BUFFER_SIZE / SUB_BLOCK_SIZE;
/// Number of `u64` words in a slot's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = SUB_BLOCKS_PER_SLOT / 64;

/// The placement of one contiguous cached run inside the ring. Lives in a file's
/// extent map keyed by the run's first file block. Never spans more than one slot.
#[derive(Clone, Copy)]
struct Extent {
    /// The run's first 4 KB block in the file (also its key in the file's extent
    /// map). Stored so the extent is self-describing - callers don't have to thread
    /// the map key alongside it.
    first_block_in_file: usize,
    /// Ring slot physically holding this run's bytes.
    slot_idx: u32,
    /// 4 KB sub-block offset of the run within that slot (0..512).
    sub_block_in_slot: u16,
    /// How many contiguous 4 KB blocks the run covers (1..=512).
    block_count: u16,
    /// The slot's `generation` when this run was placed. A later reader compares
    /// it to the slot's live generation to detect a slot recycled out from under it.
    generation: u32,
}

impl Extent {
    /// The run's last 4 KB block in the file (inclusive).
    fn last_block_in_file(&self) -> usize {
        self.first_block_in_file + self.block_count as usize - 1
    }
}

/// One `(file, first_block_in_file)` packed into a slot, recorded so the evictor
/// can drop the run's extent when it recycles the slot.
struct Tenant {
    /// The file the packed run belongs to.
    location: FileLocation,
    /// The run's first 4 KB block in the file - the key into `location`'s extent map.
    first_block_in_file: usize,
}

/// A worker's bump cursor over a shared fill slot. The slot is held as a
/// [`ReadBuffer`] (a reader pin) so runs already packed into it stay readable and
/// pinned while the worker keeps filling its tail. One per worker, owned by its
/// [`MemoryContext`](crate::memory::context).
pub struct FillCursor {
    /// The slot currently being packed, pinned. `None` before the first miss and
    /// after [`FileMemoryCache::clear`].
    buffer: Option<Arc<ReadBuffer>>,
    /// Ring slot index of `buffer`.
    slot_idx: usize,
    /// `buffer`'s start address, cached so the hot path skips a deref.
    slot_ptr: usize,
    /// The slot's `generation` snapshotted when this buffer was taken.
    generation: u32,
    /// Bump pointer: the next free sub-block in `buffer` (0..=512).
    next_sub_block: u16,
}

impl FillCursor {
    /// An empty cursor - the next miss takes a fresh fill buffer.
    pub fn empty() -> Self {
        FillCursor {
            buffer: None,
            slot_idx: 0,
            slot_ptr: 0,
            generation: 0,
            next_sub_block: 0,
        }
    }
}

/// The result of a [`FileMemoryCache::get`] over one resolved run: the looked-up bytes
/// (a zero-copy view into the slot the run lives in) plus the block(s) still
/// missing from it. `missing` is empty on a hit; otherwise `data` only becomes
/// valid once every [`MissingBlock`] has been read & committed.
pub struct CacheLookup {
    /// Zero-copy view of this run's bytes inside its ring slot (kept alive by the
    /// pin the `Bytes` owns). Only valid to read once `missing` is filled.
    data: Bytes,
    /// The block(s) of this run not yet resident - read each into its slot and
    /// commit. Empty on a hit.
    missing: Vec<MissingBlock>,
}

impl CacheLookup {
    /// Block(s) that must be read & committed before [`data`](Self::into_data) is
    /// valid. Empty on a hit.
    pub fn missing(&self) -> &[MissingBlock] {
        &self.missing
    }

    /// Take the looked-up bytes - valid to read once every [`missing`](Self::missing)
    /// block has been filled. A zero-copy view into the cache slot, kept alive by
    /// the returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// One contiguous run of missing sub-blocks that must be read from disk to satisfy
/// a lookup. Covers whole sub-blocks, so the read targets [`dest`] (a region
/// inside the pinned slot) directly - no intermediate buffer - and once it lands
/// the run is marked valid wholesale via [`commit`].
///
/// Holds an [`Arc`] on the slot pin shared with the lookup's `data`, so the slot
/// can't be evicted while a read targeting it is in flight - even if the owning
/// query is cancelled and drops its [`CacheLookup`] first.
///
/// [`dest`]: MissingBlock::dest
/// [`commit`]: MissingBlock::commit
#[derive(Clone)]
pub struct MissingBlock {
    /// Absolute file offset this run's bytes are read from (a multiple of
    /// [`SUB_BLOCK_SIZE`]). Unlike a slot offset, this is the true file position.
    file_offset: usize,
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
    /// A block covering `sub_block_count` sub-blocks starting at `first_sub_block`
    /// of the slot at `slot_ptr`, whose bytes come from `file_offset`, sharing the
    /// slot pin.
    fn new(
        file_offset: usize,
        slot_ptr: usize,
        slot_idx: usize,
        first_sub_block: usize,
        sub_block_count: usize,
        pin: Arc<ReadBuffer>,
    ) -> Self {
        MissingBlock {
            file_offset,
            dest: slot_ptr + first_sub_block * SUB_BLOCK_SIZE,
            slot_idx,
            first_sub_block,
            sub_block_count,
            _pin: pin,
        }
    }

    /// File offset this block's bytes are read from (always a multiple of
    /// [`SUB_BLOCK_SIZE`], the direct-I/O alignment).
    pub fn file_offset(&self) -> usize {
        self.file_offset
    }

    /// Number of bytes to read - a multiple of `SUB_BLOCK_SIZE`.
    pub fn len(&self) -> usize {
        self.sub_block_count * SUB_BLOCK_SIZE
    }

    /// Whether this block covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.sub_block_count == 0
    }

    /// Carve out a sub-block covering `[rel_offset, rel_offset + len)` *within*
    /// this block - both relative to this block's start and both
    /// `SUB_BLOCK_SIZE`-aligned. The sub-block shares the slot pin, reads into the
    /// matching slice of the same pinned slot, and [`commit`](Self::commit)s only
    /// its own sub-blocks.
    ///
    /// Used to split one missing run across transports: bytes already in the local
    /// disk cache are read from there, the holes are fetched over HTTP, and each
    /// part fills its own slice of the slot.
    pub fn carve_sub_block(&self, rel_offset: usize, len: usize) -> MissingBlock {
        debug_assert_eq!(rel_offset % SUB_BLOCK_SIZE, 0);
        debug_assert_eq!(len % SUB_BLOCK_SIZE, 0);
        debug_assert!(rel_offset + len <= self.len());
        MissingBlock {
            file_offset: self.file_offset + rel_offset,
            dest: self.dest + rel_offset,
            slot_idx: self.slot_idx,
            first_sub_block: self.first_sub_block + rel_offset / SUB_BLOCK_SIZE,
            sub_block_count: len / SUB_BLOCK_SIZE,
            _pin: self._pin.clone(),
        }
    }

    /// The destination to read this block's [`len`](Self::len) bytes into: a
    /// 4 KB-aligned region `[dest, dest+len)` inside the pinned slot - a valid
    /// O_DIRECT target. The slot stays alive for the read because this block holds
    /// a pin (`_pin`), and only this block's (currently invalid) sub-blocks live
    /// here, so the read never races a reader of the slot's valid bytes.
    pub fn dest(&self) -> *mut u8 {
        self.dest as *mut u8
    }

    /// Mark this block's sub-blocks valid - call once its bytes have been read into
    /// the slot.
    pub fn commit(&self) {
        memory_ctx()
            .file_memory_cache()
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

/// A slot's validity bitmap: bit `sub_block` set means sub-block `sub_block` (a
/// 4 KB span of the slot) has been read into it. Keeps all the word/bit indexing -
/// and the memory ordering that makes the in-place fill sound - in one place.
#[derive(Default)]
struct ValidBitmap([AtomicU64; BITMAP_WORDS]);

impl ValidBitmap {
    /// Is sub-block `sub_block` present? `Acquire` pairs with `set`'s `Release`, so
    /// a reader that observes the bit is guaranteed to see the sub-block's bytes.
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

/// Per-slot CLOCK metadata, validity, and tenancy. One per ring slot.
struct Entry {
    /// CLOCK second-chance bit: set on access, cleared by the eviction sweep.
    ref_bit: AtomicBool,
    /// Bumped on every recycle and snapshotted into each [`Extent`] placed here, so
    /// a reader that pins the slot can detect it was reused out from under its
    /// placement (the snapshot no longer matches).
    generation: AtomicU32,
    /// Which of the slot's 512 × 4 KB sub-blocks have actually been read in.
    valid: ValidBitmap,
    /// The runs currently packed here, one per `(file, first_block_in_file)`, so
    /// eviction can drop them from their files' maps. Written only by the worker filling the
    /// slot (under its fill pin); read only by the evictor (under `try_write`).
    tenants: UnsafeCell<Vec<Tenant>>,
    /// Whether the cache owns this slot (`true`, a published fill/cache buffer) or
    /// the free pool does (`false`). The evictor must not hand out a pooled slot, and
    /// a freshly-rotated cache slot can still have an empty `tenants`, so this can't
    /// be derived from `tenants.is_empty()`.
    bound: AtomicBool,
}

/// One resolved run on the [`get`](FileMemoryCache::get) path, starting at the block the
/// caller asked about: either a run already mapped to a slot (its [`CacheLookup`]
/// built directly, plus the next file block to resume at), or an uncovered run of
/// `block_count` blocks that must be allocated.
enum Resolved {
    Present {
        lookup: CacheLookup,
        next_block_in_file: usize,
    },
    Miss {
        block_count: usize,
    },
}

/// A CLOCK-eviction cache over the shared [`Ring`](super::Ring) that packs many
/// small reads per slot.
///
/// Runs are bucketed by [`FileLocation`], so the same cache serves both local
/// files and remote HTTP objects - only the transport that fills a missing block
/// differs.
pub struct FileMemoryCache {
    /// Per-file index of what's cached and where it physically lives: file → (a
    /// run's first 4 KB block → its [`Extent`] placement). Each file's extent map
    /// is sorted by `first_block_in_file` and its extents never overlap. Both levels
    /// are `RwLock`'d - the file map changes rarely (open/prune a file), a file's
    /// extent map per cached run.
    ///
    /// ```text
    ///   foo.parquet ─┬─ block 0  → Extent { slot 7, sub-block 130, 3 blocks }
    ///                └─ block 40 → Extent { slot 3, sub-block 88,  2 blocks }
    ///   bar.parquet ─── block 12 → Extent { slot 7, sub-block 200, 1 block  }
    /// ```
    file_maps: RwLock<HashMap<FileLocation, RwLock<BTreeMap<usize, Extent>>>>,
    /// Per-slot metadata, indexed by ring slot. `UnsafeCell` because `tenants` is
    /// mutated through a shared `&self` under the pin/exclusivity discipline.
    entries: Box<[UnsafeCell<Entry>]>,
    /// The CLOCK sweep hand: the next ring slot index [`evict`](Self::evict) considers.
    hand: AtomicUsize,
}

unsafe impl Send for FileMemoryCache {}
unsafe impl Sync for FileMemoryCache {}

impl FileMemoryCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            file_maps: Default::default(),
            entries: (0..capacity)
                .map(|_| {
                    UnsafeCell::new(Entry {
                        ref_bit: Default::default(),
                        generation: Default::default(),
                        valid: Default::default(),
                        tenants: UnsafeCell::new(Vec::new()),
                        bound: Default::default(),
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hand: Default::default(),
        }
    }

    /// Look up the file byte range `[offset, offset + len)` of `location`. The
    /// range is resolved into a sequence of contiguous runs in file order, yielding
    /// one [`CacheLookup`] per run: read & fill every lookup's
    /// [`missing`](CacheLookup::missing) blocks, then concatenate the runs' bytes.
    pub fn get(&self, location: &FileLocation, offset: usize, len: usize) -> Vec<CacheLookup> {
        if len == 0 {
            return Vec::new();
        }
        let end = offset + len;
        let first_block_in_file = offset / SUB_BLOCK_SIZE;
        let last_block_in_file = (end - 1) / SUB_BLOCK_SIZE;

        let mut lookups = Vec::new();
        let mut block_in_file = first_block_in_file;
        while block_in_file <= last_block_in_file {
            match self.classify_next_run(location, block_in_file, last_block_in_file, offset, end) {
                Resolved::Present { lookup, next_block_in_file } => {
                    block_in_file = next_block_in_file;
                    lookups.push(lookup);
                }
                Resolved::Miss { block_count } => {
                    // `None` means a racing worker placed an extent over this block
                    // between classify and allocate; re-classify the same block
                    // (it's now a Present run, so we still make progress).
                    match self.allocate_run(location, block_in_file, block_count, offset, end) {
                        Some((claimed, lookup)) => {
                            lookups.push(lookup);
                            block_in_file += claimed;
                        }
                        None => continue,
                    }
                }
            }
        }
        lookups
    }

    /// Resolve the next contiguous run starting at file block `block_in_file` (up to
    /// `last_block_in_file`): if a live extent covers it, return a [`Resolved::Present`]
    /// with the run's lookup already built; otherwise return a [`Resolved::Miss`]
    /// spanning the uncovered gap up to the next extent. `offset`/`end` are the
    /// caller's byte range, used to clip the run's bytes.
    fn classify_next_run(
        &self,
        location: &FileLocation,
        block_in_file: usize,
        last_block_in_file: usize,
        offset: usize,
        end: usize,
    ) -> Resolved {
        let file_maps = self.file_maps.read().unwrap();
        let Some(extents_lock) = file_maps.get(location) else {
            return Resolved::Miss {
                block_count: last_block_in_file - block_in_file + 1,
            };
        };
        let extents = extents_lock.read().unwrap();

        if let Some(extent) = find_extent_covering_block(&extents, block_in_file) {
            if let Some(resolved) =
                self.try_resolve_present_run(extent, block_in_file, last_block_in_file, offset, end)
            {
                return resolved;
            }
            // Stale (recycled) or contended extent: treat its covered span as a
            // miss; `allocate_run` removes the dead extent under the write lock.
            return Resolved::Miss {
                block_count: extent.last_block_in_file().min(last_block_in_file) - block_in_file + 1,
            };
        }

        // Uncovered: the gap runs until the next extent (or the request's end).
        let next_extent_first = extents
            .range(block_in_file + 1..)
            .next()
            .map(|(&k, _)| k)
            .unwrap_or(usize::MAX);
        let gap_end = next_extent_first.saturating_sub(1).min(last_block_in_file);
        Resolved::Miss {
            block_count: gap_end - block_in_file + 1,
        }
    }

    /// Pin the slot of `extent` and, if its generation still matches, resolve the
    /// maximal run of equal validity starting at file block `block_in_file` (bounded
    /// by the extent and `last_block_in_file`) into a [`Resolved::Present`] lookup.
    /// Returns `None` if the slot is contended (being written) or was recycled.
    fn try_resolve_present_run(
        &self,
        extent: Extent,
        block_in_file: usize,
        last_block_in_file: usize,
        offset: usize,
        end: usize,
    ) -> Option<Resolved> {
        let slot_idx = extent.slot_idx as usize;
        let buffer = memory_ctx().ring().try_read(slot_idx)?;
        if !self.is_live(extent) {
            return None;
        }
        let entry = self.entry(slot_idx);
        entry.ref_bit.store(true, Ordering::Relaxed);

        let run_end = extent.last_block_in_file().min(last_block_in_file);
        // Slot sub-block holding file block `b`.
        let sub_block_of = |b: usize| extent.sub_block_in_slot as usize + (b - extent.first_block_in_file);
        let first_sub_block = sub_block_of(block_in_file);

        let first_valid = entry.valid.is_set(first_sub_block);
        let mut last = block_in_file;
        while last < run_end && entry.valid.is_set(sub_block_of(last + 1)) == first_valid {
            last += 1;
        }
        let block_count = last - block_in_file + 1;

        let lookup = build_lookup(
            Arc::new(buffer),
            block_in_file,
            first_sub_block,
            block_count,
            !first_valid,
            offset,
            end,
        );
        Some(Resolved::Present {
            lookup,
            next_block_in_file: block_in_file + block_count,
        })
    }

    /// Allocate the uncovered run `[first_block_in_file, +block_count)` into this
    /// worker's fill slot, register its extent, and return `(blocks claimed, lookup)`.
    /// Claims fewer blocks than requested when the fill buffer's tail is shorter;
    /// returns `None` when a racing worker already covered the run's first block (the
    /// caller re-classifies).
    fn allocate_run(
        &self,
        location: &FileLocation,
        first_block_in_file: usize,
        block_count: usize,
        offset: usize,
        end: usize,
    ) -> Option<(usize, CacheLookup)> {
        let fill_cursor = memory_ctx().fill_cursor();
        if fill_cursor.buffer.is_none() || fill_cursor.next_sub_block as usize == SUB_BLOCKS_PER_SLOT
        {
            // May evict (touching the maps), so hold no map lock here.
            self.rotate_fill_buffer(fill_cursor);
        }
        let start_sub_block = fill_cursor.next_sub_block as usize;
        let want = block_count.min(SUB_BLOCKS_PER_SLOT - start_sub_block);
        let slot_idx = fill_cursor.slot_idx;
        let generation = fill_cursor.generation;

        // Claim the free prefix of the run under the file's write lock, re-creating
        // the file's entry if it was pruned (same retry shape as the old miss path).
        let claimed = loop {
            let file_maps = self.file_maps.read().unwrap();
            let Some(extents_lock) = file_maps.get(location) else {
                drop(file_maps);
                self.file_maps
                    .write()
                    .unwrap()
                    .entry(location.clone())
                    .or_default();
                continue;
            };
            let mut extents = extents_lock.write().unwrap();
            let claimed = self.count_free_blocks(&mut extents, first_block_in_file, want);
            if claimed > 0 {
                // Record the tenant *before* inserting the extent: should anything
                // between the two ever unwind, a tenant-without-extent is harmless
                // (the evictor skips it) whereas an extent-without-tenant would leak
                // (the evictor only prunes runs it finds in the tenant list).
                self.tenants_mut(slot_idx).push(Tenant {
                    location: location.clone(),
                    first_block_in_file,
                });
                extents.insert(
                    first_block_in_file,
                    Extent {
                        first_block_in_file,
                        slot_idx: slot_idx as u32,
                        sub_block_in_slot: start_sub_block as u16,
                        block_count: claimed as u16,
                        generation,
                    },
                );
                // Mark the slot referenced so a freshly packed run survives one CLOCK
                // sweep, the way the old positional region-bind did on its slot.
                self.entry(slot_idx).ref_bit.store(true, Ordering::Relaxed);
            }
            break claimed;
        };
        if claimed == 0 {
            return None;
        }

        fill_cursor.next_sub_block = (start_sub_block + claimed) as u16;
        let lookup = build_lookup(
            fill_cursor.buffer.clone().unwrap(),
            first_block_in_file,
            start_sub_block,
            claimed,
            true,
            offset,
            end,
        );
        Some((claimed, lookup))
    }

    /// Count how many blocks of the run `[first_block_in_file, +want)` are free of
    /// any *live* extent (its claimable prefix), removing stale (recycled) extents
    /// in the way. The caller places its extent over exactly this prefix, preserving
    /// the non-overlap invariant. `extents` is the file's extent map.
    fn count_free_blocks(
        &self,
        extents: &mut BTreeMap<usize, Extent>,
        first_block_in_file: usize,
        want: usize,
    ) -> usize {
        let end_block = first_block_in_file + want;

        // A predecessor extent may already cover `first_block_in_file`.
        if let Some(predecessor) = find_extent_covering_block(extents, first_block_in_file) {
            if self.is_live(predecessor) {
                return 0; // live coverage raced in; caller re-classifies as a hit
            }
            extents.remove(&predecessor.first_block_in_file); // stale → drop it
        }

        // Walk forward, dropping stale extents, stopping at the first live one.
        loop {
            let next_extent = extents
                .range(first_block_in_file + 1..end_block)
                .next()
                .map(|(_, &extent)| extent);
            match next_extent {
                None => return want,
                Some(next_extent) => {
                    if self.is_live(next_extent) {
                        return next_extent.first_block_in_file - first_block_in_file;
                    }
                    extents.remove(&next_extent.first_block_in_file);
                }
            }
        }
    }

    /// Point `fill_cursor` at a fresh fill buffer: a free ring slot, its bitmap and
    /// tenant list reset while held exclusively, published as readable. May evict -
    /// must run with no `file_maps` lock held.
    fn rotate_fill_buffer(&self, fill_cursor: &mut FillCursor) {
        let write_buffer = memory_ctx().get_write_buffer(false);
        let slot_idx = write_buffer.slot_idx;
        let slot_ptr = write_buffer.ptr as usize;
        let entry = self.entry(slot_idx);
        // Exclusively held (WRITING): reset metadata before any reader can pin it.
        entry.valid.clear();
        self.tenants_mut(slot_idx).clear();
        entry.bound.store(true, Ordering::Relaxed);
        // The slot's generation was already bumped by whoever recycled it (eviction
        // or `clear`); snapshot it so this fill's extents carry the live value.
        let generation = entry.generation.load(Ordering::Acquire);
        let buffer = ReadBuffer::from(write_buffer); // used=1, Release; non-zeroed

        fill_cursor.buffer = Some(Arc::new(buffer));
        fill_cursor.slot_idx = slot_idx;
        fill_cursor.slot_ptr = slot_ptr;
        fill_cursor.generation = generation;
        fill_cursor.next_sub_block = 0;
    }

    /// Borrow ring slot `idx`'s cache metadata. The atomics (`valid`, `ref_bit`,
    /// `generation`, `bound`) are always safe to touch; `tenants` is sound to touch
    /// only under the fill pin (filler) or `try_write` exclusivity (evictor).
    fn entry(&self, idx: usize) -> &Entry {
        unsafe { &*self.entries[idx].get() }
    }

    /// Mutable access to slot `idx`'s tenant list. Sound only under the fill pin
    /// (filler) or `try_write` exclusivity (evictor) - the one `unsafe` for the
    /// `tenants` invariant lives here. See [`entry`](Self::entry).
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell; see above
    fn tenants_mut(&self, idx: usize) -> &mut Vec<Tenant> {
        unsafe { &mut *self.entry(idx).tenants.get() }
    }

    /// Whether slot `extent.slot_idx`'s live generation still matches the one this
    /// extent snapshotted - i.e. the placement hasn't been recycled. `Acquire` pairs
    /// with the `Release` bump in [`recycle_slot`](Self::recycle_slot)/[`evict`](Self::evict),
    /// so a match guarantees the slot still holds this placement's bytes.
    fn is_live(&self, extent: Extent) -> bool {
        self.entry(extent.slot_idx as usize)
            .generation
            .load(Ordering::Acquire)
            == extent.generation
    }

    /// Reset a slot's validity and unbind it, bumping its generation so any
    /// surviving placement resolves to a miss. Sound only while the slot is held
    /// exclusively (`try_write` succeeded); the caller must have already taken or
    /// cleared its tenant list. The `Release` bump is the edge paired with the
    /// `Acquire` loads in [`is_live`](Self::is_live).
    fn recycle_slot(&self, idx: usize) {
        let entry = self.entry(idx);
        entry.valid.clear();
        entry.bound.store(false, Ordering::Relaxed);
        entry.generation.fetch_add(1, Ordering::Release);
    }

    /// Evict a slot using the (second-chance) CLOCK algorithm and return it as a
    /// writable buffer. Drops every run packed in the slot from its file's extent
    /// map (only those still pointing at this slot) and bumps the slot's generation
    /// so a reader pinning a surviving placement resolves to a miss.
    pub fn evict(&self) -> WriteBuffer {
        // TEMPORARY livelock guard. When a query's working set exceeds the ring,
        // every worker spins here finding nothing evictable (perf proved ~98% CPU
        // in this loop, ~0 forward progress, and the loop has no cancellation
        // point). After 100 fruitless iterations, warn and sleep to give peer
        // workers / ingest a chance to release slots; if a full ring sweep after
        // the sleep still finds nothing, the cache is genuinely exhausted, so
        // panic to abort the offending query rather than peg all cores forever.
        let mut iterations: u64 = 0;
        let mut panic_at: Option<u64> = None;
        loop {
            iterations += 1;
            if panic_at.is_none() && iterations == 100 {
                tracing::warn!(iterations, "FileCache::evict: no memory left to evict, sleeping");
                std::thread::sleep(std::time::Duration::from_secs(1));
                panic_at = Some(iterations + memory_ctx().ring().len() as u64);
            } else if panic_at.is_some_and(|limit| iterations >= limit) {
                panic!(
                    "FileCache::evict: still no evictable memory after sleeping \
                     ({iterations} iterations) — aborting query (cache exhausted \
                     by an oversized working set)"
                );
            }

            let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % memory_ctx().ring().len();
            let entry = self.entry(slot_idx);
            // Give recently-used slots a second chance.
            if entry.ref_bit.swap(false, Ordering::Relaxed) {
                continue;
            }
            let Some(write_buffer) = memory_ctx().ring().try_write(slot_idx) else {
                continue;
            };
            // Exclusive now (WRITING; every reader pin dropped - including the
            // filler's, so the tenant list is complete and visible: the last
            // reader's Release drop synchronizes with this try_write's Acquire).
            if !entry.bound.load(Ordering::Relaxed) {
                // A free buffer parked in a pool, not a cache slot. Handing it out
                // would alias the pool's copy; release it without pooling it again
                // (its pool listing stays valid) and keep scanning.
                std::mem::forget(write_buffer);
                memory_ctx()
                    .ring()
                    .set_slot_used(slot_idx, 0, Ordering::Release);
                continue;
            }

            let tenants = std::mem::take(self.tenants_mut(slot_idx));
            let mut emptied_files: Vec<FileLocation> = Vec::new();
            {
                let file_maps = self.file_maps.read().unwrap();
                for tenant in tenants {
                    let Some(extents_lock) = file_maps.get(&tenant.location) else {
                        continue;
                    };
                    let mut extents = extents_lock.write().unwrap();
                    // Remove only if the extent still names this slot: a run re-homed
                    // to another slot by a racing miss must survive. We hold WRITING,
                    // so the slot can't be concurrently refilled - any extent that
                    // names it is one of ours and is being evicted, so the generation
                    // necessarily matches and isn't worth re-checking.
                    if extents
                        .get(&tenant.first_block_in_file)
                        .is_some_and(|extent| extent.slot_idx as usize == slot_idx)
                    {
                        extents.remove(&tenant.first_block_in_file);
                    }
                    if extents.is_empty() {
                        emptied_files.push(tenant.location);
                    }
                }
            }
            self.prune_empty_file_locations(emptied_files);

            // Recycle under WRITING (before returning), so no reader ever observes
            // the slot still carrying the old generation.
            self.recycle_slot(slot_idx);
            return write_buffer;
        }
    }

    /// Remove `file_maps` entries for the given locations whose extent map is now
    /// empty. `file_maps` is keyed by a never-reused [`FileLocation`] (an
    /// `Arc<File>` / `Arc<RemoteFile>`), so without this a dead entry - pinning its
    /// `Arc` and the file/connection it holds - lingers for every file ever opened.
    /// Re-checks emptiness under the write lock so a concurrent `get` that just
    /// re-cached a run isn't dropped.
    fn prune_empty_file_locations(&self, locations: Vec<FileLocation>) {
        if locations.is_empty() {
            return;
        }
        let mut file_maps = self.file_maps.write().unwrap();
        for location in locations {
            if let Some(extents_lock) = file_maps.get(&location)
                && extents_lock.read().unwrap().is_empty()
            {
                file_maps.remove(&location);
            }
        }
    }

    /// Register a [`FileLocation`] (a local fd or a remote object) so its runs can
    /// be cached. Clears any runs from a previous registration of an equal location
    /// (e.g. a reused fd number) by dropping its extent map - the next lookup then
    /// misses. The orphaned tenants in their slots self-clean on eviction (their
    /// extent is gone, so the per-slot guard skips them).
    pub fn open_entry(&self, location: FileLocation) {
        // The `file_maps` write lock serializes with `allocate_run`'s read, so a
        // racing miss either sees the fresh empty map or has its run dropped here.
        self.file_maps
            .write()
            .unwrap()
            .insert(location, Default::default());
    }

    /// Evict every cached run: drop all extent maps and recycle the ring slots they
    /// referenced, so subsequent reads miss and re-read from disk. Registered file
    /// descriptors stay open (their maps are just emptied). Returns the number of
    /// extents dropped.
    ///
    /// Intended for benchmarking true cold reads (`SELECT drop_cache()`). Sound only
    /// while no query is in flight: it recycles bound slots, which must not race a
    /// reader holding a pin (an actively-filling slot another worker still pins is
    /// left intact - its runs are already dropped from the map, so it serves no
    /// stale hits).
    pub fn clear(&self) -> usize {
        // Release this worker's fill pin first, so its own fill slot is recyclable
        // below rather than being skipped as pinned.
        memory_ctx().fill_cursor().buffer = None;

        let mut slots = HashSet::new();
        let mut extents_dropped = 0;
        let file_maps = self.file_maps.read().unwrap();
        for extents_lock in file_maps.values() {
            let drained = std::mem::take(&mut *extents_lock.write().unwrap());
            for (_first_block_in_file, extent) in drained {
                slots.insert(extent.slot_idx as usize);
                extents_dropped += 1;
            }
        }
        drop(file_maps);

        for slot_idx in slots {
            // A slot another worker is still filling stays pinned → try_write fails →
            // skip it (its runs are already gone from the maps).
            if let Some(write_buffer) = memory_ctx().ring().try_write(slot_idx) {
                self.tenants_mut(slot_idx).clear();
                self.recycle_slot(slot_idx);
                drop(write_buffer); // returns the slot to the free pool (dirty)
            }
        }
        extents_dropped
    }
}

/// Build a zero-copy [`Bytes`] view of a placed run, clipped to the requested byte
/// range `[offset, end)`. `first_block_in_file` is the run's first file block,
/// `first_sub_block` its sub-block within the slot. The slice keeps `pin` alive.
fn slice_run_bytes(
    pin: &Arc<ReadBuffer>,
    first_block_in_file: usize,
    first_sub_block: usize,
    block_count: usize,
    offset: usize,
    end: usize,
) -> Bytes {
    let run_start_offset = first_block_in_file * SUB_BLOCK_SIZE;
    let slice_start_offset = run_start_offset.max(offset);
    let slice_end_offset = ((first_block_in_file + block_count) * SUB_BLOCK_SIZE).min(end);
    let head_offset = slice_start_offset - run_start_offset;
    let slot_start_offset = first_sub_block * SUB_BLOCK_SIZE + head_offset;
    let slot_end_offset = slot_start_offset + (slice_end_offset - slice_start_offset);
    Bytes::from_owner(SlotPin(pin.clone())).slice(slot_start_offset..slot_end_offset)
}

/// Find the extent that covers file block `block_in_file`, if any: the
/// greatest-keyed extent starting at or before it that reaches it. Extents never
/// overlap, so at most one qualifies. Returns the whole [`Extent`] (which carries
/// its own first block), so callers don't need the map key separately.
fn find_extent_covering_block(
    extents: &BTreeMap<usize, Extent>,
    block_in_file: usize,
) -> Option<Extent> {
    let (_, &extent) = extents.range(..=block_in_file).next_back()?;
    (extent.last_block_in_file() >= block_in_file).then_some(extent)
}

/// Build a [`CacheLookup`] for the run at `first_sub_block` of the slot `pin`
/// holds, clipped to `[offset, end)`. When `needs_read` the run's bytes aren't
/// valid yet, so it carries one refill [`MissingBlock`] over the whole run;
/// otherwise `missing` is empty (a hit). The slot and its base address come from
/// `pin` itself, so the read target can't drift from the data view.
fn build_lookup(
    pin: Arc<ReadBuffer>,
    first_block_in_file: usize,
    first_sub_block: usize,
    block_count: usize,
    needs_read: bool,
    offset: usize,
    end: usize,
) -> CacheLookup {
    let data = slice_run_bytes(&pin, first_block_in_file, first_sub_block, block_count, offset, end);
    let missing = if needs_read {
        vec![MissingBlock::new(
            first_block_in_file * SUB_BLOCK_SIZE,
            pin.ptr as usize,
            pin.slot_idx,
            first_sub_block,
            block_count,
            pin,
        )]
    } else {
        vec![]
    };
    CacheLookup { data, missing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    const SB: usize = SUB_BLOCK_SIZE;

    /// A local location used as the cache key throughout these tests. One shared
    /// open file, so every call keys the same cache bucket.
    #[allow(non_snake_case)]
    fn FD() -> FileLocation {
        use std::sync::OnceLock;
        static FILE: OnceLock<std::sync::Arc<std::fs::File>> = OnceLock::new();
        FileLocation::Local(
            FILE.get_or_init(|| std::sync::Arc::new(std::fs::File::open("/dev/null").unwrap()))
                .clone(),
        )
    }

    /// A fresh, independent local location (distinct cache bucket).
    fn new_file() -> FileLocation {
        FileLocation::Local(std::sync::Arc::new(
            std::fs::File::open("/dev/null").unwrap(),
        ))
    }

    fn cache() -> &'static FileMemoryCache {
        memory_ctx().file_memory_cache()
    }

    /// Drop this worker's fill pin so its current fill slot becomes evictable.
    fn release_fill_cursor() {
        memory_ctx().fill_cursor().buffer = None;
    }

    /// A deterministic byte for absolute file offset `off`, so cache contents are
    /// predictable for any slice regardless of where they were packed.
    fn pattern_at(off: usize) -> u8 {
        off.wrapping_mul(7).wrapping_add(1) as u8
    }

    /// Read the pattern into every missing block of `lookups` and commit it.
    fn fill_pattern(lookups: &[CacheLookup]) {
        for lookup in lookups {
            for block in lookup.missing() {
                for i in 0..block.len() {
                    unsafe { *block.dest().add(i) = pattern_at(block.file_offset() + i) };
                }
                block.commit();
            }
        }
    }

    /// Whether any lookup in the set still has a hole.
    fn has_misses(lookups: &[CacheLookup]) -> bool {
        lookups.iter().any(|l| !l.missing().is_empty())
    }

    /// Concatenate every lookup's bytes in file order.
    fn assemble(lookups: Vec<CacheLookup>) -> Vec<u8> {
        let mut out = Vec::new();
        for lookup in lookups {
            out.extend_from_slice(&lookup.into_data());
        }
        out
    }

    fn assert_pattern(bytes: &[u8], file_start: usize) {
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, pattern_at(file_start + i), "byte {i}");
        }
    }

    /// Every missing block across a set of lookups, cloned.
    fn missing_blocks(lookups: &[CacheLookup]) -> Vec<MissingBlock> {
        lookups
            .iter()
            .flat_map(|l| l.missing().iter().cloned())
            .collect()
    }

    #[test]
    fn miss_then_fill_then_hit() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        let miss = cache().get(&FD(), 0, 10);
        assert!(has_misses(&miss), "fresh range should miss");
        fill_pattern(&miss);
        drop(miss);

        let hit = cache().get(&FD(), 0, 10);
        assert!(!has_misses(&hit), "filled range should hit");
        assert_pattern(&assemble(hit), 0);
    }

    #[test]
    fn coalesces_contiguous_missing_sub_blocks_into_one_block() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Three sub-blocks, none present → one freshly-packed run, one block.
        let miss = cache().get(&FD(), 0, 2 * SB + 1);
        let blocks = missing_blocks(&miss);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].sub_block_count, 3);
        assert_eq!(blocks[0].len(), 3 * SB);
    }

    #[test]
    fn partial_validity_only_reports_the_holes() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Fill sub-block 0 only.
        let first = cache().get(&FD(), 0, 10);
        fill_pattern(&first);
        drop(first);

        // Ask for sub-blocks 0..=2; only 1 and 2 are missing.
        let again = cache().get(&FD(), 0, 2 * SB + 1);
        let blocks = missing_blocks(&again);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file_offset(), SB);
        assert_eq!(blocks[0].sub_block_count, 2);
    }

    #[test]
    fn consecutive_misses_pack_into_one_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Two disjoint reads of the same file pack contiguously into the fill slot.
        let a = cache().get(&FD(), 0, 10);
        let b = cache().get(&FD(), 50 * SB, 50 * SB + 10);
        let a = &a[0].missing()[0];
        let b = &b[0].missing()[0];
        assert_eq!(a.slot_idx, b.slot_idx, "should share the fill slot");
        assert_eq!(a.first_sub_block, 0);
        assert_eq!(b.first_sub_block, 1, "second read packed right after the first");
    }

    #[test]
    fn unrelated_files_pack_into_one_slot() {
        init_test_free_pool(8);
        let a = new_file();
        let b = new_file();
        cache().open_entry(a.clone());
        cache().open_entry(b.clone());

        let ra = cache().get(&a, 0, SB);
        let rb = cache().get(&b, 0, SB);

        assert_eq!(
            ra[0].missing()[0].slot_idx,
            rb[0].missing()[0].slot_idx,
            "reads from different files share one packed slot"
        );
    }

    #[test]
    fn a_read_past_the_fill_buffer_uses_a_new_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // One read spanning more than a whole slot (513 sub-blocks) splits at the
        // 2 MB boundary into two runs in two slots.
        let miss = cache().get(&FD(), 0, SUB_BLOCKS_PER_SLOT * SB + 1);
        let blocks = missing_blocks(&miss);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].sub_block_count, SUB_BLOCKS_PER_SLOT);
        assert_eq!(blocks[1].sub_block_count, 1);
        assert_ne!(blocks[0].slot_idx, blocks[1].slot_idx);

        fill_pattern(&miss);
        assert_pattern(&assemble(miss), 0);
    }

    #[test]
    fn overlap_dedup_only_fetches_the_holes() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Cache blocks 0..=3.
        let first = cache().get(&FD(), 0, 4 * SB);
        fill_pattern(&first);
        drop(first);

        // Read blocks 2..=5: 2,3 hit; only 4,5 are fetched.
        let second = cache().get(&FD(), 2 * SB, 4 * SB);
        let blocks = missing_blocks(&second);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file_offset(), 4 * SB);
        assert_eq!(blocks[0].sub_block_count, 2);
    }

    #[test]
    fn hit_reassembles_bytes_across_runs() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let miss = cache().get(&FD(), SB - 5, SB + 10);
        fill_pattern(&miss);
        drop(miss);

        let bytes = assemble(cache().get(&FD(), SB - 5, SB + 10));

        assert_pattern(&bytes, SB - 5);
        assert_eq!(bytes.len(), SB + 10);
    }

    #[test]
    fn hit_returns_the_exact_requested_byte_range() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let miss = cache().get(&FD(), 0, SB);
        fill_pattern(&miss);
        drop(miss);

        let bytes = assemble(cache().get(&FD(), 3, 6));

        assert_eq!(bytes.len(), 6);
        assert_pattern(&bytes, 3);
    }

    #[test]
    fn committed_bytes_serve_an_overlapping_later_lookup() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let first = cache().get(&FD(), 0, 2 * SB);
        fill_pattern(&first);
        drop(first);

        let bytes = assemble(cache().get(&FD(), SB / 2, SB));

        assert_pattern(&bytes, SB / 2);
        assert_eq!(bytes.len(), SB);
    }

    #[test]
    fn into_data_yields_the_freshly_filled_range() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        let miss = cache().get(&FD(), 10, SB);
        fill_pattern(&miss);
        let bytes = assemble(miss);

        assert_eq!(bytes.len(), SB);
        assert_pattern(&bytes, 10);
    }

    #[test]
    fn interleaved_gaps_split_then_fill_to_a_full_hit() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, 1)); // sub-block 0
        fill_pattern(&cache().get(&FD(), 2 * SB, 2 * SB + 1)); // sub-block 2

        let gaps = cache().get(&FD(), 0, 4 * SB);
        fill_pattern(&gaps);
        drop(gaps);
        let bytes = assemble(cache().get(&FD(), 0, 4 * SB));

        assert_pattern(&bytes, 0);
    }

    #[test]
    fn a_partially_cached_range_keeps_the_cached_bytes() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, 1)); // sub-block 0

        let m = cache().get(&FD(), 0, 2 * SB);
        assert!(has_misses(&m));
        let blocks = missing_blocks(&m);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file_offset(), SB); // only the hole
        fill_pattern(&m);
        let bytes = assemble(m);

        assert_pattern(&bytes, 0);
    }

    #[test]
    fn get_assembles_a_range_spanning_cached_and_uncached_runs() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        // Cache [0, SB); leave [SB, 2*SB) uncached.
        fill_pattern(&cache().get(&FD(), 0, SB));

        let parts = cache().get(&FD(), 0, 2 * SB);
        assert!(has_misses(&parts), "the second run is a hole");
        fill_pattern(&parts);

        assert_pattern(&assemble(parts), 0);
    }

    #[test]
    fn reopening_a_file_forgets_its_cached_runs() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));

        cache().open_entry(FD());

        assert!(has_misses(&cache().get(&FD(), 0, SB)));
    }

    /// Regression for the parquetsink OOM: the CLOCK evictor must not reclaim a ring
    /// slot that is still parked in the free pool (unbound).
    #[test]
    fn evict_does_not_steal_a_slot_from_the_free_pool() {
        init_test_free_pool(2);
        cache().open_entry(FD());
        // A real cached run binds a slot; its set ref_bit makes the CLOCK hand skip
        // it once, so the evictor reaches a slot that is still in the pool.
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();

        let evicted = cache().evict();
        let still_pooled: Vec<usize> =
            std::iter::from_fn(|| memory_ctx().pop_free_idx(false)).collect();

        assert!(
            !still_pooled.contains(&evicted.slot_idx),
            "evict() handed out slot {} while the pool still lists it {:?}",
            evicted.slot_idx,
            still_pooled,
        );
    }

    #[test]
    fn evicting_a_slot_drops_every_tenants_extent() {
        init_test_free_pool(16);
        let files: Vec<FileLocation> = (0..3).map(|_| new_file()).collect();
        for f in &files {
            cache().open_entry(f.clone());
            fill_pattern(&cache().get(f, 0, SB)); // all pack into one fill slot
        }
        release_fill_cursor();

        // The three runs share one slot; evicting it drops all three extents and
        // prunes their now-empty file_maps entries.
        cache().evict();

        let file_maps = cache().file_maps.read().unwrap();
        for f in &files {
            assert!(
                !file_maps.contains_key(f),
                "an evicted run's file_maps entry was left behind - leak",
            );
        }
    }

    #[test]
    fn a_recycled_slot_makes_stale_placements_miss() {
        init_test_free_pool(4);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();

        // Evict the slot the run lived in; its generation bumps.
        cache().evict();

        // The same range must now miss, not read recycled bytes.
        assert!(has_misses(&cache().get(&FD(), 0, SB)));
    }

    #[test]
    fn reading_a_file_after_its_entry_was_pruned_does_not_panic() {
        init_test_free_pool(8);
        let f = new_file();
        cache().open_entry(f.clone());
        fill_pattern(&cache().get(&f, 0, SB));
        release_fill_cursor();

        cache().evict();
        assert!(
            !cache().file_maps.read().unwrap().contains_key(&f),
            "evicting the last run should prune the entry",
        );

        // Reading f again must tolerate the absent entry: re-create it + miss.
        assert!(has_misses(&cache().get(&f, 0, SB)));
        assert!(
            cache().file_maps.read().unwrap().contains_key(&f),
            "re-reading a pruned file should re-create its entry",
        );
    }

    #[test]
    fn clear_forgets_everything_and_re_reads() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));

        let dropped = cache().clear();

        assert!(dropped >= 1);
        assert!(has_misses(&cache().get(&FD(), 0, SB)));
    }
}
