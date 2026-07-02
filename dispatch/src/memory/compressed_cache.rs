//! A CLOCK page cache that *packs* many small file reads into shared ring buffer
//! slots, tracking validity at 4 KB block granularity.
//!
//! Reads are not positional: a missed read is bump-allocated as a contiguous run
//! of 4 KB blocks at a per-worker cursor inside whatever shared slot that worker is
//! currently filling, so hundreds of unrelated reads from many files share one slot
//! and the working set is measured in *bytes*, not slots. The cache therefore
//! records each cached run's *physical placement* - which slot and which block
//! within it - in a per-file [`Extent`] map, decoupled from the file offset.
//!
//! ## Extents
//!
//! Each file maps `first_file_block` (= `file_offset / 4096`) → [`Extent`]. Extents
//! are non-overlapping by construction: allocation only ever covers blocks no live
//! extent already claims. A slot's [`Entry::valid`] bitmap records which of its
//! 512 × 4 KB blocks have actually been read, so a run can be present in the map yet
//! still filling.
//!
//! ## Multi-tenant slots and eviction
//!
//! A slot holds runs from arbitrarily many files. CLOCK eviction reclaims a whole
//! slot, so it must invalidate every run living in it. A per-slot tenant list - the
//! `(file, first_file_block)` of every run packed into the slot - records what to
//! drop. It is written only by the single worker filling the slot (while it holds
//! the fill pin) and read only by the evictor after `try_write` (which needs every
//! pin dropped), so the ring's reader-pin atomics provide the happens-before with no
//! new lock. See [`ReadBuffer`]'s `Drop` for the publication edge.
//!
//! A reader holds the file's extent-map read lock across its `try_read` pin, and the
//! evictor removes an extent (under that same map's write lock) before recycling its
//! slot. So a reader never pins a slot recycled out from under the extent it just
//! read: either the read lock is held and the extent is still present (the evictor's
//! removal is blocked behind it), or the extent is already gone and the lookup is a
//! miss.
//!
//! ## Lookups
//!
//! [`CompressedCache::get`] takes a file byte range and returns one [`CacheLookup`] per
//! contiguous run it resolves to - a *hit* run already resident in one slot, a
//! *refill* run mapped but not yet read, or a freshly *allocated* miss run. Each
//! lookup carries its `data` (a zero-copy view into the slot it lives in) plus its
//! `missing` blocks (empty on a hit). Concatenating every lookup's `data` in order
//! reproduces the requested bytes once every [`MissingExtent`] has been read into
//! its slot at [`MissingExtent::dest`] and marked valid via [`MissingExtent::commit`].
//! Callers feed the fragments into a scattered reader, so more (smaller) fragments
//! are fine.

use crate::io::FileLocation;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range is
/// rounded out to whole blocks.
const BLOCK_SIZE: usize = 4096;
/// Number of 4 KB blocks per 2 MB slot (512).
const BLOCKS_PER_SLOT: usize = BUFFER_SIZE / BLOCK_SIZE;
/// Number of `u64` words in a slot's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = BLOCKS_PER_SLOT / 64;

/// One cached run: a contiguous stretch of a file living in a contiguous region of
/// one 2 MB ring slot. The same run seen on two 4 KB-block rulers - file blocks and
/// slot blocks - offset from each other:
///
/// ```text
///   file block:   … 48   49   50   51 …    first_file_block = 48, block_count = 4
///                    │    │    │    │
///   slot block:     130  131  132  133 …   first_slot_block = 130   (inside slot_idx)
/// ```
///
/// So the run's Nth block is file block `first_file_block + N` and slot block
/// `first_slot_block + N`. Never spans more than one slot.
#[derive(Clone, Copy)]
struct Extent {
    /// The run's first file block. Also its key in the file's extent map.
    first_file_block: usize,
    /// The ring slot holding the run's bytes.
    slot_idx: u32,
    /// The run's first block within that slot (0..512).
    first_slot_block: u16,
    /// Length of the run in 4 KB blocks - the same count on both rulers.
    block_count: u16,
}

impl Extent {
    /// The run's last file block (inclusive).
    fn last_file_block(&self) -> usize {
        self.first_file_block + self.block_count as usize - 1
    }

    /// The slot block holding file block `file_block`: the run's slot-block start
    /// plus how far `file_block` sits into the run. Meaningful only for a
    /// `file_block` the run covers.
    fn slot_block_of(&self, file_block: usize) -> usize {
        self.first_slot_block as usize + (file_block - self.first_file_block)
    }

    /// A zero-copy view of the run's bytes that fall within the requested byte range
    /// `[start_offset, end_offset)`, read straight from its slot. Maps the file window
    /// onto the matching slice of the slot; `pin` must hold that slot and keeps it
    /// alive for as long as the returned [`Bytes`] lives.
    fn clipped_bytes(
        &self,
        pin: &Arc<ReadBuffer>,
        start_offset: usize,
        end_offset: usize,
    ) -> Bytes {
        let run_start = self.first_file_block * BLOCK_SIZE;
        let clip_start = run_start.max(start_offset);
        let clip_end =
            ((self.first_file_block + self.block_count as usize) * BLOCK_SIZE).min(end_offset);
        let slot_start = self.first_slot_block as usize * BLOCK_SIZE + (clip_start - run_start);
        Bytes::from_owner(SlotPin(pin.clone()))
            .slice(slot_start..slot_start + (clip_end - clip_start))
    }
}

/// One `(file, first_file_block)` packed into a slot, recorded so the evictor can
/// drop the run's extent when it recycles the slot.
struct Tenant {
    /// The file the packed run belongs to.
    location: FileLocation,
    /// The run's first file block - the key into `location`'s extent map.
    first_file_block: usize,
}

/// A worker's bump cursor over a shared fill slot. The slot is held as a
/// [`ReadBuffer`] (a reader pin) so runs already packed into it stay readable and
/// pinned while the worker keeps filling its tail. One per worker, owned by its
/// [`MemoryContext`](crate::memory::context).
pub struct FillCursor {
    /// The slot currently being packed, pinned. `None` before the first miss and
    /// after [`CompressedCache::clear`].
    buffer: Option<Arc<ReadBuffer>>,
    /// Ring slot index of `buffer`.
    slot_idx: usize,
    /// Bump pointer: the next free block in `buffer` (0..=512).
    next_slot_block: u16,
}

impl FillCursor {
    /// An empty cursor - the next miss takes a fresh fill buffer.
    pub fn empty() -> Self {
        FillCursor {
            buffer: None,
            slot_idx: 0,
            next_slot_block: 0,
        }
    }
}

/// The result of a [`CompressedCache::get`] over one resolved run: the looked-up bytes
/// (a zero-copy view into the slot the run lives in) plus the part(s) still missing
/// from it. `missing` is empty on a hit; otherwise `data` only becomes valid once
/// every [`MissingExtent`] has been read & committed.
pub struct CacheLookup {
    /// Zero-copy view of this run's bytes inside its ring slot (kept alive by the
    /// pin the `Bytes` owns). Only valid to read once `missing` is filled.
    data: Bytes,
    /// The part(s) of this run not yet resident - read each into its slot and
    /// commit. Empty on a hit.
    missing: Vec<MissingExtent>,
}

impl CacheLookup {
    /// The run's not-yet-resident part(s); each must be read & committed before
    /// [`data`](Self::into_data) is valid. Empty on a hit.
    pub fn missing(&self) -> &[MissingExtent] {
        &self.missing
    }

    /// Take the looked-up bytes - valid to read once every [`missing`](Self::missing)
    /// part has been filled. A zero-copy view into the cache slot, kept alive by
    /// the returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// A run still missing from the cache: the [`Extent`] to fill (its file→slot mapping)
/// plus a pin keeping its slot alive while the read is in flight. The read targets
/// [`dest`] - a region inside the pinned slot - directly, with no intermediate
/// buffer, and once it lands [`commit`] marks the whole run valid.
///
/// The pin is shared with the lookup's `data`, so the slot can't be evicted while a
/// read into it is outstanding - even if the owning query is cancelled first.
///
/// [`dest`]: MissingExtent::dest
/// [`commit`]: MissingExtent::commit
#[derive(Clone)]
pub struct MissingExtent {
    /// The run to read: which file blocks map to which slot blocks.
    extent: Extent,
    /// Keeps the destination slot pinned for the read's whole lifetime.
    _pin: Arc<ReadBuffer>,
}

impl MissingExtent {
    /// The run `extent`, backed by the slot `pin` holds; `extent.slot_idx` is `pin`'s
    /// slot, and `pin`'s base address gives [`dest`](Self::dest).
    fn new(extent: Extent, pin: Arc<ReadBuffer>) -> Self {
        MissingExtent { extent, _pin: pin }
    }

    /// File offset the run's bytes are read from - always a multiple of
    /// [`BLOCK_SIZE`], the direct-I/O alignment.
    pub fn file_offset(&self) -> usize {
        self.extent.first_file_block * BLOCK_SIZE
    }

    /// Number of bytes to read - a multiple of `BLOCK_SIZE`.
    pub fn len(&self) -> usize {
        self.extent.block_count as usize * BLOCK_SIZE
    }

    /// Whether the run covers zero bytes. Pairs with [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.extent.block_count == 0
    }

    /// Split off the sub-run covering `[rel_offset, rel_offset + len)` within this run
    /// (both relative to its start and both `BLOCK_SIZE`-aligned). The sub-run shares
    /// the slot pin, reads into the matching slice of the same slot, and
    /// [`commit`](Self::commit)s only its own blocks. Lets one missing run be filled
    /// from several transports, each part writing its own slice.
    pub fn carve(&self, rel_offset: usize, len: usize) -> MissingExtent {
        debug_assert_eq!(rel_offset % BLOCK_SIZE, 0);
        debug_assert_eq!(len % BLOCK_SIZE, 0);
        debug_assert!(rel_offset + len <= self.len());
        let blocks_in = rel_offset / BLOCK_SIZE;
        MissingExtent {
            extent: Extent {
                first_file_block: self.extent.first_file_block + blocks_in,
                slot_idx: self.extent.slot_idx,
                first_slot_block: self.extent.first_slot_block + blocks_in as u16,
                block_count: (len / BLOCK_SIZE) as u16,
            },
            _pin: self._pin.clone(),
        }
    }

    /// The destination to read the run's [`len`](Self::len) bytes into: a 4 KB-aligned
    /// region `[dest, dest+len)` inside the pinned slot - a valid O_DIRECT target. The
    /// slot stays alive because this holds a pin, and only this run's (currently
    /// invalid) blocks live here, so the read never races a reader of valid bytes.
    pub fn dest(&self) -> *mut u8 {
        (self._pin.ptr as usize + self.extent.first_slot_block as usize * BLOCK_SIZE) as *mut u8
    }

    /// Mark the run's blocks valid - call once its bytes have been read into the slot.
    pub fn commit(&self) {
        memory_ctx()
            .compressed_cache()
            .entry(self.extent.slot_idx as usize)
            .valid
            .set(
                self.extent.first_slot_block as usize,
                self.extent.block_count as usize,
            );
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

/// A slot's validity bitmap: bit `slot_block` set means that 4 KB block of the slot
/// has been read into it. Keeps all the word/bit indexing - and the memory ordering
/// that makes the in-place fill sound - in one place.
#[derive(Default)]
struct ValidBitmap([AtomicU64; BITMAP_WORDS]);

impl ValidBitmap {
    /// Is slot block `slot_block` present? `Acquire` pairs with `set`'s `Release`, so
    /// a reader that observes the bit is guaranteed to see the block's bytes.
    fn is_set(&self, slot_block: usize) -> bool {
        self.0[slot_block / 64].load(Ordering::Acquire) & (1 << (slot_block % 64)) != 0
    }

    /// Mark slot blocks `[first, first + count)` present. `Release` so it only
    /// becomes visible after the bytes have landed in the slot.
    fn set(&self, first: usize, count: usize) {
        for slot_block in first..first + count {
            self.0[slot_block / 64].fetch_or(1 << (slot_block % 64), Ordering::Release);
        }
    }

    /// Reset every block to absent. Sound only while the slot is held exclusively
    /// (no concurrent readers).
    fn clear(&self) {
        for word in &self.0 {
            word.store(0, Ordering::Relaxed);
        }
    }
}

/// Per-slot validity and tenancy. One per ring slot. The CLOCK state (recency
/// counter, owner, sweep hand) lives in the shared [`Clock`](super::clock::Clock).
struct Entry {
    /// Which of the slot's 512 × 4 KB blocks have actually been read in.
    valid: ValidBitmap,
    /// The runs currently packed here, one per `(file, first_file_block)`, so
    /// eviction can drop them from their files' maps. Written only by the worker
    /// filling the slot (under its fill pin); read only by the evictor (under
    /// `try_write`).
    tenants: UnsafeCell<Vec<Tenant>>,
}

/// One resolved run: its [`CacheLookup`] and the next file block to resolve from.
struct ResolvedRun {
    lookup: CacheLookup,
    next_file_block: usize,
}

/// A CLOCK-eviction cache over the shared [`Ring`](super::Ring) that packs many
/// small reads per slot.
///
/// Runs are bucketed by [`FileLocation`], so the same cache serves both local
/// files and remote HTTP objects - only the transport that fills a missing block
/// differs.
pub struct CompressedCache {
    /// Per-file index of what's cached and where it physically lives: file → (a
    /// run's first file block → its [`Extent`] placement). Each file's extent map is
    /// sorted by `first_file_block` and its extents never overlap. Both levels are
    /// `RwLock`'d - the file map changes rarely (open/prune a file), a file's extent
    /// map per cached run.
    ///
    /// ```text
    ///   foo.parquet ─┬─ block 0  → Extent { slot 7, slot block 130, 3 blocks }
    ///                └─ block 40 → Extent { slot 3, slot block 88,  2 blocks }
    ///   bar.parquet ─── block 12 → Extent { slot 7, slot block 200, 1 block  }
    /// ```
    file_maps: RwLock<HashMap<FileLocation, RwLock<BTreeMap<usize, Extent>>>>,
    /// Per-slot metadata, indexed by ring slot. `UnsafeCell` because `tenants` is
    /// mutated through a shared `&self` under the pin/exclusivity discipline.
    entries: Box<[UnsafeCell<Entry>]>,
}

unsafe impl Send for CompressedCache {}
unsafe impl Sync for CompressedCache {}

impl CompressedCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            file_maps: Default::default(),
            entries: (0..capacity)
                .map(|_| {
                    UnsafeCell::new(Entry {
                        valid: Default::default(),
                        tenants: UnsafeCell::new(Vec::new()),
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
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
        let end_offset = offset + len;
        let first_file_block = offset / BLOCK_SIZE;
        let last_file_block = (end_offset - 1) / BLOCK_SIZE;

        let mut lookups = Vec::new();
        let mut file_block = first_file_block;
        while file_block <= last_file_block {
            // Resolve one run: a hit/refill from an existing extent, or allocate the
            // uncovered run into the fill slot. Allocation declines only when another
            // worker just covered `file_block`, so retrying resolves it as a hit.
            let run = loop {
                if let Some(run) = self.resolve_from_extent(
                    location,
                    file_block,
                    last_file_block,
                    offset,
                    end_offset,
                ) {
                    break run;
                }
                if let Some(run) =
                    self.allocate_run(location, file_block, last_file_block, offset, end_offset)
                {
                    break run;
                }
            };
            file_block = run.next_file_block;
            lookups.push(run.lookup);
        }
        lookups
    }

    /// Resolve the run starting at `file_block` from the extent covering it, if
    /// resident: pin its slot and take the maximal span of equal validity (bounded by
    /// the extent and `last_file_block`). A valid span is a hit (empty `missing`); an
    /// invalid one is a refill carrying a [`MissingExtent`]. `None` when no extent
    /// covers `file_block`, or its slot is mid-reclaim - either way, pack it.
    fn resolve_from_extent(
        &self,
        location: &FileLocation,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<ResolvedRun> {
        let file_maps = self.file_maps.read().unwrap();
        let extents = file_maps.get(location)?.read().unwrap();
        let covering = find_extent_covering(&extents, file_block)?;

        let pin = Arc::new(memory_ctx().ring().try_read(covering.slot_idx as usize)?);
        memory_ctx().clock().touch(covering.slot_idx as usize);
        let valid = &self.entry(covering.slot_idx as usize).valid;

        // Grow the run over blocks of the same validity, bounded by the covering extent.
        let run_last_file_block = covering.last_file_block().min(last_file_block);
        let is_valid = valid.is_set(covering.slot_block_of(file_block));
        let mut last = file_block;
        while last < run_last_file_block
            && valid.is_set(covering.slot_block_of(last + 1)) == is_valid
        {
            last += 1;
        }

        let extent = Extent {
            first_file_block: file_block,
            slot_idx: covering.slot_idx,
            first_slot_block: covering.slot_block_of(file_block) as u16,
            block_count: (last - file_block + 1) as u16,
        };
        let data = extent.clipped_bytes(&pin, start_offset, end_offset);
        let missing = if is_valid {
            Vec::new()
        } else {
            vec![MissingExtent::new(extent, pin)]
        };
        Some(ResolvedRun {
            lookup: CacheLookup { data, missing },
            next_file_block: file_block + extent.block_count as usize,
        })
    }

    /// Handle a miss at `file_block`: bump-allocate the uncovered run into this
    /// worker's fill slot and register its extent, returning the refill lookup - a
    /// view over the reserved slot region plus the [`MissingExtent`] whose bytes must
    /// be read in. The run is capped by whichever runs out first, the request or the
    /// fill slot's remaining room, so a read spilling past the slot boundary continues
    /// in the next run. `None` if another worker covered `file_block` first (re-resolve
    /// it as a hit).
    fn allocate_run(
        &self,
        location: &FileLocation,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<ResolvedRun> {
        let cursor = memory_ctx().fill_cursor();
        if cursor.buffer.is_none() || cursor.next_slot_block as usize == BLOCKS_PER_SLOT {
            // Rotation may evict, so no map lock is held across it.
            self.rotate_fill_buffer(cursor);
        }
        let first_slot_block = cursor.next_slot_block as usize;
        // Blocks the run may cover: whichever runs out first, the request or the room
        // left in the fill slot.
        let requested_blocks =
            (last_file_block - file_block + 1).min(BLOCKS_PER_SLOT - first_slot_block);

        let extent = self.claim_run(
            location,
            cursor.slot_idx,
            first_slot_block,
            file_block,
            requested_blocks,
        )?;

        cursor.next_slot_block = extent.first_slot_block + extent.block_count;
        let pin = cursor.buffer.clone().unwrap();
        let data = extent.clipped_bytes(&pin, start_offset, end_offset);
        Some(ResolvedRun {
            lookup: CacheLookup {
                data,
                missing: vec![MissingExtent::new(extent, pin)],
            },
            next_file_block: file_block + extent.block_count as usize,
        })
    }

    /// Reserve a run of up to `requested_blocks` free blocks at `file_block` in
    /// `location`'s extent map, place it at `first_slot_block` of slot `slot_idx`, and
    /// record its tenant + extent.
    ///
    /// `None` when another worker already covered `file_block`: two workers can miss
    /// the same block and both try to pack it; the first to take the file's write lock
    /// inserts an extent, so the second finds the block covered and backs off (its
    /// caller then re-resolves it as a hit).
    fn claim_run(
        &self,
        location: &FileLocation,
        slot_idx: usize,
        first_slot_block: usize,
        file_block: usize,
        requested_blocks: usize,
    ) -> Option<Extent> {
        // Reserve within one file's extent map, held under its own write lock.
        let reserve = |extents_lock: &RwLock<BTreeMap<usize, Extent>>| -> Option<Extent> {
            let mut extents = extents_lock.write().unwrap();
            let block_count = claimable_blocks(&extents, file_block, requested_blocks);
            if block_count == 0 {
                return None;
            }
            let extent = Extent {
                first_file_block: file_block,
                slot_idx: slot_idx as u32,
                first_slot_block: first_slot_block as u16,
                block_count: block_count as u16,
            };
            // Record the tenant before the extent: an extent with no tenant would leak
            // (eviction only drops runs listed as tenants), while a tenant with no
            // extent is harmless (eviction skips it).
            self.tenants_mut(slot_idx).push(Tenant {
                location: location.clone(),
                first_file_block: file_block,
            });
            extents.insert(file_block, extent);
            // Mark it referenced so the fresh run survives one CLOCK sweep.
            memory_ctx().clock().touch(slot_idx);
            Some(extent)
        };

        // Fast path: the file's map already exists (`open_entry` registered it), so
        // reserve under the cheap outer read lock, which blocks a concurrent prune from
        // dropping the file mid-reservation. The read guard is a named binding so it
        // drops at the block's end - a thread can't hold this `RwLock` for read and
        // write at once, so it must be released before the miss path below.
        {
            let file_maps = self.file_maps.read().unwrap();
            if let Some(extents_lock) = file_maps.get(location) {
                return reserve(extents_lock);
            }
        }
        // Rare: a prior eviction pruned the whole file. Recreate its map and reserve
        // under one outer write lock - no lock upgrade, and no prune can race in.
        reserve(
            self.file_maps
                .write()
                .unwrap()
                .entry(location.clone())
                .or_default(),
        )
    }

    /// Point the fill cursor at a fresh, empty slot: reset its bitmap and tenant list
    /// while it is held exclusively, then publish it as readable.
    fn rotate_fill_buffer(&self, cursor: &mut FillCursor) {
        let write_buffer = memory_ctx().get_write_buffer(false);
        let slot_idx = write_buffer.slot_idx;
        let entry = self.entry(slot_idx);
        entry.valid.clear();
        self.tenants_mut(slot_idx).clear();
        memory_ctx().clock().bind(slot_idx, Owner::Compressed);
        cursor.buffer = Some(Arc::new(ReadBuffer::from(write_buffer))); // used = 1, Release
        cursor.slot_idx = slot_idx;
        cursor.next_slot_block = 0;
    }

    /// Borrow ring slot `idx`'s cache metadata. `valid` is always safe to touch;
    /// `tenants` is sound to touch only under the fill pin (filler) or `try_write`
    /// exclusivity (evictor).
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

    /// Reset a slot's validity and release it from the clock. Sound only while the
    /// slot is held exclusively (`try_write` succeeded); the caller must have already
    /// taken or cleared its tenant list.
    fn recycle_slot(&self, idx: usize) {
        let entry = self.entry(idx);
        entry.valid.clear();
        memory_ctx().clock().release(idx);
    }

    /// Reclaim compressed slot `slot_idx` if it can be taken exclusively: drop every
    /// run packed in it from its file's extent map, recycle it, and return it
    /// writable. `None` when a reader still pins it or it is no longer compressed.
    pub(crate) fn reclaim(&self, slot_idx: usize) -> Option<WriteBuffer> {
        let write_buffer = memory_ctx().ring().try_write(slot_idx)?;
        if memory_ctx().clock().owner(slot_idx) != Owner::Compressed {
            write_buffer.release_in_place();
            return None;
        }

        // Exclusive now, so the tenant list is complete and stable.
        let tenants = std::mem::take(self.tenants_mut(slot_idx));
        let mut emptied_files = Vec::new();
        {
            let file_maps = self.file_maps.read().unwrap();
            for tenant in tenants {
                let Some(extents_lock) = file_maps.get(&tenant.location) else {
                    continue;
                };
                let mut extents = extents_lock.write().unwrap();
                // Skip a run re-placed into another slot after this tenant was recorded.
                if extents
                    .get(&tenant.first_file_block)
                    .is_some_and(|extent| extent.slot_idx as usize == slot_idx)
                {
                    extents.remove(&tenant.first_file_block);
                }
                if extents.is_empty() {
                    emptied_files.push(tenant.location);
                }
            }
        }
        self.prune_empty_file_locations(emptied_files);
        self.recycle_slot(slot_idx);
        Some(write_buffer)
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
        // Drop any decompressed pages cached under this (possibly reused) location
        // for the same reason the compressed map is reset below: a reopened fd may
        // now name a different file, so its old pages must not serve a later read.
        memory_ctx().decompressed_cache().invalidate(&location);
        // The `file_maps` write lock serializes with `pack_gap`'s read, so a racing
        // miss either sees the fresh empty map or has its run dropped here.
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
        // below rather than skipped as pinned.
        memory_ctx().fill_cursor().buffer = None;

        // Drain every file's extents, collecting the slots they lived in.
        let mut slots = HashSet::new();
        let mut dropped = 0;
        let file_maps = self.file_maps.read().unwrap();
        for extents_lock in file_maps.values() {
            for (_, extent) in std::mem::take(&mut *extents_lock.write().unwrap()) {
                slots.insert(extent.slot_idx as usize);
                dropped += 1;
            }
        }
        drop(file_maps);

        // Recycle each slot we can take exclusively; one another worker is still
        // filling stays pinned and is left intact (its runs are already dropped).
        for slot_idx in slots {
            if let Some(write_buffer) = memory_ctx().ring().try_write(slot_idx) {
                self.tenants_mut(slot_idx).clear();
                self.recycle_slot(slot_idx);
                drop(write_buffer); // returns the slot to the free pool
            }
        }
        dropped
    }
}

/// The extent covering file block `file_block`, if any: the greatest-keyed extent at
/// or before it that reaches it. Extents never overlap, so at most one qualifies.
fn find_extent_covering(extents: &BTreeMap<usize, Extent>, file_block: usize) -> Option<Extent> {
    let (_, &extent) = extents.range(..=file_block).next_back()?;
    (extent.last_file_block() >= file_block).then_some(extent)
}

/// How many blocks from `file_block` are free of any extent, up to `want` - the
/// claimable prefix of the gap. Zero if `file_block` is already covered.
fn claimable_blocks(extents: &BTreeMap<usize, Extent>, file_block: usize, want: usize) -> usize {
    if find_extent_covering(extents, file_block).is_some() {
        return 0;
    }
    let next_extent = extents
        .range(file_block + 1..)
        .next()
        .map(|(&first, _)| first)
        .unwrap_or(usize::MAX);
    (next_extent - file_block).min(want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    const SB: usize = BLOCK_SIZE;

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

    fn cache() -> &'static CompressedCache {
        memory_ctx().compressed_cache()
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

    /// Every missing run across a set of lookups, cloned.
    fn missing_blocks(lookups: &[CacheLookup]) -> Vec<MissingExtent> {
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
    fn coalesces_contiguous_missing_blocks_into_one_extent() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Three blocks, none present → one freshly-packed run, one missing extent.
        let miss = cache().get(&FD(), 0, 2 * SB + 1);
        let blocks = missing_blocks(&miss);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].extent.block_count, 3);
        assert_eq!(blocks[0].len(), 3 * SB);
    }

    #[test]
    fn partial_validity_only_reports_the_holes() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Fill block 0 only.
        let first = cache().get(&FD(), 0, 10);
        fill_pattern(&first);
        drop(first);

        // Ask for blocks 0..=2; only 1 and 2 are missing.
        let again = cache().get(&FD(), 0, 2 * SB + 1);
        let blocks = missing_blocks(&again);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file_offset(), SB);
        assert_eq!(blocks[0].extent.block_count, 2);
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
        assert_eq!(
            a.extent.slot_idx, b.extent.slot_idx,
            "should share the fill slot"
        );
        assert_eq!(a.extent.first_slot_block, 0);
        assert_eq!(
            b.extent.first_slot_block, 1,
            "second read packed right after the first"
        );
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
            ra[0].missing()[0].extent.slot_idx,
            rb[0].missing()[0].extent.slot_idx,
            "reads from different files share one packed slot"
        );
    }

    #[test]
    fn a_read_past_the_fill_buffer_uses_a_new_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // One read spanning more than a whole slot (513 blocks) splits at the 2 MB
        // boundary into two runs in two slots.
        let miss = cache().get(&FD(), 0, BLOCKS_PER_SLOT * SB + 1);
        let blocks = missing_blocks(&miss);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].extent.block_count as usize, BLOCKS_PER_SLOT);
        assert_eq!(blocks[1].extent.block_count, 1);
        assert_ne!(blocks[0].extent.slot_idx, blocks[1].extent.slot_idx);

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
        assert_eq!(blocks[0].extent.block_count, 2);
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

    /// The CLOCK evictor must not reclaim a ring slot that is still parked in
    /// the free pool (unbound).
    #[test]
    fn evict_does_not_steal_a_slot_from_the_free_pool() {
        init_test_free_pool(2);
        cache().open_entry(FD());
        // A real cached run binds a slot; its set ref_bit makes the CLOCK hand skip
        // it once, so the evictor reaches a slot that is still in the pool.
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();

        let evicted = memory_ctx().evict();
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
        memory_ctx().evict();

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

        // Evict the slot the run lived in; its extent is dropped.
        memory_ctx().evict();

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

        memory_ctx().evict();
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
