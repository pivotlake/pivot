//! A CLOCK page cache that *packs* many small file reads into shared ring buffer
//! slots, tracking validity at 4 KB block granularity.
//!
//! Reads are not positional: a missed read is bump-allocated as a contiguous extent
//! of 4 KB blocks at a per-worker cursor inside whatever shared slot that worker is
//! currently filling, so hundreds of unrelated reads from many files share one slot
//! and the working set is measured in *bytes*, not slots. The cache therefore
//! records each cached extent's *physical placement* - which slot and which block
//! within it - in a per-file [`Extent`] map, decoupled from the file offset.
//!
//! ## Extents
//!
//! Each file maps `first_file_block` (= `file_offset / 4096`) → [`Extent`]. Extents
//! are non-overlapping by construction: allocation only ever covers blocks no live
//! extent already claims. A slot's [`SlotMetadata::valid`] bitmap records which of its
//! 512 × 4 KB blocks have actually been read, so an extent can be present in the map
//! yet still filling.
//!
//! ## Multi-tenant slots and eviction
//!
//! A slot holds extents from arbitrarily many files. CLOCK eviction reclaims a whole
//! slot, so it must invalidate every extent living in it. A per-slot tenant list -
//! the `(file, first_file_block)` of every extent packed into the slot - records what
//! to drop. It is written only by the single worker filling the slot (while it holds
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
//! [`CompressedCache::get`] takes a file byte range and returns one [`CacheLookup`]
//! per contiguous fragment it resolves to - a *hit* already resident in one slot, a
//! *refill* mapped but not yet read, or a freshly *allocated* miss. Each
//! lookup carries its `data` (a zero-copy view into the slot it lives in) plus its
//! `missing` blocks (empty on a hit). A new extent owns its fill; an invalid extent
//! already in the map follows that owner's request. Concatenating every lookup's
//! `data` in order reproduces the requested bytes once every [`MissingExtent`] is
//! committed. Callers feed the fragments into a scattered reader, so more (smaller)
//! fragments are fine.

use crate::io::OpenFile;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::fill_cursor::FillCursor;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::{ProbedSlot, WriteBuffer};
use ahash::HashMap;
use bytes::Bytes;
use crossbeam_deque::{Injector, Steal};
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range is
/// rounded out to whole blocks.
const BLOCK_SIZE: usize = 4096;
/// Number of 4 KB blocks per 2 MB slot (512).
const BLOCKS_PER_SLOT: usize = BUFFER_SIZE / BLOCK_SIZE;
/// Number of `u64` words in a slot's validity bitmap (8 → 512 bits).
const BITMAP_WORDS: usize = BLOCKS_PER_SLOT / 64;

/// One cached extent: a mapping from a contiguous range of file blocks to a
/// contiguous range of blocks within one 2 MB ring slot. The same extent appears on
/// two offset 4 KB-block rulers - file blocks and slot blocks:
///
/// ```text
///   file block:   … 48   49   50   51 …    first_file_block = 48, block_count = 4
///                    │    │    │    │
///   slot block:     130  131  132  133 …   first_slot_block = 130   (inside slot_idx)
/// ```
///
/// So the extent's Nth block is file block `first_file_block + N` and slot block
/// `first_slot_block + N`. Never spans more than one slot.
#[derive(Clone)]
struct Extent {
    /// The extent's first file block. Also its key in the file's extent map.
    first_file_block: usize,
    /// The ring slot holding the extent's bytes.
    slot_idx: u32,
    /// The extent's first block within that slot (0..512).
    first_slot_block: u16,
    /// Length of the extent in 4 KB blocks - the same count on both rulers.
    block_count: u16,
    /// Set when the worker filling this extent fails. Followers retain this
    /// shared flag after the failed extent is removed from the map.
    failed: Arc<AtomicBool>,
    /// Workers waiting for this extent's fill to finish. The owner drains the
    /// queue after either committing bytes or reporting failure.
    subscribing_workers: Arc<Injector<usize>>,
}

impl Extent {
    /// The extent's last file block (inclusive).
    fn last_file_block(&self) -> usize {
        self.first_file_block + self.block_count as usize - 1
    }

    /// The slot block holding file block `file_block`: the extent's slot-block start
    /// plus how far `file_block` sits into the extent. Meaningful only for a
    /// `file_block` the extent covers.
    fn slot_block_of(&self, file_block: usize) -> usize {
        self.first_slot_block as usize + (file_block - self.first_file_block)
    }

    /// A zero-copy view of the extent's bytes that fall within the requested byte
    /// range `[start_offset, end_offset)`, read straight from its slot. Maps the file
    /// window onto the matching slice of the slot; `pin` must hold that slot and
    /// keeps it alive for as long as the returned [`Bytes`] lives.
    fn clipped_bytes(
        &self,
        pin: &Arc<ReadBuffer>,
        start_offset: usize,
        end_offset: usize,
    ) -> Bytes {
        let extent_start = self.first_file_block * BLOCK_SIZE;
        let clip_start = extent_start.max(start_offset);
        let clip_end =
            ((self.first_file_block + self.block_count as usize) * BLOCK_SIZE).min(end_offset);
        let slot_start = self.first_slot_block as usize * BLOCK_SIZE + (clip_start - extent_start);
        Bytes::from_owner(SlotPin(pin.clone()))
            .slice(slot_start..slot_start + (clip_end - clip_start))
    }
}

/// One `(file, first_file_block)` packed into a slot, recorded so the evictor can
/// drop the extent when it recycles the slot.
struct Tenant {
    /// The file the packed extent belongs to.
    open_file: OpenFile,
    /// The extent's first file block - the key into `open_file`'s extent map.
    first_file_block: usize,
}

/// One fragment of a [`CompressedCache::get`] result: looked-up bytes (a zero-copy
/// view into the slot that holds them) plus the fragment's missing part, if any.
/// A fragment is resolved whole, so it is either fully resident (a hit) or has one
/// [`MissingExtent`] to resolve; its fill owner reads it while later lookups wait
/// for that owner. `data` only becomes valid once the extent has committed.
pub struct CacheLookup {
    /// Zero-copy view of this fragment's bytes inside its ring slot (kept alive by
    /// the pin the `Bytes` owns). Only valid to read once `missing` is filled.
    data: Bytes,
    /// The fragment's not-yet-resident part - read it into its slot and commit.
    /// `None` on a hit.
    missing: Option<MissingExtent>,
}

impl CacheLookup {
    /// The fragment's not-yet-resident part. Its fill owner must read and commit
    /// it; a follower waits for that owner. `None` on a hit.
    pub fn missing(&self) -> Option<&MissingExtent> {
        self.missing.as_ref()
    }

    /// Take the looked-up bytes - valid to read once every [`missing`](Self::missing)
    /// part has been filled. A zero-copy view into the cache slot, kept alive by
    /// the returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// An extent still missing from the cache: its file→slot mapping, whether this
/// lookup owns the fill, and a pin keeping its slot alive. The owning read targets
/// [`dest`] directly, with no intermediate buffer, and once it lands [`commit`]
/// marks the whole extent valid. A follower retains the same mapping and waits for
/// either those valid bits or the shared failure flag.
///
/// The pin is shared with the lookup's `data`, so the slot can't be evicted while a
/// read into it is outstanding - even if the owning query is cancelled first.
///
/// [`dest`]: MissingExtent::dest
/// [`commit`]: MissingExtent::commit
#[derive(Clone)]
pub struct MissingExtent {
    /// The extent to read: which file blocks map to which slot blocks.
    extent: Extent,
    /// The fill this lookup is responsible for, when it inserted the extent.
    /// `None` for a follower: an invalid extent found in the map is already
    /// being filled elsewhere.
    ownership: Option<Arc<FillOwnership>>,
    /// Keeps the destination slot pinned for the read's whole lifetime.
    _pin: Arc<ReadBuffer>,
}

/// The fill an owning lookup answers for. Registering the extent published it
/// to every other reader of the range as "a fill is in flight", and they
/// subscribe and wait for it to commit or fail. A fill can be abandoned with
/// no code of the owner's running: the read that would have issued it unwinds
/// after a panic on the way to the ring, or is dropped from a backlog with the
/// dataflow that queued it. So the ownership itself settles the extent when
/// dropped: unless a commit or an explicit removal settled it first, it
/// publishes failure, removes the extent so a later reader starts a fill of
/// its own, and wakes the followers, who then fail the way they do for a fill
/// whose read failed.
struct FillOwnership {
    open_file: OpenFile,
    extent: Extent,
    settled: AtomicBool,
}

impl FillOwnership {
    /// Give the fill up: publish failure, take the extent out of the map so a
    /// later reader starts a fill of its own, and wake the followers.
    fn remove_from_cache(&self) {
        self.settled.store(true, Ordering::Release);
        memory_ctx()
            .compressed_cache()
            .remove_extent(&self.open_file, &self.extent);
        self.extent.wake_subscribers();
    }
}

impl Drop for FillOwnership {
    fn drop(&mut self) {
        if self.settled.load(Ordering::Acquire) || !crate::memory::has_memory_context() {
            return;
        }
        self.remove_from_cache();
    }
}

impl Extent {
    /// Wake every worker that subscribed to this extent's outcome.
    fn wake_subscribers(&self) {
        loop {
            match self.subscribing_workers.steal() {
                Steal::Success(worker_id) => crate::waker::waker_set().notify_worker(worker_id),
                Steal::Retry => continue,
                Steal::Empty => return,
            }
        }
    }
}

impl MissingExtent {
    /// The missing `extent`, backed by the slot `pin` holds; `extent.slot_idx` is `pin`'s
    /// slot, and `pin`'s base address gives [`dest`](Self::dest).
    fn new(extent: Extent, pin: Arc<ReadBuffer>, ownership: Option<Arc<FillOwnership>>) -> Self {
        MissingExtent {
            extent,
            ownership,
            _pin: pin,
        }
    }

    /// The extent `open_file`'s lookup just inserted, owned by that lookup.
    fn owned(extent: Extent, pin: Arc<ReadBuffer>, open_file: &OpenFile) -> Self {
        let ownership = FillOwnership {
            open_file: open_file.clone(),
            extent: extent.clone(),
            settled: AtomicBool::new(false),
        };
        Self::new(extent, pin, Some(Arc::new(ownership)))
    }

    /// Whether this lookup is responsible for filling the extent. A non-owner
    /// follows the request that inserted the extent instead of issuing another.
    pub(crate) fn fill_owner(&self) -> bool {
        self.ownership.is_some()
    }

    /// Note that the fill reached an outcome the followers can see, so dropping
    /// the ownership has nothing left to publish.
    fn settle(&self) {
        if let Some(ownership) = &self.ownership {
            ownership.settled.store(true, Ordering::Release);
        }
    }

    /// Register a worker to be woken when the owning fill finishes.
    pub(crate) fn subscribe(&self, worker_id: usize) {
        self.extent.subscribing_workers.push(worker_id);
        // If the outcome was published and its subscriber queue drained just
        // before this push, no owner remains to wake us. Notify ourselves so
        // the worker reaches its next normal completions pass without sleeping.
        if self.failed() || self.is_committed() {
            crate::waker::waker_set().notify_worker(worker_id);
        }
    }

    /// Whether every block covered by this view has been committed.
    pub(crate) fn is_committed(&self) -> bool {
        let valid = &memory_ctx()
            .compressed_cache()
            .slot_metadata(self.extent.slot_idx as usize)
            .valid;
        (0..self.extent.block_count as usize)
            .all(|block| valid.is_set(self.extent.first_slot_block as usize + block))
    }

    /// Whether the worker owning the shared fill reported failure.
    pub(crate) fn failed(&self) -> bool {
        self.extent.failed.load(Ordering::Acquire)
    }

    /// File offset the extent's bytes are read from - always a multiple of
    /// [`BLOCK_SIZE`], the direct-I/O alignment.
    pub fn file_offset(&self) -> usize {
        self.extent.first_file_block * BLOCK_SIZE
    }

    /// Number of bytes to read - a multiple of `BLOCK_SIZE`.
    pub fn len(&self) -> usize {
        self.extent.block_count as usize * BLOCK_SIZE
    }

    /// Whether the extent covers zero bytes. Pairs with [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.extent.block_count == 0
    }

    /// Split off the sub-extent covering `[rel_offset, rel_offset + len)` within this
    /// extent (both relative to its start and both `BLOCK_SIZE`-aligned). The sub-extent shares
    /// the slot pin, reads into the matching slice of the same slot, and
    /// [`commit`](Self::commit)s only its own blocks. Lets one missing extent be filled
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
                failed: self.extent.failed.clone(),
                subscribing_workers: self.extent.subscribing_workers.clone(),
            },
            ownership: self.ownership.clone(),
            _pin: self._pin.clone(),
        }
    }

    /// The destination to read the extent's [`len`](Self::len) bytes into: a 4 KB-aligned
    /// region `[dest, dest+len)` inside the pinned slot - a valid O_DIRECT target. The
    /// slot stays alive because this holds a pin, and only this extent's (currently
    /// invalid) blocks live here, so the read never races a reader of valid bytes.
    pub fn dest(&self) -> *mut u8 {
        (self._pin.ptr as usize + self.extent.first_slot_block as usize * BLOCK_SIZE) as *mut u8
    }

    /// Mark the extent's blocks valid - call once its bytes have been read into the slot.
    pub fn commit(&self) {
        self.commit_prefix(self.len());
    }

    /// Mark valid only the leading blocks the first `bytes` bytes landed in -
    /// every block that received at least one byte. For a run whose block-aligned
    /// tail overhangs the end of its file, the fully-unwritten blocks stay
    /// invalid, so a later lookup re-reads them instead of consuming whatever the
    /// slot already held.
    pub fn commit_prefix(&self, bytes: usize) {
        let blocks = bytes
            .div_ceil(BLOCK_SIZE)
            .min(self.extent.block_count as usize);
        memory_ctx()
            .compressed_cache()
            .slot_metadata(self.extent.slot_idx as usize)
            .valid
            .set(self.extent.first_slot_block as usize, blocks);
        self.settle();
    }

    /// Whether a contiguous `bytes`-byte prefix reaches the extent's final 4 KB
    /// block. Because the transfer starts at the extent's beginning, reaching
    /// the final block means every extent block received at least one byte. It
    /// does not mean every byte in the extent, or every logical byte requested,
    /// was read. Since an extent contains exactly the blocks intersecting its
    /// logical range, a prefix ending at EOF before the final block means that
    /// logical range is out of bounds against the current file size (or became
    /// stale because the file shrank after the range was formed).
    pub fn prefix_reaches_last_block(&self, bytes: usize) -> bool {
        bytes.div_ceil(BLOCK_SIZE) >= self.extent.block_count as usize
    }

    /// Mark this fill failed, remove its extent, and wake its followers. A later
    /// lookup can then allocate a fresh extent and issue a new request.
    pub(crate) fn remove_from_cache(&self, open_file: &OpenFile) {
        match &self.ownership {
            Some(ownership) => {
                debug_assert_eq!(&ownership.open_file, open_file);
                ownership.remove_from_cache();
            }
            None => {
                memory_ctx()
                    .compressed_cache()
                    .remove_extent(open_file, &self.extent);
                self.extent.wake_subscribers();
            }
        }
    }

    pub(crate) fn wake_subscribers(&self) {
        self.extent.wake_subscribers();
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
struct SlotMetadata {
    /// Which of the slot's 512 × 4 KB blocks have actually been read in.
    valid: ValidBitmap,
    /// The extents currently packed here, one per `(file, first_file_block)`, so
    /// eviction can drop them from their files' maps. Written only by the worker
    /// filling the slot (under its fill pin); read only by the evictor (under
    /// `try_write`).
    tenants: UnsafeCell<Vec<Tenant>>,
}

/// One step while resolving a cache lookup: the fragment found or allocated at the
/// current file block, and the next file block the lookup should visit.
struct CacheLookupStep {
    lookup: CacheLookup,
    next_file_block: usize,
}

/// A CLOCK-eviction cache over the shared [`Ring`](super::Ring) that packs many
/// small reads per slot.
///
/// Extents are bucketed by [`OpenFile`], so the same cache serves both local
/// files and remote HTTP objects - only the transport that fills a missing block
/// differs.
pub struct CompressedCache {
    /// Per-file index of what's cached and where it physically lives: file → (a
    /// cached extent's first file block → its [`Extent`] placement). Each per-file map
    /// is sorted by `first_file_block`, and its extents never overlap. Both the outer
    /// file map and each per-file extent map are independently `RwLock`'d.
    ///
    /// ```text
    ///   foo.parquet ─┬─ block 0  → Extent { slot 7, slot block 130, 3 blocks }
    ///                └─ block 40 → Extent { slot 3, slot block 88,  2 blocks }
    ///   bar.parquet ─── block 12 → Extent { slot 7, slot block 200, 1 block  }
    /// ```
    file_maps: RwLock<HashMap<OpenFile, RwLock<BTreeMap<usize, Extent>>>>,
    /// Per-slot metadata, indexed by ring slot. `UnsafeCell` because `tenants` is
    /// mutated through a shared `&self` under the pin/exclusivity discipline.
    slot_metadatas: Box<[UnsafeCell<SlotMetadata>]>,
}

unsafe impl Send for CompressedCache {}
unsafe impl Sync for CompressedCache {}

impl CompressedCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            file_maps: Default::default(),
            slot_metadatas: (0..capacity)
                .map(|_| {
                    UnsafeCell::new(SlotMetadata {
                        valid: Default::default(),
                        tenants: UnsafeCell::new(Vec::new()),
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    /// Look up the file byte range `[offset, offset + len)` of `open_file`. The
    /// range is resolved into a sequence of contiguous fragments in file order,
    /// yielding one [`CacheLookup`] per fragment: read and fill every lookup's
    /// [`missing`](CacheLookup::missing) blocks, then concatenate the fragments' bytes.
    pub fn get(&self, open_file: &OpenFile, offset: usize, len: usize) -> Vec<CacheLookup> {
        if len == 0 {
            return Vec::new();
        }
        let end_offset = offset + len;
        let first_file_block = offset / BLOCK_SIZE;
        let last_file_block = (end_offset - 1) / BLOCK_SIZE;

        let mut lookups = Vec::new();
        let mut file_block = first_file_block;
        while file_block <= last_file_block {
            // Resolve one lookup fragment from an existing extent, or allocate an
            // extent for an uncovered fragment. Allocation declines only when another
            // worker just covered `file_block`, so retrying resolves it as a hit.
            let step = loop {
                if let Some(step) = self.try_resolve_lookup_for_location(
                    open_file,
                    file_block,
                    last_file_block,
                    offset,
                    end_offset,
                ) {
                    break step;
                }
                if let Some(step) = self.try_allocate_lookup(
                    open_file,
                    file_block,
                    last_file_block,
                    offset,
                    end_offset,
                ) {
                    break step;
                }
            };
            file_block = step.next_file_block;
            lookups.push(step.lookup);
        }
        lookups
    }

    /// Mark `missing` failed and take it out of `open_file`'s map. Runs from an
    /// unwinding owner's drop as well, so a lock another thread poisoned is
    /// used anyway: the map is only ever read and written under it here, and
    /// leaving the extent published would be the worse outcome.
    fn remove_extent(&self, open_file: &OpenFile, missing: &Extent) {
        let file_maps = self
            .file_maps
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(extents_lock) = file_maps.get(open_file) else {
            missing.failed.store(true, Ordering::Release);
            return;
        };
        let mut extents = extents_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Publish failure while holding the map write lock that excludes a new
        // lookup from discovering this extent. New readers can allocate a fresh
        // fill after removal; existing followers retain this shared flag.
        missing.failed.store(true, Ordering::Release);
        let key = extents
            .range(..=missing.first_file_block)
            .next_back()
            .filter(|(_, extent)| {
                extent.last_file_block() >= missing.first_file_block
                    && extent.slot_idx == missing.slot_idx
                    && extent.slot_block_of(missing.first_file_block)
                        == missing.first_slot_block as usize
            })
            .map(|(&key, _)| key);
        if let Some(key) = key {
            extents.remove(&key);
        }
    }

    /// Try to resolve the lookup step at `file_block` from an existing extent for
    /// `open_file`. Pins the extent's slot and takes the maximal span of equal
    /// validity, bounded by the extent and `last_file_block`. A valid span is a hit
    /// (empty `missing`); an invalid span is a refill carrying a [`MissingExtent`].
    ///
    /// Returns `None` when the location has no extent map (its last extent may have
    /// been evicted and the empty map pruned), no extent in its map covers
    /// `file_block`, or the covering extent's slot is already held exclusively for
    /// reclamation. The caller then falls back to allocation, which recreates a
    /// pruned map or claims the uncovered range; if another worker claims it first,
    /// the caller retries this lookup path.
    fn try_resolve_lookup_for_location(
        &self,
        open_file: &OpenFile,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<CacheLookupStep> {
        let file_maps = self.file_maps.read().unwrap();
        let extents = file_maps.get(open_file)?.read().unwrap();
        let found_extent = find_extent_covering(&extents, file_block)?;

        let pin = Arc::new(
            memory_ctx()
                .ring()
                .try_read(found_extent.slot_idx as usize)?,
        );
        memory_ctx().clock().touch(found_extent.slot_idx as usize);
        let valid = &self.slot_metadata(found_extent.slot_idx as usize).valid;

        // Grow the lookup over blocks of the same validity, bounded by the covering
        // extent.
        let last_lookup_file_block = found_extent.last_file_block().min(last_file_block);
        let is_valid = valid.is_set(found_extent.slot_block_of(file_block));
        let mut last_matching_file_block = file_block;
        while last_matching_file_block < last_lookup_file_block
            && valid.is_set(found_extent.slot_block_of(last_matching_file_block + 1)) == is_valid
        {
            last_matching_file_block += 1;
        }

        let extent = Extent {
            first_file_block: file_block,
            slot_idx: found_extent.slot_idx,
            first_slot_block: found_extent.slot_block_of(file_block) as u16,
            block_count: (last_matching_file_block - file_block + 1) as u16,
            failed: found_extent.failed.clone(),
            subscribing_workers: found_extent.subscribing_workers.clone(),
        };
        let data = extent.clipped_bytes(&pin, start_offset, end_offset);
        let missing = (!is_valid).then(|| MissingExtent::new(extent.clone(), pin, None));
        Some(CacheLookupStep {
            lookup: CacheLookup { data, missing },
            next_file_block: file_block + extent.block_count as usize,
        })
    }

    /// Handle a miss at `file_block`: bump-allocate an extent in this worker's fill
    /// slot and register it, returning the refill lookup: a view over the reserved
    /// slot region plus the [`MissingExtent`] whose bytes must be read in. The extent
    /// is capped by whichever runs out first, the request or the fill slot's remaining
    /// room, so a read spilling past the slot boundary continues in the next lookup.
    /// `None` if another worker covered `file_block` first (re-resolve it as a hit).
    fn try_allocate_lookup(
        &self,
        open_file: &OpenFile,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<CacheLookupStep> {
        let cursor = memory_ctx().compressed_fill_cursor();
        if cursor.is_exhausted() {
            // Rotation may evict, so no map lock is held across it.
            self.rotate_fill_buffer(cursor);
        }
        let first_slot_block = cursor.next_byte / BLOCK_SIZE;
        // Blocks the extent may cover: whichever runs out first, the request or the
        // room left in the fill slot.
        let requested_blocks =
            (last_file_block - file_block + 1).min(BLOCKS_PER_SLOT - first_slot_block);

        let extent = self.try_add_extent_for_location(
            open_file,
            cursor.slot_idx,
            first_slot_block,
            file_block,
            requested_blocks,
        )?;

        cursor.next_byte = (extent.first_slot_block + extent.block_count) as usize * BLOCK_SIZE;
        let pin = cursor.buffer.clone().unwrap();
        let data = extent.clipped_bytes(&pin, start_offset, end_offset);
        Some(CacheLookupStep {
            lookup: CacheLookup {
                data,
                missing: Some(MissingExtent::owned(extent.clone(), pin, open_file)),
            },
            next_file_block: file_block + extent.block_count as usize,
        })
    }

    /// Add an extent of up to `requested_blocks` available blocks for `open_file`,
    /// recreating its per-file map first if eviction pruned it.
    ///
    /// `None` when another worker already covered `file_block`: two workers can miss
    /// the same block and both try to pack it; the first to take the file's write lock
    /// inserts an extent, so the second finds the block covered and backs off (its
    /// caller then re-resolves it as a hit).
    fn try_add_extent_for_location(
        &self,
        open_file: &OpenFile,
        slot_idx: usize,
        first_slot_block: usize,
        file_block: usize,
        requested_blocks: usize,
    ) -> Option<Extent> {
        // Fast path: the file's map already exists (`open_entry` registered it), so
        // register under the cheap outer read lock, which blocks a concurrent prune
        // from dropping the file mid-registration. The read guard is a named binding
        // so it drops at the block's end - a thread can't hold this `RwLock` for read
        // and write at once, so it must be released before the slow path below.
        {
            let file_maps = self.file_maps.read().unwrap();
            if let Some(extents_lock) = file_maps.get(open_file) {
                return self.try_add_extent_to_map(
                    open_file,
                    extents_lock,
                    slot_idx,
                    first_slot_block,
                    file_block,
                    requested_blocks,
                );
            }
        }
        // Rare: a prior eviction pruned the whole file. Recreate its map and register
        // under one outer write lock - no lock upgrade, and no prune can race in.
        let mut file_maps = self.file_maps.write().unwrap();
        let extents_lock = file_maps.entry(open_file.clone()).or_default();
        self.try_add_extent_to_map(
            open_file,
            extents_lock,
            slot_idx,
            first_slot_block,
            file_block,
            requested_blocks,
        )
    }

    /// Atomically add a new extent to an existing per-file map: choose the available
    /// block prefix, record the slot tenant, insert the mapping, and touch the slot.
    /// The map's write lock serializes overlapping misses, so only the first worker
    /// to cover `file_block` succeeds.
    fn try_add_extent_to_map(
        &self,
        open_file: &OpenFile,
        extents_lock: &RwLock<BTreeMap<usize, Extent>>,
        slot_idx: usize,
        first_slot_block: usize,
        file_block: usize,
        requested_blocks: usize,
    ) -> Option<Extent> {
        let mut extents = extents_lock.write().unwrap();
        let block_count = count_available_blocks(&extents, file_block, requested_blocks);
        if block_count == 0 {
            return None;
        }
        let extent = Extent {
            first_file_block: file_block,
            slot_idx: slot_idx as u32,
            first_slot_block: first_slot_block as u16,
            block_count: block_count as u16,
            failed: Arc::new(AtomicBool::new(false)),
            subscribing_workers: Arc::new(Injector::new()),
        };
        // Record the tenant before the extent: an extent with no tenant would leak
        // (eviction only drops extents listed as tenants), while a tenant with no
        // extent is harmless (eviction skips it).
        self.tenants_mut(slot_idx).push(Tenant {
            open_file: open_file.clone(),
            first_file_block: file_block,
        });
        extents.insert(file_block, extent.clone());
        // Mark it referenced so the fresh extent survives one CLOCK sweep.
        memory_ctx().clock().touch(slot_idx);
        Some(extent)
    }

    /// Point the fill cursor at a fresh, empty slot: reset its bitmap and tenant list
    /// while it is held exclusively, then publish it as readable.
    fn rotate_fill_buffer(&self, cursor: &mut FillCursor) {
        let write_buffer = memory_ctx().get_write_buffer(false);
        let slot_idx = write_buffer.slot_idx;
        let metadata = self.slot_metadata(slot_idx);
        metadata.valid.clear();
        self.tenants_mut(slot_idx).clear();
        memory_ctx().clock().bind(slot_idx, Owner::Compressed);
        cursor.buffer = Some(Arc::new(ReadBuffer::from(write_buffer))); // used = 1, Release
        cursor.slot_idx = slot_idx;
        cursor.next_byte = 0;
    }

    /// Borrow ring slot `idx`'s cache metadata. `valid` is always safe to touch;
    /// `tenants` is sound to touch only under the fill pin (filler) or `try_write`
    /// exclusivity (evictor).
    fn slot_metadata(&self, idx: usize) -> &SlotMetadata {
        unsafe { &*self.slot_metadatas[idx].get() }
    }

    /// Mutable access to slot `idx`'s tenant list. Sound only under the fill pin
    /// (filler) or `try_write` exclusivity (evictor) - the one `unsafe` for the
    /// `tenants` invariant lives here. See [`slot_metadata`](Self::slot_metadata).
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell; see above
    fn tenants_mut(&self, idx: usize) -> &mut Vec<Tenant> {
        unsafe { &mut *self.slot_metadata(idx).tenants.get() }
    }

    /// Reset a slot's validity and release it from the clock. Sound only while the
    /// slot is held exclusively (`try_write` succeeded); the caller must have already
    /// taken or cleared its tenant list.
    fn recycle_slot(&self, idx: usize) {
        let metadata = self.slot_metadata(idx);
        metadata.valid.clear();
        memory_ctx().clock().release(idx);
    }

    /// Reclaim compressed slot `slot_idx` if it can be taken exclusively: drop every
    /// extent packed in it from its file's extent map, recycle it, and return it
    /// writable. `None` when a reader still pins it or it is no longer compressed.
    ///
    /// Each dropped extent's decompressed twins (if any) are reinforced: with the
    /// compressed copy gone they are the last in-memory copy of those bytes,
    /// and losing them too would turn the next read into disk IO.
    pub(crate) fn reclaim(&self, slot_idx: usize) -> Option<WriteBuffer> {
        // The probe guard releases the hold in place if anything below unwinds
        // (e.g. a poisoned map lock): a slot the cache still references must
        // never fall into the free pool via WriteBuffer::drop.
        let probe = ProbedSlot::new(memory_ctx().ring().try_write(slot_idx)?);
        if memory_ctx().clock().owner(slot_idx) != Some(Owner::Compressed) {
            return None;
        }

        // Exclusive now, so the tenant list is complete and stable.
        let tenants = std::mem::take(self.tenants_mut(slot_idx));
        let mut emptied_files = Vec::new();
        {
            let file_maps = self.file_maps.read().unwrap();
            for tenant in tenants {
                let Some(extents_lock) = file_maps.get(&tenant.open_file) else {
                    continue;
                };
                let mut extents = extents_lock.write().unwrap();
                // Skip an extent re-placed into another slot after this tenant was
                // recorded.
                if let Some(extent) = extents
                    .get(&tenant.first_file_block)
                    .filter(|extent| extent.slot_idx as usize == slot_idx)
                {
                    // Each dropped extent's decompressed twins become the last
                    // in-memory copy of those bytes; reinforcing takes the
                    // decompressed cache's read locks under our map locks, which
                    // is safe because the decompressed evictor only walks OUR
                    // maps after dropping its own (see its reclaim) - the two
                    // lock sets are never taken in the opposite order.
                    memory_ctx().decompressed_cache().reinforce_range(
                        &tenant.open_file,
                        extent.first_file_block * BLOCK_SIZE,
                        extent.block_count as usize * BLOCK_SIZE,
                    );
                    extents.remove(&tenant.first_file_block);
                }
                if extents.is_empty() {
                    emptied_files.push(tenant.open_file);
                }
            }
        }
        self.prune_empty_file_locations(emptied_files);
        self.recycle_slot(slot_idx);
        Some(probe.take_for_reuse())
    }

    /// Reinforce every cached extent overlapping the file byte range
    /// `[offset, offset + len)` - called by the decompressed cache when it
    /// evicts a block over that range, leaving these extents the last in-memory
    /// copy of it. Takes only read locks; slot granularity, so a multi-tenant
    /// slot's other extents shelter under the same reinforcement.
    pub(crate) fn reinforce_range(&self, open_file: &OpenFile, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let first_file_block = offset / BLOCK_SIZE;
        let last_file_block = (offset + len - 1) / BLOCK_SIZE;

        let file_maps = self.file_maps.read().unwrap();
        let Some(extents_lock) = file_maps.get(open_file) else {
            return;
        };
        let extents = extents_lock.read().unwrap();
        // The extent covering `first_file_block` may start before it, so begin the
        // walk at the greatest key at or before it and skip a leading extent that
        // doesn't actually reach the range.
        let walk_start = extents
            .range(..=first_file_block)
            .next_back()
            .map(|(&first, _)| first)
            .unwrap_or(first_file_block);
        for (_, extent) in extents.range(walk_start..=last_file_block) {
            if extent.last_file_block() >= first_file_block {
                memory_ctx().clock().reinforce(extent.slot_idx as usize);
            }
        }
    }

    /// Remove `file_maps` entries for the given locations whose extent map is now
    /// empty. `file_maps` is keyed by a never-reused [`OpenFile`] (an
    /// `LocalFile` / `Arc<RemoteFile>`), so without this a dead entry - pinning its
    /// `Arc` and the file/connection it holds - lingers for every file ever opened.
    /// Re-checks emptiness under the write lock so a concurrent `get` that just
    /// cached a new extent isn't dropped.
    fn prune_empty_file_locations(&self, locations: Vec<OpenFile>) {
        if locations.is_empty() {
            return;
        }
        let mut file_maps = self.file_maps.write().unwrap();
        for open_file in locations {
            if let Some(extents_lock) = file_maps.get(&open_file)
                && extents_lock.read().unwrap().is_empty()
            {
                file_maps.remove(&open_file);
            }
        }
    }

    /// Register a [`OpenFile`] (a local fd or a remote object) so its extents can
    /// be cached. Clears any extents from a previous registration of an equal file
    /// (e.g. a reused fd number) by dropping its extent map - the next lookup then
    /// misses. The orphaned tenants in their slots self-clean on eviction (their
    /// extent is gone, so the per-slot guard skips them).
    pub fn open_entry(&self, open_file: OpenFile) {
        // Drop any decompressed pages cached under this (possibly reused) file
        // for the same reason the compressed map is reset below: a reopened fd may
        // now name a different file, so its old pages must not serve a later read.
        memory_ctx().decompressed_cache().invalidate(&open_file);
        // The `file_maps` write lock serializes with
        // `try_add_extent_for_location`'s read, so a racing miss either sees the
        // fresh empty map or has its extent dropped here.
        self.file_maps
            .write()
            .unwrap()
            .insert(open_file, Default::default());
    }

    /// Evict every cached extent: drop all extent maps and recycle the ring slots they
    /// referenced, so subsequent reads miss and re-read from disk. Registered file
    /// descriptors stay open (their maps are just emptied). Returns the number of
    /// extents dropped.
    ///
    /// Intended for benchmarking true cold reads (`SELECT drop_cache()`). Sound only
    /// while no query is in flight: it recycles bound slots, which must not race a
    /// reader holding a pin (an actively-filling slot another worker still pins is
    /// left intact - its extents are already dropped from the map, so it serves no
    /// stale hits).
    pub fn clear(&self) -> usize {
        // Release this worker's fill pin first, so its own fill slot is recyclable
        // below rather than skipped as pinned.
        memory_ctx().compressed_fill_cursor().buffer = None;

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

        // Recycle each slot we can take exclusively and that is still ours; one
        // another worker is still filling stays pinned and is left intact (its
        // extents are already dropped), and one already recycled and re-bound to a
        // new owner in the window since the drain must not be ripped away from it.
        for slot_idx in slots {
            if let Some(write_buffer) = memory_ctx().ring().try_write(slot_idx) {
                let probe = ProbedSlot::new(write_buffer);
                if memory_ctx().clock().owner(slot_idx) != Some(Owner::Compressed) {
                    continue; // the probe's drop releases the hold in place
                }
                self.tenants_mut(slot_idx).clear();
                self.recycle_slot(slot_idx);
                drop(probe.take_for_reuse()); // returns the slot to the free pool
            }
        }
        dropped
    }
}

/// The extent covering file block `file_block`, if any: the greatest-keyed extent at
/// or before it that reaches it. Extents never overlap, so at most one qualifies.
fn find_extent_covering(extents: &BTreeMap<usize, Extent>, file_block: usize) -> Option<Extent> {
    let (_, extent) = extents.range(..=file_block).next_back()?;
    (extent.last_file_block() >= file_block).then(|| extent.clone())
}

/// How many blocks from `file_block` are free of any extent, up to `want` - the
/// available prefix of the gap. Zero if `file_block` is already covered.
fn count_available_blocks(
    extents: &BTreeMap<usize, Extent>,
    file_block: usize,
    want: usize,
) -> usize {
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
    use crate::io::LocalFile;
    use crate::memory::context::{init_test_free_pool, memory_ctx};

    const SB: usize = BLOCK_SIZE;

    /// A local file used as the cache key throughout these tests. One shared
    /// open file, so every call keys the same cache bucket.
    #[allow(non_snake_case)]
    fn FD() -> OpenFile {
        use std::sync::OnceLock;
        static FILE: OnceLock<LocalFile> = OnceLock::new();
        OpenFile::Local(
            FILE.get_or_init(|| LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
                .clone(),
        )
    }

    /// A fresh, independent local file (distinct cache bucket).
    fn new_file() -> OpenFile {
        OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
    }

    fn cache() -> &'static CompressedCache {
        memory_ctx().compressed_cache()
    }

    /// Drop this worker's fill pin so its current fill slot becomes evictable.
    fn release_fill_cursor() {
        memory_ctx().compressed_fill_cursor().buffer = None;
    }

    /// A deterministic byte for absolute file offset `off`, so cache contents are
    /// predictable for any slice regardless of where they were packed.
    fn pattern_at(off: usize) -> u8 {
        off.wrapping_mul(7).wrapping_add(1) as u8
    }

    /// Read the pattern into every missing block of `lookups` and commit it.
    fn fill_pattern(lookups: &[CacheLookup]) {
        for lookup in lookups {
            if let Some(block) = lookup.missing() {
                for i in 0..block.len() {
                    unsafe { *block.dest().add(i) = pattern_at(block.file_offset() + i) };
                }
                block.commit();
            }
        }
    }

    /// Whether any lookup in the set still has a hole.
    fn has_misses(lookups: &[CacheLookup]) -> bool {
        lookups.iter().any(|l| l.missing().is_some())
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

    /// Every missing extent across a set of lookups, cloned.
    fn missing_extents(lookups: &[CacheLookup]) -> Vec<MissingExtent> {
        lookups
            .iter()
            .filter_map(|l| l.missing().cloned())
            .collect()
    }

    /// Insert one resident decompressed block over `[offset, offset + len)` of
    /// `FD()` into the shared context's decompressed cache, returning its slot.
    /// Reserves a full slot's worth of bytes so each block occupies its own
    /// slot, then drops the fill pin so the slot is ordinarily evictable.
    fn insert_decompressed(offset: usize, len: usize) -> usize {
        use crate::memory::decompressed_cache::BlockKey;
        let cache = memory_ctx().decompressed_cache();
        let key = BlockKey {
            open_file: FD(),
            offset,
            len,
        };
        let reservation = cache.reserve(&key, crate::memory::BUFFER_SIZE);
        drop(cache.insert(
            key,
            vec![Bytes::from(vec![0u8])],
            reservation,
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        ));
        let cursor = memory_ctx().decompressed_fill_cursor();
        let slot = cursor.slot_idx;
        cache.release_cursor(cursor);
        slot
    }

    /// Cache one whole slot of `FD()` compressed bytes at slot-aligned file
    /// offset `slot_number * BUFFER_SIZE`, read a few times since so it holds
    /// lives like any page an earlier query used. The contents are irrelevant,
    /// so the extent is committed without writing them.
    fn insert_stale_compressed_slot(slot_number: usize) {
        let lookups = cache().get(&FD(), slot_number * BUFFER_SIZE, BUFFER_SIZE);
        let missing = lookups[0].missing().unwrap();
        missing.commit();
        for _ in 0..3 {
            memory_ctx().clock().touch(missing.extent.slot_idx as usize);
        }
    }

    /// Insert one decompressed block over `[offset, offset + len)` of `FD()` and
    /// hand back the pinned views a reader of it would hold; while they live the
    /// block's slot cannot be reclaimed.
    fn insert_pinned_decompressed(offset: usize, len: usize) -> Vec<Bytes> {
        use crate::memory::decompressed_cache::BlockKey;
        insert_decompressed(offset, len);
        let key = BlockKey {
            open_file: FD(),
            offset,
            len,
        };
        memory_ctx().decompressed_cache().get(&key).unwrap()
    }

    /// Take every slot still in the free pool as working memory, so the next
    /// request has to evict.
    fn take_every_free_slot() -> Vec<WriteBuffer> {
        std::iter::from_fn(|| memory_ctx().pop_free_idx(true))
            .map(|idx| memory_ctx().ring().try_write(idx).unwrap())
            .collect()
    }

    /// An owning lookup dropped before it commits (a read that unwound or was
    /// dropped with its dataflow before it was issued) fails its followers and
    /// frees the range for a fresh fill, instead of leaving them waiting on a
    /// fill nobody will make.
    #[test]
    fn an_abandoned_fill_fails_its_followers_and_frees_the_range() {
        init_test_free_pool(8);
        let file = new_file();
        let owner = cache().get(&file, 0, BLOCK_SIZE);
        let follower = cache().get(&file, 0, BLOCK_SIZE);
        let followed = follower[0].missing().unwrap().clone();
        assert!(owner[0].missing().unwrap().fill_owner());
        assert!(!followed.fill_owner());

        drop(owner);

        assert!(followed.failed());
        let fresh = cache().get(&file, 0, BLOCK_SIZE);
        assert!(fresh[0].missing().unwrap().fill_owner());
    }

    /// A committed fill settles its ownership: dropping the owner afterwards
    /// leaves the extent in place for the followers and later hits.
    #[test]
    fn a_committed_fill_survives_its_owner_being_dropped() {
        init_test_free_pool(8);
        let file = new_file();
        let owner = cache().get(&file, 0, BLOCK_SIZE);
        let followed = cache().get(&file, 0, BLOCK_SIZE)[0]
            .missing()
            .unwrap()
            .clone();
        fill_pattern(&owner);

        drop(owner);

        assert!(followed.is_committed());
        assert!(!followed.failed());
        assert!(!has_misses(&cache().get(&file, 0, BLOCK_SIZE)));
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
    fn an_unwinding_reclaim_leaves_the_slot_cached_and_out_of_the_pool() {
        // Poison the outer map lock so reclaim unwinds mid-probe; the slot the
        // cache still references must not fall into the free pool (it would be
        // handed out as scratch memory under two owners). The pool's only slot
        // becomes the fill slot, so a leak into the pool is directly visible.
        init_test_free_pool(1);
        cache().open_entry(FD());
        let miss = cache().get(&FD(), 0, 10);
        fill_pattern(&miss);
        drop(miss);
        let slot = memory_ctx().compressed_fill_cursor().slot_idx;
        release_fill_cursor();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _poisoning = cache().file_maps.write().unwrap();
            panic!("poison the file map lock");
        }));

        let reclaim = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache().reclaim(slot);
        }));

        assert!(reclaim.is_err(), "reclaim unwinds on the poisoned lock");
        assert_eq!(
            memory_ctx().pop_free_idx(false),
            None,
            "the slot must not reach the free pool"
        );
        assert_eq!(memory_ctx().clock().owner(slot), Some(Owner::Compressed));
        assert!(
            memory_ctx().ring().try_write(slot).is_some(),
            "the probe hold must have been released"
        );
    }

    #[test]
    fn coalesces_contiguous_missing_blocks_into_one_extent() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Three blocks, none present → one freshly packed extent.
        let miss = cache().get(&FD(), 0, 2 * SB + 1);
        let extents = missing_extents(&miss);
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].extent.block_count, 3);
        assert_eq!(extents[0].len(), 3 * SB);
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
        let extents = missing_extents(&again);
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].file_offset(), SB);
        assert_eq!(extents[0].extent.block_count, 2);
    }

    #[test]
    fn consecutive_misses_pack_into_one_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // Two disjoint reads of the same file pack contiguously into the fill slot.
        let a = cache().get(&FD(), 0, 10);
        let b = cache().get(&FD(), 50 * SB, 50 * SB + 10);
        let a = &a[0].missing().unwrap();
        let b = &b[0].missing().unwrap();
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
            ra[0].missing().unwrap().extent.slot_idx,
            rb[0].missing().unwrap().extent.slot_idx,
            "reads from different files share one packed slot"
        );
    }

    #[test]
    fn a_read_past_the_fill_buffer_uses_a_new_slot() {
        init_test_free_pool(8);
        cache().open_entry(FD());

        // One read spanning more than a whole slot (513 blocks) splits at the 2 MB
        // boundary into two extents in two slots.
        let miss = cache().get(&FD(), 0, BLOCKS_PER_SLOT * SB + 1);
        let extents = missing_extents(&miss);
        assert_eq!(extents.len(), 2);
        assert_eq!(extents[0].extent.block_count as usize, BLOCKS_PER_SLOT);
        assert_eq!(extents[1].extent.block_count, 1);
        assert_ne!(extents[0].extent.slot_idx, extents[1].extent.slot_idx);

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
        let extents = missing_extents(&second);
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].file_offset(), 4 * SB);
        assert_eq!(extents[0].extent.block_count, 2);
    }

    #[test]
    fn hit_reassembles_bytes_across_lookups() {
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
        let extents = missing_extents(&m);
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].file_offset(), SB); // only the hole
        fill_pattern(&m);
        let bytes = assemble(m);

        assert_pattern(&bytes, 0);
    }

    #[test]
    fn get_assembles_a_range_spanning_cached_and_uncached_lookups() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        // Cache [0, SB); leave [SB, 2*SB) uncached.
        fill_pattern(&cache().get(&FD(), 0, SB));

        let parts = cache().get(&FD(), 0, 2 * SB);
        assert!(has_misses(&parts), "the second lookup is a hole");
        fill_pattern(&parts);

        assert_pattern(&assemble(parts), 0);
    }

    #[test]
    fn reopening_a_file_forgets_its_cached_extents() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));

        cache().open_entry(FD());

        assert!(has_misses(&cache().get(&FD(), 0, SB)));
    }

    #[test]
    fn get_write_buffer_evicts_compressed_data_when_the_pool_is_empty() {
        init_test_free_pool(1);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();

        let write = memory_ctx().get_write_buffer(false);

        assert_eq!(memory_ctx().clock().owned(Owner::Compressed), 0);
        drop(write);
        assert!(has_misses(&cache().get(&FD(), 0, SB)));
    }

    /// The CLOCK evictor must not reclaim a ring slot that is still parked in
    /// the free pool (unbound).
    #[test]
    fn evict_does_not_steal_a_slot_from_the_free_pool() {
        init_test_free_pool(2);
        cache().open_entry(FD());
        // A real cached extent binds a slot; its set ref_bit makes the CLOCK hand skip
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
    fn evicting_a_slot_drops_all_tenant_extents() {
        init_test_free_pool(16);
        let files: Vec<OpenFile> = (0..3).map(|_| new_file()).collect();
        for f in &files {
            cache().open_entry(f.clone());
            fill_pattern(&cache().get(f, 0, SB)); // all pack into one fill slot
        }
        release_fill_cursor();

        // The three extents share one slot; evicting it drops all three mappings and
        // prunes their now-empty file_maps entries.
        memory_ctx().evict();

        let file_maps = cache().file_maps.read().unwrap();
        for f in &files {
            assert!(
                !file_maps.contains_key(f),
                "an evicted extent's file_maps entry was left behind - leak",
            );
        }
    }

    #[test]
    fn a_recycled_slot_makes_stale_placements_miss() {
        init_test_free_pool(4);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();

        // Evict the slot the extent lived in; its mapping is dropped.
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
            "evicting the last extent should prune the entry",
        );

        // Reading f again must tolerate the absent entry: re-create it + miss.
        assert!(has_misses(&cache().get(&f, 0, SB)));
        assert!(
            cache().file_maps.read().unwrap().contains_key(&f),
            "re-reading a pruned file should re-create its entry",
        );
    }

    // -- share-routed eviction and cross-tier reinforcement --

    #[test]
    fn evict_takes_decompressed_while_compressed_is_under_target() {
        init_test_free_pool(16);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB)); // 1 compressed slot
        release_fill_cursor();
        for i in 0..10 {
            insert_decompressed((10 + i) * SB, SB); // 10 decompressed slots: share 9%
        }

        memory_ctx().evict();

        assert!(
            !has_misses(&cache().get(&FD(), 0, SB)),
            "the under-target compressed extent must survive"
        );
        assert_eq!(memory_ctx().clock().owned(Owner::Decompressed), 9);
    }

    /// The preferred tier still has a page with lives to spend beside a pinned
    /// zero-life one: the hand keeps sweeping it, so the ageing page gives way
    /// once drained and the other tier is left alone.
    #[test]
    fn a_tier_still_ageing_keeps_the_sweep_despite_a_pinned_candidate() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let lookups = cache().get(&FD(), 0, BUFFER_SIZE);
        let missing = lookups[0].missing().unwrap();
        let pinned_slot = missing.extent.slot_idx as usize;
        missing.commit();
        release_fill_cursor();
        let _reader = memory_ctx().ring().try_read(pinned_slot).unwrap();
        insert_stale_compressed_slot(1); // a second compressed page, holding more lives
        release_fill_cursor();
        insert_decompressed(10 * SB, SB); // 2 of 3 compressed: over target, preferred
        let clock = memory_ctx().clock();
        assert_eq!(clock.preferred_victim_tier(0), Owner::Compressed);

        drop(memory_ctx().evict());

        assert_eq!(
            clock.owned(Owner::Compressed),
            1,
            "the ageing page gave way"
        );
        assert_eq!(
            clock.owned(Owner::Decompressed),
            1,
            "the other tier was left alone"
        );
        assert_eq!(
            clock.owner(pinned_slot),
            Some(Owner::Compressed),
            "the pinned page survives"
        );
    }

    /// The preferred tier's zero-life slots are held by readers and nothing
    /// else of it is left to age, so its hand can find nothing there until
    /// they are done. One such revolution hands the sweep to the other tier,
    /// whose page then gives way, instead of draining the pinned tier for the
    /// full lives ceiling first.
    #[test]
    fn a_revolution_of_pinned_candidates_hands_the_sweep_to_the_other_tier() {
        init_test_free_pool(8);
        let ring_len = memory_ctx().ring().len();
        cache().open_entry(FD());
        let mut readers = Vec::new();
        for offset in [0, BUFFER_SIZE] {
            let lookups = cache().get(&FD(), offset, BUFFER_SIZE);
            let missing = lookups[0].missing().unwrap();
            let slot = missing.extent.slot_idx as usize;
            missing.commit();
            release_fill_cursor();
            readers.push(memory_ctx().ring().try_read(slot).unwrap());
        }
        for i in 0..3 {
            insert_decompressed((10 + i) * SB, SB); // 2 of 5 compressed: over target, preferred
        }
        let clock = memory_ctx().clock();
        assert_eq!(clock.preferred_victim_tier(0), Owner::Compressed);
        let hand_before = clock.hand_position(Owner::Compressed, 0);
        // One revolution per life to age the pinned pages to zero, then the one
        // that finds them pinned and hands over.
        let lives = (0..ring_len)
            .filter(|&slot| clock.owner(slot) == Some(Owner::Compressed))
            .map(|slot| clock.refs(slot) as usize)
            .max()
            .unwrap();

        drop(memory_ctx().evict());

        assert_eq!(
            clock.owned(Owner::Compressed),
            2,
            "the pinned pages survive"
        );
        assert_eq!(
            clock.owned(Owner::Decompressed),
            2,
            "the other tier gave the victim"
        );
        assert!(
            clock.hand_position(Owner::Compressed, 0) - hand_before <= (lives + 1) * ring_len,
            "the pinned tier was swept only until its pages were found pinned"
        );
    }

    #[test]
    fn evict_takes_compressed_while_over_its_target_share() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();
        insert_decompressed(10 * SB, SB); // 1 of each: share 50%, over target

        memory_ctx().evict();

        assert_eq!(memory_ctx().clock().owned(Owner::Compressed), 0);
        assert_eq!(memory_ctx().clock().owned(Owner::Decompressed), 1);
    }

    /// The share ratio only moves when an eviction succeeds. With the compressed
    /// tier at its target share it points at the decompressed tier, and when
    /// every decompressed page is pinned by the running query's readers nothing
    /// there can be reclaimed, so the ratio never moves again. A request for
    /// working memory must still be served from the stale compressed pages
    /// rather than failing as if the ring were exhausted.
    #[test]
    fn working_memory_takes_stale_compressed_pages_when_every_decompressed_page_is_pinned() {
        init_test_free_pool(128);
        cache().open_entry(FD());
        // 30 of 100 cached slots: exactly the compressed tier's default target share.
        for slot_number in 0..30 {
            insert_stale_compressed_slot(slot_number);
        }
        release_fill_cursor();
        let _pinned: Vec<Vec<Bytes>> = (0..70)
            .map(|i| insert_pinned_decompressed((100 + i) * BUFFER_SIZE, SB))
            .collect();
        let _working_memory = take_every_free_slot();

        let _buffer = memory_ctx().get_write_buffer(true);

        assert_eq!(
            memory_ctx().clock().owned(Owner::Compressed),
            29,
            "the buffer must come from the stale compressed tier"
        );
    }

    #[test]
    fn evicting_a_compressed_extent_reinforces_its_decompressed_twin() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        fill_pattern(&cache().get(&FD(), 0, SB));
        release_fill_cursor();
        // Same file range as the compressed extent: its decompressed twin.
        let twin_slot = insert_decompressed(0, SB);
        assert_eq!(memory_ctx().clock().refs(twin_slot), 1);

        memory_ctx().evict(); // share 50% takes the compressed slot

        let bump = memory_ctx().clock().reinforce_bump();
        assert_eq!(memory_ctx().clock().owned(Owner::Compressed), 0);
        assert_eq!(
            memory_ctx().clock().refs(twin_slot),
            1 + bump,
            "the now-last copy gained a reinforced bump"
        );
    }

    #[test]
    fn evicting_a_decompressed_block_reinforces_its_compressed_source() {
        init_test_free_pool(8);
        cache().open_entry(FD());
        let lookups = cache().get(&FD(), 0, SB);
        fill_pattern(&lookups);
        let source_slot = lookups[0].missing().unwrap().extent.slot_idx as usize;
        drop(lookups);
        release_fill_cursor();
        let block_slot = insert_decompressed(0, SB);
        let refs_before = memory_ctx().clock().refs(source_slot);

        memory_ctx().decompressed_cache().reclaim(block_slot);

        assert_eq!(
            memory_ctx().clock().refs(source_slot),
            refs_before + memory_ctx().clock().reinforce_bump(),
            "the now-last copy gained a reinforced bump"
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
