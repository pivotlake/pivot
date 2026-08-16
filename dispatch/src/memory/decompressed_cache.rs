//! A cache of decompressed blocks, packed into shared ring slots and sharing the
//! ring's CLOCK ([`Clock`](super::clock::Clock)) with the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache).
//!
//! ## Packing
//!
//! Each worker owns a [`FillCursor`]: a bump pointer over a slot it holds pinned
//! (as a [`ReadBuffer`]). [`reserve`](DecompressedCache::reserve) carves a page's
//! byte range out of the cursor, rotating to a fresh slot when the current one
//! fills - so a page may straddle two cursor-consecutive slots, and a page larger
//! than a slot spans several. The caller decompresses directly into the reserved
//! region and then [`insert`](DecompressedCache::insert)s it. The forward-map
//! insert is what makes the block visible: decompression is synchronous and
//! readers can only discover a block through the map, so the map's `RwLock` is
//! the happens-before between the byte writes and any read - no validity bitmap
//! is needed (unlike the compressed cache, whose extents are visible while their
//! read IO is still in flight).
//!
//! Writing into a read-pinned slot is sound for the same reason it is in the
//! compressed cache's fill path: the reserved region is not yet in the map (no
//! reader can reach it), the bump pointer makes regions disjoint, and the worker
//! is the slot's only filler.
//!
//! ## Maps
//!
//! `files` maps a [`OpenFile`] to its own `RwLock`'d table of `offset ->
//! block`, ordered by offset, so a `get` only contends on one file's lock and
//! [`get_range`](DecompressedCache::get_range) can walk a byte range in file
//! order. `tenants` is a per-slot index - one lockless entry per ring slot, like
//! the compressed cache's per-slot tenant list - naming every block with bytes in
//! the slot, so the evictor (which only has a slot index) can find them. A slot's
//! tenant list is written only by its filler (under the fill pin) and taken by
//! the evictor (under `try_write`); the pin-drop `Release` pairing with
//! `try_write`'s `Acquire` publishes it. A tenant is recorded at reserve time,
//! BEFORE the forward-map insert, so on every path (including a lost dup race or
//! an unwound decompress) the invariant holds: the tenant list covers every map
//! entry referencing the slot. Stale tenants (blocks since evicted, dups,
//! failures) are filtered by the map lookup at reclaim time.
//!
//! ## Eviction
//!
//! The clock picks a slot and calls [`reclaim`](DecompressedCache::reclaim): take
//! the slot with `try_write`, take its tenant list, and drop every tenant block
//! whose current map entry still references this slot. A block straddling into
//! another slot dies whole; its bytes there stay unreachable until that slot is
//! itself evicted. No multi-slot
//! acquisition is needed: each returned `Bytes` view pins exactly its own slot,
//! so a view stays valid for as long as it is held regardless of what happens to
//! the block's other slots.
//!
//! ## Range queries
//!
//! A block's key is opaque to this module - just a file, a source byte offset, and
//! a source byte span - but the caller (the Parquet reader) always uses a Parquet
//! page's on-disk span as that key, and pages within one column chunk sit back to
//! back with no gaps. That's what [`get_range`](DecompressedCache::get_range)
//! relies on: given a column chunk's own `(offset, len)` it can report which of
//! that range is already decompressed and which byte stretches are gaps, and every
//! gap is a whole number of complete pages, without this module ever knowing what
//! a "page" is. The caller re-derives each page's header from the
//! [`Segment::Cached`] blob it gets back (stored here as an opaque, caller-supplied
//! byte string).

use crate::env::get_env_var_with_default;
use crate::io::OpenFile;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::fill_cursor::FillCursor;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Identity of a cached decompressed block: the file, the byte offset of its
/// source bytes, and their span. A page's header offset is unique within its file,
/// so the map key is just the offset; `len` (the source span) is instead checked
/// against the stored block on lookup, so a lookup never serves bytes decompressed
/// from a differently-sized region at the same offset.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub open_file: OpenFile,
    pub offset: usize,
    pub len: usize,
}

/// One contiguous piece of a block's decompressed bytes inside a ring slot.
#[derive(Clone, Copy)]
struct BlockExtent {
    slot: usize,
    offset: u32,
    len: u32,
}

/// A cached decompressed block: where its bytes live (one extent per slot it
/// touches, in byte order), the length of its compressed source in the file
/// (this block's `BlockKey::len` at insert time - what lets
/// [`get`](DecompressedCache::get) reject a same-offset/different-size request,
/// and what lets [`get_range`](DecompressedCache::get_range) find the next file
/// byte after this block), and the caller-supplied header bytes describing it
/// (opaque to this module - see the module doc).
struct DecompressedBlock {
    extents: Vec<BlockExtent>,
    compressed_len: usize,
    header: Vec<Bytes>,
    /// Caller-supplied count of live cached blocks for whatever grouping the
    /// caller tracks (the Parquet reader counts a row group's cached pages to
    /// schedule scans cache-first). Counted at construction, uncounted on drop,
    /// so it exactly tracks residency through every removal path: eviction,
    /// invalidation, and clear.
    live_blocks: Arc<AtomicUsize>,
}

impl DecompressedBlock {
    fn new(
        extents: Vec<BlockExtent>,
        compressed_len: usize,
        header: Vec<Bytes>,
        live_blocks: Arc<AtomicUsize>,
    ) -> Self {
        live_blocks.fetch_add(1, Ordering::Relaxed);
        Self {
            extents,
            compressed_len,
            header,
            live_blocks,
        }
    }
}

impl DecompressedBlock {
    /// Pin every extent's slot with `try_read`, refreshing the clock, and return
    /// one zero-copy view per extent. `None` (no pins taken) if any slot is
    /// mid-reclaim (`WRITING`).
    fn pin_views(&self) -> Option<Vec<Bytes>> {
        let mut views = Vec::with_capacity(self.extents.len());
        for extent in &self.extents {
            let read = memory_ctx().ring().try_read(extent.slot)?;
            memory_ctx().clock().touch(extent.slot);
            views.push(
                Bytes::from_owner(read)
                    .slice(extent.offset as usize..(extent.offset + extent.len) as usize),
            );
        }
        Some(views)
    }
}

impl Drop for DecompressedBlock {
    fn drop(&mut self) {
        self.live_blocks.fetch_sub(1, Ordering::Relaxed);
    }
}

/// One file's cached blocks, ordered by source offset so a byte range can be walked
/// in file order.
type FileBlocks = RwLock<BTreeMap<usize, DecompressedBlock>>;

/// Keeps a packed slot pinned for as long as a `Bytes` view into it lives.
struct PinnedSlot(Arc<ReadBuffer>);

impl AsRef<[u8]> for PinnedSlot {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// One reserved piece of a slot: the pin keeping the slot alive, and the byte
/// range the reservation owns within it.
struct RegionSegment {
    pin: Arc<ReadBuffer>,
    slot: usize,
    offset: usize,
    len: usize,
}

/// A page's reserved output region, handed out by
/// [`reserve`](DecompressedCache::reserve) and consumed by
/// [`insert`](DecompressedCache::insert).
pub enum Reservation {
    /// Byte ranges bump-allocated inside shared, read-pinned cache slots.
    Packed(PackedRegion),
    /// Whole pooled slots, used when the cache is disabled or the page is empty:
    /// never inserted, and the slots return to the pool when the views drop.
    Transient {
        buffers: Vec<WriteBuffer>,
        len: usize,
    },
}

impl Reservation {
    /// The reserved region as writable slices, in byte order - the shape the
    /// scattered decompressor fills. Derived from the pins' raw pointers, never
    /// through a `&mut` of a slot: other regions of the same slot are
    /// concurrently readable through their own views, and only the bump
    /// allocation makes this region exclusively ours.
    pub fn as_mut_slices(&mut self) -> Vec<&mut [u8]> {
        match self {
            Reservation::Packed(region) => region
                .segments
                .iter()
                .map(|segment| unsafe {
                    std::slice::from_raw_parts_mut(
                        segment.pin.ptr.add(segment.offset) as *mut u8,
                        segment.len,
                    )
                })
                .collect(),
            Reservation::Transient { buffers, .. } => {
                buffers.iter_mut().map(|b| b.as_mut()).collect()
            }
        }
    }
}

pub struct PackedRegion {
    segments: Vec<RegionSegment>,
}

/// One resolved piece of a [`get_range`](DecompressedCache::get_range) query, in
/// file order.
pub enum Segment {
    /// A page already decompressed here: its identity (offset, span), its
    /// caller-supplied header bytes, and its decompressed bytes - pinned for as
    /// long as this is held, so it can't be evicted before the caller uses it.
    Cached {
        offset: usize,
        span: usize,
        header: Vec<Bytes>,
        data: Vec<Bytes>,
    },
    /// A byte stretch with no cached block. Always a whole number of complete
    /// pages: every boundary here is either the query's own edge or a cached
    /// block's known span, and both are always page boundaries (see the module
    /// doc).
    Gap { offset: usize, len: usize },
}

/// Append a gap of `len` bytes at `offset`, merging into the previous segment if
/// it's also a gap - always correct when two gaps are adjacent, since nothing but
/// another gap or the range's own edge can sit between them.
fn push_gap(segments: &mut Vec<Segment>, offset: usize, len: usize) {
    if len == 0 {
        return;
    }
    if let Some(Segment::Gap { len: prev, .. }) = segments.last_mut() {
        *prev += len;
    } else {
        segments.push(Segment::Gap { offset, len });
    }
}

/// A cache of decompressed blocks, shared across workers behind an `Arc` like the
/// [`CompressedCache`](super::compressed_cache::CompressedCache).
pub struct DecompressedCache {
    /// Per-file: file -> (source offset -> block), offset-ordered. `get`/`get_range`
    /// only lock one file's table.
    files: RwLock<HashMap<OpenFile, FileBlocks>>,
    /// Per-slot tenant index, indexed by ring slot: the key of every block with
    /// bytes in the slot. `UnsafeCell` (like the compressed cache's per-slot
    /// `tenants`): written only by the slot's filler under the fill pin, taken by
    /// the evictor under `try_write`, so it needs no lock. May hold stale keys
    /// (evicted straddlers, lost dup races, unwound decompresses) - the map
    /// lookup at reclaim time filters them.
    tenants: Box<[UnsafeCell<Vec<BlockKey>>]>,
    /// Cached blocks across all files, maintained by [`claim`](Self::claim) and
    /// every removal path. Backs [`is_empty`](Self::is_empty).
    blocks: AtomicUsize,
    /// Kill switch (`PIVOT_DECOMPRESSED_CACHE`, default on). When off, `reserve`
    /// hands out transient pooled slots and `get` is a no-op.
    enabled: bool,
}

// SAFETY: `tenants`' `UnsafeCell`s are only accessed by a slot's single filler
// (under the fill pin) or by the evictor while the slot is held exclusively
// (`try_write`), so there is never concurrent access to one entry. Same
// discipline as [`CompressedCache`].
unsafe impl Send for DecompressedCache {}
unsafe impl Sync for DecompressedCache {}

impl DecompressedCache {
    /// `capacity` is the ring's slot count: one tenant list per slot.
    pub fn new(capacity: usize) -> Self {
        Self {
            files: RwLock::new(HashMap::default()),
            tenants: (0..capacity).map(|_| UnsafeCell::new(Vec::new())).collect(),
            blocks: AtomicUsize::new(0),
            enabled: get_env_var_with_default("PIVOT_DECOMPRESSED_CACHE", true),
        }
    }

    /// Mutable access to slot `idx`'s tenant list. Sound only under the fill pin
    /// (filler) or `try_write` exclusivity (evictor) - the one `unsafe` for the
    /// tenants invariant.
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell; see above
    fn tenants_mut(&self, idx: usize) -> &mut Vec<BlockKey> {
        unsafe { &mut *self.tenants[idx].get() }
    }

    /// Reserve `len` bytes of output space for the page `key`, to be decompressed
    /// into and then [`insert`](Self::insert)ed. Packed reservations carve the
    /// range out of this worker's fill cursor (rotating to fresh slots as
    /// needed, so a page may straddle slots); each touched slot gets `key`
    /// recorded as a tenant NOW, before any map insert, so the evictor's
    /// tenant-covers-every-entry invariant holds even if decompression fails or
    /// the dup race is lost. Disabled cache or an empty page falls back to
    /// transient pooled slots.
    pub fn reserve(&self, key: &BlockKey, len: usize) -> Reservation {
        if !self.enabled || len == 0 {
            let buffers = (0..len.div_ceil(BUFFER_SIZE))
                .map(|_| memory_ctx().get_write_buffer(false))
                .collect();
            return Reservation::Transient { buffers, len };
        }
        let cursor = memory_ctx().decompressed_fill_cursor();
        let mut segments = Vec::with_capacity(1);
        let mut remaining = len;
        while remaining > 0 {
            if cursor.is_exhausted() {
                // Rotation may evict, so it must hold no map lock (it doesn't).
                self.rotate_fill_slot(cursor);
            }
            let take = remaining.min(BUFFER_SIZE - cursor.next_byte);
            let offset = cursor.next_byte;
            cursor.next_byte += take;
            self.tenants_mut(cursor.slot_idx).push(key.clone());
            segments.push(RegionSegment {
                pin: cursor.buffer.clone().expect("rotated above"),
                slot: cursor.slot_idx,
                offset,
                len: take,
            });
            remaining -= take;
        }
        Reservation::Packed(PackedRegion { segments })
    }

    /// Point this worker's fill cursor at a fresh, empty slot: reset its tenant
    /// list while it is held exclusively, bind it, then make it readable.
    fn rotate_fill_slot(&self, cursor: &mut FillCursor) {
        let write_buffer = memory_ctx().get_write_buffer(false);
        let slot_idx = write_buffer.slot_idx;
        self.tenants_mut(slot_idx).clear();
        memory_ctx().clock().bind(slot_idx, Owner::Decompressed);
        cursor.buffer = Some(Arc::new(ReadBuffer::from(write_buffer))); // used = 1, Release
        cursor.slot_idx = slot_idx;
        cursor.next_byte = 0;
    }

    /// Drop this worker's fill pin (if any), making its cursor slot ordinarily
    /// evictable. The slot stays bound and resident; the next `reserve` rotates.
    pub(crate) fn release_cursor(&self, cursor: &mut FillCursor) {
        cursor.buffer.take();
    }

    /// Insert the decompressed page written into `reservation` and return the
    /// views to hand the decoder - the freshly written bytes ARE the page the
    /// caller is producing, so this is how they reach it. `header` is opaque to
    /// this cache (see the module doc) - it's simply handed back verbatim from a
    /// [`get_range`](Self::get_range) hit. `live_blocks` is the caller's
    /// residency counter for this block's grouping. A transient reservation, or
    /// a packed one that loses the dup race (another worker inserted the same
    /// key first), returns readable views without caching; a lost race's
    /// reserved bytes simply stay unreachable until their slots recycle.
    pub fn insert(
        &self,
        key: BlockKey,
        header: Vec<Bytes>,
        reservation: Reservation,
        live_blocks: Arc<AtomicUsize>,
    ) -> Vec<Bytes> {
        let region = match reservation {
            Reservation::Transient { buffers, len } => {
                return build_transient_views(buffers, len);
            }
            Reservation::Packed(region) => region,
        };
        let views: Vec<Bytes> = region
            .segments
            .iter()
            .map(|s| Bytes::from_owner(PinnedSlot(s.pin.clone())).slice(s.offset..s.offset + s.len))
            .collect();
        let extents: Vec<BlockExtent> = region
            .segments
            .iter()
            .map(|s| BlockExtent {
                slot: s.slot,
                offset: s.offset as u32,
                len: s.len as u32,
            })
            .collect();
        self.claim(&key, extents, header, live_blocks);
        views
    }

    /// Insert the block's forward entry under the file's write lock, creating the
    /// file's table if needed. A key whose `offset` is already cached loses the
    /// dup race: the incumbent stays, the newcomer is dropped.
    fn claim(
        &self,
        key: &BlockKey,
        extents: Vec<BlockExtent>,
        header: Vec<Bytes>,
        live_blocks: Arc<AtomicUsize>,
    ) {
        loop {
            {
                let files = self.files.read().unwrap();
                if let Some(table) = files.get(&key.open_file) {
                    let mut table = table.write().unwrap();
                    if table.contains_key(&key.offset) {
                        return;
                    }
                    table.insert(
                        key.offset,
                        DecompressedBlock::new(extents, key.len, header, live_blocks),
                    );
                    self.blocks.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
            // File absent: create its (empty) table, then retry the claim above.
            self.files
                .write()
                .unwrap()
                .entry(key.open_file.clone())
                .or_default();
        }
    }

    /// Look up a block. On a hit, pins each of its extents' slots with
    /// `try_read`, refreshes the clock, and returns one zero-copy `Bytes` view
    /// per extent. Misses (returning `None`, releasing any pins taken) if the
    /// block is absent, its span doesn't match `key.len`, or a slot is
    /// mid-reclaim (`WRITING`).
    pub fn get(&self, key: &BlockKey) -> Option<Vec<Bytes>> {
        if !self.enabled {
            return None;
        }
        let files = self.files.read().unwrap();
        let table = files.get(&key.open_file)?.read().unwrap();
        let block = table
            .get(&key.offset)
            .filter(|b| b.compressed_len == key.len)?;
        block.pin_views()
    }

    /// Resolve the file byte range `[offset, offset+len)` against the cache:
    /// segments in file order, `Segment::Cached` for each block found (pinned) and
    /// `Segment::Gap` for each byte stretch not covered. When the cache is disabled
    /// or nothing in the file is cached, this is exactly `[Segment::Gap{offset,
    /// len}]` - the range is entirely unresolved, same as if this cache didn't
    /// exist.
    pub fn get_range(&self, open_file: &OpenFile, offset: usize, len: usize) -> Vec<Segment> {
        if len == 0 {
            return Vec::new();
        }
        // Lock-free: skip the `files` `RwLock` entirely when nothing anywhere is
        // cached yet (disabled, or simply not warmed up) - the common case on a
        // cold read, where this would otherwise be a guaranteed-empty lock
        // acquisition and hashmap lookup on every column of every row group.
        if !self.enabled || self.is_empty() {
            return vec![Segment::Gap { offset, len }];
        }
        let end = offset + len;
        let mut segments = Vec::new();
        let mut cursor = offset;

        let files = self.files.read().unwrap();
        if let Some(table) = files.get(open_file) {
            let table = table.read().unwrap();
            for (&block_offset, block) in table.range(offset..end) {
                push_gap(&mut segments, cursor, block_offset - cursor);
                cursor = block_offset + block.compressed_len;
                match block.pin_views() {
                    Some(data) => segments.push(Segment::Cached {
                        offset: block_offset,
                        span: block.compressed_len,
                        header: block.header.clone(),
                        data,
                    }),
                    // Mid-reclaim: treat this span as if it weren't cached at all.
                    None => push_gap(&mut segments, block_offset, block.compressed_len),
                }
            }
        }
        push_gap(&mut segments, cursor, end - cursor);
        segments
    }

    /// Reinforce every cached block overlapping the file byte range
    /// `[offset, offset + len)` - called by the compressed cache when it evicts
    /// runs over that range, leaving these blocks the last in-memory copy of it.
    /// Takes only read locks. The range is a compressed run (4 KB aligned), so a
    /// block straddling its start is checked for separately from the in-range
    /// walk.
    pub(crate) fn reinforce_range(&self, open_file: &OpenFile, offset: usize, len: usize) {
        if !self.enabled || len == 0 {
            return;
        }
        let end = offset + len;
        let files = self.files.read().unwrap();
        let Some(table) = files.get(open_file) else {
            return;
        };
        let table = table.read().unwrap();
        let reinforce_block = |block: &DecompressedBlock| {
            for extent in &block.extents {
                memory_ctx().clock().reinforce(extent.slot);
            }
        };
        if let Some((&block_offset, block)) = table.range(..offset).next_back()
            && block_offset + block.compressed_len > offset
        {
            reinforce_block(block);
        }
        for (_, block) in table.range(offset..end) {
            reinforce_block(block);
        }
    }

    /// The clock picked `slot` as a victim. Take it exclusively (`try_write`),
    /// take its tenant list, and drop every tenant block whose current map entry
    /// still references this slot (the identity guard: stale tenants - evicted
    /// straddlers, lost dup races, unwound decompresses, keys re-homed since -
    /// simply miss). A straddling block dies whole; its bytes in other slots
    /// stay unreachable until those slots recycle. Returns the slot writable;
    /// `None` if a reader still pins it or it is no longer this cache's.
    pub fn reclaim(&self, slot: usize) -> Option<WriteBuffer> {
        let victim = memory_ctx().ring().try_write(slot)?;
        if memory_ctx().clock().owner(slot) != Some(Owner::Decompressed) {
            // Raced to the free pool or another cache between the sweep and try_write.
            victim.release_in_place();
            return None;
        }
        let dropped = self.purge_slot_tenants(slot);
        memory_ctx().clock().release(slot);

        // The compressed source of these bytes is now the last in-memory copy;
        // reinforce it so both copies rarely die close together (which is what
        // turns the next read into disk IO). MUST stay after this cache's map
        // locks are released: the compressed evictor reinforces our blocks while
        // holding its own maps, which is only cycle-free because we never hold
        // ours when walking its.
        for (open_file, offset, len) in dropped {
            memory_ctx()
                .compressed_cache()
                .reinforce_range(&open_file, offset, len);
        }
        Some(victim)
    }

    /// Drop every tenant block of `slot` whose current map entry still references
    /// it, returning the dropped blocks' source ranges. Sound only while `slot`
    /// is held exclusively (`try_write` succeeded).
    fn purge_slot_tenants(&self, slot: usize) -> Vec<(OpenFile, usize, usize)> {
        let keys = std::mem::take(self.tenants_mut(slot));
        let mut dropped = Vec::new();
        let mut emptied_files = Vec::new();
        {
            let files = self.files.read().unwrap();
            for key in keys {
                let Some(table) = files.get(&key.open_file) else {
                    continue;
                };
                let mut table = table.write().unwrap();
                let Some(block) = table.get(&key.offset) else {
                    continue;
                };
                // Identity guard: the entry at this key may be a different,
                // re-inserted block that lives elsewhere - it must survive.
                if !block.extents.iter().any(|extent| extent.slot == slot) {
                    continue;
                }
                dropped.push((key.open_file.clone(), key.offset, block.compressed_len));
                table.remove(&key.offset);
                self.blocks.fetch_sub(1, Ordering::Relaxed);
                if table.is_empty() {
                    emptied_files.push(key.open_file);
                }
            }
        }
        for open_file in emptied_files {
            self.prune_empty_file(&open_file);
        }
        dropped
    }

    /// Remove `open_file`'s table if it is now empty, so its never-reused
    /// `OpenFile` (a `LocalFile`/`Arc<RemoteFile>` pinning the file/connection
    /// it holds) isn't kept alive for every file ever opened. Re-checks emptiness
    /// under the outer write lock so a concurrent `claim` that just re-created the
    /// table isn't dropped.
    fn prune_empty_file(&self, open_file: &OpenFile) {
        let mut files = self.files.write().unwrap();
        if let Some(table) = files.get(open_file)
            && table.read().unwrap().is_empty()
        {
            files.remove(open_file);
        }
    }

    /// Whether the cache holds no blocks at all. `get_write_buffer` checks this
    /// to decide whether eviction would have to fall through to the compressed
    /// cache, and `get_range` uses it as its cold fast path.
    pub fn is_empty(&self) -> bool {
        self.blocks.load(Ordering::Relaxed) == 0
    }

    /// Drop every block belonging to `open_file` (e.g. when its fd is reopened) so a
    /// stale block can't serve a later read. The slots stay `Decompressed` with
    /// stale tenants and are recycled lazily by the clock (via
    /// [`reclaim`](Self::reclaim), whose identity guard skips them).
    pub fn invalidate(&self, open_file: &OpenFile) {
        if !self.enabled {
            return;
        }
        if let Some(table) = self.files.write().unwrap().remove(open_file) {
            self.blocks
                .fetch_sub(table.into_inner().unwrap().len(), Ordering::Relaxed);
        }
    }

    /// Drop every cached block, returning how many were dropped. Releases this
    /// worker's fill cursor, then eagerly recycles every slot it can take
    /// exclusively, using the same guarded purge as [`reclaim`](Self::reclaim)
    /// (a slot re-packed by a concurrent worker in the window keeps whatever its
    /// current map entries say). Reader-pinned slots (including other workers'
    /// fill cursors) stay bound and recycle lazily. Backs `drop_cache()`.
    pub fn clear(&self) -> usize {
        self.release_cursor(memory_ctx().decompressed_fill_cursor());
        let files = std::mem::take(&mut *self.files.write().unwrap());
        let mut dropped = 0;
        let mut slots = std::collections::HashSet::new();
        for (_, table) in files {
            for (_, block) in table.into_inner().unwrap() {
                dropped += 1;
                for extent in &block.extents {
                    slots.insert(extent.slot);
                }
            }
        }
        self.blocks.fetch_sub(dropped, Ordering::Relaxed);
        for slot in slots {
            let Some(buffer) = memory_ctx().ring().try_write(slot) else {
                continue;
            };
            if memory_ctx().clock().owner(slot) != Some(Owner::Decompressed) {
                buffer.release_in_place();
                continue;
            }
            // The map was drained, so tenants are stale unless a concurrent
            // worker re-packed the slot - the guard inside handles both.
            self.purge_slot_tenants(slot);
            memory_ctx().clock().release(slot);
            drop(buffer); // returns the slot to the pool
        }
        dropped
    }
}

/// Build `WriteBuffer`-backed (uncached, pooled-on-drop) views from whole
/// buffers holding `total_len` bytes.
fn build_transient_views(write_buffers: Vec<WriteBuffer>, total_len: usize) -> Vec<Bytes> {
    write_buffers
        .into_iter()
        .enumerate()
        .map(|(i, buffer)| {
            let len = (total_len - i * BUFFER_SIZE).min(BUFFER_SIZE);
            Bytes::from_owner(buffer).slice(..len)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalFile;
    use crate::memory::context::{init_test_free_pool, memory_ctx};
    use std::sync::{Arc, OnceLock};

    /// A fresh, independent local file (distinct cache key bucket).
    fn new_file() -> OpenFile {
        OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
    }

    fn key_for(open_file: &OpenFile) -> BlockKey {
        BlockKey {
            open_file: open_file.clone(),
            offset: 0,
            len: 1,
        }
    }

    /// One shared local file, reused across tests that key by `offset` alone.
    #[allow(non_snake_case)]
    fn FD() -> OpenFile {
        static FILE: OnceLock<LocalFile> = OnceLock::new();
        OpenFile::Local(
            FILE.get_or_init(|| LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
                .clone(),
        )
    }

    /// Keys over `FD()`, distinguished by `offset`.
    fn test_key(offset: usize) -> BlockKey {
        BlockKey {
            open_file: FD(),
            offset,
            len: 1,
        }
    }

    fn for_test(enabled: bool) -> DecompressedCache {
        // Matches the 128-slot ring `init_test_free_pool` installs. Built without
        // touching `memory_ctx` so `get_misses_for_an_absent_key` (which never inits
        // a pool) keeps working.
        DecompressedCache {
            files: RwLock::new(HashMap::default()),
            tenants: (0..128).map(|_| UnsafeCell::new(Vec::new())).collect(),
            blocks: AtomicUsize::new(0),
            enabled,
        }
    }

    /// Placeholder header bytes: their content is opaque to this cache, only their
    /// identity as "whatever the caller passed" matters here.
    fn make_header(tag: u8) -> Vec<Bytes> {
        vec![Bytes::from(vec![tag])]
    }

    /// A fresh live-block counter (the shape the caller passes into `insert`).
    fn counter() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }

    /// Reserve `len` decompressed bytes for `key`, fill them with `pattern`, and
    /// publish. Returns the views the decoder would get.
    fn insert_page(
        cache: &DecompressedCache,
        key: BlockKey,
        len: usize,
        pattern: u8,
    ) -> Vec<Bytes> {
        let mut reservation = cache.reserve(&key, len);
        for slice in reservation.as_mut_slices() {
            slice.fill(pattern);
        }
        cache.insert(key, make_header(pattern), reservation, counter())
    }

    /// Drop this worker's fill pin so packed slots become ordinarily evictable.
    fn release_cursor(cache: &DecompressedCache) {
        cache.release_cursor(memory_ctx().decompressed_fill_cursor());
    }

    /// The slot the worker's cursor currently points at (where small pages pack).
    fn cursor_slot() -> usize {
        memory_ctx().decompressed_fill_cursor().slot_idx
    }

    #[test]
    fn get_misses_for_an_absent_key() {
        let cache = for_test(true);

        assert!(cache.get(&test_key(0)).is_none());
    }

    #[test]
    fn two_pages_pack_into_one_slot() {
        init_test_free_pool(4);
        let cache = for_test(true);

        insert_page(&cache, test_key(0), 100, 0xAA);
        insert_page(&cache, test_key(100), 100, 0xBB);

        let a = cache.get(&test_key(0)).unwrap();
        let b = cache.get(&test_key(100)).unwrap();
        assert_eq!(memory_ctx().clock().owned(Owner::Decompressed), 1);
        assert!(a.iter().flat_map(|v| v.iter()).all(|&x| x == 0xAA));
        assert!(b.iter().flat_map(|v| v.iter()).all(|&x| x == 0xBB));
    }

    #[test]
    fn a_straddling_page_reads_back_its_bytes() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), BUFFER_SIZE - 7, 0x11); // leave a 7-byte tail

        let views = insert_page(&cache, test_key(1000), 100, 0x22); // 7 + 93 across two slots

        let hit = cache.get(&test_key(1000)).unwrap();
        assert_eq!(hit.len(), 2, "two extents across the slot boundary");
        assert_eq!(hit.iter().map(|v| v.len()).sum::<usize>(), 100);
        assert!(hit.iter().flat_map(|v| v.iter()).all(|&x| x == 0x22));
        assert!(views.iter().flat_map(|v| v.iter()).all(|&x| x == 0x22));
    }

    #[test]
    fn a_hit_is_ring_backed_and_copy_free() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let key = test_key(0);
        let mut reservation = cache.reserve(&key, 8);
        let ptr = reservation.as_mut_slices()[0].as_ptr();
        cache.insert(key, make_header(0), reservation, counter());

        let hit = cache.get(&test_key(0)).unwrap();

        assert_eq!(hit[0].as_ptr(), ptr);
    }

    #[test]
    fn a_different_length_at_the_same_offset_misses() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let key = BlockKey {
            len: 100,
            ..test_key(0)
        };
        insert_page(&cache, key, 8, 0);

        let other = BlockKey {
            len: 200,
            ..test_key(0)
        };

        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn a_lost_dup_race_returns_readable_views_and_keeps_the_incumbent() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), 50, 0xAA);

        let dup_views = insert_page(&cache, test_key(0), 50, 0xBB);

        assert!(dup_views.iter().flat_map(|v| v.iter()).all(|&x| x == 0xBB));
        let hit = cache.get(&test_key(0)).unwrap();
        assert!(
            hit.iter().flat_map(|v| v.iter()).all(|&x| x == 0xAA),
            "the first insert stays cached"
        );
    }

    #[test]
    fn an_unpublished_reservation_leaves_the_cache_consistent() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let reservation = cache.reserve(&test_key(0), 64);
        drop(reservation); // decompress failed / unwound: never published
        let slot = cursor_slot();
        release_cursor(&cache);

        let reclaimed = cache.reclaim(slot);

        assert!(reclaimed.is_some(), "only a stale tenant, freely recycled");
        assert!(cache.is_empty());
    }

    #[test]
    fn reclaim_evicts_every_tenant_of_the_slot() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), 100, 1);
        insert_page(&cache, test_key(100), 100, 2);
        let slot = cursor_slot();
        release_cursor(&cache);

        let reclaimed = cache.reclaim(slot);

        assert_eq!(reclaimed.unwrap().slot_idx, slot);
        assert!(cache.get(&test_key(0)).is_none());
        assert!(cache.get(&test_key(100)).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn evicting_one_slot_of_a_straddler_kills_the_block_whole() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), BUFFER_SIZE - 7, 0x11);
        insert_page(&cache, test_key(1000), 100, 0x22); // straddles: 7 in slot A, 93 in slot B
        let second_slot = cursor_slot();
        release_cursor(&cache);

        // Evict the straddler's SECOND slot: both blocks referencing it die whole.
        let reclaimed = cache.reclaim(second_slot);

        assert!(reclaimed.is_some());
        assert!(cache.get(&test_key(1000)).is_none(), "straddler died whole");
        assert!(
            cache.get(&test_key(0)).is_some(),
            "the first slot's other page is untouched"
        );
    }

    #[test]
    fn a_stale_partner_slot_recycles_cleanly() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), BUFFER_SIZE - 7, 0x11);
        insert_page(&cache, test_key(1000), 100, 0x22);
        let first_slot = {
            let files = cache.files.read().unwrap();
            let table = files.get(&FD()).unwrap().read().unwrap();
            table.get(&0).unwrap().extents[0].slot
        };
        let second_slot = cursor_slot();
        release_cursor(&cache);
        cache.reclaim(second_slot); // strands the straddler's bytes in first_slot

        let reclaimed = cache.reclaim(first_slot);

        assert!(reclaimed.is_some(), "stale tenant filtered, slot recycled");
        assert!(cache.is_empty());
    }

    #[test]
    fn reclaiming_a_stale_orphan_spares_the_reinserted_block() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), 64, 0xAA);
        let orphan_slot = cursor_slot();
        cache.invalidate(&FD()); // the block is gone; its tenant is now stale
        release_cursor(&cache); // force the re-insert into a fresh slot
        insert_page(&cache, test_key(0), 64, 0xBB);
        release_cursor(&cache);

        let reclaimed = cache.reclaim(orphan_slot).unwrap();

        assert_eq!(reclaimed.slot_idx, orphan_slot);
        assert_eq!(
            memory_ctx().clock().owner(orphan_slot),
            None,
            "the orphan must be fully released, not returned while still owned"
        );
        assert!(
            cache.get(&test_key(0)).is_some(),
            "the re-inserted block at the same key must survive"
        );
    }

    #[test]
    fn evicting_a_files_last_block_releases_its_file_handle() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let open_file = new_file();
        let OpenFile::Local(file) = &open_file else {
            unreachable!()
        };
        insert_page(&cache, key_for(&open_file), 8, 0);
        let slot = cursor_slot();
        release_cursor(&cache);

        cache.reclaim(slot);

        assert_eq!(
            file.strong_count(),
            1,
            "evicting a file's last block drops the cache's clone of its fd, not just the block"
        );
    }

    #[test]
    fn reclaim_is_none_for_an_unowned_slot() {
        init_test_free_pool(1);
        let cache = for_test(true);

        assert!(cache.reclaim(0).is_none());
    }

    #[test]
    fn is_empty_tracks_the_block_count() {
        init_test_free_pool(2);
        let cache = for_test(true);
        assert!(cache.is_empty());

        insert_page(&cache, test_key(0), 8, 0);
        assert!(!cache.is_empty());
        cache.invalidate(&FD());

        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_drops_only_the_matching_location() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let a = new_file();
        let b = new_file();
        insert_page(&cache, key_for(&a), 8, 0);
        insert_page(&cache, key_for(&b), 8, 0);

        cache.invalidate(&a);

        assert!(cache.get(&key_for(&a)).is_none());
        assert!(cache.get(&key_for(&b)).is_some());
    }

    #[test]
    fn clear_drops_everything() {
        init_test_free_pool(4);
        let cache = for_test(true);
        insert_page(&cache, test_key(0), 8, 0);
        insert_page(&cache, test_key(100), 8, 0);

        let dropped = cache.clear();

        assert_eq!(dropped, 2);
        assert!(cache.is_empty());
        assert_eq!(memory_ctx().clock().owned(Owner::Decompressed), 0);
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        init_test_free_pool(1);
        let cache = for_test(false);

        drop(insert_page(&cache, test_key(0), 8, 0));

        assert!(cache.get(&test_key(0)).is_none());
        assert!(cache.is_empty());
    }

    // -- get_range --

    /// Extract each segment's kind as `(offset, len)` for easy assertion, plus a tag
    /// identifying `Cached` segments by their header byte.
    #[derive(Debug, PartialEq, Eq)]
    enum Shape {
        Gap(usize, usize),
        Cached(usize, usize, u8),
    }
    fn shapes(segments: &[Segment]) -> Vec<Shape> {
        segments
            .iter()
            .map(|s| match s {
                Segment::Gap { offset, len } => Shape::Gap(*offset, *len),
                Segment::Cached {
                    offset,
                    span,
                    header,
                    ..
                } => Shape::Cached(*offset, *span, header[0][0]),
            })
            .collect()
    }

    #[test]
    fn get_range_over_an_empty_cache_is_one_gap() {
        init_test_free_pool(1);
        let cache = for_test(true);

        let segments = cache.get_range(&FD(), 100, 50);

        assert_eq!(shapes(&segments), vec![Shape::Gap(100, 50)]);
    }

    #[test]
    fn get_range_is_one_gap_when_disabled() {
        init_test_free_pool(1);
        let cache = for_test(false);
        insert_page(&cache, test_key(0), 8, 9);

        let segments = cache.get_range(&FD(), 0, 10);

        assert_eq!(shapes(&segments), vec![Shape::Gap(0, 10)]);
    }

    #[test]
    fn get_range_finds_a_hit_surrounded_by_gaps() {
        init_test_free_pool(2);
        let cache = for_test(true);
        // A page at file offset 10, span 5 (so it occupies [10, 15)).
        let key = BlockKey {
            len: 5,
            ..test_key(10)
        };
        insert_page(&cache, key, 8, 7);
        release_cursor(&cache);

        let segments = cache.get_range(&FD(), 0, 20);

        assert_eq!(
            shapes(&segments),
            vec![
                Shape::Gap(0, 10),
                Shape::Cached(10, 5, 7),
                Shape::Gap(15, 5)
            ]
        );
    }

    #[test]
    fn get_range_has_no_gap_between_adjacent_hits() {
        init_test_free_pool(2);
        let cache = for_test(true);
        insert_page(
            &cache,
            BlockKey {
                len: 5,
                ..test_key(0)
            },
            8,
            1,
        );
        insert_page(
            &cache,
            BlockKey {
                len: 5,
                ..test_key(5)
            },
            8,
            2,
        );
        release_cursor(&cache);

        let segments = cache.get_range(&FD(), 0, 10);

        assert_eq!(
            shapes(&segments),
            vec![Shape::Cached(0, 5, 1), Shape::Cached(5, 5, 2)]
        );
    }
}
