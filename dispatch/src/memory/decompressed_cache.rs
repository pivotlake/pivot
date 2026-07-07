//! A cache of decompressed blocks, sharing the ring's CLOCK
//! ([`Clock`](super::clock::Clock)) with the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache), and modelled on
//! it: a cached block sits in its ring slot(s) at `used == 0`, owned via the clock
//! (no held pin). A [`get`](DecompressedCache::get) reconstructs a zero-copy view
//! with `try_read`; eviction acquires the slots with `try_write` first and only
//! then removes the block - never the drop-into-pool-then-grab dance.
//!
//! ## Maps
//!
//! `files` maps a [`FileLocation`] to its own `RwLock`'d table of `offset -> block`,
//! ordered by offset (like
//! [`CompressedCache`](super::compressed_cache::CompressedCache)'s `file_maps`), so a
//! `get` only contends on one file's lock and [`get_range`](DecompressedCache::get_range)
//! can walk a byte range in file order. `reverse` is a per-slot index - one entry per
//! ring slot, no lock, like the compressed cache's per-slot tenant list - naming the
//! block that owns each slot, so the evictor (which only has a slot index) can find
//! the block. A slot's reverse entry is written and read only while the slot is held
//! exclusively (the inserter's `WRITING`, or the evictor's `try_write`), which is what
//! lets it skip a lock.
//!
//! ## Eviction
//!
//! The cache claims each block's slots in the shared clock as
//! [`Owner::Decompressed`], whose lower tier max makes them age out before
//! compressed pages. The clock picks a slot and calls
//! [`reclaim`](DecompressedCache::reclaim): it `try_write`s the victim, reads which
//! block owns it, then `try_write`s the block's other slots (non-blocking, so any
//! order is deadlock-free). If any slot is still pinned by a reader it releases the
//! rest and bails, so the block stays cached. The compressed source pages age out
//! more slowly, so an evicted decompressed block falls back to an in-memory
//! re-decompress rather than a disk read.
//!
//! ## Range queries
//!
//! A block's key is opaque to this module - just a file, a source byte offset, and a
//! source byte span - but the caller (the Parquet reader) always uses a Parquet
//! page's on-disk span as that key, and pages within one column chunk sit back to
//! back with no gaps. That's what [`get_range`](DecompressedCache::get_range) relies
//! on: given a column chunk's own `(offset, len)` - itself always a whole run of
//! pages - it can report which of that range is already decompressed and which byte
//! stretches are gaps, and every gap is guaranteed to be a whole number of complete
//! pages, without this module ever knowing what a "page" is. The caller re-derives
//! each page's header from the [`Segment::Cached`] blob it gets back (this module
//! stores it as an opaque, caller-supplied byte string) instead of this module
//! knowing anything about Parquet.

use crate::env::get_env_var_with_default;
use crate::io::FileLocation;
use crate::memory::clock::Owner;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use ahash::HashMap;
use bytes::Bytes;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Valid byte length of slot `i` of a block whose decompressed bytes total
/// `decompressed_len`: every slot is full (`BUFFER_SIZE`) except the last, which holds the
/// remainder. Lets a block store one length instead of a per-slot `Vec`.
fn slot_len(decompressed_len: usize, i: usize) -> usize {
    (decompressed_len - i * BUFFER_SIZE).min(BUFFER_SIZE)
}

/// Identity of a cached decompressed block: the file, the byte offset of its
/// source bytes, and their span. A page's header offset is unique within its file,
/// so the map key is just the offset; `len` (the source span) is instead checked
/// against the stored block on lookup, so a lookup never serves bytes decompressed
/// from a differently-sized region at the same offset.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub location: FileLocation,
    pub offset: usize,
    pub len: usize,
}

/// A cached decompressed block: the ring slots holding its bytes (one per 2 MB), its
/// length in each of the two domains it straddles, and the caller-supplied header
/// bytes describing it (opaque to this module - see the module doc).
struct DecompressedBlock {
    slots: Vec<usize>,
    /// Length of the decompressed bytes held in `slots` (per-slot lengths come
    /// from [`slot_len`]).
    decompressed_len: usize,
    /// Length of the block's compressed source in the file (this block's
    /// `BlockKey::len` at insert time) - what lets [`get`](DecompressedCache::get)
    /// reject a same-offset/different-size request, and what lets
    /// [`get_range`](DecompressedCache::get_range) find the next file byte after
    /// this block.
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
        slots: Vec<usize>,
        decompressed_len: usize,
        compressed_len: usize,
        header: Vec<Bytes>,
        live_blocks: Arc<AtomicUsize>,
    ) -> Self {
        live_blocks.fetch_add(1, Ordering::Relaxed);
        Self {
            slots,
            decompressed_len,
            compressed_len,
            header,
            live_blocks,
        }
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
    files: RwLock<HashMap<FileLocation, FileBlocks>>,
    /// Per-slot reverse index, indexed by ring slot: the key of the block that owns
    /// the slot, or `None` if unowned. `UnsafeCell` (like the compressed cache's
    /// per-slot `tenants`): an entry is written/read only while its slot is held
    /// exclusively (the inserter's `WRITING`, or the evictor's `try_write`), so it
    /// needs no lock. Lets the evictor, holding only a slot index, find the block.
    reverse: Box<[UnsafeCell<Option<BlockKey>>]>,
    /// Kill switch (`PIVOT_DECOMPRESSED_CACHE`, default on). When off, `get` is a
    /// no-op and `insert` returns the bytes uncached.
    enabled: bool,
}

// SAFETY: `reverse`'s `UnsafeCell`s are only accessed while the corresponding ring
// slot is held exclusively (`WRITING`), so there is never concurrent access to one
// entry. Same discipline as [`CompressedCache`].
unsafe impl Send for DecompressedCache {}
unsafe impl Sync for DecompressedCache {}

impl DecompressedCache {
    /// `capacity` is the ring's slot count: one reverse entry per slot.
    pub fn new(capacity: usize) -> Self {
        Self {
            files: RwLock::new(HashMap::default()),
            reverse: (0..capacity).map(|_| UnsafeCell::new(None)).collect(),
            enabled: get_env_var_with_default("PIVOT_DECOMPRESSED_CACHE", true),
        }
    }

    /// Mutable access to slot `idx`'s reverse entry. Sound only while the slot is
    /// held exclusively (the inserter's `WRITING`, or the evictor's `try_write`) -
    /// the one `unsafe` for the reverse invariant.
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell; see above
    fn reverse_mut(&self, idx: usize) -> &mut Option<BlockKey> {
        unsafe { &mut *self.reverse[idx].get() }
    }

    /// Look up a block. On a hit, pins each of its slots with `try_read`, refreshes
    /// the clock, and returns one zero-copy `Bytes` view per slot. Misses (returning
    /// `None`, releasing any pins taken) if the block is absent, its span doesn't
    /// match `key.len`, or a slot is mid-reclaim (`WRITING`).
    pub fn get(&self, key: &BlockKey) -> Option<Vec<Bytes>> {
        if !self.enabled {
            return None;
        }
        let files = self.files.read().unwrap();
        let table = files.get(&key.location)?.read().unwrap();
        let block = table
            .get(&key.offset)
            .filter(|b| b.compressed_len == key.len)?;
        self.read_block(block)
    }

    /// Resolve the file byte range `[offset, offset+len)` against the cache:
    /// segments in file order, `Segment::Cached` for each block found (pinned) and
    /// `Segment::Gap` for each byte stretch not covered. When the cache is disabled
    /// or nothing in the file is cached, this is exactly `[Segment::Gap{offset,
    /// len}]` - the range is entirely unresolved, same as if this cache didn't
    /// exist.
    pub fn get_range(&self, location: &FileLocation, offset: usize, len: usize) -> Vec<Segment> {
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
        if let Some(table) = files.get(location) {
            let table = table.read().unwrap();
            for (&block_offset, block) in table.range(offset..end) {
                push_gap(&mut segments, cursor, block_offset - cursor);
                cursor = block_offset + block.compressed_len;
                match self.read_block(block) {
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

    /// Give one extra life to every cached block overlapping the file byte range
    /// `[offset, offset + len)` - called by the compressed cache when it evicts
    /// runs over that range, leaving these blocks the last in-memory copy of it.
    /// Takes only read locks. The range is a compressed run (4 KB aligned), so a
    /// block straddling its start is checked for separately from the in-range
    /// walk.
    pub(crate) fn reinforce_range(&self, location: &FileLocation, offset: usize, len: usize) {
        if !self.enabled || len == 0 {
            return;
        }
        let end = offset + len;
        let files = self.files.read().unwrap();
        let Some(table) = files.get(location) else {
            return;
        };
        let table = table.read().unwrap();
        let reinforce_block = |block: &DecompressedBlock| {
            for &slot in &block.slots {
                memory_ctx().clock().reinforce(slot);
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

    /// Pin every slot of `block` with `try_read`, refreshing the clock, and return
    /// one zero-copy view per slot. `None` (no pins taken) if any slot is
    /// mid-reclaim (`WRITING`).
    fn read_block(&self, block: &DecompressedBlock) -> Option<Vec<Bytes>> {
        let mut views = Vec::with_capacity(block.slots.len());
        for (i, &slot) in block.slots.iter().enumerate() {
            let read = memory_ctx().ring().try_read(slot)?;
            memory_ctx().clock().touch(slot);
            views.push(Bytes::from_owner(read).slice(..slot_len(block.decompressed_len, i)));
        }
        Some(views)
    }

    /// Cache the page just decompressed into `write_buffers` (one per 2 MB slot,
    /// `decompressed_len` decompressed bytes across them) and return the bytes to hand the
    /// decoder. `header` is opaque to this cache (see the module doc) - it's simply
    /// handed back verbatim from a [`get_range`] hit. `live_blocks` is the caller's
    /// residency counter for this block's grouping: incremented while the block is
    /// cached, decremented when it leaves (a transient copy is never counted). If
    /// it caches, each buffer is converted to a read pin so its slot becomes
    /// resident at `used == 0` once the decoder drops the returned `Bytes`;
    /// otherwise (cache off, empty page, or the block already cached) the
    /// `WriteBuffer`-backed bytes are returned and their slots return to the pool
    /// on drop.
    ///
    /// [`get_range`]: DecompressedCache::get_range
    pub fn insert(
        &self,
        key: BlockKey,
        header: Vec<Bytes>,
        write_buffers: Vec<WriteBuffer>,
        decompressed_len: usize,
        live_blocks: Arc<AtomicUsize>,
    ) -> Vec<Bytes> {
        if !self.enabled || write_buffers.is_empty() {
            return build_transient_views(write_buffers, decompressed_len);
        }
        let slots: Vec<usize> = write_buffers.iter().map(|b| b.slot_idx).collect();

        // Claim `offset` under the file's write lock (so two workers decompressing
        // the same page don't both cache it); on a dup, leave this copy transient.
        if !self.claim(&key, &slots, decompressed_len, header, live_blocks) {
            return build_transient_views(write_buffers, decompressed_len);
        }

        // Record the reverse index and clock ownership while the slots are still
        // `WRITING` (held by `write_buffers`), so the evictor can't touch them yet -
        // which is also what makes writing each slot's reverse entry sound.
        for &slot in &slots {
            *self.reverse_mut(slot) = Some(key.clone());
            memory_ctx().clock().bind(slot, Owner::Decompressed);
        }

        // Convert each buffer to a read pin (`used == 1`); when the decoder drops
        // the returned views the slot drops to `used == 0` - resident, not pooled.
        write_buffers
            .into_iter()
            .enumerate()
            .map(|(i, buffer)| {
                Bytes::from_owner(ReadBuffer::from(buffer)).slice(..slot_len(decompressed_len, i))
            })
            .collect()
    }

    /// Insert the block's forward entry under the file's write lock, creating the
    /// file's table if needed. Returns `false` if `offset` is already cached.
    fn claim(
        &self,
        key: &BlockKey,
        slots: &[usize],
        decompressed_len: usize,
        header: Vec<Bytes>,
        live_blocks: Arc<AtomicUsize>,
    ) -> bool {
        loop {
            {
                let files = self.files.read().unwrap();
                if let Some(table) = files.get(&key.location) {
                    let mut table = table.write().unwrap();
                    if table.contains_key(&key.offset) {
                        return false;
                    }
                    table.insert(
                        key.offset,
                        DecompressedBlock::new(
                            slots.to_vec(),
                            decompressed_len,
                            key.len,
                            header,
                            live_blocks,
                        ),
                    );
                    return true;
                }
            }
            // File absent: create its (empty) table, then retry the claim above.
            self.files
                .write()
                .unwrap()
                .entry(key.location.clone())
                .or_default();
        }
    }

    /// The clock picked `slot` as a victim. Take it exclusively (`try_write`), read
    /// which block owns it, and evict that whole block: `try_write` every one of its
    /// slots and only then remove it and recycle them. If any slot is still pinned by
    /// a reader, release the ones taken without pooling them (the block stays cached)
    /// and return `None`. Returns the requested slot writable; the block's other
    /// slots return to the pool.
    pub fn reclaim(&self, slot: usize) -> Option<WriteBuffer> {
        // Take the victim first; only with it held `WRITING` is its reverse entry
        // sound to read (the same discipline `CompressedCache` uses for `tenants`).
        let victim = memory_ctx().ring().try_write(slot)?;
        if memory_ctx().clock().owner(slot) != Some(Owner::Decompressed) {
            // Raced to the free pool or another cache between the sweep and try_write.
            victim.release_in_place();
            return None;
        }
        // Borrow (don't clone) the owning block's key: we hold `slot` `WRITING` for
        // the rest of this call, so its reverse entry can't change under us. Sound to
        // read now that the owner is `Decompressed` - `insert` sets the entry before
        // binding the owner and the evict paths clear it after releasing, both under
        // `WRITING`, so a `Decompressed` slot always has one.
        let key = self
            .reverse_mut(slot)
            .as_ref()
            .expect("a Decompressed slot always has a reverse entry");

        // The block's full slot set. If its forward entry is gone (invalidate/clear
        // dropped it), this slot is an orphan: recycle just this one - no other
        // reader can reach an unmapped block, so no atomic group is needed.
        let block_slots = {
            let files = self.files.read().unwrap();
            files.get(&key.location).and_then(|table| {
                table
                    .read()
                    .unwrap()
                    .get(&key.offset)
                    .map(|b| b.slots.clone())
            })
        };
        let Some(block_slots) = block_slots else {
            return Some(self.recycle_orphan(slot, victim));
        };
        // The entry at `key` may be a *different* block: invalidate/clear dropped
        // the victim's block and the same key was re-inserted with fresh slots. The
        // victim is then a stale orphan, and the re-inserted block - which does not
        // hold the victim - must survive untouched.
        if !block_slots.contains(&slot) {
            return Some(self.recycle_orphan(slot, victim));
        }

        // Acquire the block's other slots; the victim is already held, and
        // `try_write` is non-blocking so any order is deadlock-free. Any miss (a
        // reader still pinning a slot) releases all taken and bails, leaving the
        // block cached. A slot whose reverse entry no longer names `key` was
        // recycled and re-homed to another block in the window since `block_slots`
        // was read - releasing it would rip it out of its new block - so bail the
        // same way; a later sweep sees the consistent end state.
        let mut buffers = vec![victim];
        for &block_slot in block_slots.iter().filter(|&&s| s != slot) {
            let acquired = memory_ctx()
                .ring()
                .try_write(block_slot)
                .filter(|_| self.reverse_mut(block_slot).as_ref() == Some(key));
            match acquired {
                Some(buffer) => buffers.push(buffer),
                None => {
                    for buffer in buffers {
                        buffer.release_in_place();
                    }
                    return None;
                }
            }
        }

        // Hold all the block's slots exclusively: drop the block and recycle them.
        // Each slot's reverse entry is cleared under its own `WRITING` hold.
        let dying = (key.location.clone(), key.offset, key.len);
        self.remove_block(key);
        for &s in &block_slots {
            *self.reverse_mut(s) = None;
            memory_ctx().clock().release(s);
        }

        // The compressed source of these bytes is now the last in-memory copy;
        // reinforce it so both copies rarely die close together (which is what
        // turns the next read into disk IO). MUST stay after this cache's map
        // locks are released: the compressed evictor reinforces our blocks while
        // holding its own maps, which is only cycle-free because we never hold
        // ours when walking its.
        let (location, offset, compressed_len) = dying;
        memory_ctx()
            .compressed_cache()
            .reinforce_range(&location, offset, compressed_len);

        // Hand `slot` back; the rest return to the pool when their buffers drop.
        let pos = buffers
            .iter()
            .position(|b| b.slot_idx == slot)
            .expect("the victim was pushed into the buffer set first");
        Some(buffers.swap_remove(pos))
    }

    /// Remove the block at `key`, then prune the file's table if that empties it.
    fn remove_block(&self, key: &BlockKey) {
        let mut emptied_file = false;
        if let Some(table) = self.files.read().unwrap().get(&key.location) {
            let mut table = table.write().unwrap();
            table.remove(&key.offset);
            emptied_file = table.is_empty();
        }
        if emptied_file {
            self.prune_empty_file(&key.location);
        }
    }

    /// Remove `location`'s table if it is now empty, so its never-reused
    /// `FileLocation` (an `Arc<File>`/`Arc<RemoteFile>` pinning the file/connection
    /// it holds) isn't kept alive for every file ever opened. Re-checks emptiness
    /// under the outer write lock so a concurrent `claim` that just re-created the
    /// table isn't dropped.
    fn prune_empty_file(&self, location: &FileLocation) {
        let mut files = self.files.write().unwrap();
        if let Some(table) = files.get(location)
            && table.read().unwrap().is_empty()
        {
            files.remove(location);
        }
    }

    /// Recycle a single orphan slot already held `WRITING` (`buffer`): its block was
    /// dropped from the forward map by [`invalidate`](Self::invalidate)/
    /// [`clear`](Self::clear), leaving the slot bound `Decompressed`. Clears its
    /// reverse entry (sound: held `WRITING`) and releases it from the clock.
    fn recycle_orphan(&self, slot: usize, buffer: WriteBuffer) -> WriteBuffer {
        *self.reverse_mut(slot) = None;
        memory_ctx().clock().release(slot);
        buffer
    }

    /// Whether the cache currently owns no ring slots (mapped or orphaned by a
    /// pending invalidate/clear - a slot counts until the clock reclaims it).
    /// `get_write_buffer` checks this to decide whether eviction would have to
    /// fall through to the compressed cache.
    pub fn is_empty(&self) -> bool {
        memory_ctx().clock().owned(Owner::Decompressed) == 0
    }

    /// Drop every block belonging to `location` (e.g. when its fd is reopened) so a
    /// stale block can't serve a later read. The slots stay `Decompressed` and are
    /// recycled lazily by the clock (via [`recycle_orphan`](Self::recycle_orphan)).
    pub fn invalidate(&self, location: &FileLocation) {
        if !self.enabled {
            return;
        }
        // Drop the file's forward entries only. The slots stay bound `Decompressed`
        // with their reverse entries set; the clock recycles each as an orphan (its
        // forward entry now gone) under `try_write` - the only point a slot's reverse
        // entry is sound to clear.
        self.files.write().unwrap().remove(location);
    }

    /// Drop every cached block, returning how many were dropped. Frees each free
    /// slot to the pool immediately; a slot still pinned by a reader stays
    /// `Decompressed` and is recycled lazily by the clock. Backs `drop_cache()`.
    pub fn clear(&self) -> usize {
        let files = std::mem::take(&mut *self.files.write().unwrap());
        let mut dropped = 0;
        for (location, table) in files {
            for (offset, block) in table.into_inner().unwrap() {
                dropped += 1;
                let key = BlockKey {
                    location: location.clone(),
                    offset,
                    len: block.compressed_len,
                };
                for &slot in &block.slots {
                    // Recycle only slots we can take exclusively AND that still
                    // belong to this block: a reader-pinned slot stays an orphan
                    // for the clock to recycle later, and a slot already recycled
                    // and re-homed to a new owner in the window since the drain
                    // must not be ripped away from it.
                    let acquired = memory_ctx().ring().try_write(slot).filter(|_| {
                        memory_ctx().clock().owner(slot) == Some(Owner::Decompressed)
                            && self.reverse_mut(slot).as_ref() == Some(&key)
                    });
                    if let Some(buffer) = acquired {
                        *self.reverse_mut(slot) = None;
                        memory_ctx().clock().release(slot);
                        drop(buffer); // returns the slot to the pool
                    }
                }
            }
        }
        dropped
    }
}

/// Build `WriteBuffer`-backed (uncached, pooled-on-drop) views from the buffers.
fn build_transient_views(write_buffers: Vec<WriteBuffer>, decompressed_len: usize) -> Vec<Bytes> {
    write_buffers
        .into_iter()
        .enumerate()
        .map(|(i, buffer)| Bytes::from_owner(buffer).slice(..slot_len(decompressed_len, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::context::{init_test_free_pool, memory_ctx};
    use std::sync::{Arc, OnceLock};

    /// A fresh write buffer over one ring slot (the shape the decompressor feeds
    /// `insert`).
    fn ring_write() -> WriteBuffer {
        memory_ctx().get_write_buffer(false)
    }

    /// A fresh, independent local location (distinct cache key bucket).
    fn new_file() -> FileLocation {
        FileLocation::Local(Arc::new(std::fs::File::open("/dev/null").unwrap()))
    }

    fn key_for(location: &FileLocation) -> BlockKey {
        BlockKey {
            location: location.clone(),
            offset: 0,
            len: 1,
        }
    }

    /// One shared local location, reused across tests that key by `offset` alone.
    #[allow(non_snake_case)]
    fn FD() -> FileLocation {
        static FILE: OnceLock<Arc<std::fs::File>> = OnceLock::new();
        FileLocation::Local(
            FILE.get_or_init(|| Arc::new(std::fs::File::open("/dev/null").unwrap()))
                .clone(),
        )
    }

    /// Keys over `FD()`, distinguished by `offset`.
    fn test_key(offset: usize) -> BlockKey {
        BlockKey {
            location: FD(),
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
            reverse: (0..128).map(|_| UnsafeCell::new(None)).collect(),
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

    #[test]
    fn reclaiming_a_stale_orphan_spares_the_reinserted_block() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let first = ring_write();
        let orphan_slot = first.slot_idx;
        drop(cache.insert(test_key(0), make_header(0), vec![first], 1, counter()));
        cache.invalidate(&FD()); // the block is gone; its slot is now an orphan
        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        ));

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
    fn get_misses_for_an_absent_key() {
        let cache = for_test(true);

        assert!(cache.get(&test_key(0)).is_none());
    }

    #[test]
    fn a_hit_is_ring_backed_and_copy_free() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let buffer = ring_write();
        let ptr = buffer.as_slice().as_ptr();
        cache.insert(test_key(0), make_header(0), vec![buffer], 1, counter());

        let hit = cache.get(&test_key(0)).unwrap();

        assert_eq!(hit[0].as_ptr(), ptr);
    }

    #[test]
    fn a_different_length_at_the_same_offset_misses() {
        init_test_free_pool(2);
        let cache = for_test(true);
        cache.insert(
            BlockKey {
                len: 100,
                ..test_key(0)
            },
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        );

        let other = BlockKey {
            len: 200,
            ..test_key(0)
        };

        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn insert_is_a_noop_when_the_offset_is_already_cached() {
        init_test_free_pool(2);
        let cache = for_test(true);
        cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        );
        let dup = ring_write();
        let dup_slot = dup.slot_idx;

        // The dup is returned transient (WriteBuffer-backed): dropping it pools the
        // slot, which the next allocation hands back.
        drop(cache.insert(test_key(0), make_header(0), vec![dup], 1, counter()));

        assert_eq!(memory_ctx().get_write_buffer(false).slot_idx, dup_slot);
    }

    #[test]
    fn reclaim_evicts_the_owning_block_and_returns_the_slot() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        drop(cache.insert(test_key(0), make_header(0), vec![buffer], 1, counter())); // drop the view → resident

        let reclaimed = cache.reclaim(slot);

        assert_eq!(
            reclaimed.unwrap().slot_idx,
            slot,
            "the freed slot is handed back"
        );
        assert!(cache.is_empty(), "its block was evicted");
    }

    #[test]
    fn a_multi_slot_block_is_evicted_atomically_from_any_slot() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let a = ring_write();
        let b = ring_write();
        let hi = a.slot_idx.max(b.slot_idx);
        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![a, b],
            BUFFER_SIZE + 1,
            counter(),
        )); // spans 2 slots

        let reclaimed = cache.reclaim(hi);

        assert_eq!(
            reclaimed.unwrap().slot_idx,
            hi,
            "the victim slot is handed back"
        );
        assert!(
            cache.is_empty(),
            "the whole block, both slots, was reclaimed"
        );
    }

    #[test]
    fn evicting_a_files_last_block_releases_its_file_handle() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let location = new_file();
        let FileLocation::Local(file) = &location else {
            unreachable!()
        };
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        drop(cache.insert(
            key_for(&location),
            make_header(0),
            vec![buffer],
            1,
            counter(),
        ));

        cache.reclaim(slot);

        assert_eq!(
            Arc::strong_count(file),
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
    fn invalidate_drops_only_the_matching_location() {
        init_test_free_pool(4);
        let cache = for_test(true);
        let a = new_file();
        let b = new_file();
        cache.insert(
            key_for(&a),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        );
        cache.insert(
            key_for(&b),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        );

        cache.invalidate(&a);

        assert!(cache.get(&key_for(&a)).is_none());
        assert!(cache.get(&key_for(&b)).is_some());
    }

    #[test]
    fn clear_drops_everything() {
        init_test_free_pool(4);
        let cache = for_test(true);
        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        ));
        drop(cache.insert(
            test_key(1),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        ));

        let dropped = cache.clear();

        assert_eq!(dropped, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        init_test_free_pool(1);
        let cache = for_test(false);

        // insert returns the bytes transient; nothing is cached.
        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            counter(),
        ));

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
        cache.insert(
            test_key(0),
            make_header(9),
            vec![ring_write()],
            1,
            counter(),
        );

        let segments = cache.get_range(&FD(), 0, 10);

        assert_eq!(shapes(&segments), vec![Shape::Gap(0, 10)]);
    }

    #[test]
    fn get_range_finds_a_hit_surrounded_by_gaps() {
        init_test_free_pool(1);
        let cache = for_test(true);
        // A page at file offset 10, span 5 (so it occupies [10, 15)).
        let key = BlockKey {
            len: 5,
            ..test_key(10)
        };
        cache.insert(key, make_header(7), vec![ring_write()], 1, counter());

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
        cache.insert(
            BlockKey {
                len: 5,
                ..test_key(0)
            },
            make_header(1),
            vec![ring_write()],
            1,
            counter(),
        );
        cache.insert(
            BlockKey {
                len: 5,
                ..test_key(5)
            },
            make_header(2),
            vec![ring_write()],
            1,
            counter(),
        );

        let segments = cache.get_range(&FD(), 0, 10);

        assert_eq!(
            shapes(&segments),
            vec![Shape::Cached(0, 5, 1), Shape::Cached(5, 5, 2)]
        );
    }

    #[test]
    fn get_range_folds_a_mid_reclaim_block_into_its_gap() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        let key = BlockKey {
            len: 5,
            ..test_key(10)
        };
        drop(cache.insert(key, make_header(7), vec![buffer], 1, counter()));
        // Hold the slot exclusively, simulating a concurrent reclaim in flight.
        let _held = memory_ctx().ring().try_write(slot).unwrap();

        let segments = cache.get_range(&FD(), 0, 20);

        assert_eq!(shapes(&segments), vec![Shape::Gap(0, 20)]);
    }

    // -- live-block counting --

    #[test]
    fn an_inserted_block_counts_as_live() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let live = counter();

        cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        );

        assert_eq!(live.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reclaiming_a_block_uncounts_it() {
        init_test_free_pool(1);
        let cache = for_test(true);
        let live = counter();
        let buffer = ring_write();
        let slot = buffer.slot_idx;
        drop(cache.insert(test_key(0), make_header(0), vec![buffer], 1, live.clone()));

        cache.reclaim(slot);

        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_duplicate_insert_is_not_counted() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let live = counter();
        cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        );

        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        ));

        assert_eq!(live.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn invalidate_uncounts_the_dropped_blocks() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let live = counter();
        cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        );
        cache.insert(
            test_key(5),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        );

        cache.invalidate(&FD());

        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn clear_uncounts_everything() {
        init_test_free_pool(2);
        let cache = for_test(true);
        let live = counter();
        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        ));
        drop(cache.insert(
            test_key(5),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        ));

        cache.clear();

        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_disabled_cache_never_counts() {
        init_test_free_pool(1);
        let cache = for_test(false);
        let live = counter();

        drop(cache.insert(
            test_key(0),
            make_header(0),
            vec![ring_write()],
            1,
            live.clone(),
        ));

        assert_eq!(live.load(Ordering::Relaxed), 0);
    }
}
