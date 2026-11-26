//! [`Cache<V>`]: entries of one derived form, packed into shared ring slots.
//!
//! ```text
//!   files: FileMap                       tenants: SlotTenants        ring
//!   file A ─┬─ 4096  → Entry{ extents:[(17,0,700k)],  meta }    slot 17: [A@4096, A@90000]
//!           └─ 90000 → Entry{ extents:[(17,700k,1.3M)], meta }
//!   file B ─── 0     → Entry{ extents:[(17,2M-97k,97k),(42,0,50k)], meta }   slot 42: [B@0]
//! ```
//!
//! An entry at rest holds no pin, only placements, so the clock can take any
//! slot. The map insert is what publishes an entry: the builder writes the
//! bytes, then `insert` takes the file's write lock and adds the entry, so the
//! lock is the happens-before between the writes and any reader; no validity
//! bitmap is needed.
//!
//! Eviction takes a whole slot and drops every entry with bytes in it. An
//! entry straddling into another slot dies whole; its bytes there become dead
//! space until that slot is itself reclaimed. Each view pins exactly the slots
//! it reads, so a view stays valid however long it is held.
//!
//! A reader that knows it will not ask for an entry again can drop it (see
//! [`Cache::drop_entry`]). The entry's slots then go back to the free pool
//! as soon as no entry is left in them and nothing reads them, instead of
//! lying at rest until the clock takes them.

use crate::env::get_env_var_with_default;
use crate::io::OpenFile;
use crate::memory::cache::cached::Cached;
use crate::memory::cache::file_map::FileMap;
use crate::memory::cache::fill::{CacheFill, CarvedSlab, SlotExtent};
use crate::memory::cache::id::CacheId;
use crate::memory::cache::ring_cache::{Reclaimed, RingCache, SourceRange};
use crate::memory::cache::slot_tenants::SlotTenants;
use crate::memory::context::memory_ctx;
use crate::memory::slab::{FrozenSlab, Slab, SlabBacking};
use crate::memory::write_buffer::ProbedSlot;
use crossbeam_queue::SegQueue;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Identity of a cache entry: the file bytes it was derived from. The map key
/// is `(file, offset)`; `span` is checked on lookup so an entry derived from a
/// differently sized region at the same offset (an earlier version of the
/// file) is never served.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub file: OpenFile,
    pub offset: usize,
    pub span: usize,
}

impl CacheKey {
    pub fn new(file: OpenFile, offset: usize, span: usize) -> Self {
        Self { file, offset, span }
    }
}

/// One segment of a range lookup, in file order.
pub enum Segment<V> {
    /// An entry the cache holds over `[offset, offset + span)`, rebuilt and
    /// pinned for as long as it is held.
    Cached {
        offset: usize,
        span: usize,
        value: V,
    },
    /// A stretch of the range no entry covers.
    Gap { offset: usize, len: usize },
}

/// A cached value at rest: where its bytes are and what rebuilding it needs.
struct Entry<V: Cached> {
    /// The source span, from the key, for the lookup check and for walking a
    /// range in file order.
    span: usize,
    extents: Vec<SlotExtent>,
    meta: V::Meta,
}

impl<V: Cached> Entry<V> {
    /// Pin every extent's slot, touch the clock, and rebuild the value. `None`
    /// (no pins kept) if any slot is mid-reclaim.
    fn pin_and_rebuild(&self) -> Option<V> {
        let mut slabs = Vec::with_capacity(self.extents.len());
        for extent in &self.extents {
            let pin = std::sync::Arc::new(memory_ctx().ring().try_read(extent.slot as usize)?);
            memory_ctx().clock().touch(extent.slot as usize);
            let ptr = unsafe { (pin.ptr as *mut u8).add(extent.offset as usize) };
            slabs.push(Slab::new(ptr, extent.len as usize, SlabBacking::Cached(pin)).freeze());
        }
        Some(V::from_slabs(&self.meta, slabs))
    }

    fn lives_in(&self, slot: usize) -> bool {
        self.extents
            .iter()
            .any(|extent| extent.slot as usize == slot)
    }
}

/// What a look at a slot that held a dropped entry found.
#[derive(PartialEq, Eq)]
pub(crate) enum SlotProbe {
    /// Something reads the slot: it is to be looked at again.
    StillRead,
    /// The slot was given back, still holds an entry, or is no longer
    /// the cache's: there is nothing more to do about it.
    Settled,
}

/// One file's entries, by source offset.
type EntryMap<V> = BTreeMap<usize, Entry<V>>;

pub struct Cache<V: Cached> {
    id: CacheId,
    files: FileMap<EntryMap<V>>,
    tenants: SlotTenants<CacheKey>,
    /// Entries across every file; the cold fast path of `get_range`.
    entries: AtomicUsize,
    /// For each slot, how many entries are counted as lying in it: one more
    /// with each entry published, one fewer with each removed. It only
    /// says which slots a drop on request may have emptied. Whether a slot
    /// is empty is decided under its exclusive hold, from its tenants.
    entries_per_slot: Box<[AtomicU32]>,
    /// The slots a drop on request left with no entry counted in them,
    /// which were still read when last looked at. Each is looked at again
    /// with the next drop.
    slots_to_recycle: SegQueue<usize>,
    /// When off, `insert` builds into working memory and `get` always misses.
    enabled: bool,
}

impl<V: Cached> Cache<V> {
    /// `slot_count` is the ring's size. `kill_switch` names the environment
    /// variable that disables the cache (default on).
    pub fn new(id: CacheId, slot_count: usize, kill_switch: &str) -> Self {
        Self::with_enabled(id, slot_count, get_env_var_with_default(kill_switch, true))
    }

    pub fn with_enabled(id: CacheId, slot_count: usize, enabled: bool) -> Self {
        Self {
            id,
            files: FileMap::new(),
            tenants: SlotTenants::new(slot_count),
            entries: AtomicUsize::new(0),
            entries_per_slot: (0..slot_count).map(|_| AtomicU32::new(0)).collect(),
            slots_to_recycle: SegQueue::new(),
            enabled,
        }
    }

    /// Build the entry for `key` straight into the cache and return it. The
    /// builder carves slabs from the fill and writes them; the slabs it
    /// carved, in order, are what the value is rebuilt over. A builder error
    /// caches nothing: the carved bytes stay dead until their slots recycle.
    /// A key already cached by then (a duplicate-insert race) keeps the
    /// incumbent; the freshly built value is still returned, readable.
    pub fn insert<E>(
        &self,
        key: CacheKey,
        build: impl FnOnce(&mut CacheFill) -> Result<V::Meta, E>,
    ) -> Result<V, E> {
        if !self.enabled {
            let mut fill = CacheFill::transient();
            let meta = build(&mut fill)?;
            return Ok(V::from_slabs(
                &meta,
                Self::collect_slabs(fill.into_carved_slabs()),
            ));
        }
        let cursor = memory_ctx().fill_cursor(self.id);
        let mut fill = CacheFill::packed(self.id, cursor, &self.tenants, &key);
        let meta = build(&mut fill)?;
        let carved = fill.into_carved_slabs();
        let extents = carved
            .iter()
            .map(|carved| carved.extent.expect("a packed fill places every slab"))
            .collect();
        let value = V::from_slabs(&meta, Self::collect_slabs(carved));
        self.claim(key, extents, meta);
        Ok(value)
    }

    fn collect_slabs(carved: Vec<CarvedSlab>) -> Vec<FrozenSlab> {
        carved.into_iter().map(|carved| carved.slab).collect()
    }

    /// Publish the entry under the file's write lock. A key whose offset is
    /// already present loses: the incumbent stays.
    fn claim(&self, key: CacheKey, extents: Vec<SlotExtent>, meta: V::Meta) {
        let inserted = self.files.with_table_or_create(&key.file, |table| {
            if table.contains_key(&key.offset) {
                return false;
            }
            for extent in &extents {
                self.entries_per_slot[extent.slot as usize].fetch_add(1, Ordering::Relaxed);
            }
            table.insert(
                key.offset,
                Entry {
                    span: key.span,
                    extents,
                    meta,
                },
            );
            true
        });
        if inserted {
            self.entries.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The entry for `key`, pinned and rebuilt. Misses if absent, if the
    /// cached span differs from `key.span`, or if a slot is mid-reclaim.
    pub fn get(&self, key: &CacheKey) -> Option<V> {
        if !self.enabled {
            return None;
        }
        self.files
            .with_table(&key.file, |table| {
                table
                    .get(&key.offset)
                    .filter(|entry| entry.span == key.span)
                    .and_then(Entry::pin_and_rebuild)
            })
            .flatten()
    }

    /// Resolve `[offset, offset + len)` of `file` into segments in file order:
    /// each entry found (pinned) and each stretch between them. Entries are
    /// keyed by the spans the caller derived them from, so the gaps are whole
    /// units of whatever the caller caches (whole pages, for the page cache).
    pub fn get_range(&self, file: &OpenFile, offset: usize, len: usize) -> Vec<Segment<V>> {
        let mut segments = SegmentList::new(offset, len);
        if len == 0 {
            return segments.finish();
        }
        // Cold fast path: nothing is cached anywhere, so skip the locks.
        if !self.enabled || self.is_empty() {
            return segments.finish();
        }
        self.files.with_table(file, |table| {
            for (&entry_offset, entry) in table.range(offset..offset + len) {
                match entry.pin_and_rebuild() {
                    Some(value) => segments.push_cached(entry_offset, entry.span, value),
                    // Mid-reclaim: as good as absent.
                    None => segments.skip(entry_offset, entry.span),
                }
            }
        });
        segments.finish()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.load(Ordering::Relaxed) == 0
    }

    /// Count one entry fewer as lying in `slot`. Returns whether none is
    /// counted in it from here on.
    fn count_entry_removed_from(&self, slot: usize) -> bool {
        let count_one_fewer = |count: u32| count.checked_sub(1);
        let entries = &self.entries_per_slot[slot];
        entries.fetch_update(Ordering::Relaxed, Ordering::Relaxed, count_one_fewer) == Ok(1)
    }

    /// Count `entry` as lying in none of its slots any more.
    fn count_entry_removed(&self, entry: &Entry<V>) {
        for extent in &entry.extents {
            self.count_entry_removed_from(extent.slot as usize);
        }
    }

    /// Drop the entry for `key`, which its reader will not ask for again,
    /// and give the slots it lay in back to the free pool if they hold no
    /// other entry. Does nothing if the cache holds no such entry.
    ///
    /// A slot is only looked at once no entry is counted in it: looking
    /// takes it exclusively for a moment, which a reader of another entry
    /// in it would see as a miss. And it can only be given back while
    /// nothing reads it, where the reader that drops an entry often still
    /// holds its bytes. Such a slot is looked at again with every later
    /// drop (see [`recycle_emptied_slots`](Self::recycle_emptied_slots)).
    pub fn drop_entry(&self, key: &CacheKey) {
        if !self.enabled {
            return;
        }
        let removed = self.files.with_table_mut(&key.file, |table| {
            let is_cached = table
                .get(&key.offset)
                .is_some_and(|entry| entry.span == key.span);
            let entry = is_cached.then(|| table.remove(&key.offset)).flatten()?;
            Some((entry, table.is_empty()))
        });
        let Some(Some((entry, has_emptied_file))) = removed else {
            return;
        };
        self.entries.fetch_sub(1, Ordering::Relaxed);
        if has_emptied_file {
            self.files.prune_if_empty(&key.file, BTreeMap::is_empty);
        }
        for extent in &entry.extents {
            if self.count_entry_removed_from(extent.slot as usize) {
                self.slots_to_recycle.push(extent.slot as usize);
            }
        }
        self.recycle_emptied_slots();
    }

    /// Give back to the free pool every slot that a drop on request left
    /// with no entry counted in it, that holds none, and that nothing
    /// reads. A slot that is still read is kept to be looked at again.
    pub fn recycle_emptied_slots(&self) {
        for _ in 0..self.slots_to_recycle.len() {
            let Some(slot) = self.slots_to_recycle.pop() else {
                return;
            };
            if self.recycle_slot_if_emptied(slot) == SlotProbe::StillRead {
                self.slots_to_recycle.push(slot);
            }
        }
    }

    /// Take `slot` exclusively and, if it is this cache's and no entry
    /// lies in it, release it to the free pool.
    fn recycle_slot_if_emptied(&self, slot: usize) -> SlotProbe {
        // A slot the clock took since is working memory or another cache's
        // by now, and would never stop being held.
        if memory_ctx().clock().owner(slot) != Some(self.id) {
            return SlotProbe::Settled;
        }
        let Some(buffer) = memory_ctx().ring().try_write(slot) else {
            return SlotProbe::StillRead;
        };
        // Dropped without being taken, the probe leaves the slot cached.
        let probe = ProbedSlot::new(buffer);
        if memory_ctx().clock().owner(slot) != Some(self.id) {
            return SlotProbe::Settled;
        }
        let holds_entry = self.tenants.list(slot).iter().any(|key| {
            let lives_here = self.files.with_table(&key.file, |table| {
                table
                    .get(&key.offset)
                    .is_some_and(|entry| entry.lives_in(slot))
            });
            lives_here == Some(true)
        });
        if holds_entry {
            return SlotProbe::Settled;
        }
        self.tenants.clear(slot);
        self.entries_per_slot[slot].store(0, Ordering::Relaxed);
        memory_ctx().clock().release(slot);
        drop(probe.take_for_reuse()); // back to the free pool
        SlotProbe::Settled
    }

    /// Drop every entry of `slot` still placed there, returning their source
    /// ranges. Stale tenants (dropped another way, lost races, failed builds,
    /// re-inserted elsewhere) are filtered by the placement check. Sound only
    /// while `slot` is held exclusively.
    fn purge_slot(&self, slot: usize) -> Vec<SourceRange> {
        let mut dropped = Vec::new();
        let mut emptied_files = Vec::new();
        for key in self.tenants.take(slot) {
            let removed = self.files.with_table_mut(&key.file, |table| {
                let still_here = table
                    .get(&key.offset)
                    .is_some_and(|entry| entry.lives_in(slot));
                if !still_here {
                    return None;
                }
                let entry = table.remove(&key.offset).expect("checked present");
                Some((entry, table.is_empty()))
            });
            if let Some(Some((entry, emptied))) = removed {
                self.entries.fetch_sub(1, Ordering::Relaxed);
                self.count_entry_removed(&entry);
                dropped.push(SourceRange::new(key.file.clone(), key.offset, entry.span));
                if emptied {
                    emptied_files.push(key.file);
                }
            }
        }
        for file in emptied_files {
            self.files.prune_if_empty(&file, BTreeMap::is_empty);
        }
        self.entries_per_slot[slot].store(0, Ordering::Relaxed);
        dropped
    }

    /// Take `slot` exclusively if it is still ours, purge it, release it.
    fn take_slot(&self, slot: usize) -> Option<Reclaimed> {
        // The probe releases the hold in place if anything below unwinds: a
        // slot the cache still references must never reach the free pool.
        let probe = ProbedSlot::new(memory_ctx().ring().try_write(slot)?);
        if memory_ctx().clock().owner(slot) != Some(self.id) {
            return None;
        }
        let dropped = self.purge_slot(slot);
        memory_ctx().clock().release(slot);
        Some(Reclaimed {
            buffer: probe.take_for_reuse(),
            dropped,
        })
    }
}

impl<V: Cached> RingCache for Cache<V> {
    fn id(&self) -> CacheId {
        self.id
    }

    fn reclaim(&self, slot: usize) -> Option<Reclaimed> {
        self.take_slot(slot)
    }

    fn reinforce_range(&self, range: &SourceRange) {
        if !self.enabled || range.len == 0 {
            return;
        }
        self.files.with_table(&range.file, |table| {
            // An entry starting before the range may still reach into it.
            if let Some((&entry_offset, entry)) = table.range(..range.offset).next_back()
                && entry_offset + entry.span > range.offset
            {
                for extent in &entry.extents {
                    memory_ctx().clock().reinforce(extent.slot as usize);
                }
            }
            for (_, entry) in table.range(range.offset..range.end()) {
                for extent in &entry.extents {
                    memory_ctx().clock().reinforce(extent.slot as usize);
                }
            }
        });
    }

    fn invalidate(&self, file: &OpenFile) {
        if let Some(table) = self.files.remove(file) {
            self.entries.fetch_sub(table.len(), Ordering::Relaxed);
            for entry in table.values() {
                self.count_entry_removed(entry);
            }
        }
    }

    fn clear(&self) -> usize {
        // Release this worker's fill pin first so its own fill slot can be
        // recycled below rather than skipped as pinned.
        memory_ctx().fill_cursor(self.id).release();
        let mut dropped = 0;
        let mut slots = std::collections::HashSet::new();
        for (_, table) in self.files.drain() {
            for (_, entry) in table {
                dropped += 1;
                self.count_entry_removed(&entry);
                slots.extend(entry.extents.iter().map(|extent| extent.slot as usize));
            }
        }
        self.entries.fetch_sub(dropped, Ordering::Relaxed);
        // A slot a peer still pins (its fill cursor, a reader) stays bound and
        // recycles lazily; one re-packed by a peer since the drain keeps
        // whatever its current entries say, which the purge's placement check
        // honours.
        for slot in slots {
            if let Some(reclaimed) = self.take_slot(slot) {
                drop(reclaimed.buffer); // back to the free pool
            }
        }
        dropped
    }
}

/// Accumulates a range lookup's segments in file order, filling the stretches
/// between entries with gaps and merging adjacent gaps.
struct SegmentList<V> {
    segments: Vec<Segment<V>>,
    cursor: usize,
    end: usize,
}

impl<V> SegmentList<V> {
    fn new(offset: usize, len: usize) -> Self {
        Self {
            segments: Vec::new(),
            cursor: offset,
            end: offset + len,
        }
    }

    fn push_cached(&mut self, offset: usize, span: usize, value: V) {
        self.push_gap_up_to(offset);
        self.segments.push(Segment::Cached {
            offset,
            span,
            value,
        });
        self.cursor = offset + span;
    }

    /// An entry that could not be used: its span is a gap.
    fn skip(&mut self, offset: usize, span: usize) {
        self.push_gap_up_to(offset + span);
    }

    fn push_gap_up_to(&mut self, offset: usize) {
        let len = offset.saturating_sub(self.cursor);
        if len == 0 {
            return;
        }
        match self.segments.last_mut() {
            Some(Segment::Gap { len: previous, .. }) => *previous += len,
            _ => self.segments.push(Segment::Gap {
                offset: self.cursor,
                len,
            }),
        }
        self.cursor = offset;
    }

    fn finish(mut self) -> Vec<Segment<V>> {
        self.push_gap_up_to(self.end);
        self.segments
    }
}
