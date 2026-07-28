//! Open-addressing hash table with linear probing, optimized for GROUP BY.
//!
//! See [`BaseHashTable`] for the full design rationale.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use tracing::debug;

/// Maximum occupied fraction before the caller grows or replaces the table.
pub const MAX_LOAD_FACTOR: f64 = 0.7;

/// A candidate key that has not necessarily been copied into durable storage.
///
/// Probing compares this form directly with table keys. [`persist`](Self::persist)
/// is called only after an empty slot is found, avoiding an allocation or string
/// copy when the group already exists.
pub trait LiveKey {
    type Persisted: PersistedKey;

    /// Compare this live key against a persisted key in the table.
    fn eq_persisted(&self, other: &Self::Persisted) -> bool;

    /// Convert the candidate into the representation stored by the table.
    fn persist(self) -> Self::Persisted;
}

/// A key representation that can live in raw hash-table storage.
///
/// # Safety
///
/// An all-zero byte region with the size and alignment of `Self` must be a valid
/// value. Empty table slots are zero-filled and may be viewed as keys before
/// their hash is checked. The `Copy` bound ensures keys have no destructor and
/// can be relocated byte-for-byte when a table grows. Any out-of-line data
/// referenced by a key must remain valid for at least as long as the table.
pub unsafe trait PersistedKey: Copy + Clone + Default + Send + Sync + 'static {
    /// Whether this key holds out-of-line data (an arena string blob) that
    /// `eq_persisted` will chase. `true` for string-bearing keys, `false` for
    /// fixed-width integer keys. Used to decide whether the merge's blob prefetch
    /// is worth its per-row cost (it is only when there's actually a cold blob).
    const HAS_BLOB: bool = false;

    /// Prefetch the arena blob this key's `eq_persisted` will read, hiding the
    /// scattered cache miss of a non-inline string before the `memcmp`. No-op for
    /// keys with no out-of-line data (integers, inline strings).
    #[inline(always)]
    fn prefetch_blob(&self, _arena: &SharedArena) {}
}

/// A persisted key can be inserted into another table without conversion.
impl<P: PersistedKey + PartialEq> LiveKey for P {
    type Persisted = Self;

    #[inline(always)]
    fn eq_persisted(&self, other: &Self) -> bool {
        self == other
    }

    fn persist(self) -> Self::Persisted {
        self
    }
}

/// Borrowed fields of one table entry.
///
/// This is a view rather than a reference to a Rust struct because entry layout
/// and stride are computed at query construction time.
pub struct EntryView<'a, K, S: ?Sized> {
    /// The stored hash (never 0 for an occupied entry).
    pub hash: u64,
    /// The persisted group key.
    pub key: &'a K,
    /// The group's aggregation state.
    pub state: &'a S,
}

/// Maximum number of entries allowed for a table with `len` total slots.
fn max_load_for_len(len: usize) -> usize {
    (len as f64 * MAX_LOAD_FACTOR).round() as usize
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Compute a reciprocal that replaces division by `d` with a widening multiply:
/// `n / d == (n * reciprocal(d)) >> 64`.
///
/// With `m = ceil(2^64 / d)`, this formula is exact while `n < 2^64 / d`.
/// A slab contains fewer than `2^21` entries, so every practical table index is
/// well inside that range.
fn reciprocal(d: u64) -> u64 {
    assert!(d >= 2, "an entry never fills half a slab");
    ((1u128 << 64).div_ceil(d as u128)) as u64
}

/// `n / d` via the precomputed [`reciprocal`] `m`: one widening multiply, no
/// hardware divide.
#[inline(always)]
fn fast_div(n: usize, m: u64) -> usize {
    (((n as u128) * (m as u128)) >> 64) as usize
}

/// Pre-adjust each slab base so a global slot index maps to
/// `adjusted_bases[slab] + index * stride`.
///
/// The stored integer may wrap below the mapped address. It is never
/// dereferenced until the matching global byte offset has been added back.
fn adjusted_bases(slabs: &[Slab], entries_per_slab: usize, stride: usize) -> Vec<usize> {
    slabs
        .iter()
        .enumerate()
        .map(|(s, slab)| (slab.ptr as usize).wrapping_sub(s * entries_per_slab * stride))
        .collect()
}

/// Runtime offsets, stride, and alignment for one table or scatter row.
///
/// Fields are placed in descending alignment order. Equal alignments retain the
/// logical hash, key, state order. This minimizes padding and preserves the
/// previous layout for fixed-size aggregation states.
pub(super) struct EntryLayout {
    pub(super) hash_offset: usize,
    pub(super) key_offset: usize,
    pub(super) state_offset: usize,
    pub(super) stride: usize,
    pub(super) align: usize,
}

/// The exact bytes one table entry occupies for this key type and signature,
/// for entry-footprint heuristics (merge-partition sizing).
pub fn entry_stride<K, V: AggregationValue>(ctx: &V::Context) -> usize {
    entry_layout::<K, V>(V::entry_state_meta(ctx)).stride
}

pub(super) fn entry_layout<K, V: AggregationValue>(state_meta: V::EntryStateMeta) -> EntryLayout {
    // Each tuple is (alignment, size), in logical hash, key, state order.
    let fields = [
        (align_of::<u64>(), size_of::<u64>()),
        (align_of::<K>(), size_of::<K>()),
        (
            V::entry_state_align(state_meta),
            V::entry_state_size(state_meta),
        ),
    ];
    let mut order = [0usize, 1, 2];
    order.sort_by_key(|&f| std::cmp::Reverse(fields[f].0));
    let mut offsets = [0usize; 3];
    let mut cursor = 0;
    for &f in &order {
        cursor = align_up(cursor, fields[f].0);
        offsets[f] = cursor;
        cursor += fields[f].1;
    }
    let align = fields.iter().map(|&(a, _)| a).max().unwrap();
    let stride = align_up(cursor, align);
    assert!(stride <= BUFFER_SIZE, "entry stride exceeds one slab");
    EntryLayout {
        hash_offset: offsets[0],
        key_offset: offsets[1],
        state_offset: offsets[2],
        stride,
        align,
    }
}

/// Prefetch the cache line at `ptr` into L1 (x86 `T0` / ARM `pldl1keep`).
#[inline(always)]
fn prefetch_l1_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // `prfm` is a hint with no architecturally-observable memory effect, so
    // `nomem` is correct and lets the compiler schedule freely around it in the
    // hot probe loop; the lint's pointer-with-nomem heuristic is a false positive.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// Prefetch the cache line at `ptr` into L2 only (x86 `T1` / ARM `pldl2keep`).
#[inline(always)]
fn prefetch_l2_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// A linear-probing table specialized for grouped aggregation.
///
/// # Entry layout
///
/// Entries are raw byte regions in 2 MiB slabs. Each region contains a full
/// hash, a persisted key, and aggregation state. Their physical order depends
/// on alignment; [`EntryLayout`] records their offsets.
///
/// ```text
/// slab:  [ entry 0 ][ entry 1 ][ entry 2 ] ... [ unused tail ]
/// entry: hash at hash_offset, key at key_offset, state at state_offset
///        <---------------------- stride ---------------------->
/// ```
///
/// The stride is computed once per table because
/// [`AggregationValue::EntryState`] may contain a query-defined number of
/// cells. Entries never cross slab boundaries; any remaining bytes at the end
/// of a slab are unused.
///
/// # Probing and partitioning
///
/// The initial slot comes from the high hash bits. A collision advances to the
/// next slot until an empty hash or equal key is found. Zero is the empty-slot
/// sentinel, so ordinary callers remap a zero hash to one.
///
/// High-bit placement is important to merge scheduling. Radix partitions use a
/// prefix of the same high bits, so a partition corresponds to a contiguous
/// slot range in every power-of-two table. A merge job can scan only its range,
/// even when source tables have different capacities.
///
/// The full hash is stored with the entry. Resizing and merging therefore never
/// need to hash a persisted string key again.
pub struct BaseHashTable<K: PersistedKey, V: AggregationValue> {
    mask: usize,
    slot_shift: u32,
    collisions: usize,
    /// Left-shift applied to hash before computing the slot index.
    /// Set to `PARTITIONS.trailing_zeros()` for merge maps so that the partition
    /// bits (top N) are stripped and the next top bits drive slot placement.
    hash_left_shift: u32,
    /// Whole entries per 2MB slab (`BUFFER_SIZE / stride`); entries never
    /// straddle a slab boundary.
    entries_per_slab: usize,
    /// Reciprocal used by [`fast_div`] to avoid a hardware division while
    /// locating the slab for a slot.
    entries_per_slab_reciprocal: u64,
    /// Bytes per entry; see the layout docs above.
    stride: usize,
    hash_offset: usize,
    key_offset: usize,
    state_offset: usize,
    /// Strictest field alignment, for allocating replacement slabs on resize.
    align: usize,
    /// The value's runtime view metadata (slot count for a runtime-arity
    /// signature); held once here, never per entry.
    state_meta: V::EntryStateMeta,
    /// Slab bases adjusted for addressing with a global slot index; see
    /// [`adjusted_bases`].
    adjusted_bases: Vec<usize>,
    slabs: Vec<Slab>,
    len: usize,
    max_load: usize,
    _phantom: PhantomData<(K, V)>,
}

unsafe impl<K: PersistedKey, V: AggregationValue> Send for BaseHashTable<K, V> {}

impl<K: PersistedKey, V: AggregationValue> BaseHashTable<K, V> {
    /// Create a zero-filled table with a power-of-two slot capacity.
    ///
    /// The key type and aggregation context determine the entry layout.
    pub fn new(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        hash_left_shift: u32,
        ctx: &V::Context,
    ) -> Self {
        let state_meta = V::entry_state_meta(ctx);
        let layout = entry_layout::<K, V>(state_meta);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let slabs = allocator.create_strided_slabs(expected_capacity, layout.stride, layout.align);
        BaseHashTable {
            mask: expected_capacity - 1,
            len: 0,
            max_load: max_load_for_len(expected_capacity),
            hash_left_shift,
            slot_shift: u64::BITS - expected_capacity.trailing_zeros(),
            entries_per_slab,
            entries_per_slab_reciprocal: reciprocal(entries_per_slab as u64),
            stride: layout.stride,
            hash_offset: layout.hash_offset,
            key_offset: layout.key_offset,
            state_offset: layout.state_offset,
            align: layout.align,
            state_meta,
            adjusted_bases: adjusted_bases(&slabs, entries_per_slab, layout.stride),
            slabs,
            _phantom: PhantomData,
            collisions: 0,
        }
    }

    /// Empty the table in place (zero every slot and reset the counters), keeping
    /// its capacity and backing slabs. Used by the radix "abandon" path: once the
    /// table's aggregated entries have been drained into the scatter buffers, the
    /// same storage is reused for the next window instead of reallocating.
    pub fn clear(&mut self) {
        for slab in &mut self.slabs {
            slab.zero_out();
        }
        self.len = 0;
        self.collisions = 0;
    }

    /// Total number of slots (occupied + empty). Always a power of 2.
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Returns the number of entries currently stored in the table.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Cumulative number of probe-chain collisions since the last resize.
    pub fn collisions(&self) -> usize {
        self.collisions
    }

    /// Map a hash to a slot index using the top bits:
    /// `(hash << hash_left_shift) >> slot_shift`.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.hash_left_shift) >> self.slot_shift) as usize
    }

    /// Find a global slot in the supplied set of adjusted slab bases.
    #[inline(always)]
    fn entry_ptr_in(&self, adjusted_bases: &[usize], index: usize) -> *mut u8 {
        let slab_idx = fast_div(index, self.entries_per_slab_reciprocal);
        adjusted_bases[slab_idx].wrapping_add(index * self.stride) as *mut u8
    }

    /// The address of this table's entry at `index`.
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        self.entry_ptr_in(&self.adjusted_bases, index)
    }

    /// Returns an iterator over all non-empty entries in the table.
    ///
    /// Iteration order is arbitrary (based on slot positions, not insertion order).
    /// Empty slots (hash == 0) are skipped automatically.
    pub fn iter(&self, start_offset: usize) -> HashTableIterator<'_, K, V> {
        HashTableIterator {
            layout: self.probe_layout(),
            idx: start_offset,
            _table: PhantomData,
        }
    }

    /// Returns `true` if the table has exceeded its [`MAX_LOAD_FACTOR`] threshold.
    pub fn needs_growth(&self) -> bool {
        self.len() > self.max_load
    }

    /// A register-resident snapshot of this table's entry layout and probe
    /// parameters (see [`ProbeLayout`]).
    #[inline(always)]
    fn probe_layout(&self) -> ProbeLayout<K, V> {
        ProbeLayout {
            entries_per_slab_reciprocal: self.entries_per_slab_reciprocal,
            stride: self.stride,
            hash_offset: self.hash_offset,
            key_offset: self.key_offset,
            state_offset: self.state_offset,
            mask: self.mask,
            slot_shift: self.slot_shift,
            hash_left_shift: self.hash_left_shift,
            adjusted_bases: self.adjusted_bases.as_ptr(),
            adjusted_bases_len: self.adjusted_bases.len(),
            first_adjusted_base: self.adjusted_bases[0],
            state_meta: self.state_meta,
            _phantom: PhantomData,
        }
    }

    /// A probing handle that holds the table's [`ProbeLayout`] in locals for the
    /// duration of a consume window, so the per-row probe reads its layout
    /// constants from registers (see [`Prober`]).
    #[inline(always)]
    pub fn prober(&mut self) -> Prober<'_, K, V> {
        let layout = self.probe_layout();
        Prober {
            table: self,
            layout,
        }
    }

    /// A read handle with the same register-resident [`ProbeLayout`], for the
    /// merge phase's linear slot scans.
    #[inline(always)]
    pub fn reader(&self) -> TableReader<'_, K, V> {
        TableReader {
            layout: self.probe_layout(),
            _table: PhantomData,
        }
    }

    /// Rehash all entries into fresh zeroed slabs of `new_size` slots.
    ///
    /// Every occupied entry is re-probed into the larger table and copied whole
    /// (`stride` bytes: hash, key, and aggregation state).
    ///
    /// The collision counter is reset to `self.len` so that post-resize
    /// collision tracking reflects only new probing pressure, not historical
    /// collisions from the smaller table.
    ///
    /// # Panics
    ///
    /// The underlying allocation may panic if memory allocation fails.
    #[cold]
    pub fn resize(&mut self, allocator: &mut SlabAllocator, new_size: usize) {
        debug!("Resizing map to {:?}...", new_size);
        let new_slabs = allocator.create_strided_slabs(new_size, self.stride, self.align);
        // Keep the replaced slabs alive until the copy loop below has read
        // every old entry; `old_adjusted_bases` points into them.
        let _old_slabs = mem::replace(&mut self.slabs, new_slabs);
        let old_adjusted_bases = mem::replace(
            &mut self.adjusted_bases,
            adjusted_bases(&self.slabs, self.entries_per_slab, self.stride),
        );
        let old_mask = self.mask;
        self.collisions = self.len;
        self.mask = new_size - 1;
        self.slot_shift = u64::BITS - new_size.trailing_zeros();

        for idx in 0..=old_mask {
            let old_entry = self.entry_ptr_in(&old_adjusted_bases, idx);
            let hash = unsafe { *(old_entry.add(self.hash_offset) as *const u64) };
            if hash != 0 {
                let mut new_idx = self.slot_for(hash);
                loop {
                    let new_entry = self.entry_ptr(new_idx);
                    if unsafe { *(new_entry.add(self.hash_offset) as *const u64) } == 0 {
                        unsafe { ptr::copy_nonoverlapping(old_entry, new_entry, self.stride) };
                        break;
                    }
                    new_idx = (new_idx + 1) & self.mask;
                }
            }
        }

        self.max_load = max_load_for_len(new_size);
    }
}

/// A copyable snapshot of the layout and values needed by a probe loop.
///
/// Raw entry writes prevent the compiler from proving that fields read through
/// `&BaseHashTable` do not alias those writes. Copying these values before the
/// loop lets them remain ordinary locals. `adjusted_bases` stays valid because
/// every owner of a snapshot also borrows the table, preventing resize.
struct ProbeLayout<K, V: AggregationValue> {
    entries_per_slab_reciprocal: u64,
    stride: usize,
    hash_offset: usize,
    key_offset: usize,
    state_offset: usize,
    mask: usize,
    slot_shift: u32,
    hash_left_shift: u32,
    adjusted_bases: *const usize,
    adjusted_bases_len: usize,
    /// `adjusted_bases[0]`, kept inline so a single-slab table's entry address needs
    /// neither the reciprocal multiply nor the dependent base load.
    first_adjusted_base: usize,
    state_meta: V::EntryStateMeta,
    _phantom: PhantomData<K>,
}

impl<K, V: AggregationValue> Clone for ProbeLayout<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V: AggregationValue> Copy for ProbeLayout<K, V> {}

impl<K, V: AggregationValue> ProbeLayout<K, V> {
    /// Map a hash to a slot index using the top bits.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.hash_left_shift) >> self.slot_shift) as usize
    }

    /// The address of the entry at `index`; see
    /// [`BaseHashTable::entry_ptr_in`].
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        // Nearly every in-place consume table and merge target fits one slab
        // (a slab holds ~1-2MB of entries and the radix switch caps in-place
        // growth), so take the entry address straight off the slab base: no
        // reciprocal multiply and no dependent base load on the address chain.
        // The branch is fixed per table, so it predicts perfectly either way.
        if self.adjusted_bases_len == 1 {
            return self.first_adjusted_base.wrapping_add(index * self.stride) as *mut u8;
        }
        let slab_idx = fast_div(index, self.entries_per_slab_reciprocal);
        assert!(slab_idx < self.adjusted_bases_len);
        let base = unsafe { *self.adjusted_bases.add(slab_idx) };
        base.wrapping_add(index * self.stride) as *mut u8
    }

    /// The view of the occupied entry at `entry`'s address, with `hash`
    /// already read.
    #[inline(always)]
    unsafe fn view<'a>(&self, entry: *const u8, hash: u64) -> EntryView<'a, K, V::EntryState> {
        unsafe {
            EntryView {
                hash,
                key: &*(entry.add(self.key_offset) as *const K),
                state: V::entry_state_ref(entry.add(self.state_offset), self.state_meta),
            }
        }
    }
}

/// A probing handle over a mutably borrowed table, carrying its [`ProbeLayout`]
/// in locals. The consume loop creates one per active table and probes through
/// it row after row, so the layout constants are read once per window, not
/// re-loaded per row and per probe step.
pub struct Prober<'t, K: PersistedKey, V: AggregationValue> {
    table: &'t mut BaseHashTable<K, V>,
    layout: ProbeLayout<K, V>,
}

impl<K: PersistedKey, V: AggregationValue> Prober<'_, K, V> {
    /// Prefetch the hash table slot where `hash` would land, plus the next
    /// cache lines to cover short probe chains. Brings the lines all the way
    /// into L1 (`T0`) — use this *near* the access (small lookahead).
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let ptr = self.layout.entry_ptr(self.layout.slot_for(hash)) as *const u8;
        prefetch_l1_line(ptr);
        prefetch_l1_line(ptr.wrapping_add(64));
        prefetch_l1_line(ptr.wrapping_add(128));
    }

    /// Prefetch the slot's cache line into L2 (`T1`) only. Issued *far* ahead
    /// of the access and paired with a nearer [`prefetch`](Self::prefetch)
    /// (L1) call, this software-pipelines the memory hierarchy: the line is
    /// pulled DRAM→L2 far ahead, then L2→L1 just before use, hiding the full
    /// DRAM latency that a single L1 prefetch at a short distance can't cover
    /// on a multi-GB table.
    #[inline(always)]
    pub fn prefetch_l2(&self, hash: u64) {
        prefetch_l2_line(self.layout.entry_ptr(self.layout.slot_for(hash)) as *const u8);
    }

    /// See [`BaseHashTable::needs_growth`].
    #[inline(always)]
    pub fn needs_growth(&self) -> bool {
        self.table.needs_growth()
    }

    /// Merge a state read from another table.
    ///
    /// A new group copies `src`; an existing group combines `src` with its
    /// current state.
    #[inline(always)]
    pub fn merge_from<const COUNT_COLLISIONS: bool, L>(
        &mut self,
        hash: u64,
        key: L,
        src: &V::EntryState,
        ctx: &V::Context,
    ) where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<COUNT_COLLISIONS, L, &V::EntryState, _, _>(
            hash,
            key,
            src,
            |src, dst| V::copy_entry(dst, src),
            |src, dst| V::merge_entries(dst, src, ctx),
        );
    }

    /// Grow the table 4x (updating `cap`) if it has crossed its load threshold,
    /// re-snapshotting the probe layout the resize invalidated. The merge's
    /// safety-net growth, kept on the prober so the per-row fold loop needn't
    /// give up its snapshot.
    #[inline(always)]
    pub fn grow_if_needed(&mut self, allocator: &mut SlabAllocator, cap: &mut usize) {
        if self.table.needs_growth() {
            *cap *= 4;
            self.table.resize(allocator, *cap);
            self.layout = self.table.probe_layout();
        }
    }

    /// Double the table if cumulative collision pressure is too high (see the
    /// merge's resize ratio), refreshing the probe layout on resize.
    /// `collision_ratio` is the integer threshold: resize once
    /// `collisions > collision_ratio * len`.
    #[inline(always)]
    pub fn resize_on_collisions(&mut self, allocator: &mut SlabAllocator, collision_ratio: usize) {
        if self.table.collisions() > self.table.len() * collision_ratio {
            let new_size = self.table.capacity() << 1;
            self.table.resize(allocator, new_size);
            self.layout = self.table.probe_layout();
        }
    }

    /// Probe for `hash` and `key`, then initialize or update that entry's state.
    ///
    /// A newly claimed slot calls `seed`; a matching slot calls `update`. Both
    /// closures receive the state reference produced during the probe, so no
    /// second lookup is needed.
    ///
    /// `ctx` is moved to the one closure that runs. Consume uses it for
    /// worker-local aggregation resources; merge uses it for the source state.
    ///
    /// Zero hashes are remapped to one because zero marks an empty slot. When
    /// `COUNT_COLLISIONS` is true, every unsuccessful probe step increments the
    /// table's collision counter.
    #[inline(always)]
    pub fn probe_fold<const COUNT_COLLISIONS: bool, L, X, S, U>(
        &mut self,
        mut hash: u64,
        key: L,
        ctx: X,
        seed: S,
        update: U,
    ) where
        L: LiveKey<Persisted = K>,
        S: FnOnce(X, &mut V::EntryState),
        U: FnOnce(X, &mut V::EntryState),
    {
        let layout = self.layout;
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1.
        if hash == 0 {
            hash = 1;
        }
        let mut idx = layout.slot_for(hash);
        loop {
            let entry = layout.entry_ptr(idx);
            unsafe {
                let hash_ptr = entry.add(layout.hash_offset) as *mut u64;
                if *hash_ptr == 0 {
                    *hash_ptr = hash;
                    (entry.add(layout.key_offset) as *mut K).write(key.persist());
                    self.table.len += 1;
                    seed(
                        ctx,
                        V::entry_state_mut(entry.add(layout.state_offset), layout.state_meta),
                    );
                    return;
                }
                if *hash_ptr == hash
                    && key.eq_persisted(&*(entry.add(layout.key_offset) as *const K))
                {
                    update(
                        ctx,
                        V::entry_state_mut(entry.add(layout.state_offset), layout.state_meta),
                    );
                    return;
                }
            }
            if COUNT_COLLISIONS {
                self.table.collisions += 1;
            }
            idx = (idx + 1) & layout.mask;
        }
    }
}

/// A read handle over a borrowed table, carrying its [`ProbeLayout`] in locals:
/// the merge phase's linear slot scans read thousands of consecutive slots, so
/// they too keep the layout in registers.
pub struct TableReader<'a, K: PersistedKey, V: AggregationValue> {
    layout: ProbeLayout<K, V>,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue> TableReader<'a, K, V> {
    /// See [`BaseHashTable::hash_at`].
    #[inline(always)]
    pub fn hash_at(&self, index: usize) -> u64 {
        unsafe { *(self.layout.entry_ptr(index).add(self.layout.hash_offset) as *const u64) }
    }

    /// See [`BaseHashTable::view_at`].
    #[inline(always)]
    pub fn view_at(&self, index: usize) -> EntryView<'a, K, V::EntryState> {
        let entry = self.layout.entry_ptr(index);
        unsafe {
            let hash = *(entry.add(self.layout.hash_offset) as *const u64);
            self.layout.view(entry, hash)
        }
    }
}

/// An iterator over the non-empty entries in a `BaseHashTable`.
///
/// Created by [`BaseHashTable::iter`]. Yields [`EntryView`]s of entries where
/// `hash != 0` in arbitrary order (based on slot positions, not insertion order).
pub struct HashTableIterator<'a, K: PersistedKey, V: AggregationValue> {
    layout: ProbeLayout<K, V>,
    idx: usize,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue> Iterator for HashTableIterator<'a, K, V> {
    type Item = EntryView<'a, K, V::EntryState>;

    fn next(&mut self) -> Option<Self::Item> {
        let layout = self.layout;
        while self.idx < layout.mask + 1 {
            // One address computation per slot: the hash check and the yielded
            // view read the same entry pointer.
            let entry = layout.entry_ptr(self.idx);
            self.idx += 1;
            unsafe {
                let hash = *(entry.add(layout.hash_offset) as *const u64);
                if hash != 0 {
                    return Some(layout.view(entry, hash));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::values::{
        AggregationColumnBuilders, AggregationSlot, ByValueAggregation,
    };
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::Field;

    /// A minimal additive value for exercising the probe mechanics: merging two
    /// of these sums their counts.
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    unsafe impl ByValueAggregation for Count {
        type Reader<'b> = ();
        type Context = ();
        type Columns = CountColumns;
        type SortKey = i64;
        type WorkerState = ();
        fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}
        fn value(_reader: &(), _idx: usize, _wc: &mut ()) -> Self {
            Count(1)
        }
        fn merge(self, other: Self, _ctx: &()) -> Self {
            Count(self.0 + other.0)
        }
        fn sort_key(&self, _slot: usize) -> i64 {
            self.0 as i64
        }
    }

    /// Empty output columns for the test [`Count`] value (the probe tests never
    /// materialise output).
    struct CountColumns;

    impl AggregationColumnBuilders for CountColumns {
        type Value = Count;
        type Context = ();
        fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _context: &()) -> Self {
            CountColumns
        }
        fn push_owned(&mut self, _value: &Count) {}
        fn push_entry(&mut self, _stored: &Count) {}
        fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
            (Vec::new(), Vec::new())
        }
    }

    type TestTable = BaseHashTable<u64, Count>;

    fn new_table(allocator: &mut SlabAllocator, capacity: usize) -> TestTable {
        BaseHashTable::new(allocator, capacity, 0, &())
    }

    #[test]
    fn reciprocal_division_matches_hardware_division() {
        // Representative entries-per-slab divisors: non-divisors of 2MB (24B
        // and 176B entries), exact powers of two (8B/16B entries), and odd ones.
        let divisors = [2u64, 3, 24, 176, 11915, 87381, 131072, 262144];

        for d in divisors {
            let m = reciprocal(d);
            for n in (0..1usize << 26).step_by(65_537).chain([
                0,
                d as usize - 1,
                d as usize,
                d as usize + 1,
                (1 << 40) - 1,
                1 << 40,
            ]) {
                assert_eq!(fast_div(n, m), n / d as usize, "n={n} d={d}");
            }
        }
    }

    #[test]
    fn insert_single_entry() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key, 100);
        assert_eq!(*entry.state, Count(1));
    }

    #[test]
    fn merge_duplicate_keys_sums_values() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.state, Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 2u64, &Count(1), &());

        assert_eq!(table.len(), 2);
        let keys: Vec<u64> = table.iter(0).map(|e| *e.key).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn hash_zero_is_remapped_and_retrievable() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(0, 99u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key, 99);
        assert_ne!(entry.hash, 0);
    }

    #[test]
    fn undersized_triggers_at_load_factor() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        let max_load = (16.0 * MAX_LOAD_FACTOR).round() as usize;

        for i in 0..max_load {
            table
                .prober()
                .merge_from::<false, _>(i as u64 + 1, i as u64, &Count(1), &());
            assert!(!table.needs_growth());
        }

        table
            .prober()
            .merge_from::<false, _>(max_load as u64 + 1, max_load as u64, &Count(1), &());

        assert!(table.needs_growth());
    }

    #[test]
    fn collision_counting() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<true, _>(42, 1u64, &Count(1), &());
        assert_eq!(table.collisions(), 0);

        table
            .prober()
            .merge_from::<true, _>(42, 2u64, &Count(1), &());
        assert_eq!(table.collisions(), 1);
    }

    #[test]
    fn collision_counting_disabled() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 2u64, &Count(1), &());

        assert_eq!(table.collisions(), 0);
    }

    #[test]
    fn iter_skips_empty_slots() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);

        table
            .prober()
            .merge_from::<false, _>(1, 10u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(2, 20u64, &Count(1), &());

        let entries: Vec<_> = table.iter(0).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.hash != 0));
    }

    #[test]
    fn resize_preserves_all_entries() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        for i in 0..8u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        table.resize(&mut allocator, 32);

        assert_eq!(table.len(), 8);
        assert_eq!(table.capacity(), 32);
        for i in 0..8u64 {
            let found = table.iter(0).any(|e| *e.key == i);
            assert!(found, "key {} missing after resize", i);
        }
    }

    #[test]
    fn resize_resets_collision_counter() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<true, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<true, _>(42, 2u64, &Count(1), &());
        let pre_resize_collisions = table.collisions();
        assert!(pre_resize_collisions > 0);

        table.resize(&mut allocator, 32);

        assert_eq!(table.collisions(), table.len());
    }

    #[test]
    fn many_entries_all_retrievable() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 256);

        for i in 0..100u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        assert_eq!(table.len(), 100);
        for i in 0..100u64 {
            let found = table.iter(0).any(|e| *e.key == i && e.state.0 == 1);
            assert!(found, "key {} missing or wrong value", i);
        }
    }

    #[test]
    fn wrap_around_probing() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        let last_slot_hash = u64::MAX;

        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 2u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 3u64, &Count(1), &());

        assert_eq!(table.len(), 3);
        let keys: Vec<u64> = table.iter(0).map(|e| *e.key).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn empty_table_iteration() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let table = new_table(&mut allocator, 16);

        let entries: Vec<_> = table.iter(0).collect();

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn merge_after_resize() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(99, 2u64, &Count(1), &());

        table.resize(&mut allocator, 32);
        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(200, 3u64, &Count(1), &());

        assert_eq!(table.len(), 3);
        let merged = table.iter(0).find(|e| *e.key == 1).unwrap();
        assert_eq!(*merged.state, Count(2));
        assert!(table.iter(0).any(|e| *e.key == 3));
    }

    #[test]
    fn view_at_returns_correct_slot() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        let occupied: Vec<usize> = (0..table.capacity())
            .filter(|&i| table.reader().hash_at(i) != 0)
            .collect();

        assert_eq!(occupied.len(), 1);
        assert_eq!(*table.reader().view_at(occupied[0]).key, 100);
    }

    #[test]
    fn iter_with_start_offset() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);
        for i in 0..20u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
