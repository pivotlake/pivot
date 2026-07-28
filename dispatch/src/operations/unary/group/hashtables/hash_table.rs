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

/// Maximum fraction of slots that can be occupied before the table is
/// considered undersized. Used by [`AggregatedTable`](super::AggregatedTable)
/// to decide when to stop inserting and create a new table.
pub const MAX_LOAD_FACTOR: f64 = 0.7;

/// A `LiveKey` is a key that has *not* yet been persisted, and therefore needs to explicitly be
/// persisted if it is saved within the HashTable. The HashTable may decide not to persist a key
/// if, for example, the same key already exists.
pub trait LiveKey {
    type Persisted: PersistedKey;

    /// Compare this live key against a persisted key in the table.
    fn eq_persisted(&self, other: &Self::Persisted) -> bool;

    /// Save the key
    fn persist(self) -> Self::Persisted;
}

/// A key that has been persisted in the HashTable. Any PersistedKey can be referenced from within
/// the HashTable as long as the HashTable is alive
pub trait PersistedKey: Copy + Clone + Default {
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

/// We reimplement LiveKey for any PersistentKey to allow saving an already persisted key within
/// the HashTable (for example, if it was saved in another HashTable's arena previously)
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

/// Borrowed view of an occupied table entry.
///
/// Entries have a runtime stride, so they cannot be represented by references
/// to one statically sized Rust struct.
pub struct EntryView<'a, K, S: ?Sized> {
    /// The stored hash (never 0 for an occupied entry).
    pub hash: u64,
    /// The persisted group key.
    pub key: &'a K,
    /// The group's stored aggregation value.
    pub stored: &'a S,
}

/// Maximum number of entries allowed for a table with `len` total slots.
fn max_load_for_len(len: usize) -> usize {
    (len as f64 * MAX_LOAD_FACTOR).round() as usize
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Computes a reciprocal used to replace division by `d` with a widening
/// multiply: `n / d == (n * reciprocal(d)) >> 64`.
///
/// Exact for this table's ranges: with `m = ceil(2^64 / d) = (2^64 + e) / d`
/// (`0 <= e < d`), `n * m / 2^64 = n/d + n*e/(d * 2^64)`, and the error term
/// stays below `n / 2^64`. The true quotient's fractional part is at most
/// `1 - 1/d`, so rounding can change the floor only if `n >= 2^64 / d`.
/// Here `d < 2^21`, making the result exact for every table slot index.
fn reciprocal(d: u64) -> u64 {
    assert!(d >= 2, "an entry never fills half a slab");
    ((1u128 << 64).div_ceil(d as u128)) as u64
}

/// Divides `n` using a reciprocal returned by [`reciprocal`].
#[inline(always)]
fn fast_div(n: usize, m: u64) -> usize {
    (((n as u128) * (m as u128)) >> 64) as usize
}

/// Computes slab bases adjusted by their first global entry offset.
///
/// With these bases, an entry address is `bases[slab] + index * stride`.
/// Adjusted addresses may numerically precede an allocation, but only the
/// reconstructed in-bounds address is dereferenced.
fn adjusted_bases(slabs: &[Slab], entries_per_slab: usize, stride: usize) -> Vec<usize> {
    slabs
        .iter()
        .enumerate()
        .map(|(slab_index, slab)| {
            (slab.ptr as usize).wrapping_sub(slab_index * entries_per_slab * stride)
        })
        .collect()
}

/// Runtime field offsets, stride, and alignment for one table entry.
pub(super) struct EntryLayout {
    pub(super) hash_offset: usize,
    pub(super) key_offset: usize,
    pub(super) value_offset: usize,
    pub(super) stride: usize,
    pub(super) align: usize,
}

/// Returns the byte stride for one entry of this key and value type.
pub fn entry_stride<K, V: AggregationValue + ?Sized>(ctx: &V::SharedContext) -> usize {
    entry_layout::<K, V>(V::storage_metadata(ctx)).stride
}

pub(super) fn entry_layout<K, V: AggregationValue + ?Sized>(
    metadata: V::StorageMetadata,
) -> EntryLayout {
    // Alignment and size for hash, key, and value.
    let fields = [
        (align_of::<u64>(), size_of::<u64>()),
        (align_of::<K>(), size_of::<K>()),
        (V::stored_align(), V::stored_size(metadata)),
    ];
    // A fixed comparison network lets this layout constant-fold for specialized
    // dynamic arities. Strict comparisons preserve field order on ties.
    let mut order = [0usize, 1, 2];
    if fields[order[1]].0 > fields[order[0]].0 {
        order.swap(0, 1);
    }
    if fields[order[2]].0 > fields[order[1]].0 {
        order.swap(1, 2);
    }
    if fields[order[1]].0 > fields[order[0]].0 {
        order.swap(0, 1);
    }
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
        value_offset: offsets[2],
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

/// Linear-probing hash table for GROUP BY.
///
/// # Memory Layout
///
/// Each table computes one entry layout from its key type and aggregation
/// signature. Entries never cross a 2 MB slab boundary.
///
/// ```text
/// slab
/// +----------------+----------------+---------+----------------+
/// | entry 0        | entry 1        |   ...   | unused tail    |
/// | hash | key | V | hash | key | V |         | < one stride   |
/// +----------------+----------------+---------+----------------+
/// ```
///
/// A [`Dynamic`] value makes the stride a runtime value because its cell count
/// is fixed per query rather than per Rust type. Fixed signatures use the same
/// layout algorithm and retain their compact representation.
///
/// The full hash is stored beside the key and value. Hash zero is the empty
/// sentinel; a real zero hash is remapped to one. Capacity is a power of two,
/// so linear probing wraps with a mask.
///
/// # Top-bit placement
///
/// Initial slots use the high bits of the hash. Merge partitions use the same
/// prefix, so each partition maps to a contiguous slot range even when source
/// tables have different capacities:
///
/// ```text
/// hash prefix P
///      |
///      +-- 128-slot table: slots [P * 2, P * 2 + 2)
///      +-- 256-slot table: slots [P * 4, P * 4 + 4)
/// ```
///
/// This lets one merge job scan only its ranges. Storing the full hash also
/// avoids hashing large keys again during resize and merge.
pub struct BaseHashTable<K: PersistedKey, V: AggregationValue + ?Sized> {
    mask: usize,
    shift: u32,
    collisions: usize,
    /// Left-shift applied to hash before computing the slot index.
    /// Set to `PARTITIONS.trailing_zeros()` for merge maps so that the partition
    /// bits (top N) are stripped and the next top bits drive slot placement.
    pre_shift: u32,
    /// Whole entries per 2MB slab (`BUFFER_SIZE / stride`); entries never
    /// straddle a slab boundary.
    entries_per_slab: usize,
    /// Reciprocal used by [`fast_div`] to find a slab without hardware division.
    entries_per_slab_magic: u64,
    /// Bytes per entry.
    stride: usize,
    hash_offset: usize,
    /// Strictest field alignment, for allocating replacement slabs on resize.
    align: usize,
    /// Value layout metadata stored once per table.
    metadata: V::StorageMetadata,
    /// Slab addresses adjusted by their first global entry offset.
    bases: Vec<usize>,
    slabs: Vec<Slab>,
    length: usize,
    max_load: usize,
    // A raw-pointer marker: a plain `(K, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(K, *const V)>,
}

unsafe impl<K: PersistedKey, V: AggregationValue + ?Sized> Send for BaseHashTable<K, V> {}

impl<K: PersistedKey, V: AggregationValue + ?Sized> BaseHashTable<K, V> {
    /// Creates a zeroed table with power-of-two capacity.
    pub fn new(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        pre_shift: u32,
        ctx: &V::SharedContext,
    ) -> Self {
        let metadata = V::storage_metadata(ctx);
        let layout = entry_layout::<K, V>(metadata);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let slabs = allocator.create_strided_slabs(expected_capacity, layout.stride, layout.align);
        BaseHashTable {
            mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            pre_shift,
            shift: u64::BITS - expected_capacity.trailing_zeros(),
            entries_per_slab,
            entries_per_slab_magic: reciprocal(entries_per_slab as u64),
            stride: layout.stride,
            hash_offset: layout.hash_offset,
            align: layout.align,
            metadata,
            bases: adjusted_bases(&slabs, entries_per_slab, layout.stride),
            slabs,
            _phantom: PhantomData,
            collisions: 0,
        }
    }

    /// Clears entries and counters while retaining the allocated slabs.
    pub fn clear(&mut self) {
        for slab in &mut self.slabs {
            slab.zero_out();
        }
        self.length = 0;
        self.collisions = 0;
    }

    /// Total number of slots (occupied + empty). Always a power of 2.
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Returns the number of entries currently stored in the table.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Cumulative number of probe-chain collisions since the last resize.
    pub fn collisions(&self) -> usize {
        self.collisions
    }

    /// Map a hash to a slot index using the top bits: `(hash << pre_shift) >> shift`.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.pre_shift) >> self.shift) as usize
    }

    /// Returns an entry address using the supplied adjusted slab bases.
    ///
    /// Entries never cross slab boundaries. [`fast_div`] identifies the slab,
    /// and the adjusted base allows the byte address to use the global index
    /// directly.
    #[inline(always)]
    fn entry_ptr_in(&self, bases: &[usize], index: usize) -> *mut u8 {
        let slab_idx = fast_div(index, self.entries_per_slab_magic);
        bases[slab_idx].wrapping_add(index * self.stride) as *mut u8
    }

    /// The address of this table's entry at `index`.
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        self.entry_ptr_in(&self.bases, index)
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
    pub fn undersized(&self) -> bool {
        self.len() > self.max_load
    }

    /// Copies entry layout and probe parameters into a [`ProbeLayout`].
    #[inline(always)]
    fn probe_layout(&self) -> ProbeLayout<K, V> {
        self.probe_layout_with_metadata(self.metadata)
    }

    /// Builds a probe layout from caller-provided value metadata.
    ///
    /// Arity-specialized loops pass constant metadata here so field offsets and
    /// stride arithmetic can be constant-folded. The metadata must describe
    /// the layout used to construct this table.
    #[inline(always)]
    fn probe_layout_with_metadata(&self, metadata: V::StorageMetadata) -> ProbeLayout<K, V> {
        let layout = entry_layout::<K, V>(metadata);
        debug_assert_eq!(layout.stride, self.stride);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        ProbeLayout {
            magic: reciprocal(entries_per_slab as u64),
            stride: layout.stride,
            hash_offset: layout.hash_offset,
            key_offset: layout.key_offset,
            value_offset: layout.value_offset,
            mask: self.mask,
            shift: self.shift,
            pre_shift: self.pre_shift,
            bases: self.bases.as_ptr(),
            bases_len: self.bases.len(),
            first_base: self.bases[0],
            metadata,
            _phantom: PhantomData,
        }
    }

    /// Creates a probing handle with a local layout snapshot.
    #[inline(always)]
    pub fn prober(&mut self) -> Prober<'_, K, V> {
        self.prober_with_metadata(self.metadata)
    }

    /// Creates a probing handle for an arity-specialized loop.
    #[inline(always)]
    pub fn prober_with_metadata(&mut self, metadata: V::StorageMetadata) -> Prober<'_, K, V> {
        let layout = self.probe_layout_with_metadata(metadata);
        Prober {
            table: self,
            layout,
        }
    }

    /// Creates a read handle with a local layout snapshot.
    #[inline(always)]
    pub fn reader(&self) -> TableReader<'_, K, V> {
        self.reader_with_metadata(self.metadata)
    }

    /// [`reader`](Self::reader) with caller-supplied metadata; see
    /// [`probe_layout_with_metadata`](Self::probe_layout_with_metadata).
    #[inline(always)]
    pub fn reader_with_metadata(&self, metadata: V::StorageMetadata) -> TableReader<'_, K, V> {
        TableReader {
            layout: self.probe_layout_with_metadata(metadata),
            _table: PhantomData,
        }
    }

    /// Rehash all entries into fresh zeroed slabs of `new_size` slots.
    ///
    /// Every occupied entry is re-probed into the larger table and copied whole
    /// (`stride` bytes: hash, key, and stored value alike).
    ///
    /// The collision counter is reset to `self.length` so that post-resize
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
        // every old entry; `old_bases` points into them.
        let _old_slabs = mem::replace(&mut self.slabs, new_slabs);
        let old_bases = mem::replace(
            &mut self.bases,
            adjusted_bases(&self.slabs, self.entries_per_slab, self.stride),
        );
        let old_mask = self.mask;
        self.collisions = self.length;
        self.mask = new_size - 1;
        self.shift = u64::BITS - new_size.trailing_zeros();

        for idx in 0..=old_mask {
            let old_entry = self.entry_ptr_in(&old_bases, idx);
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

/// Copyable snapshot of an entry layout and probe parameters.
///
/// Hot loops copy this value into locals so raw entry writes cannot force
/// repeated loads from the table object. Its `bases` pointer remains valid for
/// the lifetime of the borrowing reader or prober.
struct ProbeLayout<K, V: AggregationValue + ?Sized> {
    magic: u64,
    stride: usize,
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    mask: usize,
    shift: u32,
    pre_shift: u32,
    bases: *const usize,
    bases_len: usize,
    /// Cached first slab base for the common single-slab case.
    first_base: usize,
    metadata: V::StorageMetadata,
    _phantom: PhantomData<K>,
}

impl<K, V: AggregationValue + ?Sized> Clone for ProbeLayout<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V: AggregationValue + ?Sized> Copy for ProbeLayout<K, V> {}

impl<K, V: AggregationValue + ?Sized> ProbeLayout<K, V> {
    /// Map a hash to a slot index using the top bits.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.pre_shift) >> self.shift) as usize
    }

    /// The address of the entry at `index`; see
    /// [`BaseHashTable::entry_ptr_in`].
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        // Skip slab selection when the table uses a single slab.
        if self.bases_len == 1 {
            return self.first_base.wrapping_add(index * self.stride) as *mut u8;
        }
        let slab_idx = fast_div(index, self.magic);
        assert!(slab_idx < self.bases_len);
        let base = unsafe { *self.bases.add(slab_idx) };
        base.wrapping_add(index * self.stride) as *mut u8
    }

    /// The view of the occupied entry at `entry`'s address, with `hash`
    /// already read.
    #[inline(always)]
    unsafe fn view<'a>(&self, entry: *const u8, hash: u64) -> EntryView<'a, K, V> {
        unsafe {
            EntryView {
                hash,
                key: &*(entry.add(self.key_offset) as *const K),
                stored: V::from_entry(entry.add(self.value_offset), self.metadata),
            }
        }
    }
}

/// Mutable table handle that keeps the probe layout local across many probes.
pub struct Prober<'t, K: PersistedKey, V: AggregationValue + ?Sized> {
    table: &'t mut BaseHashTable<K, V>,
    layout: ProbeLayout<K, V>,
}

impl<K: PersistedKey, V: AggregationValue + ?Sized> Prober<'_, K, V> {
    /// Prefetches the initial slot and two following cache lines into L1.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let ptr = self.layout.entry_ptr(self.layout.slot_for(hash)) as *const u8;
        prefetch_l1_line(ptr);
        prefetch_l1_line(ptr.wrapping_add(64));
        prefetch_l1_line(ptr.wrapping_add(128));
    }

    /// Prefetches the initial slot into L2 for a longer lookahead.
    #[inline(always)]
    pub fn prefetch_l2(&self, hash: u64) {
        prefetch_l2_line(self.layout.entry_ptr(self.layout.slot_for(hash)) as *const u8);
    }

    /// See [`BaseHashTable::undersized`].
    #[inline(always)]
    pub fn undersized(&self) -> bool {
        self.table.undersized()
    }

    /// Inserts or merges a stored partial value from another table.
    #[inline(always)]
    pub fn merge_from<const COUNT_COLLISIONS: bool, L>(
        &mut self,
        hash: u64,
        key: L,
        source: &V,
        ctx: &V::SharedContext,
    ) where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<COUNT_COLLISIONS, L, &V, _, _>(
            hash,
            key,
            source,
            |source, stored| stored.copy_from(source),
            |source, stored| stored.merge_from(source, ctx),
        );
    }

    /// Grows the table fourfold after it crosses the load threshold.
    #[inline(always)]
    pub fn grow_if_full(&mut self, allocator: &mut SlabAllocator, capacity: &mut usize) {
        if self.table.undersized() {
            *capacity *= 4;
            self.table.resize(allocator, *capacity);
            // Preserve constant metadata across the uncommon resize branch.
            self.layout = self.table.probe_layout_with_metadata(self.layout.metadata);
        }
    }

    /// Doubles the table when collision pressure exceeds the given ratio.
    #[inline(always)]
    pub fn resize_on_collisions(&mut self, allocator: &mut SlabAllocator, collision_ratio: usize) {
        if self.table.collisions() > self.table.len() * collision_ratio {
            let new_size = self.table.capacity() << 1;
            self.table.resize(allocator, new_size);
            // See `grow_if_full` for why this uses the prober's metadata.
            self.layout = self.table.probe_layout_with_metadata(self.layout.metadata);
        }
    }

    /// Probes for a key, then seeds an empty slot or updates the matching slot.
    ///
    /// `context` is moved into whichever closure runs. This avoids a second
    /// lookup and lets update paths, such as string MIN and MAX, skip work when
    /// the new row does not change the group.
    ///
    /// Hash zero is remapped to one because zero marks an empty slot. With
    /// `COUNT_COLLISIONS`, each occupied non-match increments the table's
    /// collision counter.
    #[inline(always)]
    pub fn probe_fold<const COUNT_COLLISIONS: bool, L, X, S, U>(
        &mut self,
        mut hash: u64,
        key: L,
        context: X,
        seed: S,
        update: U,
    ) where
        L: LiveKey<Persisted = K>,
        S: FnOnce(X, &mut V),
        U: FnOnce(X, &mut V),
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
                    self.table.length += 1;
                    seed(
                        context,
                        V::from_entry_mut(entry.add(layout.value_offset), layout.metadata),
                    );
                    return;
                }
                if *hash_ptr == hash
                    && key.eq_persisted(&*(entry.add(layout.key_offset) as *const K))
                {
                    update(
                        context,
                        V::from_entry_mut(entry.add(layout.value_offset), layout.metadata),
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

/// Read-only table handle with a local probe-layout snapshot.
pub struct TableReader<'a, K: PersistedKey, V: AggregationValue + ?Sized> {
    layout: ProbeLayout<K, V>,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue + ?Sized> TableReader<'a, K, V> {
    /// See [`BaseHashTable::hash_at`].
    #[inline(always)]
    pub fn hash_at(&self, index: usize) -> u64 {
        unsafe { *(self.layout.entry_ptr(index).add(self.layout.hash_offset) as *const u64) }
    }

    /// See [`BaseHashTable::view_at`].
    #[inline(always)]
    pub fn view_at(&self, index: usize) -> EntryView<'a, K, V> {
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
pub struct HashTableIterator<'a, K: PersistedKey, V: AggregationValue + ?Sized> {
    layout: ProbeLayout<K, V>,
    idx: usize,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue + ?Sized> Iterator for HashTableIterator<'a, K, V> {
    type Item = EntryView<'a, K, V>;

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
        AggregationSlot, OwnedValue, ValueColumnBuilder,
    };
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::Field;

    /// A minimal additive value for exercising the probe mechanics: merging two
    /// of these sums their counts.
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    impl OwnedValue for Count {
        type Reader<'b> = ();
        type SharedContext = ();
        type ColumnBuilder = CountColumnBuilder;
        type SortKey = i64;
        type WorkerContext = ();
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
    struct CountColumnBuilder;

    impl ValueColumnBuilder for CountColumnBuilder {
        type Value = Count;
        type Context = ();
        fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _context: &()) -> Self {
            CountColumnBuilder
        }
        fn push(&mut self, _value: &Count) {}
        fn push_stored(&mut self, _stored: &Count) {}
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
        assert_eq!(*entry.stored, Count(1));
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
        assert_eq!(*entry.stored, Count(3));
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
            assert!(!table.undersized());
        }

        table
            .prober()
            .merge_from::<false, _>(max_load as u64 + 1, max_load as u64, &Count(1), &());

        assert!(table.undersized());
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
            let found = table.iter(0).any(|e| *e.key == i && e.stored.0 == 1);
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
        assert_eq!(*merged.stored, Count(2));
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
