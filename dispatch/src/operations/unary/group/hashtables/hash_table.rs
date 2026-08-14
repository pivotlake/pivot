//! Open-addressing hash table with linear probing, optimized for GROUP BY.
//!
//! See [`BaseHashTable`] for the full design rationale.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::TableReader;
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// Maximum fraction of slots that can be occupied before the table is
/// considered undersized. Used to size the merge's result tables so they fill
/// without resizing.
pub const MAX_LOAD_FACTOR: f64 = 0.7;

/// How full a worker's table gets before it is retired and replaced.
///
/// Higher than [`MAX_LOAD_FACTOR`] because the two densities are paid for
/// differently. A worker's table is written once and then read once, whole, by
/// the merge, so its empty slots are read bandwidth the merge spends on
/// nothing: at 0.7 that is 43% more bytes than the entries need. The extra
/// probing a denser table costs walks *consecutive* slots, so it buys those
/// bytes back for a few comparisons inside cache lines already fetched, rather
/// than for more misses.
const SPILL_LOAD_FACTOR: f64 = 0.7;

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
#[cfg(test)]
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
    (len as f64 * SPILL_LOAD_FACTOR).round() as usize
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
pub(super) fn reciprocal(d: u64) -> u64 {
    assert!(d >= 2, "an entry never fills half a slab");
    ((1u128 << 64).div_ceil(d as u128)) as u64
}

/// Divides `n` using a reciprocal returned by [`reciprocal`].
#[inline(always)]
pub(super) fn fast_div(n: usize, m: u64) -> usize {
    (((n as u128) * (m as u128)) >> 64) as usize
}

/// Computes slab bases adjusted by their first global entry offset.
///
/// With these bases, an entry address is `bases[slab] + index * stride`.
/// Adjusted addresses may numerically precede an allocation, but only the
/// reconstructed in-bounds address is dereferenced.
pub(super) fn adjusted_bases(slabs: &[Slab], entries_per_slab: usize, stride: usize) -> Vec<usize> {
    slabs
        .iter()
        .enumerate()
        .map(|(slab_index, slab)| {
            (slab.ptr as usize).wrapping_sub(slab_index * entries_per_slab * stride)
        })
        .collect()
}

/// Byte offset of the stored value inside an entry of this key and value type.
pub(crate) fn value_offset_for<K, V: AggregationValue + ?Sized>(
    metadata: V::StorageMetadata,
) -> usize {
    entry_layout::<K, V>(metadata).value_offset
}

/// Runtime field offsets, stride, and alignment for one table entry.
pub(super) struct EntryLayout {
    pub(super) hash_offset: usize,
    pub(super) key_offset: usize,
    pub(super) value_offset: usize,
    pub(super) stride: usize,
    pub(super) align: usize,
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
pub(crate) fn prefetch_l1_line(ptr: *const u8) {
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
/// A [`Dynamic`](crate::operations::unary::group::values::Dynamic) value
/// makes the stride a runtime value because its cell count
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
    slot_mask: usize,
    slot_shift: u32,
    /// Left-shift applied to hash before computing the slot index.
    /// Set to `PARTITIONS.trailing_zeros()` for merge maps so that the partition
    /// bits (top N) are stripped and the next top bits drive slot placement.
    hash_left_shift: u32,
    /// Bytes per entry.
    entry_stride: usize,
    /// Value layout metadata stored once per table.
    storage_metadata: V::StorageMetadata,
    /// Slab addresses adjusted by their first global entry offset.
    adjusted_slab_bases: Vec<usize>,
    /// Owns the table's memory; only ever read through `adjusted_slab_bases`.
    _slabs: Vec<Slab>,
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
        hash_left_shift: u32,
        ctx: &V::SharedContext,
    ) -> Self {
        let storage_metadata = V::storage_metadata(ctx);
        let layout = entry_layout::<K, V>(storage_metadata);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let slabs = allocator.create_strided_slabs(expected_capacity, layout.stride, layout.align);
        BaseHashTable {
            slot_mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            hash_left_shift,
            slot_shift: u64::BITS - expected_capacity.trailing_zeros(),
            entry_stride: layout.stride,
            storage_metadata,
            adjusted_slab_bases: adjusted_bases(&slabs, entries_per_slab, layout.stride),
            _slabs: slabs,
            _phantom: PhantomData,
        }
    }

    /// Total number of slots (occupied + empty). Always a power of 2.
    pub fn capacity(&self) -> usize {
        self.slot_mask + 1
    }

    /// Returns the number of entries currently stored in the table.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns an iterator over all non-empty entries in the table.
    ///
    /// Iteration order is arbitrary (based on slot positions, not insertion order).
    /// Empty slots (hash == 0) are skipped automatically.
    #[cfg(test)]
    pub fn iter(&self, start_offset: usize) -> HashTableIterator<'_, K, V> {
        HashTableIterator {
            reader: self.reader::<0>(),
            idx: start_offset,
        }
    }

    /// Returns `true` if the table has exceeded its [`SPILL_LOAD_FACTOR`] threshold.
    pub fn undersized(&self) -> bool {
        self.len() > self.max_load
    }

    /// Returns a copyable reader over this table's slots and entries.
    ///
    /// `N` is the specialized slot count of the surrounding
    /// [`dispatch_arity`](AggregationValue::dispatch_arity) body: a nonzero
    /// `N` derives the reader's layout from that constant so its field
    /// offsets and stride arithmetic constant-fold, while `N == 0` (the
    /// runtime-arity fallback, and every fixed-size value) reads the table's
    /// stored metadata.
    #[inline(always)]
    pub(crate) fn reader<const N: usize>(&self) -> TableReader<'_, K, V> {
        let metadata = if N == 0 {
            self.storage_metadata
        } else {
            V::metadata_for_arity::<N>()
        };
        unsafe { self.reader_with_unchecked_lifetime(metadata) }
    }

    /// Builds a reader without tying its lifetime to this particular borrow.
    ///
    /// The prober already holds an exclusive table borrow for the resulting
    /// lifetime. It uses this constructor so the copied reader and that
    /// mutable borrow can live in the same struct.
    ///
    /// # Safety
    ///
    /// The returned reader must not outlive this table. It must be replaced
    /// before accessing entries after any operation that changes the table's
    /// slabs, capacity, or entry layout.
    #[inline(always)]
    unsafe fn reader_with_unchecked_lifetime<'table>(
        &self,
        metadata: V::StorageMetadata,
    ) -> TableReader<'table, K, V> {
        let layout = entry_layout::<K, V>(metadata);
        debug_assert_eq!(layout.stride, self.entry_stride);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        unsafe {
            TableReader::new(
                layout,
                entries_per_slab,
                self.slot_mask,
                self.slot_shift,
                self.hash_left_shift,
                &self.adjusted_slab_bases,
                metadata,
            )
        }
    }

    /// Builds a bucket-grouped view of this table's occupied slots at the
    /// given resolution, folding every stored hash into the worker's
    /// distinct-count sketch on the way.
    pub fn build_sorted_run(
        &self,
        hll: &mut super::super::hll::Hll,
        bucket_bits: u32,
    ) -> super::SortedRun {
        let reader = self.reader::<0>();
        let capacity = self.capacity();
        // One slack element keeps the unconditional store below in bounds
        // when the cursor has already reached the entry count.
        let mut entries = vec![(0u64, 0u32); self.len() + 1];
        let mut count = 0usize;
        for idx in 0..capacity {
            let hash = reader.hash_of(reader.entry_ptr(idx));
            entries[count] = (hash, idx as u32);
            count += (hash != 0) as usize;
        }
        debug_assert_eq!(count, self.len());
        super::SortedRun::from_slot_scan(&entries[..count], hll, bucket_bits)
    }

    /// Creates a probing handle with a local layout snapshot.
    #[inline(always)]
    pub fn prober(&mut self) -> Prober<'_, K, V> {
        self.prober_with_metadata(self.storage_metadata)
    }

    /// Creates a probing handle for an arity-specialized loop.
    #[inline(always)]
    pub fn prober_with_metadata(&mut self, metadata: V::StorageMetadata) -> Prober<'_, K, V> {
        let reader = unsafe { self.reader_with_unchecked_lifetime(metadata) };
        Prober {
            table: self,
            reader,
        }
    }
}

/// Mutable table handle that keeps the probe reader local across many probes.
pub struct Prober<'t, K: PersistedKey, V: AggregationValue + ?Sized> {
    table: &'t mut BaseHashTable<K, V>,
    reader: TableReader<'t, K, V>,
}

impl<K: PersistedKey, V: AggregationValue + ?Sized> Prober<'_, K, V> {
    /// Prefetches the initial slot and two following cache lines into L1.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let ptr = self.reader.entry_ptr(self.reader.slot_for(hash)) as *const u8;
        prefetch_l1_line(ptr);
        prefetch_l1_line(ptr.wrapping_add(64));
        prefetch_l1_line(ptr.wrapping_add(128));
    }

    /// Prefetches the initial slot into L2 for a longer lookahead.
    #[inline(always)]
    pub fn prefetch_l2(&self, hash: u64) {
        prefetch_l2_line(self.reader.entry_ptr(self.reader.slot_for(hash)) as *const u8);
    }

    /// See [`BaseHashTable::undersized`].
    #[inline(always)]
    pub fn undersized(&self) -> bool {
        self.table.undersized()
    }

    /// Inserts or merges a stored partial value, for tests that build tables
    /// with chosen hashes.
    #[cfg(test)]
    pub fn merge_from<L>(&mut self, hash: u64, key: L, source: &V, ctx: &V::SharedContext)
    where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<L, &V, _, _>(
            hash,
            key,
            source,
            |source, stored| stored.copy_from(source),
            |source, stored| stored.merge_from(source, ctx),
        );
    }

    /// Probes for a key, then seeds an empty slot or updates the matching slot.
    ///
    /// `context` is moved into whichever closure runs. This avoids a second
    /// lookup and lets update paths, such as string MIN and MAX, skip work when
    /// the new row does not change the group.
    ///
    /// Hash zero is remapped to one because zero marks an empty slot.
    ///
    /// Returns the address of the entry the key resolved to, which lets a
    /// caller remember it and fold a later repeat of the same key straight into
    /// it.
    #[inline(always)]
    pub fn probe_fold<L, X, S, U>(
        &mut self,
        mut hash: u64,
        key: L,
        context: X,
        seed: S,
        update: U,
    ) -> *mut u8
    where
        L: LiveKey<Persisted = K>,
        S: FnOnce(X, &mut V),
        U: FnOnce(X, &mut V),
    {
        let reader = self.reader;
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1.
        if hash == 0 {
            hash = 1;
        }
        let mut idx = reader.slot_for(hash);
        loop {
            let entry = reader.entry_ptr(idx);
            unsafe {
                let hash_ptr = entry.add(reader.hash_offset) as *mut u64;
                if *hash_ptr == 0 {
                    *hash_ptr = hash;
                    (entry.add(reader.key_offset) as *mut K).write(key.persist());
                    self.table.length += 1;
                    seed(
                        context,
                        V::from_entry_mut(entry.add(reader.value_offset), reader.storage_metadata),
                    );
                    return entry;
                }
                if *hash_ptr == hash
                    && key.eq_persisted(&*(entry.add(reader.key_offset) as *const K))
                {
                    update(
                        context,
                        V::from_entry_mut(entry.add(reader.value_offset), reader.storage_metadata),
                    );
                    return entry;
                }
            }
            idx = (idx + 1) & reader.slot_mask;
        }
    }
}

/// An iterator over the non-empty entries in a `BaseHashTable`.
///
/// Created by [`BaseHashTable::iter`]. Yields [`EntryView`]s of entries where
/// `hash != 0` in arbitrary order (based on slot positions, not insertion order).
#[cfg(test)]
pub struct HashTableIterator<'a, K: PersistedKey, V: AggregationValue + ?Sized> {
    reader: TableReader<'a, K, V>,
    idx: usize,
}

#[cfg(test)]
impl<'a, K: PersistedKey, V: AggregationValue + ?Sized> Iterator for HashTableIterator<'a, K, V> {
    type Item = EntryView<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let reader = self.reader;
        while self.idx < reader.slot_mask + 1 {
            // One address computation per slot: the hash check and the yielded
            // view read the same entry pointer.
            let entry = reader.entry_ptr(self.idx);
            self.idx += 1;
            unsafe {
                let hash = *(entry.add(reader.hash_offset) as *const u64);
                if hash != 0 {
                    return Some(reader.view(entry, hash));
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
        AggregationSlot, AggregationValue, ArityBody, ValueColumnBuilder,
    };
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::Field;

    /// A minimal additive value for exercising the probe mechanics: merging two
    /// of these sums their counts.
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    impl AggregationValue for Count {
        type Owned = Self;
        type StorageMetadata = ();
        type Reader<'b> = ();
        type SharedContext = ();
        type ColumnBuilder = CountColumnBuilder;
        type SortKey = i64;
        type WorkerContext = ();
        fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}
        fn storage_metadata(_ctx: &()) {}
        fn metadata_for_arity<const N: usize>() {}
        fn dispatch_arity<Ret>(_metadata: (), body: impl ArityBody<Ret>) -> Ret {
            body.run::<0>()
        }
        fn stored_size(_metadata: ()) -> usize {
            size_of::<Self>()
        }
        fn stored_align() -> usize {
            align_of::<Self>()
        }
        unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a Self {
            unsafe { &*(ptr as *const Self) }
        }
        unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut Self {
            unsafe { &mut *(ptr as *mut Self) }
        }
        fn seed(&mut self, _reader: &(), _idx: usize, _wc: &mut ()) {
            *self = Count(1);
        }
        fn update(&mut self, _reader: &(), _idx: usize, _wc: &mut (), _ctx: &()) {
            self.0 += 1;
        }
        fn merge_from(&mut self, source: &Self, _ctx: &()) {
            self.0 += source.0;
        }
        fn copy_from(&mut self, source: &Self) {
            *self = *source;
        }
        fn sort_key(&self, _slot: usize) -> i64 {
            self.0 as i64
        }
        fn to_owned(&self, _ctx: &(), _wc: &mut Option<()>) -> Self {
            *self
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

        table.prober().merge_from(42, 100u64, &Count(1), &());

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

        table.prober().merge_from(42, 100u64, &Count(1), &());
        table.prober().merge_from(42, 100u64, &Count(1), &());
        table.prober().merge_from(42, 100u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.stored, Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table.prober().merge_from(42, 1u64, &Count(1), &());
        table.prober().merge_from(42, 2u64, &Count(1), &());

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

        table.prober().merge_from(0, 99u64, &Count(1), &());

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
        let max_load = (16.0 * SPILL_LOAD_FACTOR).round() as usize;

        for i in 0..max_load {
            table
                .prober()
                .merge_from(i as u64 + 1, i as u64, &Count(1), &());
            assert!(!table.undersized());
        }

        table
            .prober()
            .merge_from(max_load as u64 + 1, max_load as u64, &Count(1), &());

        assert!(table.undersized());
    }

    #[test]
    fn iter_skips_empty_slots() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);

        table.prober().merge_from(1, 10u64, &Count(1), &());
        table.prober().merge_from(2, 20u64, &Count(1), &());

        let entries: Vec<_> = table.iter(0).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.hash != 0));
    }

    #[test]
    fn many_entries_all_retrievable() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 256);

        for i in 0..100u64 {
            table.prober().merge_from(i + 1, i, &Count(1), &());
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
            .merge_from(last_slot_hash, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from(last_slot_hash, 2u64, &Count(1), &());
        table
            .prober()
            .merge_from(last_slot_hash, 3u64, &Count(1), &());

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
    fn view_at_returns_correct_slot() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table.prober().merge_from(42, 100u64, &Count(1), &());

        let reader = table.reader::<0>();
        let occupied: Vec<usize> = (0..table.capacity())
            .filter(|&i| reader.hash_at(i) != 0)
            .collect();

        assert_eq!(occupied.len(), 1);
        assert_eq!(*reader.view_at(occupied[0]).key, 100);
    }

    #[test]
    fn iter_with_start_offset() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);
        for i in 0..20u64 {
            table.prober().merge_from(i + 1, i, &Count(1), &());
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
