//! Open-addressing hash table with linear probing, optimized for GROUP BY.
//!
//! See [`BaseHashTable`] for the full design rationale.

use crate::memory::SlabAllocator;
use crate::memory::{MultiSlabBuffer, SlabBuffer};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut};
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
pub trait PersistedKey: Copy + Clone + Default {}

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

/// An aggregation value stored alongside each key in the hash table.
///
/// Must be `Copy` + `Default` so entries can be zero-initialized and moved
/// cheaply during resize.
pub trait Value: Copy + Clone + Default {
    /// The value for a single occurrence of a key (e.g. count = 1).
    fn single() -> Self;
    /// Combine two values (e.g. sum counts).
    fn merge(self, v: Self) -> Self;
}

/// Supplies per-row keys and values to [`BaseHashTable::merge_batch`].
///
/// Passed as a single `&mut` so the implementation can hold a mutable borrow
/// (e.g. of a key arena) internally without it escaping through a closure return
/// — the methods hand back owned keys/values or a bool, never a borrow tied to
/// that internal state.
pub trait BatchRowSource<K: PersistedKey, V: Value> {
    /// The persisted key for row `i` (allocating in any backing arena as needed).
    fn persisted(&mut self, i: usize) -> K;
    /// Whether row `i`'s key equals the already-persisted key `persisted`.
    fn key_eq(&mut self, i: usize, persisted: &K) -> bool;
    /// The aggregate value contributed by row `i`.
    fn value(&mut self, i: usize) -> V;
}

/// A single slot in the hash table, storing the full hash, key, and value.
///
/// `hash == 0` marks an empty slot. Real zero hashes are remapped to 1
/// by [`BaseHashTable::merge`] to preserve this invariant.
#[derive(Copy, Clone, Default)]
pub struct Entry<K: PersistedKey, V: Value> {
    hash: u64,
    key: K,
    value: V,
}

impl<K: PersistedKey, V: Value> Entry<K, V> {
    /// The stored hash (0 = empty slot).
    #[inline]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// The persisted group key.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// The aggregation value (e.g. count).
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

/// Maximum number of entries allowed for a table with `len` total slots.
fn max_load_for_len(len: usize) -> usize {
    (len as f64 * MAX_LOAD_FACTOR).round() as usize
}

/// Remap the empty-slot sentinel: `hash == 0` marks an empty slot, so a real
/// zero hash is bumped to 1 before it is stored or probed.
#[inline(always)]
fn remap_zero(hash: u64) -> u64 {
    if hash == 0 { 1 } else { hash }
}

/// A linear probing hash table optimized for never rehashing, exposing a very raw interface allowing
/// maximum control by the caller.
///
/// # Memory Layout
///
/// Uses a single contiguous buffer of `Entry<K, V>` structs, generic over the
/// backing allocation (Vec, mmap, etc.):
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                         Buffer: Allocation<Entry<K,V>>                  │
/// ├─────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
/// │  Entry[0]   │  Entry[1]   │             │  Entry[3]   │      ...       │
/// │ hash|key|val│ hash|key|val│             │ hash|key|val│                │
/// │  8B | xB|xB │  8B | xB|xB │             │  8B | xB|xB │                │
/// └─────────────┴─────────────┴─────────────┴─────────────┴────────────────┘
///  8 bytes + Key + Value per entry (e.g., for `ArenaKey` and `Count` this would be 32 bytes-
///  2 entries per 64-byte cache line)
/// ```
///
/// Each `Entry` contains:
/// - `hash: u64` - Full 64-bit hash (0 = empty slot sentinel)
/// - `value: V` - The aggregation value (e.g., count)
/// - `key: K` - The persisted key (e.g., ArenaKey with pointer + length)
///
/// The buffer is allocated with 64-byte alignment to ensure cache-line-aligned access.
/// Capacity is always a power of 2, which allows wrapping via `& mask` during
/// linear probing and efficient top-bit slot placement via `hash >> shift`.
///
/// # How It Works
///
/// Uses open addressing with linear probing:
///
/// 1. **Insert/Merge**: Compute `slot = hash >> shift` (top bits of the hash).
///    Top-bit placement is required so that the partition (top `log2(PARTITIONS)`
///    bits) is a prefix of the slot index, enabling the merge phase to scan
///    tables of different sizes by partition (see module-level docs in `group`).
///    If occupied and different key, probe linearly (slot+1, slot+2, ...) until
///    finding an empty slot (hash == 0) or matching key. On match, merge values
///    instead of inserting.
///
/// 2. **Empty detection**: `hash == 0` marks empty slots. Real zero hashes are
///    converted to 1 to preserve this invariant.
///
/// 3. **Resize**: Handled externally — callers check load pressure and call
///    [`resize_with`](BaseHashTable::resize_with) to replace the buffer.
///
/// # Why Not Swiss Tables (hashbrown / std HashMap)
///
/// Swiss Tables use a two-array layout — a `ctrl[]` byte array for
/// SIMD-accelerated probing and a separate `slots[]` array for key-value
/// data:
///
/// ```text
/// Swiss Table layout:
/// ┌──────────────────────────┐    ┌─────────────────────────────────────────┐
/// │   ctrl[] (1 byte each)   │    │         slots[] (key+value only)        │
/// │ [h2|h2|h2|h2|h2|h2|h2|h2]│    │ [kv0|kv1|kv2|kv3|kv4|kv5|kv6|kv7|...]   │
/// └──────────────────────────┘    └─────────────────────────────────────────┘
///    SIMD-probed metadata              actual data (accessed on match)
/// ```
///
/// This is excellent for general-purpose use, but the GROUP BY merge
/// phase requires properties that Swiss Tables cannot provide:
///
/// ## 1. Top-bit slot placement enables partitioned merging
///
/// This table places entries using the **top** bits of the hash
/// (`slot = hash >> shift`). Because the partition index is also derived
/// from the top bits (`partition = hash >> (64 - log2(PARTITIONS))`), the
/// partition is always a **prefix** of the slot index. This means entries
/// for partition P occupy a contiguous, predictable slot range in any
/// power-of-2 table, regardless of size:
///
/// ```text
/// 128-slot table:  partition 0 = slots [0, 2)    partition 1 = slots [2, 4)   ...
/// 256-slot table:  partition 0 = slots [0, 4)    partition 1 = slots [4, 8)   ...
/// ```
///
/// The merge phase exploits this: it scans only the relevant slot range
/// per partition, across tables of mixed sizes, without touching entries
/// from other partitions. **Swiss Tables use the lower bits for slot
/// placement**, so entries from the same partition are scattered across
/// the entire table. A Swiss Table merge would require either a full scan
/// of every source table (touching all entries to check partition
/// membership) or an O(n) pre-partitioning pass — both losing the
/// cache-locality advantage.
///
/// ## 2. Stored hashes avoid rehashing large keys
///
/// Large strings (URLs, paths) are expensive to hash. We store the full
/// 64-bit hash inline with each entry, so resize and cross-table merge
/// never re-hash a key. Swiss Tables only store 7 bits (h2) in their
/// control byte — to avoid rehashing they'd need to store the full hash
/// separately, losing their space advantage over this layout.
///
/// ## 3. Inline entries give single-access probing
///
/// Each `Entry` packs hash + key + value into one struct (e.g. 32 bytes
/// for `ArenaKey` + `Count` = 2 entries per 64-byte cache line). A single
/// cache-line fetch gives the hash for comparison AND the next linear-probe
/// candidate. Swiss Tables require two separate memory accesses per probe:
/// one for the ctrl byte and one for the slot data.
///
/// # Generic Allocation
///
/// The type parameter `A` allows swapping the backing storage:
/// - `SlabBuffer<Entry<K,V>>`: Single slab from the slab allocator (for small tables)
/// - `MultiSlabBuffer<Entry<K,V>>`: Multiple slabs for tables exceeding a single buffer
pub struct BaseHashTable<
    K: PersistedKey,
    V: Value,
    A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>,
> {
    mask: usize,
    shift: u32,
    collisions: usize,
    /// Left-shift applied to hash before computing the slot index.
    /// Set to `PARTITIONS.trailing_zeros()` for merge maps so that the partition
    /// bits (top N) are stripped and the next top bits drive slot placement.
    pre_shift: u32,
    buffer: A,
    length: usize,
    max_load: usize,
    _phantom: PhantomData<(K, V)>,
}

unsafe impl<K: PersistedKey, V: Value, A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>> Send
    for BaseHashTable<K, V, A>
{
}

impl<K: PersistedKey, V: Value> BaseHashTable<K, V, MultiSlabBuffer<Entry<K, V>>> {
    /// Creates a new HashTable backed by a multi-slab buffer.
    ///
    /// `expected_capacity` must be a power of 2 and is used directly as the
    /// number of slots. The buffer is zero-initialized so that all slots
    /// start empty (`hash == 0`).
    pub fn multi_slab(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        pre_shift: u32,
    ) -> Self {
        let buffer = allocator.create_multi_slab_buffer(expected_capacity, true);

        BaseHashTable {
            mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            buffer,
            pre_shift,
            shift: u64::BITS - expected_capacity.trailing_zeros(),
            _phantom: PhantomData,
            collisions: 0,
        }
    }
}

impl<K: PersistedKey, V: Value> BaseHashTable<K, V, SlabBuffer<Entry<K, V>>> {
    /// Creates a new HashTable backed by a single slab buffer.
    ///
    /// `expected_capacity` must be a power of 2 and is used directly as the
    /// number of slots. The total byte size (`expected_capacity * size_of::<Entry>()`)
    /// must fit within a single slab (`< BUFFER_SIZE`). The buffer is
    /// zero-initialized so that all slots start empty (`hash == 0`).
    pub fn single_slab(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        pre_shift: u32,
    ) -> Self {
        let buffer = allocator.create_slab_buffer(expected_capacity, true);

        BaseHashTable {
            mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            buffer,
            _phantom: PhantomData,
            pre_shift,
            shift: u64::BITS - expected_capacity.trailing_zeros(),
            collisions: 0,
        }
    }
}

impl<K: PersistedKey, V: Value, A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>>
    BaseHashTable<K, V, A>
{
    /// Bitmask for slot indexing (`capacity - 1`).
    pub fn mask(&self) -> usize {
        self.mask
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

    /// Direct slot access by index (no bounds checking beyond the buffer's own).
    #[inline(always)]
    pub fn entry_at(&self, index: usize) -> &Entry<K, V> {
        &self.buffer[index]
    }

    /// Prefetch the hash table slot where `hash` would land, plus the next cache line
    /// to cover short probe chains. Brings the lines all the way into L1 (`T0`) —
    /// use this *near* the access (small lookahead).
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let idx = self.slot_for(hash);
        let ptr = &self.buffer[idx] as *const Entry<K, V> as *const u8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                ptr.add(64) as *const i8
            );
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                ptr.add(128) as *const i8
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Prefetch the slot's cache line into L2 (`T1`) only. Issued *far* ahead of
    /// the access and paired with a nearer [`prefetch`] (L1) call, this
    /// software-pipelines the memory hierarchy: the line is pulled DRAM→L2 far
    /// ahead, then L2→L1 just before use, hiding the full DRAM latency that a
    /// single L1 prefetch at a short distance can't cover on a multi-GB table.
    #[inline]
    pub fn prefetch_l2(&self, hash: u64) {
        let idx = self.slot_for(hash);
        let ptr = &self.buffer[idx] as *const Entry<K, V> as *const u8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Returns an iterator over all non-empty entries in the table.
    ///
    /// Iteration order is arbitrary (based on slot positions, not insertion order).
    /// Empty slots (hash == 0) are skipped automatically.
    pub fn iter(&self, start_offset: usize) -> HashTableIterator<'_, K, V, A> {
        HashTableIterator {
            hash_table: self,
            idx: start_offset,
        }
    }

    /// Returns `true` if the table has exceeded its [`MAX_LOAD_FACTOR`] threshold.
    pub fn undersized(&self) -> bool {
        self.len() > self.max_load
    }

    /// Inserts or merges an entry into the hash table.
    ///
    /// # Behavior
    ///
    /// - If `key` does not exist: persists the key and inserts a new entry with `value`
    /// - If `key` already exists: merges `value` into the existing entry via `Value::merge`
    ///
    /// # Algorithm
    ///
    /// 1. Compute initial slot via `slot_for(hash)` (top bits of the hash)
    /// 2. Linear probe until we find either:
    ///    - Empty slot (hash == 0): insert new entry here
    ///    - Matching entry (same hash AND same key): merge values
    ///
    /// # Hash Zero Handling
    ///
    /// Since `hash == 0` is the empty sentinel, any key that legitimately hashes to 0
    /// is stored with `hash = 1` instead. This is invisible to callers.
    ///
    /// # Collision Tracking
    ///
    /// When `COUNT_COLLISIONS` is true, each probe step increments the collision
    /// counter. This allows callers to monitor probe-chain pressure and decide
    /// when to resize based on the cumulative collision-to-entry ratio.
    ///
    /// # Performance
    ///
    /// - Best case: O(1) - slot is empty or immediate match
    /// - Average case: O(1) - at 70% load, expected probe length is ~1.8
    /// - Worst case: O(n) - pathological hash collisions
    #[inline(always)]
    pub fn merge<const COUNT_COLLISIONS: bool, L: LiveKey<Persisted = K>>(
        &mut self,
        mut hash: u64,
        key: L,
        value: V,
    ) {
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1
        if hash == 0 {
            hash = 1;
        }

        let mut idx = self.slot_for(hash);

        loop {
            let entry = &mut self.buffer[idx];

            if entry.hash == 0 {
                // Empty slot found - persist the key and insert
                let persisted = key.persist();
                self.buffer[idx] = Entry {
                    hash,
                    value,
                    key: persisted,
                };
                self.length += 1;
                return;
            }

            if entry.hash == hash && key.eq_persisted(&entry.key) {
                // Key exists - merge the values (e.g., add counts)
                entry.value = entry.value.merge(value);
                return;
            }

            if COUNT_COLLISIONS {
                self.collisions += 1;
            }

            // Collision with different key - linear probe to next slot
            idx = (idx + 1) & self.mask;
        }
    }

    /// Prefetch the cache line backing slot `idx` into L1.
    #[inline(always)]
    fn prefetch_entry(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let ptr = &self.buffer[idx] as *const Entry<K, V> as *const i8;
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = idx;
    }

    /// Probe row `i` (hash `hash`) against `slot` for [`merge_batch`](Self::merge_batch).
    ///
    /// Returns `true` when the row is resolved — either by claiming an empty slot
    /// (insert) or by merging into a slot holding the same key. On a collision
    /// (occupied by a different key) it advances `slots[i]` to the next slot and
    /// returns `false`, so the caller re-queues the row for the following pass.
    #[inline(always)]
    fn resolve_row<S: BatchRowSource<K, V>>(
        &mut self,
        i: usize,
        slot: usize,
        hash: u64,
        slots: &mut [usize],
        src: &mut S,
    ) -> bool {
        let stored = self.buffer[slot].hash;
        if stored == 0 {
            let key = src.persisted(i);
            let value = src.value(i);
            self.buffer[slot] = Entry { hash, key, value };
            self.length += 1;
            true
        } else if stored == hash && src.key_eq(i, &self.buffer[slot].key) {
            let value = src.value(i);
            self.buffer[slot].value = self.buffer[slot].value.merge(value);
            true
        } else {
            self.collisions += 1;
            slots[i] = (slot + 1) & self.mask;
            false
        }
    }

    /// Batched, multi-pass probe insert — the structure DuckDB uses in
    /// `FindOrCreateGroupsInternal`.
    ///
    /// The scalar [`merge`](Self::merge) resolves one row completely before the
    /// next, so a row's linear-probe chain is a dependency chain the CPU must
    /// walk one cache-missing slot at a time — terrible memory-level parallelism.
    /// Here every still-unresolved row is advanced by exactly **one** slot per
    /// pass, so within a pass the rows touch independent slots and the hardware
    /// keeps many misses in flight; collisions are deferred to the next pass over
    /// the (shrinking) unresolved set.
    ///
    /// Correctness mirrors `merge` exactly (see [`resolve_row`](Self::resolve_row)).
    /// Because the slot is re-read each pass, two rows in the same batch with the
    /// same key (same hash → same slot) resolve correctly: the first claims the
    /// slot, the rest fall into the equal-key merge branch.
    ///
    /// `slots`, `sel`, `next` are caller scratch of length ≥ `length`; `hashes`
    /// holds the row hashes (0 is remapped to 1 in place to preserve the empty
    /// sentinel). The caller MUST ensure `capacity - len > length` so every probe
    /// finds an empty slot and the passes terminate.
    #[inline]
    pub fn merge_batch<S: BatchRowSource<K, V>>(
        &mut self,
        length: usize,
        hashes: &mut [u64],
        slots: &mut [usize],
        sel: &mut [u32],
        next: &mut [u32],
        src: &mut S,
    ) {
        /// How far ahead, in row positions, to prefetch each slot.
        const PREFETCH_DIST: usize = 16;

        // Pass 1: probe every row at its home slot, computing the slot inline.
        // Rows that resolve here (insert or merge) never touch the scratch
        // arrays; only collided rows are recorded — advanced slot in `slots[i]`,
        // index in `next` — for the follow-up passes. Folding the slot
        // computation in here avoids materialising `slots`/`sel` for *every* row,
        // which is pure overhead in the common low-collision / merge-heavy case.
        let mut collided = 0usize;
        for i in 0..length {
            if i + PREFETCH_DIST < length {
                self.prefetch_entry(self.slot_for(hashes[i + PREFETCH_DIST]));
            }
            let hash = remap_zero(hashes[i]);
            hashes[i] = hash;
            let slot = self.slot_for(hash);
            if !self.resolve_row(i, slot, hash, slots, src) {
                next[collided] = i as u32;
                collided += 1;
            }
        }

        // Passes 2+: walk only the still-unresolved rows, ping-ponging the
        // worklist between `next` and `sel`. Each pass advances every row by one
        // slot, keeping their probes independent so misses stay in flight.
        let (mut work, mut spill) = (next, sel);
        let mut remaining = collided;
        while remaining > 0 {
            let mut collided = 0usize;
            for k in 0..remaining {
                if k + PREFETCH_DIST < remaining {
                    self.prefetch_entry(slots[work[k + PREFETCH_DIST] as usize]);
                }
                let i = work[k] as usize;
                if !self.resolve_row(i, slots[i], hashes[i], slots, src) {
                    spill[collided] = i as u32;
                    collided += 1;
                }
            }
            std::mem::swap(&mut work, &mut spill);
            remaining = collided;
        }
    }

    /// Rehash all entries into a new buffer of `new_size` slots.
    ///
    /// Replaces the current buffer with `buffer` (which must be zeroed and
    /// have `new_size` slots), then re-inserts every occupied entry from the
    /// old buffer using linear probing in the new, larger table.
    ///
    /// The collision counter is reset to `self.length` so that post-resize
    /// collision tracking reflects only new probing pressure, not historical
    /// collisions from the smaller table.
    ///
    /// # Panics
    ///
    /// The underlying allocation may panic if memory allocation fails.
    #[cold]
    pub fn resize_with(&mut self, buffer: A, new_size: usize) {
        debug!("Resizing map to {:?}...", new_size);
        let old_mask = self.mask;
        self.collisions = self.length;
        self.mask = new_size - 1;
        self.shift = u64::BITS - new_size.trailing_zeros();

        let old_buffer = mem::replace(&mut self.buffer, buffer);

        for idx in 0..=old_mask {
            let entry = &old_buffer[idx];
            if entry.hash != 0 {
                let mut new_idx = self.slot_for(entry.hash);
                loop {
                    if self.buffer[new_idx].hash == 0 {
                        self.buffer[new_idx] = *entry;
                        break;
                    }
                    new_idx = (new_idx + 1) & self.mask;
                }
            }
        }

        self.max_load = max_load_for_len(new_size);
    }
}

/// An iterator over the non-empty entries in a `BaseHashTable`.
///
/// Created by [`BaseHashTable::iter`]. Yields references to entries where
/// `hash != 0` in arbitrary order (based on slot positions, not insertion order).
pub struct HashTableIterator<
    'a,
    K: PersistedKey,
    V: Value,
    A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>,
> {
    hash_table: &'a BaseHashTable<K, V, A>,
    idx: usize,
}

impl<'a, K: PersistedKey, V: Value, A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>>
    Iterator for HashTableIterator<'a, K, V, A>
{
    type Item = &'a Entry<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.hash_table.mask + 1 {
            let entry = &self.hash_table.buffer[self.idx];
            self.idx += 1;
            if entry.hash != 0 {
                return Some(entry);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    impl Value for Count {
        fn single() -> Self {
            Count(1)
        }
        fn merge(self, v: Self) -> Self {
            Count(self.0 + v.0)
        }
    }

    type TestTable = BaseHashTable<u64, Count, Vec<Entry<u64, Count>>>;

    fn new_table(capacity: usize) -> TestTable {
        let buffer = vec![Entry::default(); capacity];
        BaseHashTable {
            mask: capacity - 1,
            shift: u64::BITS - capacity.trailing_zeros(),
            collisions: 0,
            pre_shift: 0,
            buffer,
            length: 0,
            max_load: max_load_for_len(capacity),
            _phantom: PhantomData,
        }
    }

    #[test]
    fn insert_single_entry() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 100u64, Count::single());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key(), 100);
        assert_eq!(*entry.value(), Count(1));
    }

    #[test]
    fn merge_duplicate_keys_sums_values() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 100u64, Count::single());
        table.merge::<false, _>(42, 100u64, Count::single());
        table.merge::<false, _>(42, 100u64, Count::single());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.value(), Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 1u64, Count::single());
        table.merge::<false, _>(42, 2u64, Count::single());

        assert_eq!(table.len(), 2);
        let entries: Vec<_> = table.iter(0).collect();
        let keys: Vec<u64> = entries.iter().map(|e| *e.key()).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn hash_zero_is_remapped_and_retrievable() {
        let mut table = new_table(16);

        table.merge::<false, _>(0, 99u64, Count::single());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key(), 99);
        assert_ne!(entry.hash(), 0);
    }

    #[test]
    fn undersized_triggers_at_load_factor() {
        let mut table = new_table(16);
        let max_load = (16.0 * MAX_LOAD_FACTOR).round() as usize;

        for i in 0..max_load {
            table.merge::<false, _>(i as u64 + 1, i as u64, Count::single());
            assert!(!table.undersized());
        }

        table.merge::<false, _>(max_load as u64 + 1, max_load as u64, Count::single());

        assert!(table.undersized());
    }

    #[test]
    fn collision_counting() {
        let mut table = new_table(16);

        table.merge::<true, _>(42, 1u64, Count::single());
        assert_eq!(table.collisions(), 0);

        table.merge::<true, _>(42, 2u64, Count::single());
        assert_eq!(table.collisions(), 1);
    }

    #[test]
    fn collision_counting_disabled() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 1u64, Count::single());
        table.merge::<false, _>(42, 2u64, Count::single());

        assert_eq!(table.collisions(), 0);
    }

    #[test]
    fn iter_skips_empty_slots() {
        let mut table = new_table(128);

        table.merge::<false, _>(1, 10u64, Count::single());
        table.merge::<false, _>(2, 20u64, Count::single());

        let entries: Vec<_> = table.iter(0).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.hash() != 0));
    }

    #[test]
    fn resize_preserves_all_entries() {
        let mut table = new_table(16);
        for i in 0..8u64 {
            table.merge::<false, _>(i + 1, i, Count::single());
        }

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, 32);

        assert_eq!(table.len(), 8);
        assert_eq!(table.capacity(), 32);
        for i in 0..8u64 {
            let found = table.iter(0).any(|e| *e.key() == i);
            assert!(found, "key {} missing after resize", i);
        }
    }

    #[test]
    fn resize_resets_collision_counter() {
        let mut table = new_table(16);
        table.merge::<true, _>(42, 1u64, Count::single());
        table.merge::<true, _>(42, 2u64, Count::single());
        let pre_resize_collisions = table.collisions();
        assert!(pre_resize_collisions > 0);

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, 32);

        assert_eq!(table.collisions(), table.len());
    }

    #[test]
    fn many_entries_all_retrievable() {
        let mut table = new_table(256);

        for i in 0..100u64 {
            table.merge::<false, _>(i + 1, i, Count::single());
        }

        assert_eq!(table.len(), 100);
        for i in 0..100u64 {
            let found = table.iter(0).any(|e| *e.key() == i && e.value().0 == 1);
            assert!(found, "key {} missing or wrong value", i);
        }
    }

    #[test]
    fn wrap_around_probing() {
        let mut table = new_table(16);
        let last_slot_hash = u64::MAX;

        table.merge::<false, _>(last_slot_hash, 1u64, Count::single());
        table.merge::<false, _>(last_slot_hash, 2u64, Count::single());
        table.merge::<false, _>(last_slot_hash, 3u64, Count::single());

        assert_eq!(table.len(), 3);
        let keys: Vec<u64> = table.iter(0).map(|e| *e.key()).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn empty_table_iteration() {
        let table = new_table(16);

        let entries: Vec<_> = table.iter(0).collect();

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn merge_after_resize() {
        let mut table = new_table(16);
        table.merge::<false, _>(42, 1u64, Count::single());
        table.merge::<false, _>(99, 2u64, Count::single());

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, 32);
        table.merge::<false, _>(42, 1u64, Count::single());
        table.merge::<false, _>(200, 3u64, Count::single());

        assert_eq!(table.len(), 3);
        let merged = table.iter(0).find(|e| *e.key() == 1).unwrap();
        assert_eq!(*merged.value(), Count(2));
        assert!(table.iter(0).any(|e| *e.key() == 3));
    }

    #[test]
    fn entry_at_returns_correct_slot() {
        let mut table = new_table(16);
        table.merge::<false, _>(42, 100u64, Count::single());

        let occupied: Vec<usize> = (0..table.capacity())
            .filter(|&i| table.entry_at(i).hash() != 0)
            .collect();

        assert_eq!(occupied.len(), 1);
        assert_eq!(*table.entry_at(occupied[0]).key(), 100);
    }

    #[test]
    fn iter_with_start_offset() {
        let mut table = new_table(128);
        for i in 0..20u64 {
            table.merge::<false, _>(i + 1, i, Count::single());
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
