//! Open-addressing hash table with linear probing, optimized for GROUP BY.
//!
//! See [`BaseHashTable`] for the full design rationale.

use crate::memory::MultiSlabBuffer;
use crate::memory::SlabAllocator;
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

/// Minimum fraction of a table's entry-buffer slabs that must come back already
/// zeroed (from the pre-zeroed pool) to use the **zeroed buffer + `hash == 0`
/// sentinel** scheme. At/above this, the dirty minority is memset (cheap) and we
/// skip the per-probe bitmap load. Below it — i.e. the pre-zeroed pool is
/// depleted, which happens for big, high-cardinality tables on a
/// bandwidth-bound box — we instead leave the buffer **dirty** and track empties
/// with an **occupancy bitmap**, avoiding a multi-gigabyte memset (the
/// "re-zeroing cliff"). This is decided per-table at allocation time from the
/// actual pool state, not statically from entry width.
const ZEROED_FRACTION_THRESHOLD: f64 = 0.5;

/// Remap a real zero hash to 1. The sentinel scheme needs this (a stored
/// `hash == 0` marks an empty slot); the bitmap scheme doesn't care, but it's
/// applied unconditionally — it's a no-op for non-zero hashes, deterministic
/// (so consume/merge stay consistent), and harmless under the bitmap (the hash
/// is only used for slot placement and the fast pre-compare, never as a
/// sentinel) — which removes a per-insert branch.
#[inline(always)]
fn remap_zero(hash: u64) -> u64 {
    if hash == 0 { 1 } else { hash }
}

/// Empty-slot tracking for a hash table: an occupancy bitmap, or the
/// `hash == 0` sentinel.
///
/// The bitmap is a plain heap `Vec<u64>`, deliberately *not* carved from the
/// ring like the entry buffer. The ring is `MADV_HUGEPAGE` (2 MB pages); a small
/// heap `Vec` is on 4 KB pages — and x86 has **separate TLBs for 2 MB vs 4 KB
/// pages**. The ~350 MB entry buffer already saturates the small 2 MB-page dTLB,
/// so a *huge-paged* bitmap competes for those same scarce entries and its
/// per-probe `bitmap[slot>>6]` access evicts the entry buffer's translations →
/// TLB-miss stalls. A 4 KB-paged bitmap uses the *separate* 4 KB dTLB the entry
/// buffer never touches, so there's no competition.
///
/// Proven on q32 by controlled experiment (this VM exposes no HW cache/TLB
/// counters): with the *same* mmap region toggled only via `madvise`,
/// `MADV_HUGEPAGE` ran ~1143 ms vs `MADV_NOHUGEPAGE` ~1123 ms — the slowdown
/// tracks **page size**, not alignment, allocation, or zeroing (perf shows the
/// huge-page variants burn ~+2.8 s task-clock of stalls with *fewer* faults).
/// The hot path never indexes the `Vec`; it derefs the cached
/// [`BaseHashTable::bitmap_ptr`] (no bounds check).
pub(crate) enum Occupancy {
    /// Sentinel scheme: no bitmap — empty slots are marked by `hash == 0`.
    Sentinel,
    /// Occupancy bitmap, one bit per slot (1 = occupied).
    Bitmap(Vec<u64>),
}

impl Occupancy {
    /// Allocate a zeroed occupancy bitmap sized for `capacity` slots.
    pub(crate) fn alloc_bitmap(capacity: usize) -> Self {
        Occupancy::Bitmap(vec![0u64; capacity.div_ceil(64)])
    }

    /// Whether the bitmap scheme is in use (vs the `hash == 0` sentinel).
    #[inline(always)]
    fn is_bitmap(&self) -> bool {
        !matches!(self, Occupancy::Sentinel)
    }

    /// Base pointer to the contiguous bitmap words, or null for the sentinel
    /// scheme. The `Vec`'s heap buffer is stable across moves of the enclosing
    /// struct, so this pointer is cached on the hot path.
    #[inline]
    fn base_ptr(&self) -> *mut u64 {
        match self {
            Occupancy::Sentinel => std::ptr::null_mut(),
            Occupancy::Bitmap(v) => v.as_ptr() as *mut u64,
        }
    }
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
    /// Empty-slot tracking: an occupancy bitmap (1 bit/slot, marks occupied), or
    /// the `hash == 0` sentinel. When the bitmap is used the (large) entry
    /// `buffer` is left **dirty** and never zeroed — only the tiny bitmap is
    /// zeroed, removing the per-query full-buffer memset that otherwise dominates
    /// hot iterations on a bandwidth-bound box whose ring can't keep the zeroed
    /// pool full. See [`Occupancy`].
    occupied: Occupancy,
    /// Cached base pointer of `occupied`'s contiguous bitmap words (null on the
    /// sentinel path). Lets the per-probe hot path read/write the bitmap with a
    /// single null check + direct deref — no enum dispatch, no bounds check —
    /// matching flat-array cost while keeping the storage off-heap. Kept in sync
    /// with `occupied` on construction and resize.
    bitmap_ptr: *mut u64,
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
        // Grab the entry buffer preferring zeroed pool memory but never memsetting,
        // then decide the empty-slot scheme from what we actually got:
        //  - single slab, or enough slabs already zeroed → zero the dirty minority
        //    (cheap) and use the `hash == 0` sentinel (no per-probe bitmap load);
        //  - otherwise (pre-zeroed pool depleted by a big table) → leave the
        //    buffer dirty and track empties with an occupancy bitmap, avoiding a
        //    multi-gigabyte memset.
        let mut buffer =
            allocator.create_multi_slab_buffer_lazy::<Entry<K, V>>(expected_capacity);
        let zeroed_frac = buffer.zeroed_slab_count() as f64 / buffer.slab_count() as f64;

        let occupied = if zeroed_frac >= ZEROED_FRACTION_THRESHOLD {
            // Enough already-zeroed slabs: memset the dirty minority (cheap) and
            // use the `hash == 0` sentinel — no per-probe bitmap load.
            buffer.zero_dirty_slabs();
            Occupancy::Sentinel
        } else {
            // Pre-zeroed pool depleted: leave the buffer dirty and track empties
            // with a zeroed bitmap, avoiding a large memset.
            Occupancy::alloc_bitmap(expected_capacity)
        };

        let bitmap_ptr = occupied.base_ptr();
        BaseHashTable {
            mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            buffer,
            occupied,
            bitmap_ptr,
            pre_shift,
            shift: u64::BITS - expected_capacity.trailing_zeros(),
            _phantom: PhantomData,
            collisions: 0,
        }
    }
}

impl<K: PersistedKey, V: Value, A: Index<usize, Output = Entry<K, V>> + IndexMut<usize>>
    BaseHashTable<K, V, A>
{
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

    /// Whether this table tracks empties with the occupancy bitmap (vs the
    /// `hash == 0` sentinel). Lets a caller allocate a matching bitmap before
    /// [`resize_with`](Self::resize_with).
    pub fn uses_bitmap(&self) -> bool {
        self.occupied.is_bitmap()
    }

    /// Map a hash to a slot index using the top bits: `(hash << pre_shift) >> shift`.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.pre_shift) >> self.shift) as usize
    }

    /// Whether slot `idx` holds an entry. Bitmap path reads the occupancy bitmap
    /// (entry bytes are dirty); sentinel path reads the entry's `hash` (`0` =
    /// empty). The branch is on a per-table-constant flag, so it predicts well.
    #[inline(always)]
    pub fn is_occupied(&self, idx: usize) -> bool {
        // One predicted null check + a direct deref of the cached, contiguous
        // bitmap (no enum dispatch, no bounds check). Null = sentinel scheme.
        if self.bitmap_ptr.is_null() {
            self.buffer[idx].hash != 0
        } else {
            unsafe { (*self.bitmap_ptr.add(idx >> 6) >> (idx & 63)) & 1 != 0 }
        }
    }

    /// Mark slot `idx` occupied (bitmap path only; the sentinel path marks a slot
    /// implicitly via the non-zero `hash` written into the entry).
    #[inline(always)]
    fn set_occupied(&mut self, idx: usize) {
        if !self.bitmap_ptr.is_null() {
            unsafe { *self.bitmap_ptr.add(idx >> 6) |= 1u64 << (idx & 63) };
        }
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
            // Bitmap path reads the occupancy word before the entry — prefetch it
            // too, else it's an un-hidden miss that regresses probe-heavy group-bys.
            if !self.bitmap_ptr.is_null() {
                let bm = self.bitmap_ptr.add(idx >> 6) as *const i8;
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(bm);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Prefetch the slot's cache line into L2 (`T1`) only. Issued *far* ahead of
    /// the access and paired with a nearer [`prefetch`](Self::prefetch) (L1) call, this
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
        hash: u64,
        key: L,
        value: V,
    ) {
        // Always remap a real zero hash to 1: required by the sentinel scheme,
        // a harmless no-op under the bitmap (see `remap_zero`).
        let hash = remap_zero(hash);
        let mut idx = self.slot_for(hash);

        loop {
            if !self.is_occupied(idx) {
                // Empty slot found - persist the key and insert
                let persisted = key.persist();
                self.buffer[idx] = Entry {
                    hash,
                    value,
                    key: persisted,
                };
                self.set_occupied(idx);
                self.length += 1;
                return;
            }

            let entry = &mut self.buffer[idx];
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
            if !self.bitmap_ptr.is_null() {
                let bm = self.bitmap_ptr.add(idx >> 6) as *const i8;
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(bm);
            }
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
    fn probe_row_for_slot<S: BatchRowSource<K, V>>(
        &mut self,
        i: usize,
        slot: usize,
        hash: u64,
        slots: &mut [usize],
        rows: &mut S,
    ) -> bool {
        if !self.is_occupied(slot) {
            let key = rows.persisted(i);
            let value = rows.value(i);
            self.buffer[slot] = Entry { hash, key, value };
            self.set_occupied(slot);
            self.length += 1;
            true
        } else if self.buffer[slot].hash == hash && rows.key_eq(i, &self.buffer[slot].key) {
            let value = rows.value(i);
            self.buffer[slot].value = self.buffer[slot].value.merge(value);
            true
        } else {
            self.collisions += 1;
            slots[i] = (slot + 1) & self.mask;
            false
        }
    }

    /// Insert/merge a whole batch of rows with a batched, multi-pass linear probe.
    ///
    /// The scalar [`merge`](Self::merge) walks one row's probe chain to the end
    /// before starting the next, so each cache-missing slot is a serial
    /// dependency. Instead, this advances every unresolved row by just **one**
    /// slot per pass, so the many probes in a pass hit independent slots and the
    /// CPU keeps their misses in flight.
    ///
    /// Steps:
    /// 1. **Pass 1** — for each row, compute its home slot and try to resolve it
    ///    there via [`probe_row_for_slot`](Self::probe_row_for_slot): claim an
    ///    empty slot (insert), or merge into a slot holding the same key. A row
    ///    that collides (slot held by a *different* key) has its probe advanced
    ///    one slot and its index pushed onto the `unresolved` worklist.
    /// 2. **Passes 2+** — re-probe only the `unresolved` rows at their advanced
    ///    slots; any that still collide spill into `unresolved_scratch`. Swap the
    ///    two worklists and repeat over the shrinking set until none remain.
    ///
    /// Re-reading the slot each pass keeps duplicate keys within the batch
    /// correct: the first row claims the slot, the rest take the equal-key merge.
    ///
    /// `slots`, `unresolved`, `unresolved_scratch` are caller scratch of length
    /// ≥ `length`; `hashes` holds the row hashes (0 is remapped to 1 in place to
    /// keep the empty-slot sentinel). The caller MUST ensure `capacity - len >
    /// length`, so every probe eventually finds an empty slot and the passes
    /// terminate.
    #[inline]
    pub fn merge_batch<S: BatchRowSource<K, V>>(
        &mut self,
        length: usize,
        hashes: &mut [u64],
        slots: &mut [usize],
        unresolved: &mut [u32],
        unresolved_scratch: &mut [u32],
        rows: &mut S,
    ) {
        /// How far ahead, in row positions, to prefetch each slot.
        const PREFETCH_DIST: usize = 16;

        // Pass 1: probe every row at its home slot, computing the slot inline.
        // Rows that resolve here (insert or merge) never touch the scratch
        // arrays; only collided rows are recorded — advanced slot in `slots[i]`,
        // index onto `unresolved` — for the follow-up passes. Folding the slot
        // computation in here avoids materialising a slot for *every* row, which
        // is pure overhead in the common low-collision / merge-heavy case.
        let mut collided = 0usize;
        for i in 0..length {
            if i + PREFETCH_DIST < length {
                self.prefetch_entry(self.slot_for(hashes[i + PREFETCH_DIST]));
            }
            // Always remap zero (required by sentinel, harmless under bitmap).
            let hash = remap_zero(hashes[i]);
            hashes[i] = hash;
            let slot = self.slot_for(hash);
            if !self.probe_row_for_slot(i, slot, hash, slots, rows) {
                unresolved[collided] = i as u32;
                collided += 1;
            }
        }

        // Passes 2+: walk only the still-unresolved rows, ping-ponging the
        // worklist between the two scratch buffers. Each pass advances every row
        // by one slot, keeping their probes independent so misses stay in flight.
        let (mut work, mut spill) = (unresolved, unresolved_scratch);
        let mut remaining = collided;
        while remaining > 0 {
            let mut collided = 0usize;
            for k in 0..remaining {
                if k + PREFETCH_DIST < remaining {
                    self.prefetch_entry(slots[work[k + PREFETCH_DIST] as usize]);
                }
                let i = work[k] as usize;
                if !self.probe_row_for_slot(i, slots[i], hashes[i], slots, rows) {
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
    /// `new_occupied` must be a zeroed bitmap iff this table uses the bitmap
    /// scheme (and [`Occupancy::Sentinel`] for the sentinel scheme, where
    /// `buffer` must be zeroed); the resize preserves the table's existing
    /// scheme. The caller owns allocation since `resize_with` has no allocator.
    #[cold]
    pub fn resize_with(&mut self, buffer: A, new_occupied: Occupancy, new_size: usize) {
        debug!("Resizing map to {:?}...", new_size);
        debug_assert_eq!(new_occupied.is_bitmap(), self.occupied.is_bitmap());
        let old_mask = self.mask;
        self.collisions = self.length;
        self.mask = new_size - 1;
        self.shift = u64::BITS - new_size.trailing_zeros();

        let old_buffer = mem::replace(&mut self.buffer, buffer);
        let old_occupied = mem::replace(&mut self.occupied, new_occupied);
        // Re-cache the hot-path bitmap pointer for the new (post-resize) bitmap
        // before any is_occupied/set_occupied below uses it.
        self.bitmap_ptr = self.occupied.base_ptr();
        let old_bitmap = old_occupied.base_ptr();

        for idx in 0..=old_mask {
            // Occupancy in the OLD table: bitmap path reads the old bitmap;
            // sentinel path reads the old entry's hash.
            let occupied = if old_bitmap.is_null() {
                old_buffer[idx].hash != 0
            } else {
                unsafe { (*old_bitmap.add(idx >> 6) >> (idx & 63)) & 1 != 0 }
            };
            if !occupied {
                continue;
            }
            let entry = &old_buffer[idx];
            let mut new_idx = self.slot_for(entry.hash);
            loop {
                if !self.is_occupied(new_idx) {
                    self.buffer[new_idx] = *entry;
                    self.set_occupied(new_idx);
                    break;
                }
                new_idx = (new_idx + 1) & self.mask;
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
            let idx = self.idx;
            self.idx += 1;
            if self.hash_table.is_occupied(idx) {
                return Some(&self.hash_table.buffer[idx]);
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
        fn merge(self, v: Self) -> Self {
            Count(self.0 + v.0)
        }
    }

    type TestTable = BaseHashTable<u64, Count, Vec<Entry<u64, Count>>>;

    fn new_table(capacity: usize) -> TestTable {
        // Vec-backed test tables use the zeroed-buffer + `hash == 0` sentinel
        // scheme (no bitmap); the bitmap path needs ring-backed slabs and is
        // covered end-to-end by the query oracle tests.
        let buffer = vec![Entry::default(); capacity];
        BaseHashTable {
            mask: capacity - 1,
            shift: u64::BITS - capacity.trailing_zeros(),
            collisions: 0,
            pre_shift: 0,
            buffer,
            occupied: Occupancy::Sentinel,
            bitmap_ptr: std::ptr::null_mut(),
            length: 0,
            max_load: max_load_for_len(capacity),
            _phantom: PhantomData,
        }
    }

    #[test]
    fn insert_single_entry() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 100u64, Count(1));

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key(), 100);
        assert_eq!(*entry.value(), Count(1));
    }

    #[test]
    fn merge_duplicate_keys_sums_values() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 100u64, Count(1));
        table.merge::<false, _>(42, 100u64, Count(1));
        table.merge::<false, _>(42, 100u64, Count(1));

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.value(), Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 1u64, Count(1));
        table.merge::<false, _>(42, 2u64, Count(1));

        assert_eq!(table.len(), 2);
        let entries: Vec<_> = table.iter(0).collect();
        let keys: Vec<u64> = entries.iter().map(|e| *e.key()).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn hash_zero_is_stored_and_retrievable() {
        // A real zero hash is handled invisibly (remapped on the sentinel path,
        // stored as-is on the bitmap path) and the entry stays retrievable.
        let mut table = new_table(16);

        table.merge::<false, _>(0, 99u64, Count(1));

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key(), 99);
    }

    #[test]
    fn undersized_triggers_at_load_factor() {
        let mut table = new_table(16);
        let max_load = (16.0 * MAX_LOAD_FACTOR).round() as usize;

        for i in 0..max_load {
            table.merge::<false, _>(i as u64 + 1, i as u64, Count(1));
            assert!(!table.undersized());
        }

        table.merge::<false, _>(max_load as u64 + 1, max_load as u64, Count(1));

        assert!(table.undersized());
    }

    #[test]
    fn collision_counting() {
        let mut table = new_table(16);

        table.merge::<true, _>(42, 1u64, Count(1));
        assert_eq!(table.collisions(), 0);

        table.merge::<true, _>(42, 2u64, Count(1));
        assert_eq!(table.collisions(), 1);
    }

    #[test]
    fn collision_counting_disabled() {
        let mut table = new_table(16);

        table.merge::<false, _>(42, 1u64, Count(1));
        table.merge::<false, _>(42, 2u64, Count(1));

        assert_eq!(table.collisions(), 0);
    }

    #[test]
    fn iter_skips_empty_slots() {
        let mut table = new_table(128);

        table.merge::<false, _>(1, 10u64, Count(1));
        table.merge::<false, _>(2, 20u64, Count(1));

        let entries: Vec<_> = table.iter(0).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.hash() != 0));
    }

    #[test]
    fn resize_preserves_all_entries() {
        let mut table = new_table(16);
        for i in 0..8u64 {
            table.merge::<false, _>(i + 1, i, Count(1));
        }

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, Occupancy::Sentinel, 32);

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
        table.merge::<true, _>(42, 1u64, Count(1));
        table.merge::<true, _>(42, 2u64, Count(1));
        let pre_resize_collisions = table.collisions();
        assert!(pre_resize_collisions > 0);

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, Occupancy::Sentinel, 32);

        assert_eq!(table.collisions(), table.len());
    }

    #[test]
    fn many_entries_all_retrievable() {
        let mut table = new_table(256);

        for i in 0..100u64 {
            table.merge::<false, _>(i + 1, i, Count(1));
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

        table.merge::<false, _>(last_slot_hash, 1u64, Count(1));
        table.merge::<false, _>(last_slot_hash, 2u64, Count(1));
        table.merge::<false, _>(last_slot_hash, 3u64, Count(1));

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
        table.merge::<false, _>(42, 1u64, Count(1));
        table.merge::<false, _>(99, 2u64, Count(1));

        let new_buf = vec![Entry::default(); 32];
        table.resize_with(new_buf, Occupancy::Sentinel, 32);
        table.merge::<false, _>(42, 1u64, Count(1));
        table.merge::<false, _>(200, 3u64, Count(1));

        assert_eq!(table.len(), 3);
        let merged = table.iter(0).find(|e| *e.key() == 1).unwrap();
        assert_eq!(*merged.value(), Count(2));
        assert!(table.iter(0).any(|e| *e.key() == 3));
    }

    #[test]
    fn entry_at_returns_correct_slot() {
        let mut table = new_table(16);
        table.merge::<false, _>(42, 100u64, Count(1));

        let occupied: Vec<usize> = (0..table.capacity())
            .filter(|&i| table.is_occupied(i))
            .collect();

        assert_eq!(occupied.len(), 1);
        assert_eq!(*table.entry_at(occupied[0]).key(), 100);
    }

    #[test]
    fn iter_with_start_offset() {
        let mut table = new_table(128);
        for i in 0..20u64 {
            table.merge::<false, _>(i + 1, i, Count(1));
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
