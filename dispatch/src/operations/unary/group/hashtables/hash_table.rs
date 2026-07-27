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

/// A borrowed read of one occupied table entry: its hash, persisted key, and
/// stored value. What [`BaseHashTable::iter`] yields and
/// [`TableReader::view_at`] returns. A view rather than a struct reference
/// because an entry is a byte region at a per-table stride, not a Rust struct
/// (the stride is a runtime value; see [`BaseHashTable`]).
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

/// The magic reciprocal for dividing by `d` with a widening multiply:
/// `n / d == (n * reciprocal(d)) >> 64` — the same strength reduction the
/// compiler applies to a compile-time divisor.
///
/// Exact for this table's ranges: with `m = ceil(2^64 / d) = (2^64 + e) / d`
/// (`0 <= e < d`), `n * m / 2^64 = n/d + n*e/(d * 2^64)`, and the error term
/// stays below `n / 2^64`. The true quotient's fractional part is at most
/// `1 - 1/d`, so the floor can only be pushed over when `n / 2^64 >= 1/d`,
/// i.e. `n >= 2^64 / d`. Here `d = entries_per_slab < 2^21` (a slab is 2MB and
/// an entry at least 8 bytes... at least 2 entries per slab), so the result is
/// exact for every `n < 2^43` — far above any slot index.
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

/// Each slab's base address minus its first slot's global byte offset
/// (`s * entries_per_slab * stride`), so an entry address is
/// `bases[slab] + index * stride` with no per-access subtract (see
/// [`BaseHashTable::entry_ptr_in`]). Wrapping: an adjusted base may
/// arithmetically precede its mapping; only the re-added sum is dereferenced.
fn adjusted_bases(slabs: &[Slab], entries_per_slab: usize, stride: usize) -> Vec<usize> {
    slabs
        .iter()
        .enumerate()
        .map(|(s, slab)| (slab.ptr as usize).wrapping_sub(s * entries_per_slab * stride))
        .collect()
}

/// The runtime layout of one entry: where each field starts, the entry stride,
/// and the strictest field alignment. Fields are placed in descending alignment
/// order (ties keep hash, key, value order), mirroring the padding-minimising
/// layout the compiler gives a sized entry struct, so a fixed-arity value's
/// entry is exactly as large as it always was.
struct EntryLayout {
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    stride: usize,
    align: usize,
}

/// The exact bytes one table entry occupies for this key type and signature,
/// for entry-footprint heuristics (merge-partition sizing).
pub fn entry_stride<K, V: AggregationValue>(ctx: &V::SharedContext) -> usize {
    entry_layout::<K, V>(V::stored_meta(ctx)).stride
}

fn entry_layout<K, V: AggregationValue>(meta: V::StoredMeta) -> EntryLayout {
    // (alignment, size) per field, in hash, key, value order.
    let fields = [
        (align_of::<u64>(), size_of::<u64>()),
        (align_of::<K>(), size_of::<K>()),
        (V::stored_align(meta), V::stored_size(meta)),
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

/// A linear probing hash table optimized for never rehashing, exposing a very raw interface allowing
/// maximum control by the caller.
///
/// # Memory Layout
///
/// Entries live in one or more 2MB slabs as raw byte regions at a fixed
/// per-table `stride`, each holding the hash, the persisted key, and the
/// group's stored aggregation value:
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                    Slabs: entries at `stride` bytes each                │
/// ├─────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
/// │  entry 0    │  entry 1    │             │  entry 3    │      ...       │
/// │ hash|key|val│ hash|key|val│             │ hash|key|val│                │
/// └─────────────┴─────────────┴─────────────┴─────────────┴────────────────┘
///  stride = 8 bytes + key + stored value, fields in descending-alignment
///  order (e.g. 24 bytes for an `Int64` key with one `i64` cell)
/// ```
///
/// The stride is a *runtime* value computed at construction rather than a
/// compile-time `size_of`, because a stored value's size may itself be fixed
/// only at query build time: a runtime-arity signature ([`Variable`]) stores
/// `n` cells inline per entry, `n` constant per query but unknown to the
/// compiler. Fixed-arity values get the same field offsets and stride a sized
/// entry struct had; they simply pay the stride multiply at probe time.
///
/// Each entry contains:
/// - `hash: u64` - Full 64-bit hash (0 = empty slot sentinel)
/// - `key: K` - The persisted key (e.g., ArenaKey with pointer + length)
/// - value - The stored aggregation value (see [`AggregationValue::Stored`])
///
/// Entries never straddle a slab boundary: each slab holds
/// `BUFFER_SIZE / stride` whole entries, with the leftover tail bytes unused.
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
///    finding an empty slot (hash == 0) or matching key. On match, fold values
///    instead of inserting.
///
/// 2. **Empty detection**: `hash == 0` marks empty slots. Real zero hashes are
///    converted to 1 to preserve this invariant.
///
/// 3. **Resize**: Handled externally — callers check load pressure and call
///    [`resize`](BaseHashTable::resize) to rehash into fresh slabs.
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
/// Each entry packs hash + key + value into one region (e.g. 32 bytes
/// for `ArenaKey` + a count = 2 entries per 64-byte cache line). A single
/// cache-line fetch gives the hash for comparison AND the next linear-probe
/// candidate. Swiss Tables require two separate memory accesses per probe:
/// one for the ctrl byte and one for the slot data.
pub struct BaseHashTable<K: PersistedKey, V: AggregationValue> {
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
    /// Magic reciprocal of `entries_per_slab` (see [`reciprocal`]), so the
    /// per-probe slab lookup divides with a widening multiply instead of a
    /// hardware `udiv`. A compile-time divisor gets this strength reduction
    /// from the compiler; a runtime one must carry it itself, or the divide's
    /// latency lands on the probe's address computation, in front of the
    /// entry load it feeds. Measured +70% on a hash-only distinct count.
    entries_per_slab_magic: u64,
    /// Bytes per entry; see the layout docs above.
    stride: usize,
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    /// Strictest field alignment, for allocating replacement slabs on resize.
    align: usize,
    /// The value's runtime view metadata (slot count for a runtime-arity
    /// signature); held once here, never per entry.
    meta: V::StoredMeta,
    /// Each slab's base address pre-adjusted by its first slot's byte offset
    /// (`slabs[s].ptr - s * entries_per_slab * stride`), so an entry address is
    /// `bases[slab] + index * stride`: the slab-index multiply and the byte
    /// multiply are independent and run in parallel, with no subtract on the
    /// dependency chain, matching the latency of a compile-time stride.
    bases: Vec<usize>,
    slabs: Vec<Slab>,
    length: usize,
    max_load: usize,
    _phantom: PhantomData<(K, V)>,
}

unsafe impl<K: PersistedKey, V: AggregationValue> Send for BaseHashTable<K, V> {}

impl<K: PersistedKey, V: AggregationValue> BaseHashTable<K, V> {
    /// Creates a new table with `expected_capacity` slots (must be a power of
    /// 2), its entry layout derived from the key type and the value's runtime
    /// metadata off `ctx`. The slabs are zeroed so all slots start empty
    /// (`hash == 0`).
    pub fn new(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        pre_shift: u32,
        ctx: &V::SharedContext,
    ) -> Self {
        let meta = V::stored_meta(ctx);
        let layout = entry_layout::<K, V>(meta);
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
            key_offset: layout.key_offset,
            value_offset: layout.value_offset,
            align: layout.align,
            meta,
            bases: adjusted_bases(&slabs, entries_per_slab, layout.stride),
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

    /// The address of the entry at `index` in the slabs behind `bases` (with
    /// this table's layout). Entries never straddle slabs: entry `i` lives in
    /// slab `i / entries_per_slab` at byte offset
    /// `(i % entries_per_slab) * stride`, but through the pre-adjusted bases
    /// this is one reciprocal multiply and one independent byte multiply — no
    /// hardware divide and no subtract on the address dependency chain, which
    /// every probe's entry load waits on.
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
            geo: self.geometry(),
            idx: start_offset,
            _table: PhantomData,
        }
    }

    /// Returns `true` if the table has exceeded its [`MAX_LOAD_FACTOR`] threshold.
    pub fn undersized(&self) -> bool {
        self.len() > self.max_load
    }

    /// A register-resident snapshot of this table's entry layout and probe
    /// parameters (see [`Geometry`]).
    #[inline(always)]
    fn geometry(&self) -> Geometry<K, V> {
        Geometry {
            magic: self.entries_per_slab_magic,
            stride: self.stride,
            hash_offset: self.hash_offset,
            key_offset: self.key_offset,
            value_offset: self.value_offset,
            mask: self.mask,
            shift: self.shift,
            pre_shift: self.pre_shift,
            bases: self.bases.as_ptr(),
            bases_len: self.bases.len(),
            meta: self.meta,
            _phantom: PhantomData,
        }
    }

    /// A probing handle that holds the table's [`Geometry`] in locals for the
    /// duration of a consume window, so the per-row probe reads its layout
    /// constants from registers (see [`Prober`]).
    #[inline(always)]
    pub fn prober(&mut self) -> Prober<'_, K, V> {
        let geo = self.geometry();
        Prober { table: self, geo }
    }

    /// A read handle with the same register-resident [`Geometry`], for the
    /// merge phase's linear slot scans.
    #[inline(always)]
    pub fn reader(&self) -> TableReader<'_, K, V> {
        TableReader {
            geo: self.geometry(),
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

/// A register-resident snapshot of a table's entry layout and probe
/// parameters. These are all runtime values (the whole point of the strided
/// table), so hot loops that touched them through `&self` would re-load them
/// from memory constantly — the probe writes entries through raw pointers,
/// which the compiler must assume may alias the table's own fields. A
/// `Geometry` is plain `Copy` locals, immune to that: snapshotted once per
/// call (or once per consume window via [`Prober`]), it keeps the layout in
/// registers exactly as a compile-time layout would be immediates.
///
/// The `bases` pointer is valid while the table's slabs are untouched; every
/// holder ties itself to the table with a borrow, and nothing resizes a table
/// while a snapshot of it is live.
struct Geometry<K, V: AggregationValue> {
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
    meta: V::StoredMeta,
    _phantom: PhantomData<K>,
}

impl<K, V: AggregationValue> Clone for Geometry<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V: AggregationValue> Copy for Geometry<K, V> {}

impl<K, V: AggregationValue> Geometry<K, V> {
    /// Map a hash to a slot index using the top bits.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.pre_shift) >> self.shift) as usize
    }

    /// The address of the entry at `index`; see
    /// [`BaseHashTable::entry_ptr_in`].
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        let slab_idx = fast_div(index, self.magic);
        assert!(slab_idx < self.bases_len);
        let base = unsafe { *self.bases.add(slab_idx) };
        base.wrapping_add(index * self.stride) as *mut u8
    }

    /// The view of the occupied entry at `entry`'s address, with `hash`
    /// already read.
    #[inline(always)]
    unsafe fn view<'a>(&self, entry: *const u8, hash: u64) -> EntryView<'a, K, V::Stored> {
        unsafe {
            EntryView {
                hash,
                key: &*(entry.add(self.key_offset) as *const K),
                stored: V::stored_ref(entry.add(self.value_offset), self.meta),
            }
        }
    }
}

/// A probing handle over a mutably borrowed table, carrying its [`Geometry`]
/// in locals. The consume loop creates one per active table and probes through
/// it row after row, so the layout constants are read once per window, not
/// re-loaded per row and per probe step.
pub struct Prober<'t, K: PersistedKey, V: AggregationValue> {
    table: &'t mut BaseHashTable<K, V>,
    geo: Geometry<K, V>,
}

impl<K: PersistedKey, V: AggregationValue> Prober<'_, K, V> {
    /// Prefetch the hash table slot where `hash` would land, plus the next
    /// cache lines to cover short probe chains. Brings the lines all the way
    /// into L1 (`T0`) — use this *near* the access (small lookahead).
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let ptr = self.geo.entry_ptr(self.geo.slot_for(hash)) as *const u8;
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
        prefetch_l2_line(self.geo.entry_ptr(self.geo.slot_for(hash)) as *const u8);
    }

    /// See [`BaseHashTable::undersized`].
    #[inline(always)]
    pub fn undersized(&self) -> bool {
        self.table.undersized()
    }

    /// Insert or fold an *owned* value — the radix scatter rows' aggregation
    /// pass. A new key stores the value; an existing one folds it in via the
    /// value's own scatter-row fold. Thin wrapper over
    /// [`probe_fold`](Self::probe_fold), carrying the value in as the context.
    #[inline(always)]
    pub fn merge<const COUNT_COLLISIONS: bool, L>(
        &mut self,
        hash: u64,
        key: L,
        value: V,
        ctx: &V::SharedContext,
    ) where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<COUNT_COLLISIONS, L, V, _, _>(
            hash,
            key,
            value,
            |value, stored| V::store(stored, value),
            |value, stored| V::merge_value(stored, value, ctx),
        );
    }

    /// Insert or fold a *stored* value read from another table's entry — the
    /// partition merge and the node merge. A new key copies the partial in; an
    /// existing one combines the two partials in place. Thin wrapper over
    /// [`probe_fold`](Self::probe_fold).
    #[inline(always)]
    pub fn merge_from<const COUNT_COLLISIONS: bool, L>(
        &mut self,
        hash: u64,
        key: L,
        src: &V::Stored,
        ctx: &V::SharedContext,
    ) where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<COUNT_COLLISIONS, L, &V::Stored, _, _>(
            hash,
            key,
            src,
            |src, stored| V::clone_stored(stored, src),
            |src, stored| V::merge_stored(stored, src, ctx),
        );
    }

    /// Grow the table 4x (updating `cap`) if it has crossed its load threshold,
    /// re-snapshotting the geometry the resize invalidated. The merge's
    /// safety-net growth, kept on the prober so the per-row fold loop needn't
    /// give up its snapshot.
    #[inline(always)]
    pub fn grow_if_full(&mut self, allocator: &mut SlabAllocator, cap: &mut usize) {
        if self.table.undersized() {
            *cap *= 4;
            self.table.resize(allocator, *cap);
            self.geo = self.table.geometry();
        }
    }

    /// Double the table if cumulative collision pressure is too high (see the
    /// merge's resize ratio), re-snapshotting the geometry on resize.
    /// `collision_ratio` is the integer threshold: resize once
    /// `collisions > collision_ratio * len`.
    #[inline(always)]
    pub fn resize_on_collisions(&mut self, allocator: &mut SlabAllocator, collision_ratio: usize) {
        if self.table.collisions() > self.table.len() * collision_ratio {
            let new_size = self.table.capacity() << 1;
            self.table.resize(allocator, new_size);
            self.geo = self.table.geometry();
        }
    }

    /// Probe for `hash`/`key`, then initialise or fold the value at the
    /// matched slot — one probe-and-fold pass, the stored-value reference
    /// handed straight from the matched entry (no second slot lookup). A
    /// freshly inserted slot calls `seed` (its value bytes are zeroed until
    /// then); an existing one calls `update`. Splitting the two avoids a
    /// per-row `is_new` branch and lets each do only its own work (a string
    /// extreme's `update` can skip persisting a loser).
    ///
    /// `ctx` is the one per-call value both arms might need — moved into
    /// whichever arm runs, so they needn't both capture it: the merge phase
    /// passes the already-materialised value; the consume path passes
    /// `&mut value_arena`. Because the key is persisted before either arm
    /// runs, and keys and values live in separate memory, the closure's
    /// value-arena borrow never aliases the key's.
    ///
    /// Since `hash == 0` is the empty sentinel, a key that legitimately
    /// hashes to 0 is stored with `hash = 1`; invisible to callers. When
    /// `COUNT_COLLISIONS` is true, each probe step increments the collision
    /// counter, so callers can resize on cumulative probe-chain pressure. At
    /// the 70% max load, the expected probe length is ~1.8 slots.
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
        S: FnOnce(X, &mut V::Stored),
        U: FnOnce(X, &mut V::Stored),
    {
        let geo = self.geo;
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1.
        if hash == 0 {
            hash = 1;
        }
        let mut idx = geo.slot_for(hash);
        loop {
            let entry = geo.entry_ptr(idx);
            unsafe {
                let hash_ptr = entry.add(geo.hash_offset) as *mut u64;
                if *hash_ptr == 0 {
                    *hash_ptr = hash;
                    (entry.add(geo.key_offset) as *mut K).write(key.persist());
                    self.table.length += 1;
                    seed(ctx, V::stored_mut(entry.add(geo.value_offset), geo.meta));
                    return;
                }
                if *hash_ptr == hash && key.eq_persisted(&*(entry.add(geo.key_offset) as *const K))
                {
                    update(ctx, V::stored_mut(entry.add(geo.value_offset), geo.meta));
                    return;
                }
            }
            if COUNT_COLLISIONS {
                self.table.collisions += 1;
            }
            idx = (idx + 1) & geo.mask;
        }
    }
}

/// A read handle over a borrowed table, carrying its [`Geometry`] in locals:
/// the merge phase's linear slot scans read thousands of consecutive slots, so
/// they too keep the layout in registers.
pub struct TableReader<'a, K: PersistedKey, V: AggregationValue> {
    geo: Geometry<K, V>,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue> TableReader<'a, K, V> {
    /// See [`BaseHashTable::hash_at`].
    #[inline(always)]
    pub fn hash_at(&self, index: usize) -> u64 {
        unsafe { *(self.geo.entry_ptr(index).add(self.geo.hash_offset) as *const u64) }
    }

    /// See [`BaseHashTable::view_at`].
    #[inline(always)]
    pub fn view_at(&self, index: usize) -> EntryView<'a, K, V::Stored> {
        let entry = self.geo.entry_ptr(index);
        unsafe {
            let hash = *(entry.add(self.geo.hash_offset) as *const u64);
            self.geo.view(entry, hash)
        }
    }
}

/// An iterator over the non-empty entries in a `BaseHashTable`.
///
/// Created by [`BaseHashTable::iter`]. Yields [`EntryView`]s of entries where
/// `hash != 0` in arbitrary order (based on slot positions, not insertion order).
pub struct HashTableIterator<'a, K: PersistedKey, V: AggregationValue> {
    geo: Geometry<K, V>,
    idx: usize,
    _table: PhantomData<&'a BaseHashTable<K, V>>,
}

impl<'a, K: PersistedKey, V: AggregationValue> Iterator for HashTableIterator<'a, K, V> {
    type Item = EntryView<'a, K, V::Stored>;

    fn next(&mut self) -> Option<Self::Item> {
        let geo = self.geo;
        while self.idx < geo.mask + 1 {
            // One address computation per slot: the hash check and the yielded
            // view read the same entry pointer.
            let entry = geo.entry_ptr(self.idx);
            self.idx += 1;
            unsafe {
                let hash = *(entry.add(geo.hash_offset) as *const u64);
                if hash != 0 {
                    return Some(geo.view(entry, hash));
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
    use crate::operations::unary::group::values::{AggregationSlot, OwnedValue, ValueColumns};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::Field;

    /// A minimal additive value for exercising the probe mechanics: merging two
    /// of these sums their counts.
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    impl OwnedValue for Count {
        type Reader<'b> = ();
        type SharedContext = ();
        type Columns = CountColumns;
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
    struct CountColumns;

    impl ValueColumns for CountColumns {
        type Value = Count;
        type Context = ();
        fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _context: &()) -> Self {
            CountColumns
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

        table.prober().merge::<false, _>(42, 100u64, Count(1), &());

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

        table.prober().merge::<false, _>(42, 100u64, Count(1), &());
        table.prober().merge::<false, _>(42, 100u64, Count(1), &());
        table.prober().merge::<false, _>(42, 100u64, Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.stored, Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table.prober().merge::<false, _>(42, 1u64, Count(1), &());
        table.prober().merge::<false, _>(42, 2u64, Count(1), &());

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

        table.prober().merge::<false, _>(0, 99u64, Count(1), &());

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
                .merge::<false, _>(i as u64 + 1, i as u64, Count(1), &());
            assert!(!table.undersized());
        }

        table
            .prober()
            .merge::<false, _>(max_load as u64 + 1, max_load as u64, Count(1), &());

        assert!(table.undersized());
    }

    #[test]
    fn collision_counting() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table.prober().merge::<true, _>(42, 1u64, Count(1), &());
        assert_eq!(table.collisions(), 0);

        table.prober().merge::<true, _>(42, 2u64, Count(1), &());
        assert_eq!(table.collisions(), 1);
    }

    #[test]
    fn collision_counting_disabled() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table.prober().merge::<false, _>(42, 1u64, Count(1), &());
        table.prober().merge::<false, _>(42, 2u64, Count(1), &());

        assert_eq!(table.collisions(), 0);
    }

    #[test]
    fn iter_skips_empty_slots() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);

        table.prober().merge::<false, _>(1, 10u64, Count(1), &());
        table.prober().merge::<false, _>(2, 20u64, Count(1), &());

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
            table.prober().merge::<false, _>(i + 1, i, Count(1), &());
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
        table.prober().merge::<true, _>(42, 1u64, Count(1), &());
        table.prober().merge::<true, _>(42, 2u64, Count(1), &());
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
            table.prober().merge::<false, _>(i + 1, i, Count(1), &());
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
            .merge::<false, _>(last_slot_hash, 1u64, Count(1), &());
        table
            .prober()
            .merge::<false, _>(last_slot_hash, 2u64, Count(1), &());
        table
            .prober()
            .merge::<false, _>(last_slot_hash, 3u64, Count(1), &());

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
        table.prober().merge::<false, _>(42, 1u64, Count(1), &());
        table.prober().merge::<false, _>(99, 2u64, Count(1), &());

        table.resize(&mut allocator, 32);
        table.prober().merge::<false, _>(42, 1u64, Count(1), &());
        table.prober().merge::<false, _>(200, 3u64, Count(1), &());

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
        table.prober().merge::<false, _>(42, 100u64, Count(1), &());

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
            table.prober().merge::<false, _>(i + 1, i, Count(1), &());
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
