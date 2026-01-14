use crate::operations::unary::group::allocation::Allocation;
use std::alloc::{Layout, alloc_zeroed};
use std::marker::PhantomData;
use std::mem;

pub const MAX_LOAD_FACTOR: f64 = 0.7;

/// A `LiveKey` is a key that has *not* yet been persisted, and therefore needs to explicitly be
/// persisted if it is saved within the HashTable. The HashTable may decide not to persist a key
/// if, for example, the same key already exists.
pub trait LiveKey: PartialEq<Self::Persisted> {
    type Persisted: PersistedKey;

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

    fn persist(self) -> Self::Persisted {
        self
    }
}

pub trait Value: Copy + Clone + Default {
    fn merge(self, v: Self) -> Self;
}

#[derive(Copy, Clone, Default)]
pub struct Entry<K: PersistedKey, V: Value> {
    hash: u64,
    key: K,
    value: V,
}

impl<K: PersistedKey, V: Value> Entry<K, V> {
    #[inline]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

fn max_load_for_len(len: usize) -> usize {
    (len as f64 * MAX_LOAD_FACTOR).round() as usize
}

/// A hash table optimized for two things:
///    1. Merging in large keys (re-hashing is too time-consuming).
///    2. Working with many HashTables (optimized to make the least amount of memory-access)
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
/// There is also always a power of 2 entries available in a given HashMap. This allows us to
/// compute a slot for a new entry by using `slot = hash & mask` instead of a more costly `%`.
///
/// # How It Works
///
/// Uses open addressing with linear probing:
///
/// 1. **Insert/Merge**: Compute `slot = hash & mask` (see why this is possible above).
///    If occupied and different key, probe linearly (slot+1, slot+2, ...) until finding an empty
///    slot (hash == 0) or matching key. On match, merge values instead of inserting.
///
/// 2. **Empty detection**: `hash == 0` marks empty slots. Real zero hashes are
///    converted to 1 to preserve this invariant.
///
/// 3. **Resize**: When length exceeds 70% of capacity, double the table and
///    relocate Entries using linear probing in the new larger table. This re-uses the same hash,
///    and is expected to be aggressively efficient for CPU caches.
///
/// # Why This Can Be Faster Than Swiss Tables (hashbrown, default std)
///
/// Swiss Tables use a two-array layout for SIMD-accelerated probing:
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
/// Swiss Tables win when:
/// - Keys are small and cheap to hash (can recompute instead of store)
/// - SIMD ctrl-byte probing avoids most data accesses
/// - Single hash table (both arrays likely in cache)
///
/// **This table wins for partitioned GROUP BY with large string keys:**
///
/// 1. **Full hash is required anyway**: Large strings (URLs, paths) are expensive
///    to hash. We store the full 64-bit hash to avoid rehashing during resize and
///    merge. Swiss Tables only store 7 bits (h2) in ctrl, requiring either rehash
///    or separate hash storage—losing their space advantage.
///
/// 2. **256 partitions kills two-array locality**: With 256 hash tables per worker,
///    Swiss Tables suffer two cache misses per probe:
///    - First: load ctrl byte (cache miss - ctrl arrays scattered across tables)
///    - Second: load slot data (another miss - data arrays also scattered)
///
///    With inline entries, ONE cache miss gives hash + key + value together.
///    The second entry in the cache line is the next probe target (linear probing).
///
/// # Generic Allocation
///
/// The `Allocation<T>` trait allows swapping the backing storage:
/// - `Vec<Entry<K,V>>`: Standard heap allocation with `alloc_zeroed` for alignment
/// - Custom mmap-backed allocations for huge tables or specialized memory management
pub struct HashTable<K: PersistedKey, V: Value, A: Allocation<Entry<K, V>>> {
    mask: usize,
    buffer: A,
    length: usize,
    max_load: usize,
    _phantom: PhantomData<(K, V)>,
}

impl<K: PersistedKey, V: Value> HashTable<K, V, Vec<Entry<K, V>>> {
    /// Creates a new HashTable with Vec-backed storage, sized to hold `expected_capacity`
    /// entries without resizing.
    ///
    /// The actual allocation is larger than `expected_capacity` because:
    /// 1. We maintain a 70% max load factor to keep probe chains short
    /// 2. We round up to a power of 2 for fast modulo via bitmask
    ///
    /// # Memory Allocation
    ///
    /// Uses `alloc_zeroed` with 64-byte (cache line) alignment. Zero-initialized memory
    /// is critical: `hash == 0` is our empty slot sentinel, so all slots start empty.
    ///
    /// # Example
    ///
    /// `expected_capacity = 100` → request 143 (100/0.7) → allocate 256 (next power of 2)
    /// This gives us 179 usable slots (256 * 0.7) before resize triggers.
    pub fn new(expected_capacity: usize) -> Self {
        // Request more than expected_capacity to stay under 70% load factor
        let capacity_to_request =
            ((expected_capacity as f64) * (1f64 / MAX_LOAD_FACTOR)).round() as usize;
        let length = capacity_to_request.next_power_of_two();

        const CACHE_LINE: usize = 64;
        let size_in_bytes = size_of::<Entry<K, V>>() * length;

        let layout =
            Layout::from_size_align(size_in_bytes, CACHE_LINE.max(align_of::<Entry<K, V>>()))
                .expect("Invalid layout");

        let entries = unsafe {
            let ptr = alloc_zeroed(layout) as *mut Entry<K, V>;

            if ptr.is_null() {
                panic!("Allocation failed");
            }

            Vec::from_raw_parts(ptr, length, length)
        };

        HashTable {
            mask: length - 1,
            length: 0,
            max_load: max_load_for_len(entries.len()),
            buffer: entries,
            _phantom: PhantomData,
        }
    }
}

impl<K: PersistedKey, V: Value, A: Allocation<Entry<K, V>>> HashTable<K, V, A> {
    /// Returns the total number of slots in the backing buffer (including empty slots).
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Returns the number of entries currently stored in the table.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns an iterator over all non-empty entries in the table.
    ///
    /// Iteration order is arbitrary (based on slot positions, not insertion order).
    /// The iterator scans all slots and skips empty ones (hash == 0).
    pub fn iter(&self) -> HashTableIterator<'_, K, V, A> {
        HashTableIterator {
            hash_table: self,
            idx: 0,
        }
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
    /// 1. Compute initial slot: `slot = hash & mask`
    /// 2. Linear probe until we find either:
    ///    - Empty slot (hash == 0): insert new entry here
    ///    - Matching entry (same hash AND same key): merge values
    /// 3. If inserting pushes us over 70% load, trigger resize
    ///
    /// # Hash Zero Handling
    ///
    /// Since `hash == 0` is the empty sentinel, any key that legitimately hashes to 0
    /// is stored with `hash = 1` instead. This is invisible to callers.
    ///
    /// # Performance
    ///
    /// - Best case: O(1) - slot is empty or immediate match
    /// - Average case: O(1) - at 70% load, expected probe length is ~1.8
    /// - Worst case: O(n) - pathological hash collisions
    pub fn merge<L: LiveKey<Persisted = K>>(&mut self, mut hash: u64, key: L, value: V) {
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1
        if hash == 0 {
            hash = 1;
        }

        let mut idx = (hash & self.mask as u64) as usize;

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
                if self.length > self.max_load {
                    self.resize(self.buffer.len() << 1);
                }
                return;
            }

            if entry.hash == hash && key == entry.key {
                // Key exists - merge the values (e.g., add counts)
                entry.value = entry.value.merge(value);
                return;
            }

            // Collision with different key - linear probe to next slot
            idx = (idx + 1) & self.mask;
        }
    }

    /// Grows the hash table to `new_size` slots and rehashes all entries in place.
    ///
    /// # Algorithm
    ///
    /// This uses an in-place strategy that avoids allocating a second buffer:
    ///
    /// 1. Expand the buffer to `new_size` (new slots are zero-initialized = empty)
    /// 2. Update the mask to `new_size - 1` for the larger table
    /// 3. Scan only the OLD portion of the buffer (indices 0..previous_len)
    /// 4. For each occupied slot, compute its new position with the updated mask
    /// 5. If it needs to move, clear the old slot and linear-probe to find its new home
    ///
    /// # Why In-Place Works
    ///
    /// When doubling size, entries either:
    /// - Stay in their current slot (if the new high bit of `hash & mask` is 0)
    /// - Move to `old_slot + old_capacity` (if the new high bit is 1)
    /// - And importantly - they never merge, since we know our HashTable was already correct up to
    ///   this point
    ///
    /// Since we only scan the old portion and new slots start empty, we never
    /// accidentally skip or double-process an entry.
    ///
    /// # Performance
    ///
    /// - Avoids allocating a second buffer (important for large tables)
    /// - Cache-friendly: sequential scan of old entries, mostly local moves
    ///
    /// # Panics
    ///
    /// The underlying allocation's `resize` may panic if memory allocation fails.
    #[cold]
    pub fn resize(&mut self, new_size: usize) {
        self.mask = new_size - 1;
        let previous_len = self.buffer.len();

        self.buffer.resize(new_size);

        let mut idx = 0;
        while idx < previous_len {
            let entry = &mut self.buffer[idx];
            if entry.hash != 0 {
                let new_idx = (entry.hash & self.mask as u64) as usize;
                // Check if it's already in the correct place, if yes we can exit early
                if idx != new_idx {
                    // Clear the old slot
                    let entry = mem::take(&mut self.buffer[idx]);

                    // Find the correct slot with linear probing
                    // In difference with `merge`, we don't need to check if a slot has our key if
                    // it's not empty, given our HashTable was correct up to this point and you
                    // can't have the same key twice
                    let mut new_idx = new_idx;
                    loop {
                        if self.buffer[new_idx].hash == 0 {
                            self.buffer[new_idx] = entry;
                            break;
                        }
                        new_idx = (new_idx + 1) & self.mask;
                    }
                }
            }

            idx += 1;
        }
        self.max_load = max_load_for_len(self.buffer.len());
    }
}

/// An iterator over the entries in a `HashTable`.
///
/// Created by [`HashTable::iter`]. Yields references to all non-empty entries
/// in arbitrary order (based on slot positions, not insertion order).
pub struct HashTableIterator<'a, K: PersistedKey, V: Value, A: Allocation<Entry<K, V>>> {
    hash_table: &'a HashTable<K, V, A>,
    idx: usize,
}

impl<'a, K: PersistedKey, V: Value, A: Allocation<Entry<K, V>>> Iterator
    for HashTableIterator<'a, K, V, A>
{
    type Item = &'a Entry<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.hash_table.buffer.len() {
            let entry = &self.hash_table.buffer[self.idx];
            self.idx += 1;
            if entry.hash != 0 {
                return Some(entry);
            }
        }
        None
    }
}
