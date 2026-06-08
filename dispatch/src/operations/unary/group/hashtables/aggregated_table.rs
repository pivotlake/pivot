use crate::RECORD_BATCH_SIZE;
use crate::memory::{MultiSlabBuffer, SlabAllocator};
use crate::operations::unary::group::PARTITIONS;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{KeyExtractor, LiveKey, ValueExtractor};
use crate::operations::unary::group::values::AggregationSlot;
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

/// One scattered row: `(hash, persisted key, per-row value contribution)`.
pub type RadixRow<K, V> = (u64, <K as KeyExtractor>::Persisted, <V as ValueExtractor>::Value);

/// Elements per chunk in a [`SlabList`]. ~`CHUNK_CAP * size_of::<RadixRow>()`
/// bytes, kept under one 2MB slab so each chunk is a single slab.
const CHUNK_CAP: usize = 1 << 11; // 2048

/// A growable, **engine-backed** (slab-pool) buffer, as a list of fixed-size
/// chunks. Appending never reallocates/copies the data (unlike `Vec`), and the
/// rows live in the pre-faulted, accounted slab pool rather than the heap — only
/// the small vector of chunk handles is on the heap. This is what lets the radix
/// scatter use our memory system instead of `std::Vec`.
pub struct SlabList<T: Copy> {
    chunks: Vec<MultiSlabBuffer<T>>,
    /// Cached base pointer of the current (last) chunk. Each chunk is a single
    /// slab (CHUNK_CAP * size_of::<T>() < 2MB), so its elements are contiguous
    /// and we can write/read sequentially off the base — avoiding the per-element
    /// `Index` (which pays a div/mod for the per-slab packing) on the hot path.
    cur_base: *mut T,
    last_len: usize,
}

// SAFETY: same as the hash table — for string keys the stored value embeds an
// `ArenaKey` (raw pointer into the `Arc`'d arena, which outlives the merge), so
// moving these to a merge worker is sound. `cur_base` points into a slab the
// `chunks` keep alive.
unsafe impl<T: Copy> Send for SlabList<T> {}

impl<T: Copy> SlabList<T> {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cur_base: std::ptr::null_mut(),
            last_len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, allocator: &mut SlabAllocator, val: T) {
        if self.chunks.is_empty() || self.last_len == CHUNK_CAP {
            let chunk = allocator.create_multi_slab_buffer(CHUNK_CAP, false);
            self.cur_base = chunk.ptr_at_index(0);
            self.chunks.push(chunk);
            self.last_len = 0;
        }
        unsafe { *self.cur_base.add(self.last_len) = val };
        self.last_len += 1;
    }

    pub fn len(&self) -> usize {
        match self.chunks.len() {
            0 => 0,
            n => (n - 1) * CHUNK_CAP + self.last_len,
        }
    }

    /// Visit every element in insertion order — sequentially off each chunk's
    /// base pointer (a contiguous single slab), so no per-element index math.
    #[inline(always)]
    pub fn for_each(&self, mut f: impl FnMut(T)) {
        let n = self.chunks.len();
        for (ci, chunk) in self.chunks.iter().enumerate() {
            let len = if ci + 1 == n { self.last_len } else { CHUNK_CAP };
            let base = chunk.ptr_at_index(0) as *const T;
            let slice = unsafe { std::slice::from_raw_parts(base, len) };
            for &val in slice {
                f(val);
            }
        }
    }
}

/// A worker's scatter output: one engine-backed buffer of raw (un-aggregated)
/// rows per partition. Aggregated once, in the merge phase.
pub struct PartitionBuffers<K: KeyExtractor, V: ValueExtractor>(pub Vec<SlabList<RadixRow<K, V>>>);
unsafe impl<K: KeyExtractor, V: ValueExtractor> Send for PartitionBuffers<K, V> {}

/// Per-worker consume state — **single-level radix** on engine buffers.
///
/// The consume phase does *no* aggregation: it scatters each row into one of
/// [`PARTITIONS`] per-partition [`SlabList`]s (slab-pool memory) by the top hash
/// bits. Each partition's rows from all workers are then aggregated exactly once
/// in a cache-resident table by the merge phase.
pub struct AggregatedTable<K: KeyExtractor, V: ValueExtractor> {
    hash_state: RandomState,
    worker_arena: WorkerArena,
    allocator: SlabAllocator,
    buffers: Vec<SlabList<RadixRow<K, V>>>,
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
}

impl<K: KeyExtractor, V: ValueExtractor> AggregatedTable<K, V> {
    pub fn new(state: RandomState, shared_arena: Arc<SharedArena>) -> Self {
        Self {
            hash_state: state,
            worker_arena: WorkerArena::new(shared_arena),
            allocator: SlabAllocator::new(false),
            buffers: (0..PARTITIONS).map(|_| SlabList::new()).collect(),
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    /// Scatter every row of `batch` into its partition buffer.
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[AggregationSlot],
    ) {
        let key_reader = K::make_reader(batch, key_cols);
        let value_reader = V::make_reader(batch, value_slots);
        let length = batch.num_rows();

        // Destructure so the per-partition buffer and the shared allocator can be
        // borrowed mutably at once (disjoint fields).
        let Self {
            hash_state,
            worker_arena,
            allocator,
            buffers,
            hashes,
        } = self;

        for i in 0..length {
            hashes[i] = K::hash(&key_reader, i, hash_state);
        }

        let shift = u64::BITS - PARTITIONS.trailing_zeros();
        for i in 0..length {
            let hash = hashes[i];
            let p = (hash >> shift) as usize;
            let key = K::live_key(&key_reader, i, worker_arena).persist();
            let value = V::value(&value_reader, i);
            buffers[p].push(allocator, (hash, key, value));
        }
    }

    /// Hand the per-partition scatter buffers to the merge phase.
    pub fn flush(self) -> PartitionBuffers<K, V> {
        self.worker_arena.flush();
        PartitionBuffers(self.buffers)
    }
}
