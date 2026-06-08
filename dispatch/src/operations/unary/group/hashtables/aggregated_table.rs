use crate::RECORD_BATCH_SIZE;
use crate::memory::{MultiSlabBuffer, SlabAllocator};
use crate::operations::unary::group::RADIX_PARTITIONS;
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    BatchRowSource, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable, ValueExtractor,
};
use crate::operations::unary::group::values::AggregationSlot;
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

/// Table-slot count at which a worker stops growing its in-place hash table and
/// switches to radix scatter (for radix-eligible keys). Set near the L2-resident
/// boundary: an in-place table up to this size (~0.8–1.8 MB depending on entry
/// width) stays in cache, so low- and medium-cardinality group-bys keep the cheap
/// in-place path and pay no scatter/partition overhead — only once the table
/// would spill cache does radix's cache-resident per-partition merge win. (Set
/// too low this regresses e.g. q42's ~2880-group date_trunc into a 4096-way radix.)
const SWITCH_THRESHOLD: usize = 32768;

/// Elements per [`SlabList`] chunk (single slab, < 2MB for the row width).
const CHUNK_CAP: usize = 1 << 11; // 2048

/// One scattered row: `(hash, persisted key, per-row value contribution)`.
pub type RadixRow<K, V> = (u64, <K as KeyExtractor>::Persisted, <V as ValueExtractor>::Value);

/// A growable, engine-backed (slab-pool) append-only buffer, as a list of
/// fixed-size chunks. Appends never reallocate; a cached base pointer makes the
/// scatter writes and aggregate reads sequential (no per-element index math).
pub struct SlabList<T: Copy> {
    chunks: Vec<MultiSlabBuffer<T>>,
    cur_base: *mut T,
    last_len: usize,
}

// SAFETY: for string keys the stored value embeds an `ArenaKey` (raw pointer into
// the `Arc`'d arena, which outlives the merge); `cur_base` points into a slab the
// `chunks` keep alive. (Strings never scatter today, but the bound is generic.)
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

    /// Visit every element in insertion order — sequentially off each chunk's base.
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

/// A worker's scatter output: one engine-backed buffer of raw rows per partition.
pub struct PartitionBuffers<K: KeyExtractor, V: ValueExtractor>(pub Vec<SlabList<RadixRow<K, V>>>);
unsafe impl<K: KeyExtractor, V: ValueExtractor> Send for PartitionBuffers<K, V> {}

/// What a worker hands the merge phase, depending on whether it crossed the
/// radix threshold during consume.
pub enum WorkerOutput<K: KeyExtractor, V: ValueExtractor> {
    /// Never switched — the full result is in this stack of in-place tables
    /// (low-cardinality, or any string group-by). Combined by the slot-range merge.
    InPlace(Vec<MultiSlabTable<K, V>>),
    /// Switched — the pre-switch stack has been folded into these per-partition
    /// scatter buffers, so the radix merge sees a single uniform source.
    Radix(PartitionBuffers<K, V>, Hll),
}

/// [`BatchRowSource`] adapter for [`BaseHashTable::merge_batch`] (in-place phase).
struct RowSrc<'r, 'b, K: KeyExtractor, V: ValueExtractor> {
    key_reader: &'r K::Reader<'b>,
    value_reader: V::Reader<'b>,
    arena: &'r mut WorkerArena,
}

impl<K: KeyExtractor, V: ValueExtractor> BatchRowSource<K::Persisted, V::Value>
    for RowSrc<'_, '_, K, V>
{
    #[inline(always)]
    fn persisted(&mut self, i: usize) -> K::Persisted {
        K::live_key(self.key_reader, i, &mut *self.arena).persist()
    }
    #[inline(always)]
    fn key_eq(&mut self, i: usize, persisted: &K::Persisted) -> bool {
        K::live_key(self.key_reader, i, &mut *self.arena).eq_persisted(persisted)
    }
    #[inline(always)]
    fn value(&mut self, i: usize) -> V::Value {
        V::value(&self.value_reader, i)
    }
}

/// Per-worker aggregation state — **adaptive in-place → radix**.
///
/// Phase 1 aggregates into a growing stack of in-place hash tables (the original
/// strategy). When the table would grow past [`SWITCH_THRESHOLD`] and the key is
/// radix-eligible ([`KeyExtractor::SUPPORTS_RADIX`]), it switches to phase 2:
/// scatter into per-partition buffers (cache-resident merge). String keys never
/// switch. At [`flush`](Self::flush) a switched worker folds its small pre-switch
/// stack into the buffers, so the merge sees one uniform source.
pub struct AggregatedTable<K: KeyExtractor, V: ValueExtractor> {
    hash_state: RandomState,
    worker_arena: WorkerArena,
    allocator: SlabAllocator,
    /// Phase 1: stack of growing in-place tables.
    tables: Vec<MultiSlabTable<K, V>>,
    /// Phase 2: per-partition scatter buffers (allocated on switch).
    buffers: Option<Vec<SlabList<RadixRow<K, V>>>>,
    hll: Hll,
    switched: bool,
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    slots: Box<[usize; RECORD_BATCH_SIZE]>,
    sel: Box<[u32; RECORD_BATCH_SIZE]>,
    sel_next: Box<[u32; RECORD_BATCH_SIZE]>,
}

impl<K: KeyExtractor, V: ValueExtractor> AggregatedTable<K, V> {
    pub fn new(state: RandomState, shared_arena: Arc<SharedArena>) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::multi_slab(&mut allocator, DEFAULT_CAPACITY, 0);
        Self {
            hash_state: state,
            worker_arena: WorkerArena::new(shared_arena),
            allocator,
            tables: vec![table],
            buffers: None,
            hll: Hll::new(),
            switched: false,
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            slots: vec![0usize; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            sel: vec![0u32; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            sel_next: vec![0u32; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[AggregationSlot],
    ) {
        let key_reader = K::make_reader(batch, key_cols);
        let value_reader = V::make_reader(batch, value_slots);
        let length = batch.num_rows();

        for i in 0..length {
            self.hashes[i] = K::hash(&key_reader, i, &self.hash_state);
        }

        if self.switched {
            self.scatter_range(0, length, &key_reader, &value_reader);
            return;
        }

        // The active table can take the whole batch iff no probe overflows it
        // mid-batch; only then is the batched path safe.
        let free = {
            let t = self.tables.last().unwrap();
            t.capacity() - t.len()
        };
        if free > length {
            self.consume_batched(length, &key_reader, value_reader);
        } else {
            self.consume_scalared(length, &key_reader, &value_reader);
        }
    }

    /// Batched multi-pass probe over the active table (it has room for the whole
    /// batch). The whole batch is consumed before any grow/switch is considered.
    #[inline(always)]
    fn consume_batched<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: V::Reader<'b>,
    ) {
        let mut src = RowSrc::<K, V> {
            key_reader,
            value_reader,
            arena: &mut self.worker_arena,
        };
        self.tables.last_mut().unwrap().merge_batch(
            length,
            &mut self.hashes[..],
            &mut self.slots[..],
            &mut self.sel[..],
            &mut self.sel_next[..],
            &mut src,
        );
        if self.tables.last().unwrap().undersized() {
            self.grow_or_switch();
        }
    }

    /// Row-by-row probe, switching tables (or to radix) when the active table
    /// overflows mid-batch. On switch, the remaining rows are scattered. A
    /// two-level software prefetch (DRAM→L2 far, L2→L1 near) hides the per-row
    /// probe latency on the large in-place tables this path handles.
    #[inline(always)]
    fn consume_scalared<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
    ) {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        let mut i = 0;
        while i < length {
            let hash = self.hashes[i];
            let overflowed = {
                let table = self.tables.last_mut().unwrap();
                if i + L2_DISTANCE < length {
                    table.prefetch_l2(self.hashes[i + L2_DISTANCE]);
                }
                if i + L1_DISTANCE < length {
                    table.prefetch(self.hashes[i + L1_DISTANCE]);
                }
                let key = K::live_key(key_reader, i, &mut self.worker_arena);
                let value = V::value(value_reader, i);
                table.merge::<false, _>(hash, key, value);
                table.undersized()
            };
            if overflowed && self.grow_or_switch() {
                self.scatter_range(i + 1, length, key_reader, value_reader);
                return;
            }
            i += 1;
        }
    }

    /// Active table is full: either grow the stack (4x) or, for a radix-eligible
    /// key that would grow past [`SWITCH_THRESHOLD`], switch to scatter. Returns
    /// `true` if it switched.
    #[inline(always)]
    fn grow_or_switch(&mut self) -> bool {
        let next_size = self.tables.last().unwrap().capacity() * 4;
        if K::SUPPORTS_RADIX && next_size > SWITCH_THRESHOLD {
            self.buffers = Some((0..RADIX_PARTITIONS).map(|_| SlabList::new()).collect());
            self.switched = true;
            true
        } else {
            self.tables
                .push(BaseHashTable::multi_slab(&mut self.allocator, next_size, 0));
            false
        }
    }

    /// Scatter rows `[start, end)` into per-partition buffers (post-switch).
    #[inline(always)]
    fn scatter_range(
        &mut self,
        start: usize,
        end: usize,
        key_reader: &K::Reader<'_>,
        value_reader: &V::Reader<'_>,
    ) {
        let shift = u64::BITS - RADIX_PARTITIONS.trailing_zeros();
        let Self {
            worker_arena,
            allocator,
            buffers,
            hll,
            hashes,
            ..
        } = self;
        let buffers = buffers.as_mut().unwrap();
        for i in start..end {
            let hash = hashes[i];
            hll.add(hash);
            let p = (hash >> shift) as usize;
            let key = K::live_key(key_reader, i, worker_arena).persist();
            let value = V::value(value_reader, i);
            buffers[p].push(allocator, (hash, key, value));
        }
    }

    /// Finalize: a switched worker folds its small pre-switch stack into the
    /// buffers (so the merge sees one source); otherwise it hands back the stack.
    pub fn flush(mut self) -> WorkerOutput<K, V> {
        if self.switched {
            let shift = u64::BITS - RADIX_PARTITIONS.trailing_zeros();
            let Self {
                tables,
                buffers,
                allocator,
                hll,
                ..
            } = &mut self;
            let buffers = buffers.as_mut().unwrap();
            for table in tables.iter() {
                for entry in table.iter(0) {
                    hll.add(entry.hash());
                    let p = (entry.hash() >> shift) as usize;
                    buffers[p].push(allocator, (entry.hash(), *entry.key(), *entry.value()));
                }
            }
        }

        self.worker_arena.flush();

        if self.switched {
            WorkerOutput::Radix(PartitionBuffers(self.buffers.take().unwrap()), self.hll)
        } else {
            WorkerOutput::InPlace(self.tables)
        }
    }
}
