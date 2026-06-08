//! Per-worker GROUP BY consume state, with an **adaptive in-place → radix**
//! strategy that needs no up-front cardinality estimate.
//!
//! A worker starts aggregating *in place* — a growing stack of hash tables. This
//! is the cheap path: no scatter, and the whole result stays in a few
//! cache-resident tables. It wins while the distinct count is small (low
//! cardinality, and every string group-by).
//!
//! Once that table would grow past [`SWITCH_THRESHOLD`] (≈ where it stops fitting
//! L2) *and* the key is radix-eligible ([`KeyExtractor::SUPPORTS_RADIX`] — ints
//! yes, strings never), the worker **drops off to radix**: it stops aggregating
//! and instead *scatters* every remaining row into one of [`RADIX_PARTITIONS`]
//! per-partition buffers by the top hash bits — no probing on the hot path. The
//! aggregation moves to the merge phase, where each partition is small enough to
//! stay cache-resident. That's the whole point at high cardinality, where a
//! single in-place table would spill to DRAM and every probe would miss cache.
//!
//! The choice is per-worker and just falls out of how fast the table fills, so a
//! low-cardinality column never switches. At [`flush`](AggregatedTable::flush) a
//! worker hands back its in-place stack and, if it switched, its scatter buffers;
//! the merge slot-range-combines both by the same top bits, with no pre-fold.

use crate::RECORD_BATCH_SIZE;
use crate::memory::{SlabAllocator, SlabVec};
use crate::operations::unary::group::RADIX_PARTITIONS;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    BatchRowSource, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable, ValueExtractor,
};
use crate::operations::unary::group::hll::Hll;
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
/// too low and a few-thousand-group aggregation regresses into a 4096-way radix.)
const SWITCH_THRESHOLD: usize = 32768;

/// Tunable thresholds for the in-place→radix switch. Production uses
/// [`DEFAULT`](RadixConfig::DEFAULT); tests build small configs so the radix path
/// (scatter buffers + an N-way merge) fits the test slab pool.
#[derive(Copy, Clone)]
pub struct RadixConfig {
    /// Table-slot count past which a radix-eligible worker switches to scatter.
    pub switch_threshold: usize,
    /// Number of scatter partitions (a power of two).
    pub partitions: usize,
}

impl RadixConfig {
    pub const DEFAULT: Self = Self {
        switch_threshold: SWITCH_THRESHOLD,
        partitions: RADIX_PARTITIONS,
    };
}

/// One scattered row: `(hash, persisted key, per-row value contribution)`.
pub type RadixRow<K, V> = (
    u64,
    <K as KeyExtractor>::Persisted,
    <V as ValueExtractor>::Value,
);

/// A worker's scatter output: one engine-backed buffer of raw rows per partition.
pub struct PartitionBuffers<K: KeyExtractor, V: ValueExtractor>(pub Vec<SlabVec<RadixRow<K, V>>>);
unsafe impl<K: KeyExtractor, V: ValueExtractor> Send for PartitionBuffers<K, V> {}

/// What a worker hands the merge phase: its in-place stack (always), the radix
/// scatter buffers (only if it switched), and its distinct-count sketch. The
/// merge slot-range-combines the stack and the buffers by the same top hash bits,
/// so a switched worker's pre-switch stack needs no pre-fold into the buffers.
pub struct AggregatedTableOutput<K: KeyExtractor, V: ValueExtractor> {
    /// In-place table stack: the full result if the worker never switched,
    /// otherwise its pre-switch tables.
    pub tables: Vec<MultiSlabTable<K, V>>,
    /// Per-partition scatter buffers — `Some` iff the worker switched to radix.
    pub buffers: Option<PartitionBuffers<K, V>>,
    /// Distinct-count sketch over the worker's rows, for sizing the merge targets.
    pub hll: Hll,
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

/// Per-worker aggregation state for the adaptive in-place → radix consume
/// strategy (see the module docs). Holds the in-place table stack and, after a
/// switch, the per-partition scatter buffers plus a distinct-count sketch.
pub struct AggregatedTable<K: KeyExtractor, V: ValueExtractor> {
    hash_state: RandomState,
    worker_arena: WorkerArena,
    allocator: SlabAllocator,
    /// Phase 1: stack of growing in-place tables.
    tables: Vec<MultiSlabTable<K, V>>,
    /// Phase 2: per-partition scatter buffers (allocated on switch).
    buffers: Option<Vec<SlabVec<RadixRow<K, V>>>>,
    /// Distinct-count sketch over scattered (post-switch) hashes, for sizing.
    hll: Hll,
    /// Have we switched to radix yet (we may never with low-cardinality)
    switched_to_radix: bool,
    /// The config of radix (when to switch, number of partitions)
    radix_cfg: RadixConfig,
    /// Scratch buffer for hashes
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    /// Scratch buffer for probe slots within the hashtable
    probe_slots: Box<[usize; RECORD_BATCH_SIZE]>,
    /// Scratch buffer for unresolved slots (slots that collided)
    unresolved_slots: Box<[u32; RECORD_BATCH_SIZE]>,
    /// Another scratch buffer for unresolved slots (slots that collided)- this is used in tangent
    /// with the original `unresolved_slots` as a sort of ping-pong (see merge_batch)
    next_unresolved_slots: Box<[u32; RECORD_BATCH_SIZE]>,
}

impl<K: KeyExtractor, V: ValueExtractor> AggregatedTable<K, V> {
    pub fn new(state: RandomState, shared_arena: Arc<SharedArena>, radix: RadixConfig) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::multi_slab(&mut allocator, DEFAULT_CAPACITY, 0);
        Self {
            hash_state: state,
            worker_arena: WorkerArena::new(shared_arena),
            allocator,
            tables: vec![table],
            buffers: None,
            hll: Hll::new(),
            switched_to_radix: false,
            radix_cfg: radix,
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            probe_slots: vec![0usize; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            unresolved_slots: vec![0u32; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            next_unresolved_slots: vec![0u32; RECORD_BATCH_SIZE]
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

        if self.switched_to_radix {
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
            &mut self.probe_slots[..],
            &mut self.unresolved_slots[..],
            &mut self.next_unresolved_slots[..],
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
        if K::SUPPORTS_RADIX && next_size > self.radix_cfg.switch_threshold {
            self.buffers = Some(
                (0..self.radix_cfg.partitions)
                    .map(|_| SlabVec::new())
                    .collect(),
            );
            self.switched_to_radix = true;
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
        let shift = u64::BITS - self.radix_cfg.partitions.trailing_zeros();
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

    /// Finalize: hand back this worker's in-place stack plus, if it switched, its
    /// scatter buffers. The merge slot-range-combines both by the same top bits,
    /// so there's no pre-fold of the stack into the buffers.
    pub fn flush(mut self) -> AggregatedTableOutput<K, V> {
        if self.switched_to_radix {
            // The scatter counted post-switch rows into the sketch; add the
            // pre-switch stack's distinct so the merge sizing sees the total.
            for table in &self.tables {
                for entry in table.iter(0) {
                    self.hll.add(entry.hash());
                }
            }
        }
        self.worker_arena.flush();
        AggregatedTableOutput {
            tables: self.tables,
            buffers: self.buffers.map(PartitionBuffers),
            hll: self.hll,
        }
    }
}
