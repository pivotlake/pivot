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
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::values::{AggregationSlot, WorkerContext};
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

/// Sub-window size for the in-place consume: each window is probed and then folded
/// (`finalize_batch`) before the next, so a deferred value's seed/update buffers
/// stay small and the cells it folds are still cache-hot from the probe. Smaller
/// than [`RECORD_BATCH_SIZE`] so the per-window working set stays cache-resident.
/// MEASUREMENT: overridable via `PIVOT_CHUNK` to sweep sizes without rebuilding.
fn consume_chunk() -> usize {
    use std::sync::OnceLock;
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("PIVOT_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
    })
}

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

    /// The same config with the radix switch disabled (the threshold can never be
    /// crossed), so the worker always folds in-place and never scatters. Used
    /// when a value holds a string extreme: the scatter path materialises every
    /// row's value, which for a string means persisting it into the arena before
    /// any comparison — so a loser would be persisted. The in-place fold compares
    /// first and persists only a winner, keeping the arena to kept strings.
    pub const fn without_radix(self) -> Self {
        Self {
            switch_threshold: usize::MAX,
            partitions: self.partitions,
        }
    }
}

/// One scattered row: `(hash, persisted key, per-row value contribution)`.
pub type RadixRow<K, V> = (u64, <K as KeyExtractor>::Persisted, V);

/// A worker's scatter output: one engine-backed buffer of raw rows per partition.
pub struct PartitionBuffers<K: KeyExtractor, V: AggregationValue>(pub Vec<SlabVec<RadixRow<K, V>>>);
unsafe impl<K: KeyExtractor, V: AggregationValue> Send for PartitionBuffers<K, V> {}

/// What a worker hands the merge phase: its in-place stack (always), the radix
/// scatter buffers (only if it switched), and its distinct-count sketch. The
/// merge slot-range-combines the stack and the buffers by the same top hash bits,
/// so a switched worker's pre-switch stack needs no pre-fold into the buffers.
pub struct AggregatedTableOutput<K: KeyExtractor, V: AggregationValue> {
    /// In-place table stack: the full result if the worker never switched,
    /// otherwise its pre-switch tables.
    pub tables: Vec<MultiSlabTable<K, V>>,
    /// Per-partition scatter buffers — `Some` iff the worker switched to radix.
    pub buffers: Option<PartitionBuffers<K, V>>,
    /// Distinct-count sketch over the worker's rows, for sizing the merge targets.
    pub hll: Hll,
    /// `K::DEDUP_BY_HASH` only: this worker saw the (single) key whose bijective
    /// hash is 0, which is excluded from the tables. Adds 1 to the distinct count.
    pub zero_hash_seen: bool,
}

/// Per-worker aggregation state for the adaptive in-place → radix consume
/// strategy (see the module docs). Holds the in-place table stack and, after a
/// switch, the per-partition scatter buffers plus a distinct-count sketch.
pub struct AggregatedTable<K: KeyExtractor, V: AggregationValue> {
    hash_state: RandomState,
    /// String *key* storage. Separate from the value's write state so a live
    /// string key (which holds `&mut key_arena` until persisted) and a
    /// string-extreme value fold (which needs `&mut worker_context`) never alias —
    /// letting consume probe and fold in one fused pass.
    key_arena: WorkerArena,
    /// The value's per-worker consume write state: `()` for a numeric signature
    /// (consume threads `&mut ()`, free), a real `WorkerArena` for a string
    /// extreme (it stores winners). Spawned by `Group` from the shared context.
    worker_context: V::WorkerContext,
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
    /// Scratch buffer for the per-row hashes computed once per batch.
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    /// Per-worker reusable key-extraction scratch (e.g. the row extractor's
    /// encode buffers). Lent to the reader each batch and reused, never
    /// reallocated. `()` for extractors that read columns directly.
    scratch: K::Scratch,
    /// `K::DEDUP_BY_HASH` only: a key whose bijective hash is exactly 0 collides
    /// with the empty-slot sentinel, so it is excluded from the table and recorded
    /// here. There is at most one such key (the hash is a bijection), so this
    /// boolean adds 0 or 1 to the distinct count at output.
    zero_hash_seen: bool,
}

impl<K: KeyExtractor, V: AggregationValue> AggregatedTable<K, V> {
    pub fn new(
        state: RandomState,
        key_arena: Arc<SharedArena>,
        worker_context: V::WorkerContext,
        radix: RadixConfig,
    ) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::multi_slab(&mut allocator, DEFAULT_CAPACITY, 0);
        Self {
            hash_state: state,
            key_arena: WorkerArena::new(key_arena),
            worker_context,
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
            scratch: K::Scratch::default(),
            zero_hash_seen: false,
        }
    }

    /// Merge all rows of `batch`. The per-batch scratch is sized to
    /// [`RECORD_BATCH_SIZE`], so an input wider than that — which happens when a
    /// GROUP BY feeds another GROUP BY (the output path emits up to
    /// `OUTPUT_CHUNK_ROWS` ≫ `RECORD_BATCH_SIZE` rows per batch) — is consumed in
    /// `RECORD_BATCH_SIZE`-row windows via zero-copy slices. Scan/filter output
    /// (already ≤ `RECORD_BATCH_SIZE`) skips slicing.
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let total = batch.num_rows();
        if total <= RECORD_BATCH_SIZE {
            self.consume_window(batch, key_cols, value_slots, key_config, shared_context);
            return;
        }
        let mut start = 0;
        while start < total {
            let len = (total - start).min(RECORD_BATCH_SIZE);
            self.consume_window(
                &batch.slice(start, len),
                key_cols,
                value_slots,
                key_config,
                shared_context,
            );
            start += len;
        }
    }

    fn consume_window(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let length = batch.num_rows();
        debug_assert!(
            !(K::DEDUP_BY_HASH && K::SUPPORTS_RADIX),
            "DEDUP_BY_HASH requires the in-place path (no radix scatter)"
        );

        // Lend the reusable scratch to the reader for this batch. Taking it out of
        // `self` (it goes back at the end) means the reader borrows a local, not
        // `self`, so the probe below still has `&mut self`. The buffers' capacity
        // persists across batches — nothing is reallocated. The readers live in
        // an inner scope so their borrow of `scratch`/`batch` ends before we hand
        // the buffers back.
        let mut scratch = std::mem::take(&mut self.scratch);
        {
            let mut key_reader = K::make_reader(batch, key_cols, key_config, &mut scratch);
            let value_reader = V::make_reader(batch, value_slots);

            // Fill every row's hash once up front (the row extractor also encodes
            // its keys here); the scalar probe then reuses `self.hashes` and can
            // prefetch ahead.
            K::prepare_and_hash(
                &mut key_reader,
                &self.hash_state,
                &mut self.hashes[..length],
            );

            if self.switched_to_radix {
                self.scatter_range(0, length, &key_reader, &value_reader);
            } else {
                // Probe + finalize in small sub-windows. A deferred value (Dynamic)
                // records each chunk's matched cells, then `finalize_batch` folds
                // them while they are still cache-hot from the probe and the
                // seed/update buffers are still tiny. An eager value folds in place
                // during the probe and its `finalize_batch` is a no-op, so chunking
                // is free for it (same total probe work).
                let chunk = consume_chunk();
                let mut start = 0;
                while start < length {
                    if self.switched_to_radix {
                        self.scatter_range(start, length, &key_reader, &value_reader);
                        break;
                    }
                    let end = (start + chunk).min(length);
                    self.consume_scalared(start, end, length, &key_reader, &value_reader, shared_context);
                    V::finalize_batch(&mut self.worker_context, &value_reader, shared_context);
                    start = end;
                }
            }
        }
        self.scratch = scratch;
    }

    /// Row-by-row probe, switching tables (or to radix) when the active table
    /// overflows mid-batch. On switch, the remaining rows are scattered. A
    /// two-level software prefetch (DRAM→L2 far, L2→L1 near) hides the per-row
    /// probe latency on the large in-place tables this path handles.
    #[inline(always)]
    fn consume_scalared<'b>(
        &mut self,
        start: usize,
        end: usize,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
        shared_context: &V::SharedContext,
    ) {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        let mut i = start;
        while i < end {
            let hash = self.hashes[i];
            // Keys-only exact COUNT(DISTINCT): a 0 hash collides with the
            // empty-slot sentinel, so don't store it (the table would remap it to
            // 1 and alias a real key) — count it out of band instead. At most one
            // key hashes to 0 (the hash is a bijection), so the flag adds 1 at
            // output. Const-gated: zero cost for every other group-by.
            if K::DEDUP_BY_HASH && hash == 0 {
                self.zero_hash_seen = true;
                i += 1;
                continue;
            }
            let overflowed = {
                let table = self.tables.last_mut().unwrap();
                if i + L2_DISTANCE < length {
                    table.prefetch_l2(self.hashes[i + L2_DISTANCE]);
                }
                if i + L1_DISTANCE < length {
                    table.prefetch(self.hashes[i + L1_DISTANCE]);
                }
                // Probe and fold in one pass. The live key persists into the key
                // arena; the value's write state (`&mut self.worker_context`) is
                // handed to whichever arm runs — `seed` materialises a new group's
                // value, `update` folds the row into an existing one (a string
                // extreme persists only if it wins). When `V::WorkerContext` is `()`
                // (a string-free `Compiled` signature) this threads `&mut ()` —
                // free, with nothing in the loop that can alias the table it
                // mutates; a string extreme threads its `WorkerArena` to store the
                // winning string.
                let key = K::live_key(key_reader, i, &mut self.key_arena);
                table.probe_fold::<false, _, _, _, _>(
                    hash,
                    key,
                    &mut self.worker_context,
                    |wc, cell| V::consume_seed(cell, value_reader, i, wc),
                    |wc, cell| V::consume_update(cell, value_reader, i, wc, shared_context),
                );
                table.undersized()
            };
            if overflowed && self.grow_or_switch() {
                // Scatter only the rest of this chunk; `consume_window`'s loop sees
                // `switched_to_radix` and scatters the remaining chunks.
                self.scatter_range(i + 1, end, key_reader, value_reader);
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
            key_arena,
            worker_context,
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
            // Only a numeric signature ever radixes (a string extreme disables the
            // switch), so `value` persists nothing into `worker_context` here even
            // when it is a `Dynamic`'s `WorkerArena` (the string arms are dead).
            let key = K::live_key(key_reader, i, key_arena).persist();
            let value = V::value(value_reader, i, worker_context);
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
        self.key_arena.flush();
        self.worker_context.flush();
        AggregatedTableOutput {
            tables: self.tables,
            buffers: self.buffers.map(PartitionBuffers),
            hll: self.hll,
            zero_hash_seen: self.zero_hash_seen,
        }
    }
}
