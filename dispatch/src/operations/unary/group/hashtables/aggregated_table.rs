//! Per-worker state for the GROUP BY consume phase.
//!
//! Each worker begins with ordinary in-place aggregation. When the active table
//! fills, it either grows another table or switches to radix scatter, depending
//! on the key type and table size.
//!
//! In-place aggregation is cheapest for low cardinality. Beyond
//! [`SWITCH_THRESHOLD`], fixed-width keys are appended to
//! [`RADIX_PARTITIONS`] buffers without probing; each smaller partition is
//! aggregated during merge. String-bearing keys use an abandon variant: the
//! current table is drained as already-deduplicated rows and then reused. This
//! avoids copying the same string for every input occurrence.
//!
//! [`flush`](AggregatedTable::flush) returns both the pre-switch table stack and
//! any scatter buffers. They use the same hash prefixes, so merge can process
//! them together without first repartitioning the tables.

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::RADIX_PARTITIONS;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable, ScatterRows,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::values::{AggregationSlot, AggregationWorkerState};
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

/// One scatter buffer per radix partition for a worker.
pub struct PartitionBuffers<K: KeyExtractor, V: AggregationValue>(
    pub Vec<<V as AggregationValue>::ScatterBuffer<<K as KeyExtractor>::Persisted>>,
);
unsafe impl<K: KeyExtractor, V: AggregationValue> Send for PartitionBuffers<K, V> {}

/// What a worker hands the merge phase: its in-place stack (always), the radix
/// scatter buffers (only if it switched), and its distinct-count sketch. The
/// merge slot-range-combines the stack and the buffers by the same top hash bits,
/// so a switched worker's pre-switch stack needs no pre-fold into the buffers.
pub struct AggregatedTableOutput<K: KeyExtractor, V: AggregationValue> {
    /// The NUMA node whose worker flushed this output; the merge groups
    /// sources by it.
    pub node: usize,
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
    /// Query-wide aggregation metadata used to construct tables and buffers.
    value_context: V::Context,
    /// String *key* storage. Separate from the value's write state so a live
    /// string key (which holds `&mut key_arena` until persisted) and a
    /// string-extreme value fold (which needs `&mut worker_state`) never alias —
    /// letting consume probe and fold in one fused pass.
    key_arena: WorkerArena,
    /// The value's per-worker consume write state: `()` for a numeric signature
    /// (consume threads `&mut ()`, free), a real `WorkerArena` for a string
    /// extreme (it stores winners). Spawned by `Group` from the shared context.
    worker_state: V::WorkerState,
    allocator: SlabAllocator,
    /// Phase 1: stack of growing in-place tables.
    tables: Vec<MultiSlabTable<K, V>>,
    /// Phase 2: per-partition scatter buffers (allocated on switch).
    buffers: Option<Vec<<V as AggregationValue>::ScatterBuffer<<K as KeyExtractor>::Persisted>>>,
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
        value_context: V::Context,
        worker_state: V::WorkerState,
        radix: RadixConfig,
    ) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::new(&mut allocator, DEFAULT_CAPACITY, 0, &value_context);
        Self {
            hash_state: state,
            value_context,
            key_arena: WorkerArena::new(key_arena),
            worker_state,
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
        value_context: &V::Context,
    ) {
        let total = batch.num_rows();
        if total <= RECORD_BATCH_SIZE {
            self.consume_window(batch, key_cols, value_slots, key_config, value_context);
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
                value_context,
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
        value_context: &V::Context,
    ) {
        let length = batch.num_rows();

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

            // A worker that switched to a *scatter* radix route (fixed-width keys)
            // scatters whole batches raw. Abandon-route keys (long strings) and
            // not-yet-switched workers keep probing/aggregating in `consume_scalared`.
            if self.switched_to_radix && !K::RADIX_ABANDON {
                self.scatter_range(0, length, &key_reader, &value_reader);
            } else {
                self.consume_scalared(length, &key_reader, &value_reader, value_context);
            }
        }
        self.scratch = scratch;
    }

    /// Row-by-row probe, growing or abandoning the active table when it overflows
    /// mid-batch. A two-level software prefetch (DRAM→L2 far, L2→L1 near) hides the
    /// per-row probe latency on the large in-place tables this path handles.
    #[inline(always)]
    fn consume_scalared<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
        value_context: &V::Context,
    ) {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        let mut i = 0;
        while i < length {
            // Keep one prober until this active table fills so its runtime
            // layout is loaded once for the whole row run.
            let overflowed = {
                let Self {
                    tables,
                    key_arena,
                    worker_state,
                    hashes,
                    zero_hash_seen,
                    ..
                } = self;
                let mut prober = tables.last_mut().unwrap().prober();
                let mut overflowed = false;
                while i < length {
                    let hash = hashes[i];
                    // Hash-only distinct uses a bijection and has no key bytes
                    // to disambiguate the empty sentinel. Record its unique
                    // zero-hash input outside the table.
                    if K::DEDUP_BY_HASH && hash == 0 {
                        *zero_hash_seen = true;
                        i += 1;
                        continue;
                    }
                    if i + L2_DISTANCE < length {
                        prober.prefetch_l2(hashes[i + L2_DISTANCE]);
                    }
                    if i + L1_DISTANCE < length {
                        prober.prefetch(hashes[i + L1_DISTANCE]);
                    }
                    // The probe returns the state reference directly to the
                    // selected arm. A new group persists its key and seeds its
                    // state; an existing group updates in place. String extrema
                    // persist a candidate only when it becomes the new winner.
                    let key = K::live_key(key_reader, i, key_arena);
                    prober.probe_fold::<false, _, _, _, _>(
                        hash,
                        key,
                        &mut *worker_state,
                        |wc, state| V::seed_entry(state, value_reader, i, wc),
                        |wc, state| V::update_entry(state, value_reader, i, wc, value_context),
                    );
                    i += 1;
                    if prober.needs_growth() {
                        overflowed = true;
                        break;
                    }
                }
                overflowed
            };
            // Fixed-width radix returns true and scatters the remaining rows.
            // An abandon or ordinary grow leaves an active table to continue
            // probing, so it returns false.
            if overflowed && self.grow_or_radix() {
                self.scatter_range(i, length, key_reader, value_reader);
                return;
            }
        }
    }

    /// Active table is full. For a radix-eligible key past [`SWITCH_THRESHOLD`],
    /// take the radix route and return whether the caller should now *scatter* the
    /// rest of the batch raw:
    /// - abandon route (long strings): drain this window's deduplicated entries to
    ///   the partition buffers and clear the table for reuse; keep probing (returns
    ///   `false`).
    /// - scatter route (fixed-width keys): allocate buffers and signal the caller to
    ///   append raw rows (returns `true`).
    ///
    /// Otherwise (not radix-eligible / below threshold) grow the stack 4x.
    #[inline(always)]
    fn grow_or_radix(&mut self) -> bool {
        let next_size = self.tables.last().unwrap().capacity() * 4;
        // A key with real bytes (`!DEDUP_BY_HASH`) is radix-eligible; a hash-only
        // key has nothing to scatter and always grows in place.
        if K::DEDUP_BY_HASH || next_size <= self.radix_cfg.switch_threshold {
            self.tables.push(BaseHashTable::new(
                &mut self.allocator,
                next_size,
                0,
                &self.value_context,
            ));
            return false;
        }
        if self.buffers.is_none() {
            self.buffers = Some(
                (0..self.radix_cfg.partitions)
                    .map(|_| V::ScatterBuffer::new(&self.value_context))
                    .collect(),
            );
        }
        self.switched_to_radix = true;
        if K::RADIX_ABANDON {
            self.abandon_active_table();
            false
        } else {
            true
        }
    }

    /// Scatter rows `[start, end)` into per-partition buffers (raw, post-switch),
    /// the radix route for fixed-width keys: a cheap append (no probe, no per-row
    /// dedup), with deduplication deferred to the merge.
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
            worker_state,
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
            let key = K::live_key(key_reader, i, key_arena).persist();
            // Initialize the state directly inside the scatter row.
            buffers[p].push_with(allocator, hash, key, |state| {
                V::seed_entry(state, value_reader, i, worker_state)
            });
        }
    }

    /// Drain the active table's aggregated entries into the per-partition scatter
    /// buffers (routing each by the same top hash bits the merge partitions on),
    /// then clear it for reuse. Unlike scattering raw rows, the entries here are
    /// already deduplicated for this window, so only one entry per distinct key
    /// moves, and its key handle is already persisted, so no string is recopied.
    /// The merge folds these per-window partials together across all windows.
    #[inline(always)]
    fn abandon_active_table(&mut self) {
        let shift = u64::BITS - self.radix_cfg.partitions.trailing_zeros();
        let Self {
            tables,
            buffers,
            allocator,
            hll,
            ..
        } = self;
        let table = tables.last_mut().unwrap();
        let buffers = buffers.as_mut().unwrap();
        for entry in table.iter(0) {
            let hash = entry.hash;
            hll.add(hash);
            let p = (hash >> shift) as usize;
            // This table entry is already a partial aggregate for its key.
            buffers[p].push_with(allocator, hash, *entry.key, |state| {
                V::copy_entry(state, entry.state)
            });
        }
        table.clear();
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
                    self.hll.add(entry.hash);
                }
            }
        }
        self.key_arena.flush();
        self.worker_state.flush();
        AggregatedTableOutput {
            node: crate::worker::current_node(),
            tables: self.tables,
            buffers: self.buffers.map(PartitionBuffers),
            hll: self.hll,
            zero_hash_seen: self.zero_hash_seen,
        }
    }
}
