//! Per-worker GROUP BY state: a stack of aggregating hash tables.
//!
//! Every row is folded into the active table, so a group is deduplicated as
//! soon as it repeats. The table grows by quadrupling up to
//! [`SPILL_CAPACITY`], past which it stops growing: the worker retires the full
//! table and starts another of the same size. Probe cost therefore stays
//! bounded however large the input's cardinality grows, at the price of
//! leaving the merge more tables to combine.
//!
//! ```text
//! consume rows --> active table --(full)--> retired onto the stack
//!                        ^                           |
//!                        +-- fresh same-size table   v
//!                                              partition merge
//! ```

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, MultiSlabTable,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::values::{AggregationSlot, ArityBody, WorkerContext};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

/// Slot count past which tables stop growing and start stacking.
///
/// Two costs pull against each other. A smaller table keeps probes closer to
/// the core but splits the input into more tables, which consolidation and
/// the merge pay for per entry copied and per source slice. A larger table
/// deduplicates more but its probes reach further out in the hierarchy.
/// Swept across the high-cardinality group-bys: 32768 beats 65536 on every
/// one of them (the stack is consolidated per worker, so the merge no longer
/// charges per table), and 16384 measures the same as 32768.
const SPILL_CAPACITY: usize = 32768;

/// Sizing for a worker's table stack.
#[derive(Copy, Clone)]
pub struct SpillConfig {
    /// Slot count past which tables stop growing and stack instead.
    pub spill_capacity: usize,
}

impl SpillConfig {
    pub const DEFAULT: Self = Self {
        spill_capacity: SPILL_CAPACITY,
    };

    /// The default sizing, overridable for capacity sweeps via
    /// `PIVOT_SPILL_CAPACITY` (a power-of-two slot count).
    pub fn from_env() -> Self {
        static SPILL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let spill_capacity = *SPILL.get_or_init(|| {
            std::env::var("PIVOT_SPILL_CAPACITY")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|capacity| capacity.is_power_of_two())
                .unwrap_or(Self::DEFAULT.spill_capacity)
        });
        Self { spill_capacity }
    }
}

/// How many recently resolved keys a worker remembers.
const MEMO_SLOTS: usize = 1024;

/// Remembers, for the current batch, which group a key's storage resolved to.
///
/// A dictionary-encoded column decodes every occurrence of a value to the same
/// view, so the second and later occurrences can fold straight into the group
/// the first one found, skipping both the probe and the byte comparison that
/// would reach the same place. A plain-encoded column gives each row its own
/// bytes, so nothing ever matches and the lookup falls through.
///
/// Two things bound how long a remembered answer stays true, and both are
/// handled by bumping the generation rather than clearing the arrays. A token
/// only names bytes within the batch that produced it, so it expires at each
/// batch boundary; and the addresses point into the active table, so they
/// expire when that table is retired.
struct KeyMemo {
    tokens: Box<[u128; MEMO_SLOTS]>,
    entries: Box<[*mut u8; MEMO_SLOTS]>,
    generations: Box<[u32; MEMO_SLOTS]>,
    generation: u32,
    /// Lookups made and answered while deciding whether to keep going.
    attempts: u32,
    hits: u32,
    /// Cleared for good once the sample shows the lookups are not paying.
    consult: bool,
}

/// Lookups to watch before deciding whether the memo earns its keep.
const MEMO_SAMPLE: u32 = 1 << 16;

/// Hit rate below which the memo is abandoned, as a reciprocal: fewer than one
/// hit in this many lookups means the token fetch and probe of the memo cost
/// more than the key comparisons they save.
const MEMO_MIN_HIT_RATE: u32 = 3;

impl KeyMemo {
    fn new() -> Self {
        Self {
            tokens: vec![0u128; MEMO_SLOTS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            entries: vec![std::ptr::null_mut(); MEMO_SLOTS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            generations: vec![0u32; MEMO_SLOTS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            generation: 1,
            attempts: 0,
            hits: 0,
            consult: true,
        }
    }

    /// Abandons everything remembered so far.
    #[inline(always)]
    fn invalidate(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        // Generation 0 is what the arrays start at, so skipping it keeps a
        // never-written slot from matching after a wrap.
        if self.generation == 0 {
            self.generation = 1;
        }
    }

    #[inline(always)]
    fn slot(hash: u64) -> usize {
        (hash as usize) & (MEMO_SLOTS - 1)
    }

    /// Whether the memo is still worth consulting for this worker.
    ///
    /// Repeated values only share a token when the column arrived
    /// dictionary-encoded. Plain-encoded input gives every row its own bytes,
    /// so nothing ever matches and every lookup is wasted; a short sample is
    /// enough to tell the two apart, after which the memo stops charging for
    /// itself.
    #[inline(always)]
    fn consulting(&self) -> bool {
        self.consult
    }

    #[inline(always)]
    fn lookup(&mut self, hash: u64, token: u128) -> Option<*mut u8> {
        let slot = Self::slot(hash);
        let hit = self.generations[slot] == self.generation && self.tokens[slot] == token;
        self.attempts += 1;
        self.hits += hit as u32;
        if self.attempts == MEMO_SAMPLE {
            self.consult = self.hits * MEMO_MIN_HIT_RATE >= self.attempts;
        }
        hit.then(|| self.entries[slot])
    }

    #[inline(always)]
    fn record(&mut self, hash: u64, token: u128, entry: *mut u8) {
        let slot = Self::slot(hash);
        self.tokens[slot] = token;
        self.entries[slot] = entry;
        self.generations[slot] = self.generation;
    }
}

/// A retired table together with the bucket-grouped view the merge reads.
pub struct SealedTable<KP: super::PersistedKey, V: AggregationValue + ?Sized> {
    pub table: MultiSlabTable<KP, V>,
    pub run: super::SortedRun,
}

impl<KP: super::PersistedKey, V: AggregationValue + ?Sized> SealedTable<KP, V> {
    fn seal(table: MultiSlabTable<KP, V>, hll: &mut Hll) -> Self {
        let run = table.build_sorted_run(hll);
        Self { table, run }
    }
}

/// One worker's merge input: its single table sealed in place, or its whole
/// stack consolidated into a dense run.
///
/// A worker that never outgrew one table (the common low and mid cardinality
/// case) hands the merge that table with a bucket-grouped slot view: one
/// sealing pass, no entry copies. A worker that stacked tables consolidates
/// them into one dense run so the merge visits one source per worker.
pub enum MergeSource<KP: super::PersistedKey, V: AggregationValue + ?Sized> {
    Sealed(SealedTable<KP, V>),
    Dense(super::DenseRun<KP, V>),
}

impl<KP: super::PersistedKey, V: AggregationValue + ?Sized> MergeSource<KP, V> {
    /// Number of entries this source contributes to the merge.
    pub fn len(&self) -> usize {
        match self {
            MergeSource::Sealed(sealed) => sealed.run.len(),
            MergeSource::Dense(run) => run.len(),
        }
    }

    /// Hash-prefix bits this source's bucket index resolves.
    pub fn bucket_bits(&self) -> u32 {
        match self {
            MergeSource::Sealed(sealed) => sealed.run.bucket_bits(),
            MergeSource::Dense(run) => run.bucket_bits(),
        }
    }
}

/// The tables and sizing data one worker produced.
pub struct AggregatedTableOutput<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// The NUMA node whose worker flushed this output; the merge groups
    /// sources by it.
    pub node: usize,
    /// Every group entry this worker produced (worker-internal repeats
    /// included; the merge combines them).
    pub source: MergeSource<K::Persisted, V>,
    /// Distinct-count sketch over the worker's rows, for sizing the merge targets.
    pub hll: Hll,
    /// `K::DEDUP_BY_HASH` only: this worker saw the (single) key whose bijective
    /// hash is 0, which is excluded from the tables. Adds 1 to the distinct count.
    pub zero_hash_seen: bool,
}

/// Per-worker aggregation state.
pub struct AggregatedTable<K: KeyExtractor, V: AggregationValue + ?Sized> {
    hash_state: RandomState,
    /// Shared value state used to construct tables and scatter rows.
    shared_context: V::SharedContext,
    /// String key storage, separate from the value arena to keep borrows disjoint.
    key_arena: WorkerArena,
    /// Per-worker value state.
    worker_context: V::WorkerContext,
    allocator: SlabAllocator,
    /// The table currently absorbing rows.
    active: MultiSlabTable<K::Persisted, V>,
    /// Tables retired at the spill capacity, consolidated at flush.
    retired: Vec<MultiSlabTable<K::Persisted, V>>,
    /// Distinct-count sketch used to size merge targets.
    hll: Hll,
    /// Slot count at which tables stop growing.
    spill_config: SpillConfig,
    /// Batch-local map from key storage to the group it resolved to.
    memo: KeyMemo,
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

impl<K: KeyExtractor, V: AggregationValue + ?Sized> AggregatedTable<K, V> {
    pub fn new(
        hash_state: RandomState,
        key_arena: Arc<SharedArena>,
        shared_context: V::SharedContext,
        worker_context: V::WorkerContext,
        spill_config: SpillConfig,
    ) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::new(&mut allocator, DEFAULT_CAPACITY, 0, &shared_context);
        Self {
            hash_state,
            shared_context,
            key_arena: WorkerArena::new(key_arena),
            worker_context,
            allocator,
            active: table,
            retired: Vec::new(),
            hll: Hll::new(),
            spill_config,
            memo: KeyMemo::new(),
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            scratch: K::Scratch::default(),
            zero_hash_seen: false,
        }
    }

    /// Consumes a batch in windows that fit the fixed hash scratch buffer.
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let total = batch.num_rows();
        if total <= RECORD_BATCH_SIZE {
            self.consume_window(batch, key_columns, value_slots, key_config, shared_context);
            return;
        }
        let mut start = 0;
        while start < total {
            let len = (total - start).min(RECORD_BATCH_SIZE);
            self.consume_window(
                &batch.slice(start, len),
                key_columns,
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
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let length = batch.num_rows();

        // Move scratch out temporarily so the key reader does not borrow `self`
        // while the table is mutated.
        let mut scratch = std::mem::take(&mut self.scratch);
        {
            let mut key_reader = K::make_reader(batch, key_columns, key_config, &mut scratch);
            let value_reader = V::make_reader(batch, value_slots);

            // Hash and encode each key once before probing.
            K::prepare_and_hash(
                &mut key_reader,
                &self.hash_state,
                &mut self.hashes[..length],
            );

            // A token names bytes inside one batch's buffers, so nothing
            // remembered from the last batch can be trusted here.
            self.memo.invalidate();
            self.consume_in_place(length, &key_reader, &value_reader, shared_context);
        }
        self.scratch = scratch;
    }

    /// Probes rows until the batch ends or the active table crosses its limit.
    #[inline(always)]
    fn consume_in_place<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
        shared_context: &V::SharedContext,
    ) {
        let metadata = V::storage_metadata(shared_context);
        let mut i = 0;
        while i < length {
            let overflowed = {
                let Self {
                    active,
                    key_arena,
                    worker_context,
                    hashes,
                    zero_hash_seen,
                    memo,
                    ..
                } = self;
                V::dispatch_arity(
                    metadata,
                    ProbeWindow::<K, V> {
                        table: active,
                        key_arena,
                        worker_context,
                        hashes,
                        zero_hash_seen,
                        memo,
                        next_row: &mut i,
                        length,
                        key_reader,
                        value_reader,
                        shared_context,
                    },
                )
            };
            // The row that crossed the limit has already been folded, so the
            // next one lands in the replacement table.
            if overflowed {
                self.retire_active_table();
            }
        }
    }

    /// Retires the full table and makes a fresh one active.
    ///
    /// The replacement is four times larger while that still fits the cache
    /// budget, and the same size once it does not. The retired table is
    /// sealed with a hash-ordered run while its entries are still cache-warm;
    /// the merge walks each partition's slice of that run densely.
    #[cold]
    fn retire_active_table(&mut self) {
        // Remembered addresses point into the table being retired.
        self.memo.invalidate();
        let capacity = self.active.capacity();
        let next_size = if capacity < self.spill_config.spill_capacity {
            capacity * 4
        } else {
            capacity
        };
        let retired = std::mem::replace(
            &mut self.active,
            BaseHashTable::new(&mut self.allocator, next_size, 0, &self.shared_context),
        );
        self.retired.push(retired);
    }

    /// Finishes worker state for the merge phase.
    pub fn flush(mut self) -> AggregatedTableOutput<K, V> {
        self.key_arena.flush();
        self.worker_context.flush();
        // Sealing and consolidation both fold every entry's hash into the
        // sketch. The sketch is idempotent per hash, so a key recurring in
        // several tables is counted once and the merged estimate is the true
        // distinct count.
        let source = if self.retired.is_empty() {
            MergeSource::Sealed(SealedTable::seal(self.active, &mut self.hll))
        } else {
            let mut tables = self.retired;
            tables.push(self.active);
            // Consolidating the stack into one dense run keeps the merge's
            // fan-in at the worker count and drops the tables' empty slots;
            // the tables are freed here, once their entries are copied out.
            MergeSource::Dense(super::DenseRun::consolidate(
                &tables,
                &mut self.allocator,
                V::storage_metadata(&self.shared_context),
                &mut self.hll,
            ))
        };
        AggregatedTableOutput {
            node: crate::worker::current_node(),
            source,
            hll: self.hll,
            zero_hash_seen: self.zero_hash_seen,
        }
    }
}

/// State passed through arity dispatch for one in-place consume window.
///
/// A nonzero `N` specializes the entry layout and slot loops. `N == 0` uses
/// runtime metadata.
struct ProbeWindow<'a, 'b, K: KeyExtractor, V: AggregationValue + ?Sized> {
    table: &'a mut MultiSlabTable<K::Persisted, V>,
    key_arena: &'a mut WorkerArena,
    worker_context: &'a mut V::WorkerContext,
    hashes: &'a [u64; RECORD_BATCH_SIZE],
    zero_hash_seen: &'a mut bool,
    memo: &'a mut KeyMemo,
    next_row: &'a mut usize,
    length: usize,
    key_reader: &'a K::Reader<'b>,
    value_reader: &'a V::Reader<'b>,
    shared_context: &'a V::SharedContext,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> ArityBody<bool> for ProbeWindow<'_, '_, K, V> {
    #[inline(always)]
    fn run<const N: usize>(self) -> bool {
        let ProbeWindow {
            table,
            key_arena,
            worker_context,
            hashes,
            zero_hash_seen,
            memo,
            next_row,
            length,
            key_reader,
            value_reader,
            shared_context,
        } = self;
        probe_rows::<N, K, V>(
            table,
            key_arena,
            worker_context,
            hashes,
            zero_hash_seen,
            memo,
            next_row,
            length,
            key_reader,
            value_reader,
            shared_context,
        )
    }
}

/// Probes and folds one consume window.
///
/// This function remains outlined so each specialized arity has its own stack
/// frame and register allocation. The call occurs once per window, not per row.
///
/// Separate reference parameters also preserve alias information across raw
/// table writes, allowing bound column pointers to remain outside the row loop.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn probe_rows<const N: usize, K: KeyExtractor, V: AggregationValue + ?Sized>(
    table: &mut MultiSlabTable<K::Persisted, V>,
    key_arena: &mut WorkerArena,
    worker_context: &mut V::WorkerContext,
    hashes: &[u64; RECORD_BATCH_SIZE],
    zero_hash_seen: &mut bool,
    memo: &mut KeyMemo,
    next_row: &mut usize,
    length: usize,
    key_reader: &K::Reader<'_>,
    value_reader: &V::Reader<'_>,
    shared_context: &V::SharedContext,
) -> bool {
    {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        // Sweep knob; the memo is only ever a win when repeated values share a
        // representation, so it is worth being able to measure both ways.
        static MEMO_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let memo_enabled = *MEMO_ENABLED
            .get_or_init(|| std::env::var("PIVOT_GROUP_KEY_MEMO").as_deref() != Ok("0"));
        let metadata = if N == 0 {
            V::storage_metadata(shared_context)
        } else {
            V::metadata_for_arity::<N>()
        };
        let value_offset = super::value_offset_for::<K::Persisted, V>(metadata);
        // Reuse one probe layout for every row in the window.
        let mut prober = if N == 0 {
            table.prober()
        } else {
            table.prober_with_metadata(V::metadata_for_arity::<N>())
        };
        let mut row = *next_row;
        let mut overflowed = false;
        while row < length {
            let hash = hashes[row];
            // Exact hash-only distinct counting handles hash zero out of band
            // because zero is the table's empty sentinel.
            if K::DEDUP_BY_HASH && hash == 0 {
                *zero_hash_seen = true;
                row += 1;
                continue;
            }
            if row + L2_DISTANCE < length {
                prober.prefetch_l2(hashes[row + L2_DISTANCE]);
            }
            if row + L1_DISTANCE < length {
                prober.prefetch(hashes[row + L1_DISTANCE]);
            }
            // A value that already resolved this batch folds straight into
            // the group it found, with no probe and no key comparison.
            let token = if memo_enabled && memo.consulting() {
                K::memo_token(key_reader, row)
            } else {
                None
            };
            if let Some(token) = token
                && let Some(entry) = memo.lookup(hash, token)
            {
                let stored = unsafe { V::from_entry_mut(entry.add(value_offset), metadata) };
                stored.update(value_reader, row, &mut *worker_context, shared_context);
                row += 1;
                continue;
            }
            // Probe once, then seed a new group or update the matching group.
            let key = K::live_key(key_reader, row, key_arena);
            let entry = prober.probe_fold::<false, _, _, _, _>(
                hash,
                key,
                &mut *worker_context,
                |worker_context, stored| stored.seed(value_reader, row, worker_context),
                |worker_context, stored| {
                    stored.update(value_reader, row, worker_context, shared_context)
                },
            );
            if let Some(token) = token {
                memo.record(hash, token, entry);
            }
            row += 1;
            if prober.undersized() {
                overflowed = true;
                break;
            }
        }
        *next_row = row;
        overflowed
    }
}
