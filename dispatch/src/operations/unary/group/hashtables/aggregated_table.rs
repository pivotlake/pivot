//! Per-worker GROUP BY state with adaptive in-place and radix modes.
//!
//! Every worker starts by aggregating into hash tables. If the active table
//! grows beyond [`SWITCH_THRESHOLD`], eligible keys switch to radix scatter.
//! Remaining rows are partitioned by the high hash bits and aggregated during
//! merge.
//!
//! ```text
//! consume rows
//!     |
//!     v
//! in-place tables -- threshold crossed --> partitioned scatter rows
//!     |                                      |
//!     +------------------+-------------------+
//!                        v
//!                  partition merge
//! ```

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::RADIX_PARTITIONS;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable, PersistedKey,
    StridedScatterRows,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::values::{AggregationSlot, ArityBody, WorkerContext};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

/// Table capacity at which eligible keys switch to radix scatter.
const SWITCH_THRESHOLD: usize = 32768;

/// Rows the deduplication windows must have folded per distinct key, over
/// all windows so far, to keep deduplicating: three rows per two keys. Below
/// that, the probe each row pays in the window table saves too little
/// scatter and merge work, and the remaining rows scatter raw.
const KEEP_DEDUP_ROWS_PER_KEYS: (usize, usize) = (3, 2);

/// Windows flushed before the fold ratio is judged. Keys often repeat in
/// bursts, so one window says little; the probes spent on these windows
/// are negligible even for keys that never repeat.
///
/// Keys that carry a blob are never judged: scattering such a row raw
/// copies its blob again, so folding stays cheaper for them however rarely
/// the keys repeat.
const WINDOWS_BEFORE_RAW_SCATTER: usize = 4;

/// Configuration for the in-place to radix transition.
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

    /// Disables radix scatter.
    ///
    /// String MIN and MAX use this mode so losing strings are compared before
    /// they are persisted in the value arena.
    pub const fn without_radix(self) -> Self {
        Self {
            switch_threshold: usize::MAX,
            partitions: self.partitions,
        }
    }
}

/// One worker's scatter buffer for each radix partition.
pub struct PartitionBuffers<KP: PersistedKey, V: AggregationValue + ?Sized>(
    pub Vec<StridedScatterRows<KP, V>>,
);
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for PartitionBuffers<KP, V> {}

/// Tables, optional scatter buffers, and sizing data produced by one worker.
pub struct AggregatedTableOutput<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// The NUMA node whose worker flushed this output; the merge groups
    /// sources by it.
    pub node: usize,
    /// In-place table stack: the full result if the worker never switched,
    /// otherwise its pre-switch tables.
    pub tables: Vec<MultiSlabTable<K::Persisted, V>>,
    /// Per-partition scatter buffers, present only after a radix transition.
    pub buffers: Option<PartitionBuffers<K::Persisted, V>>,
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
    /// Tables built before a radix transition.
    tables: Vec<MultiSlabTable<K::Persisted, V>>,
    /// Per-partition buffers allocated on the first radix transition.
    buffers: Option<Vec<StridedScatterRows<<K as KeyExtractor>::Persisted, V>>>,
    /// Distinct-count sketch used to size merge targets.
    hll: Hll,
    /// Whether this worker has entered radix mode.
    switched_to_radix: bool,
    /// Rows folded into the active table so far, compared against the
    /// distinct keys flushed from it to judge how much the windows
    /// deduplicated.
    rows_folded: usize,
    /// Distinct keys flushed from the active table to the radix partitions.
    keys_flushed: usize,
    /// Windows flushed so far.
    windows_flushed: usize,
    /// Whether rows now scatter raw, without deduplicating in the active
    /// table first. Set once the windows have barely deduplicated.
    scatter_raw: bool,
    /// Radix threshold and partition count.
    radix_config: RadixConfig,
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
        radix_config: RadixConfig,
    ) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::new(&mut allocator, DEFAULT_CAPACITY, 0, &shared_context);
        Self {
            hash_state,
            shared_context,
            key_arena: WorkerArena::new(key_arena),
            worker_context,
            allocator,
            tables: vec![table],
            buffers: None,
            hll: Hll::new(),
            switched_to_radix: false,
            rows_folded: 0,
            keys_flushed: 0,
            windows_flushed: 0,
            scatter_raw: false,
            radix_config,
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

            // Rows deduplicate in the active table until a window shows the
            // keys barely repeat; from then on they scatter directly.
            if self.scatter_raw {
                self.scatter_range(0, length, &key_reader, &value_reader);
            } else {
                self.consume_in_place(length, &key_reader, &value_reader, shared_context);
            }
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
            let window_start = i;
            let overflowed = {
                let Self {
                    tables,
                    key_arena,
                    worker_context,
                    hashes,
                    zero_hash_seen,
                    ..
                } = self;
                V::dispatch_arity(
                    metadata,
                    ProbeWindow::<K, V> {
                        table: tables.last_mut().unwrap(),
                        key_arena,
                        worker_context,
                        hashes,
                        zero_hash_seen,
                        next_row: &mut i,
                        length,
                        key_reader,
                        value_reader,
                        shared_context,
                    },
                )
            };
            self.rows_folded += i - window_start;
            // The row that crossed the limit has already been folded. Scatter
            // starts at the following row.
            if overflowed && self.grow_or_radix() {
                self.scatter_range(i, length, key_reader, value_reader);
                return;
            }
        }
    }

    /// Grows the in-place stack, or flushes the full active table's distinct
    /// keys to the radix partitions and clears it for the next window.
    ///
    /// Returns `true` when the caller should scatter the remaining rows raw:
    /// the windows so far barely deduplicated, so probing further windows
    /// would not pay for itself. Returns `false` when the caller should keep
    /// folding rows into the cleared table.
    #[inline(always)]
    fn grow_or_radix(&mut self) -> bool {
        let next_size = self.tables.last().unwrap().capacity() * 4;
        // Hash-only keys have no persisted key to scatter.
        if K::DEDUP_BY_HASH || next_size <= self.radix_config.switch_threshold {
            self.tables.push(BaseHashTable::new(
                &mut self.allocator,
                next_size,
                0,
                &self.shared_context,
            ));
            return false;
        }
        if self.buffers.is_none() {
            self.buffers = Some(
                (0..self.radix_config.partitions)
                    .map(|_| StridedScatterRows::new())
                    .collect(),
            );
        }
        self.switched_to_radix = true;
        self.keys_flushed += self.tables.last().unwrap().len();
        self.windows_flushed += 1;
        self.scatter_active_table_to_radix_partitions();
        let (rows, keys) = KEEP_DEDUP_ROWS_PER_KEYS;
        if !<K::Persisted as PersistedKey>::HAS_BLOB
            && self.windows_flushed >= WINDOWS_BEFORE_RAW_SCATTER
            && self.rows_folded * keys < self.keys_flushed * rows
        {
            self.scatter_raw = true;
            return true;
        }
        false
    }

    /// Scatters rows into radix partitions without deduplicating them.
    #[inline(always)]
    fn scatter_range<'b>(
        &mut self,
        start: usize,
        end: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
    ) {
        let shift = u64::BITS - self.radix_config.partitions.trailing_zeros();
        let Self {
            key_arena,
            worker_context,
            allocator,
            buffers,
            hll,
            hashes,
            shared_context,
            ..
        } = self;
        V::dispatch_arity(
            V::storage_metadata(shared_context),
            ScatterWindow::<K, V> {
                buffers: buffers.as_mut().unwrap(),
                key_arena,
                worker_context,
                allocator,
                hll,
                hashes,
                shift,
                start,
                end,
                key_reader,
                value_reader,
                shared_context,
            },
        );
    }

    /// Moves the active table's partial groups into radix buffers and clears it.
    #[inline(always)]
    fn scatter_active_table_to_radix_partitions(&mut self) {
        let shift = u64::BITS - self.radix_config.partitions.trailing_zeros();
        let Self {
            tables,
            buffers,
            allocator,
            hll,
            shared_context,
            ..
        } = self;
        let table = tables.last_mut().unwrap();
        let buffers = buffers.as_mut().unwrap();
        let scatter_layout =
            StridedScatterRows::<<K as KeyExtractor>::Persisted, V>::layout::<0>(shared_context);
        for entry in table.iter(0) {
            let hash = entry.hash;
            hll.add(hash);
            let partition = (hash >> shift) as usize;
            // Entries are already deduplicated within this table.
            buffers[partition].push_with(scatter_layout, allocator, hash, *entry.key, |stored| {
                stored.copy_from(entry.stored)
            });
        }
        table.clear();
    }

    /// Finishes worker state for the merge phase.
    pub fn flush(mut self) -> AggregatedTableOutput<K, V> {
        if self.switched_to_radix {
            // Add pre-transition groups to the scatter-side estimate.
            for table in &self.tables {
                for entry in table.iter(0) {
                    self.hll.add(entry.hash);
                }
            }
        }
        self.key_arena.flush();
        self.worker_context.flush();
        AggregatedTableOutput {
            node: crate::worker::current_node(),
            tables: self.tables,
            buffers: self.buffers.map(PartitionBuffers),
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
    next_row: &mut usize,
    length: usize,
    key_reader: &K::Reader<'_>,
    value_reader: &V::Reader<'_>,
    shared_context: &V::SharedContext,
) -> bool {
    {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
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
            // Probe once, then seed a new group or update the matching group.
            let key = K::live_key(key_reader, row, key_arena);
            prober.probe_fold::<false, _, _, _, _>(
                hash,
                key,
                &mut *worker_context,
                |worker_context, stored| stored.seed(value_reader, row, worker_context),
                |worker_context, stored| {
                    stored.update(value_reader, row, worker_context, shared_context)
                },
            );
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

/// State passed through arity dispatch for one scatter range.
struct ScatterWindow<'a, 'b, K: KeyExtractor, V: AggregationValue + ?Sized> {
    buffers: &'a mut Vec<StridedScatterRows<<K as KeyExtractor>::Persisted, V>>,
    key_arena: &'a mut WorkerArena,
    worker_context: &'a mut V::WorkerContext,
    allocator: &'a mut SlabAllocator,
    hll: &'a mut Hll,
    hashes: &'a [u64; RECORD_BATCH_SIZE],
    shift: u32,
    start: usize,
    end: usize,
    key_reader: &'a K::Reader<'b>,
    value_reader: &'a V::Reader<'b>,
    shared_context: &'a V::SharedContext,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> ArityBody<()> for ScatterWindow<'_, '_, K, V> {
    /// Scatter has little live state, so keeping it inline avoids a window call.
    #[inline(always)]
    fn run<const N: usize>(self) {
        let ScatterWindow {
            buffers,
            key_arena,
            worker_context,
            allocator,
            hll,
            hashes,
            shift,
            start,
            end,
            key_reader,
            value_reader,
            shared_context,
        } = self;
        // Every partition shares one row-layout snapshot.
        let scatter_layout =
            StridedScatterRows::<<K as KeyExtractor>::Persisted, V>::layout::<N>(shared_context);
        // Indexing rather than iterating: `i` is the absolute row, used to
        // read the hash, the key, and the value columns in parallel.
        #[allow(clippy::needless_range_loop)]
        for i in start..end {
            let hash = hashes[i];
            hll.add(hash);
            let partition = (hash >> shift) as usize;
            let key = K::live_key(key_reader, i, key_arena).persist();
            // Seed directly into the destination row.
            buffers[partition].push_with(scatter_layout, allocator, hash, key, |stored| {
                stored.seed(value_reader, i, worker_context)
            });
        }
    }
}
