//! Parallel GROUP BY operator.
//!
//! # Architecture
//!
//! GROUP BY is a pipeline breaker: it must consume all input before producing
//! any output. The implementation has two phases, both fully parallel:
//!
//! ## Phase 1: Consume (parallel per worker, no coordination)
//!
//! Each worker owns a [`Group`] containing a local [`AggregatedTable`]. For
//! every input [`RecordBatch`], the worker:
//!
//! 1. Hashes all keys in the group column (all workers share the same
//!    [`RandomState`] so hashes are consistent).
//! 2. Inserts each key/value into its local hash table via
//!    [`AggregatedTable::consume_batch`]. Duplicate keys within the same
//!    worker are merged immediately (e.g. counts are summed).
//! 3. If the hash table exceeds its load threshold, a new, larger table is
//!    created and subsequent rows go there. The old table is kept — its
//!    entries will be merged in phase 2.  See [Why partitioned merging works](#why-partitioned-merging-works-across-different-table-sizes).
//!
//! For high-cardinality integer keys a worker instead **switches to radix**
//! partway through: rather than growing its in-place table further it scatters
//! later rows into [`RADIX_PARTITIONS`] per-partition buffers (no probing),
//! deferring their aggregation to a finer-grained, cache-resident phase-2 merge.
//! Strings and low-cardinality keys never switch and stay fully in-place.
//!
//! When consumption finishes, each worker sends its tables (plus any radix
//! buffers) to a shared mpsc channel and transitions into a [`GroupOutputter`].
//!
//! ## Phase 2: Output / Merge (parallel via work-stealing)
//!
//! The first worker to enter the output phase drains the channel (collecting
//! tables from all workers) and publishes [`PARTITIONS`] independent
//! [`PartitionJob`]s to a shared [`Injector`]. Each partition covers a
//! disjoint range of hash values (determined by the top `log2(PARTITIONS)`
//! bits), so the jobs are embarrassingly parallel.
//!
//! All workers then steal and execute jobs. Each [`PartitionJob`]:
//!
//! 1. Scans every source table for entries belonging to its partition.
//! 2. Merges them into a single result table (see [`merge`] module for
//!    the batched, prefetched merge strategy).
//! 3. Converts the result table into an Arrow [`RecordBatch`] via the
//!    [`output`] combinator — key columns from the [`KeyExtractor`], value
//!    columns from the [`ValueExtractor`] — and sends it downstream.
//!
//! When any worker switched to radix, the same machinery runs at
//! [`RADIX_PARTITIONS`] granularity, and each job additionally aggregates its
//! partition's scatter buffers (where most of phase 1's aggregation was deferred)
//! alongside the in-place stacks.
//!
//! ## Why partitioned merging works across different table sizes
//!
//! The hash table uses **top-bit slot placement**: `slot = hash >> (64 - log2(capacity))`.
//! This means the top bits of the hash determine the slot, and the partition
//! (top 6 bits for 64 partitions) is always a prefix of the slot index.
//!
//! Because of this, entries for partition P occupy a predictable, contiguous
//! slot range in *any* power-of-2 table, regardless of its size:
//!
//! ```text
//! 128-slot table:  partition 0 = slots [0, 2)    partition 1 = slots [2, 4)   ...
//! 256-slot table:  partition 0 = slots [0, 4)    partition 1 = slots [4, 8)   ...
//! 1024-slot table: partition 0 = slots [0, 16)   partition 1 = slots [16, 32) ...
//! ```
//!
//! The ranges differ in width but always align — a 128-slot table's partition 0
//! range is a subset of a 256-slot table's partition 0 range (same top bits,
//! just fewer bits resolved). This lets the merge phase scan tables of mixed
//! sizes and route entries to the correct partition using only the hash value,
//! without needing to know the source table's capacity.
//!
//! ## Module layout
//!
//! - [`keys`] — the [`KeyExtractor`] trait and implementations
//!   ([`IntKeyExtractor`], [`StringKeyExtractor`]), each co-located with its key
//!   type (e.g. `keys::string` owns [`ArenaKey`])
//! - [`values`] — the [`ValueExtractor`] trait and implementations, each
//!   co-located with its value/aggregate type (`Count`, `AggregationRow`) plus
//!   [`AggregationKind`]/[`AggregationSlot`]
//! - [`hashtables`] — `BaseHashTable`, [`AggregatedTable`], [`MultiSlabTable`],
//!   and associated type machinery
//! - [`merge`] — partition-parallel merge of per-worker tables
//! - [`arena`] — shared string storage backing string keys
//! - [`factory`] — [`GroupFactory`] for creating per-worker [`Group`] instances

pub(crate) mod arena;
mod hll;
use hll::Hll;
mod factory;
mod keys;
mod merge;
mod output;
mod values;

pub use factory::GroupFactory;
mod hashtables;

pub use keys::{
    ArenaKey, HashOnlyIntKeyExtractor, IntKeyExtractor, IntPairKeyExtractor, KeyExtractor,
    RowKeyExtractor, RowKeySchema, StringKeyExtractor,
};
pub use values::{
    Accumulator, Aggregate, AggregationKind, AggregationRowValueExtractor, AggregationSlot,
    Compiled, Count, DistinctValueExtractor, MixedRowValueExtractor, Sum, ValueExtractor,
};

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregatedTable, AggregatedTableOutput, DEFAULT_CAPACITY, MAX_LOAD_FACTOR, MultiSlabTable,
    PartitionBuffers, RadixConfig,
};
use crate::worker::worker_waker;
use ahash::RandomState;
use arena::SharedArena;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use thiserror::Error;
use unary::pipeline_breaker::{Consumer, Outputter};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] crate::operations::channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Number of hash partitions for the output merge phase.
/// Each partition is merged independently, enabling parallel output.
const PARTITIONS: usize = 64;

/// Number of radix partitions for the scatter + merge of high-cardinality
/// (switched) workers. Larger than [`PARTITIONS`] so each radix target stays
/// cache-resident at high group counts.
const RADIX_PARTITIONS: usize = 4096;

/// Per-worker GROUP BY consumer.
///
/// During the consume phase, each worker owns a `Group` that hashes incoming
/// rows and inserts them into its local [`AggregatedTable`]. When consumption
/// finishes, the accumulated tables are sent to a shared channel and the
/// `Group` transitions into a [`GroupOutputter`] for the merge phase.
pub struct Group<K: KeyExtractor, V: ValueExtractor> {
    key_cols: Vec<usize>,
    value_slots: Vec<AggregationSlot>,
    key_config: K::Config,

    aggregated_table: AggregatedTable<K, V>,
    sender: mpsc::Sender<AggregatedTableOutput<K, V>>,
    outputter: GroupOutputter<K, V>,
}

impl<K: KeyExtractor, V: ValueExtractor> Group<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shared_arena: Arc<SharedArena>,
        state: RandomState,
        injector: Arc<Injector<PartitionJob<K, V>>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        key_config: K::Config,
        top_k: Option<(usize, usize)>,
        output_limit: Option<usize>,
        count_only: bool,
        sender: mpsc::Sender<AggregatedTableOutput<K, V>>,
        receiver: Option<mpsc::Receiver<AggregatedTableOutput<K, V>>>,
        partition_jobs_injected: Arc<AtomicBool>,
        radix: RadixConfig,
    ) -> Self {
        Self {
            key_cols,
            outputter: GroupOutputter {
                shared_arena: shared_arena.clone(),
                injector,
                receiver,
                partition_jobs_injected,
                top_k,
                output_limit,
                count_only,
                key_config: key_config.clone(),
                value_slots: value_slots.clone(),
                output_allocator: None,
            },
            sender,
            key_config,
            value_slots,
            aggregated_table: AggregatedTable::new(state, shared_arena, radix),
        }
    }
}

impl<K: KeyExtractor, V: ValueExtractor> Consumer<RecordBatch, RecordBatch> for Group<K, V> {
    type Outputter = GroupOutputter<K, V>;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        self.aggregated_table.consume_batch(
            &batch,
            &self.key_cols,
            &self.value_slots,
            &self.key_config,
        );
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        let tables = self.aggregated_table.flush();
        self.sender.send(tables).unwrap();
        Ok(Some(self.outputter))
    }
}

/// Handles the output (merge) phase of a GROUP BY.
///
/// Only one worker holds the `receiver` end of the channel (assigned by
/// [`GroupFactory`]). That worker drains the channel, collects all per-worker
/// tables, and publishes [`PARTITIONS`] [`PartitionJob`]s to the shared
/// work-stealing [`Injector`]. All workers (including the one that injected)
/// then steal and execute jobs until the injector is empty.
pub struct GroupOutputter<K: KeyExtractor, V: ValueExtractor> {
    shared_arena: Arc<SharedArena>,
    injector: Arc<Injector<PartitionJob<K, V>>>,
    receiver: Option<mpsc::Receiver<AggregatedTableOutput<K, V>>>,
    partition_jobs_injected: Arc<AtomicBool>,
    top_k: Option<(usize, usize)>,
    /// A bare `LIMIT n` directly above the group (no ORDER BY): any `n`
    /// complete groups are a valid answer, and every partition holds complete
    /// groups — so the drainer merges partitions serially and stops as soon as
    /// `n` rows have been emitted, skipping the rest of the merge entirely.
    output_limit: Option<usize>,
    /// Global `COUNT(DISTINCT)`: emit each partition's distinct-key count instead
    /// of its keys (a downstream `SUM` totals them).
    count_only: bool,
    /// The key extractor's runtime configuration, forwarded to each partition
    /// job so the output columns can be built (e.g. the row extractor's schema).
    key_config: K::Config,
    /// The query's aggregate slots, forwarded to each partition job for
    /// kind-aware value combining and output column building.
    value_slots: Vec<AggregationSlot>,
    /// One allocator per worker for the output columns of every partition this
    /// worker handles, so small per-partition outputs pack into shared buffers
    /// instead of each grabbing a fresh 2MB one. Created lazily on the first job.
    output_allocator: Option<SlabAllocator>,
}

/// A single partition's merge work unit.
///
/// Created by the [`GroupOutputter`] that holds the receiver and pushed to a
/// shared [`Injector`] for work-stealing execution. Each job merges all
/// source tables for partition `index` into one result table and sends the
/// output as a [`RecordBatch`].
pub struct PartitionJob<K: KeyExtractor, V: ValueExtractor> {
    /// Switched workers' scatter buffers (empty Vec in the all-in-place case).
    buffers: Arc<Vec<PartitionBuffers<K, V>>>,
    /// Every worker's in-place stack: switched workers' pre-switch tables and
    /// non-switched workers' full stacks. Slot-range-merged at `num_partitions`.
    tables: Arc<Vec<MultiSlabTable<K, V>>>,
    index: usize,
    arena: Arc<SharedArena>,
    partition_capacity: usize,
    /// [`PARTITIONS`] when nobody switched, else [`RADIX_PARTITIONS`].
    num_partitions: usize,
    top_k: Option<(usize, usize)>,
    count_only: bool,
    key_config: K::Config,
    value_slots: Vec<AggregationSlot>,
}

unsafe impl<K: KeyExtractor, V: ValueExtractor> Send for PartitionJob<K, V> {}

impl<K: KeyExtractor, V: ValueExtractor> PartitionJob<K, V> {
    /// Merge this partition's scatter buffers and in-place stacks into one result
    /// table and send the output batches, building columns into `allocator`.
    pub fn run<S: Sender<RecordBatch>>(
        self,
        sender: &mut S,
        allocator: &mut SlabAllocator,
    ) -> Result<usize> {
        let result_map = merge::merge_combined::<K, V>(
            self.index,
            &self.buffers,
            &self.tables,
            self.partition_capacity,
            self.num_partitions,
            &self.arena,
            &self.value_slots,
        );
        if result_map.len() == 0 {
            return Ok(0);
        }
        output::build_and_send::<K, V, _, _>(
            result_map,
            &self.arena,
            allocator,
            self.top_k,
            self.count_only,
            &self.key_config,
            &self.value_slots,
            sender,
        )
    }
}

impl<K: KeyExtractor, V: ValueExtractor> Outputter<RecordBatch> for GroupOutputter<K, V> {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
            let mut all_tables = Vec::new();
            let mut all_buffers = Vec::new();
            let mut hll = Hll::new();
            let mut zero_hash_seen = false;
            for out in rx.into_iter() {
                all_tables.extend(out.tables);
                if let Some(b) = out.buffers {
                    all_buffers.push(b);
                }
                hll.merge(&out.hll);
                zero_hash_seen |= out.zero_hash_seen;
            }

            // Pick the partition count: nobody switched -> the cheap PARTITIONS-way
            // slot-range merge (don't blow a small group-by into a 4096-way
            // merge); any switch -> RADIX_PARTITIONS so each radix target
            // stays cache-resident. Either way, one merge_combined job per partition
            // combines that partition's scatter buffers and in-place stacks.
            let (num_partitions, partition_capacity) = if all_buffers.is_empty() {
                let total: usize = all_tables.iter().map(|t| t.len()).sum();
                (
                    PARTITIONS,
                    (total / PARTITIONS)
                        .next_power_of_two()
                        .max(DEFAULT_CAPACITY),
                )
            } else {
                // Every switched worker scattered into the same partition count
                // (their RadixConfig); read it back off the buffers. Size each
                // radix target so its distinct-per-partition groups sit at
                // MAX_LOAD_FACTOR — the merge's resize threshold — so it fills
                // without ever resizing.
                let parts = all_buffers[0].0.len();
                let per_partition = hll.estimate() as f64 / parts as f64;
                let capacity = ((per_partition / MAX_LOAD_FACTOR).ceil() as usize)
                    .next_power_of_two()
                    .max(DEFAULT_CAPACITY);
                (parts, capacity)
            };

            let buffers = Arc::new(all_buffers);
            let tables = Arc::new(all_tables);

            // Bare `LIMIT n`: every partition holds complete groups, so any
            // `n` of them answer the query. Merge partitions serially right
            // here and stop as soon as enough rows have gone out — at high
            // group counts this skips nearly the whole merge. Other workers
            // see no jobs (the flag is set first) and finish immediately.
            if let Some(n) = self.output_limit {
                self.partition_jobs_injected.store(true, Ordering::Relaxed);
                worker_waker().notify();
                let allocator = self
                    .output_allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                let mut emitted = 0usize;
                for i in 0..num_partitions {
                    if emitted >= n {
                        break;
                    }
                    let job = PartitionJob {
                        buffers: buffers.clone(),
                        tables: tables.clone(),
                        index: i,
                        arena: self.shared_arena.clone(),
                        partition_capacity,
                        num_partitions,
                        top_k: self.top_k,
                        count_only: self.count_only,
                        key_config: self.key_config.clone(),
                        value_slots: self.value_slots.clone(),
                    };
                    emitted += job.run(sender, allocator).map_err(unary::Error::from)?;
                }
                return Ok(true);
            }

            for i in 0..num_partitions {
                self.injector.push(PartitionJob {
                    buffers: buffers.clone(),
                    tables: tables.clone(),
                    index: i,
                    arena: self.shared_arena.clone(),
                    partition_capacity,
                    num_partitions,
                    top_k: self.top_k,
                    count_only: self.count_only,
                    key_config: self.key_config.clone(),
                    value_slots: self.value_slots.clone(),
                });
            }

            self.partition_jobs_injected.store(true, Ordering::Relaxed);

            // Exact COUNT(DISTINCT): the keys-only consume excluded the single
            // key whose bijective hash is 0 (it collides with the empty sentinel)
            // and flagged it instead. Emit it now as one extra count row so the
            // downstream SUM includes it. Only one worker drains the channel, so
            // this fires exactly once.
            if self.count_only && zero_hash_seen {
                use arrow_array::Int64Array;
                use arrow_schema::{DataType, Field, Schema};
                let arr = Arc::new(Int64Array::from(vec![1i64]));
                let field = Field::new("v0", DataType::Int64, false);
                let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![arr])
                    .map_err(|e| unary::Error::from(Error::from(e)))?;
                sender
                    .send(batch)
                    .map_err(|e| unary::Error::from(Error::from(e)))?;
            }

            // Wake up all workers so that they can start working on partitions
            worker_waker().notify();
        }

        let steal = self.injector.steal();
        match steal {
            Steal::Success(job) => {
                let allocator = self
                    .output_allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                job.run(sender, allocator).map_err(unary::Error::from)?;
            }
            Steal::Empty => {
                if self.partition_jobs_injected.load(Ordering::Relaxed) {
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::keys::{IntKeyExtractor, StringKeyExtractor};
    use crate::operations::unary::group::values::Sum;
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    type IntExtractor = IntKeyExtractor<Int32Type>;
    type CountValue = Compiled<(Count,)>;
    type SumValue = Compiled<(Sum<Int32Type>,)>;

    fn batch_with_column(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// An `Int32` key column (0) plus an `Int32` value column (1), for `SUM`.
    fn keyed_i32_batch(keys: &[i32], vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(Int32Array::from(vals.to_vec())),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    /// A single `Utf8View` key column.
    fn string_key_batch(values: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            DataType::Utf8View,
            false,
        )]));
        let col: ArrayRef = Arc::new(StringViewArray::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn count_slots() -> Vec<AggregationSlot> {
        vec![AggregationSlot::new(AggregationKind::CountStar, 0)]
    }

    /// Common case: `Int32` keys, `COUNT(*)`, key column 0, no LIMIT, default radix.
    fn run_group(worker_batches: Vec<Vec<RecordBatch>>) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        )
    }

    #[test]
    fn batch_wider_than_record_batch_size() {
        // A single input batch wider than RECORD_BATCH_SIZE (8192) must be
        // consumed via windowing rather than overflowing the per-batch scratch.
        // This is the shape a GROUP BY feeding another GROUP BY produces.
        let values: Vec<i32> = (0..20_000).map(|i| i % 5_000).collect();
        let sender = run_group(vec![vec![batch_with_column(&values)]]);
        // 5_000 distinct keys, each appearing 4 times.
        assert_eq!(sender.total_rows(), 5_000);
    }

    /// Common case with a chosen [`RadixConfig`] — a small one forces the radix
    /// switch + scatter merge to run within the test slab pool.
    fn run_group_with_radix(
        worker_batches: Vec<Vec<RecordBatch>>,
        radix: RadixConfig,
    ) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            radix,
        )
    }

    /// General harness: choose the key/value extractors, key columns, aggregates,
    /// LIMIT pushdown, and radix config. The common-case wrappers above cover
    /// `Int32` keys + `COUNT(*)`.
    fn run_group_full<K: KeyExtractor<Config: Default>, V: ValueExtractor>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        top_k: Option<(usize, usize)>,
        radix: RadixConfig,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<K, V>::new(
                    arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    K::Config::default(),
                    top_k,
                    None,
                    false,
                    tx.clone(),
                    rx_opt.take(),
                    partition_jobs_injected.clone(),
                    radix,
                )
            })
            .collect();
        drop(tx);

        run_consumers(groups, worker_batches)
    }

    #[test]
    fn single_worker_single_batch() {
        let sender = run_group(vec![vec![batch_with_column(&[1, 2, 3, 4, 5])]]);

        assert_eq!(sender.total_rows(), 5);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn single_worker_duplicates_merged() {
        let sender = run_group(vec![vec![batch_with_column(&[1, 1, 2, 2, 3])]]);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3]);
    }

    #[test]
    fn single_worker_multiple_batches() {
        let sender = run_group(vec![vec![
            batch_with_column(&[1, 2]),
            batch_with_column(&[2, 3]),
            batch_with_column(&[3, 4]),
        ]]);

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn two_workers_disjoint() {
        let sender = run_group(vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[4, 5, 6])],
        ]);

        assert_eq!(sender.total_rows(), 6);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn two_workers_overlapping() {
        let sender = run_group(vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[2, 3, 4])],
        ]);

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn empty_input() {
        let sender = run_group(vec![vec![batch_with_column(&[])]]);

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn many_distinct_keys() {
        let values: Vec<i32> = (0..1000).collect();
        let sender = run_group(vec![vec![batch_with_column(&values)]]);

        assert_eq!(sender.total_rows(), 1000);
    }

    #[test]
    fn four_workers_all_same_key() {
        let sender = run_group(vec![
            vec![batch_with_column(&[42, 42, 42])],
            vec![batch_with_column(&[42, 42])],
            vec![batch_with_column(&[42])],
            vec![batch_with_column(&[42, 42, 42, 42])],
        ]);

        assert_eq!(sender.total_rows(), 1);
    }

    /// (key, count) pairs from a finished `COUNT(*)` group-by, sorted by key.
    fn group_counts(sender: &CollectSender) -> Vec<(i32, i64)> {
        let mut pairs: Vec<_> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn counts_aggregate_within_and_across_workers() {
        let batches = vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[2, 2, 3, 4])],
        ];

        let sender = run_group(batches);

        assert_eq!(group_counts(&sender), vec![(1, 1), (2, 3), (3, 2), (4, 1)]);
    }

    #[test]
    fn counts_survive_in_place_stack_growth() {
        let mut values: Vec<i32> = (0..3000).collect();
        values.extend(0..3000);

        let sender = run_group(vec![vec![batch_with_column(&values)]]);

        assert_eq!(
            group_counts(&sender),
            (0..3000).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn radix_switch_single_worker_aggregates_counts() {
        // A small config makes a radix-eligible (integer) worker switch to scatter
        // after ~a couple hundred keys and scatter into 16 partitions, exercising
        // the whole radix path within the test pool.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
        };
        let mut values: Vec<i32> = (0..500).collect();
        values.extend(0..500);

        let sender = run_group_with_radix(vec![vec![batch_with_column(&values)]], radix);

        assert_eq!(
            group_counts(&sender),
            (0..500).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn radix_switch_merges_across_workers() {
        // Two workers each switch to scatter; the radix merge has to combine both
        // workers' per-partition buffers and pre-switch stacks for every key.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
        };
        let worker = || vec![batch_with_column(&(0..500).collect::<Vec<_>>())];

        let sender = run_group_with_radix(vec![worker(), worker()], radix);

        assert_eq!(
            group_counts(&sender),
            (0..500).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    /// Run a keys-only global `COUNT(DISTINCT v)` (count-only output) over
    /// `Int32` values and return the total: the sum of the emitted
    /// per-partition distinct counts, plus the out-of-band 0-hash row if the
    /// key 0 (the unique preimage of hash 0 — `mix64` fixes 0) was present.
    fn run_distinct_count(worker_batches: Vec<Vec<RecordBatch>>, radix: RadixConfig) -> i64 {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<HashOnlyIntKeyExtractor<Int32Type>, DistinctValueExtractor>::new(
                    arena.clone(),
                    state.clone(),
                    injector.clone(),
                    vec![0],
                    Vec::new(),
                    (),
                    None,
                    None,
                    true,
                    tx.clone(),
                    rx_opt.take(),
                    partition_jobs_injected.clone(),
                    radix,
                )
            })
            .collect();
        drop(tx);

        let sender = run_consumers(groups, worker_batches);
        sender.i64_column(0).into_iter().sum()
    }

    #[test]
    fn distinct_count_zero_hash_key_in_place() {
        // The default config never switches at 3 keys, so the 0 key (whose
        // bijective hash collides with the empty-slot sentinel) is excluded by
        // the in-place scalar path and counted via the extra output row.
        let total = run_distinct_count(
            vec![vec![batch_with_column(&[0, 1, 2, 0, 1])]],
            RadixConfig::DEFAULT,
        );
        assert_eq!(total, 3);
    }

    #[test]
    fn distinct_count_radix_switch_exact_across_workers() {
        // Two workers over overlapping ranges (0 included in both), each with
        // enough distinct values to switch to radix scatter partway through
        // (threshold 256 × the keys-only 4x width scale = 1024 slots). The
        // count must be exact: the merge dedups scattered rows by the
        // bijective hash against both workers' pre-switch in-place stacks, and
        // the 0-hash key — seen by both workers — counts exactly once.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
        };
        let w1: Vec<i32> = (0..6000).collect();
        let w2: Vec<i32> = std::iter::once(0).chain(3000..9000).collect();
        let total = run_distinct_count(
            vec![vec![batch_with_column(&w1)], vec![batch_with_column(&w2)]],
            radix,
        );
        assert_eq!(total, 9000); // distinct(0..9000)
    }

    #[test]
    fn distinct_count_zero_hash_key_skipped_by_scatter() {
        // Feed the 0 key only *after* the worker has switched to radix, so it
        // is the post-switch scatter — not the in-place path — that must
        // exclude it (storing it would let the merge remap its 0 hash to 1 and
        // alias a real key) and record it for the out-of-band count.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
        };
        let big: Vec<i32> = (1..5000).collect(); // forces the switch
        let tail = vec![0, 0, 17, 4999]; // 0 first arrives post-switch
        let total = run_distinct_count(
            vec![vec![batch_with_column(&big), batch_with_column(&tail)]],
            radix,
        );
        assert_eq!(total, 5000); // distinct(1..5000) plus the 0 key
    }

    /// (string key, count) pairs from a finished string group-by, sorted by key.
    fn string_group_pairs(sender: &CollectSender) -> Vec<(String, i64)> {
        let mut pairs: Vec<_> = sender
            .string_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn string_keys_dedup_and_count() {
        let batches = vec![vec![string_key_batch(&["a", "b", "a", "c", "b", "a"])]];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            batches,
            vec![0],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        );

        assert_eq!(
            string_group_pairs(&sender),
            vec![
                ("a".to_string(), 3),
                ("b".to_string(), 2),
                ("c".to_string(), 1)
            ]
        );
    }

    #[test]
    fn sum_aggregates_value_per_group() {
        // key 1 -> 10+30, key 2 -> 20, key 3 -> 5+5.
        let batch = keyed_i32_batch(&[1, 2, 1, 3, 3], &[10, 20, 30, 5, 5]);
        let slots = vec![AggregationSlot::new(AggregationKind::Sum, 1)];

        let sender = run_group_full::<IntExtractor, SumValue>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            RadixConfig::DEFAULT,
        );

        assert_eq!(group_counts(&sender), vec![(1, 40), (2, 20), (3, 10)]);
    }

    /// Multi-column batch for row-key tests: an `Int16`, a `Utf8View`, and an
    /// `Int32` value column.
    fn mixed_key_batch(small: &[i16], names: &[&str], vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("small", DataType::Int16, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(arrow_array::Int16Array::from(small.to_vec())),
            Arc::new(StringViewArray::from(names.to_vec())),
            Arc::new(Int32Array::from(vals.to_vec())),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    /// Run a row-key group over `worker_batches` with the given key columns,
    /// schema and slots.
    fn run_row_key_group<V: ValueExtractor>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        schema: keys::RowKeySchema,
        value_slots: Vec<AggregationSlot>,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<keys::RowKeyExtractor, V>::new(
                    arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    schema.clone(),
                    None,
                    None,
                    false,
                    tx.clone(),
                    rx_opt.take(),
                    partition_jobs_injected.clone(),
                    RadixConfig::DEFAULT,
                )
            })
            .collect();
        drop(tx);

        run_consumers(groups, worker_batches)
    }

    #[test]
    fn row_key_mixed_int_string_counts() {
        // Keys (1, "a") ×2 across workers, (1, "bb-very-long-string-over-12") ×2
        // within one worker, (2, "a") ×1 — mixed inline and arena-backed blobs.
        let long = "bb-very-long-string-over-12";
        let batches = vec![
            vec![mixed_key_batch(
                &[1, 1, 1],
                &["a", long, long],
                &[10, 20, 30],
            )],
            vec![mixed_key_batch(&[1, 2], &["a", "a"], &[40, 50])],
        ];
        let schema = keys::RowKeySchema::new(vec![DataType::Int16, DataType::Utf8View]);

        let sender =
            run_row_key_group::<Compiled<(Count,)>>(batches, vec![0, 1], schema, count_slots());

        let mut rows: Vec<(i16, String, i64)> = sender
            .i16_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((a, b), c)| (a, b, c))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![
                (1, "a".to_string(), 2),
                (1, long.to_string(), 2),
                (2, "a".to_string(), 1),
            ]
        );
    }

    #[test]
    fn row_key_sums_per_group() {
        // SUM(val) per (small, name) group.
        let batches = vec![vec![mixed_key_batch(
            &[1, 1, 2, 1],
            &["x", "y", "x", "x"],
            &[10, 20, 30, 40],
        )]];
        let schema = keys::RowKeySchema::new(vec![DataType::Int16, DataType::Utf8View]);

        let sender = run_row_key_group::<Compiled<(Sum<Int32Type>,)>>(
            batches,
            vec![0, 1],
            schema,
            vec![AggregationSlot::new(AggregationKind::Sum, 2)],
        );

        let mut rows: Vec<(i16, String, i64)> = sender
            .i16_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((a, b), c)| (a, b, c))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![
                (1, "x".to_string(), 50),
                (1, "y".to_string(), 20),
                (2, "x".to_string(), 30),
            ]
        );
    }

    #[test]
    fn row_key_many_distinct_groups() {
        // Enough distinct (int, string) pairs to grow the in-place stack and
        // exercise the merge across mixed table sizes, with every key counted
        // twice (the worker sees its batch twice).
        let n = 1500;
        let small: Vec<i16> = (0..n).map(|i| (i % 100) as i16).collect();
        let names: Vec<String> = (0..n).map(|i| format!("k{i}")).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let vals: Vec<i32> = vec![1; n];
        let batch = || mixed_key_batch(&small, &name_refs, &vals);
        let schema = keys::RowKeySchema::new(vec![DataType::Int16, DataType::Utf8View]);

        let sender = run_row_key_group::<Compiled<(Count,)>>(
            vec![vec![batch(), batch()]],
            vec![0, 1],
            schema,
            count_slots(),
        );

        assert_eq!(sender.total_rows(), n);
        assert!(sender.i64_column(2).into_iter().all(|c| c == 2));
    }

    /// Run a grouped aggregate over the mixed-slot extractor with explicit
    /// slots (its column shapes depend on the slot kinds).
    fn run_mixed_group<const N: usize>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_col: usize,
        value_slots: Vec<AggregationSlot>,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<IntExtractor, values::MixedRowValueExtractor<N>>::new(
                    arena.clone(),
                    state.clone(),
                    injector.clone(),
                    vec![key_col],
                    value_slots.clone(),
                    (),
                    None,
                    None,
                    false,
                    tx.clone(),
                    rx_opt.take(),
                    partition_jobs_injected.clone(),
                    RadixConfig::DEFAULT,
                )
            })
            .collect();
        drop(tx);

        run_consumers(groups, worker_batches)
    }

    /// An `Int32` key column plus a `Utf8View` value column for string extremes.
    fn keyed_string_batch(keys: &[i32], vals: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(StringViewArray::from(vals.to_vec())),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    #[test]
    fn mixed_numeric_min_max_per_group() {
        // MIN(val), MAX(val), COUNT(*) per key.
        let batch = keyed_i32_batch(&[1, 2, 1, 2, 1], &[30, 5, 10, 50, 20]);
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1),
            AggregationSlot::new(AggregationKind::Max, 1),
            AggregationSlot::new(AggregationKind::CountStar, 0),
        ];

        let sender = run_mixed_group::<3>(vec![vec![batch]], 0, slots);

        let mut rows: Vec<(i32, i64, i64, i64)> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .zip(sender.i64_column(2))
            .zip(sender.i64_column(3))
            .map(|(((k, mn), mx), c)| (k, mn, mx, c))
            .collect();
        rows.sort();
        assert_eq!(rows, vec![(1, 10, 30, 3), (2, 5, 50, 2)]);
    }

    #[test]
    fn mixed_string_min_across_workers() {
        // MIN over strings, mixing inline (≤12B) and arena-backed (>12B)
        // candidates, deduped within workers and combined across them in the
        // partition merge.
        let long_low = "aaaa-long-string-over-12-bytes";
        let long_high = "zzzz-long-string-over-12-bytes";
        let w1 = keyed_string_batch(&[1, 1, 2], &["mmm", long_low, long_high]);
        let w2 = keyed_string_batch(&[1, 2, 2], &["zzz", long_high, "bbb"]);
        let slots = vec![
            AggregationSlot::new(AggregationKind::MinStr, 1),
            AggregationSlot::new(AggregationKind::CountStar, 0),
        ];

        let sender = run_mixed_group::<2>(vec![vec![w1], vec![w2]], 0, slots);

        let mut rows: Vec<(i32, String, i64)> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((k, s), c)| (k, s, c))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(1, long_low.to_string(), 3), (2, "bbb".to_string(), 3),]
        );
    }

    #[test]
    fn top_k_limits_output_keeping_partition_maxima() {
        // key 0 dominates (count 100); keys 1..1000 appear once. ORDER BY count
        // LIMIT 1 is applied per partition, collapsing ~1000 groups to at most one
        // row per partition — always the partition's largest, so key 0 survives.
        let mut values: Vec<i32> = vec![0; 100];
        values.extend(1..1000);

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some((0, 1)),
            RadixConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(0, 100)), "dominant group kept");
        assert!(pairs.len() <= PARTITIONS, "at most one row per partition");
        assert!(
            pairs.iter().all(|&(k, c)| k == 0 || c == 1),
            "every survivor is its partition's max"
        );
    }
}
