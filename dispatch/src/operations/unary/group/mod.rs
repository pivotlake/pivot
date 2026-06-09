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
//! When consumption finishes, each worker sends its `Vec<MultiSlabTable>` to a
//! shared mpsc channel and transitions into a [`GroupOutputter`].
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

pub use keys::{ArenaKey, IntKeyExtractor, IntPairKeyExtractor, KeyExtractor, StringKeyExtractor};
pub use values::{
    AggregationKind, AggregationRowValueExtractor, AggregationSlot, Compiled, Count, Sum,
    ValueExtractor,
};

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregatedTable, DEFAULT_CAPACITY, MultiSlabTable, PartitionBuffers, RadixConfig, WorkerOutput,
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

    aggregated_table: AggregatedTable<K, V>,
    sender: mpsc::Sender<WorkerOutput<K, V>>,
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
        top_k: Option<(usize, usize)>,
        sender: mpsc::Sender<WorkerOutput<K, V>>,
        receiver: Option<mpsc::Receiver<WorkerOutput<K, V>>>,
        partition_jobs_injected: Arc<AtomicBool>,
        radix: RadixConfig,
    ) -> Self {
        Self {
            key_cols,
            value_slots,
            outputter: GroupOutputter {
                shared_arena: shared_arena.clone(),
                injector,
                receiver,
                partition_jobs_injected,
                top_k,
                output_allocator: None,
            },
            sender,
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
        self.aggregated_table
            .consume_batch(&batch, &self.key_cols, &self.value_slots);
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
    receiver: Option<mpsc::Receiver<WorkerOutput<K, V>>>,
    partition_jobs_injected: Arc<AtomicBool>,
    top_k: Option<(usize, usize)>,
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
}

unsafe impl<K: KeyExtractor, V: ValueExtractor> Send for PartitionJob<K, V> {}

impl<K: KeyExtractor, V: ValueExtractor> PartitionJob<K, V> {
    /// Merge this partition's scatter buffers and in-place stacks into one result
    /// table and send the output batches, building columns into `allocator`.
    pub fn run<S: Sender<RecordBatch>>(
        self,
        sender: &mut S,
        allocator: &mut SlabAllocator,
    ) -> Result<()> {
        let result_map = merge::merge_combined::<K, V>(
            self.index,
            &self.buffers,
            &self.tables,
            self.partition_capacity,
            self.num_partitions,
            &self.arena,
        );
        if result_map.len() == 0 {
            return Ok(());
        }
        output::build_and_send::<K, V, _, _>(result_map, &self.arena, allocator, self.top_k, sender)
    }
}

impl<K: KeyExtractor, V: ValueExtractor> Outputter<RecordBatch> for GroupOutputter<K, V> {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
            let mut all_tables = Vec::new();
            let mut all_buffers = Vec::new();
            let mut hll = Hll::new();
            for out in rx.into_iter() {
                all_tables.extend(out.tables);
                if let Some(b) = out.buffers {
                    all_buffers.push(b);
                }
                hll.merge(&out.hll);
            }

            // Pick the partition count: nobody switched -> the cheap PARTITIONS-way
            // slot-range merge (don't blow a small group-by like q42 into a
            // 4096-way merge); any switch -> RADIX_PARTITIONS so each radix target
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
                // radix target for ~0.7 load (x3/2 margin) from the distinct estimate.
                let parts = all_buffers[0].0.len();
                let est = hll.estimate();
                (
                    parts,
                    (est * 3 / (2 * parts))
                        .next_power_of_two()
                        .max(DEFAULT_CAPACITY),
                )
            };

            let buffers = Arc::new(all_buffers);
            let tables = Arc::new(all_tables);
            for i in 0..num_partitions {
                self.injector.push(PartitionJob {
                    buffers: buffers.clone(),
                    tables: tables.clone(),
                    index: i,
                    arena: self.shared_arena.clone(),
                    partition_capacity,
                    num_partitions,
                    top_k: self.top_k,
                });
            }

            self.partition_jobs_injected.store(true, Ordering::Relaxed);

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
    fn run_group_full<K: KeyExtractor, V: ValueExtractor>(
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
                    top_k,
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
