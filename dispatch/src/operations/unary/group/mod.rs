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
//!    [`AggregatedTable::merge_array`]. Duplicate keys within the same
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
//! 3. Converts the result table into an Arrow [`RecordBatch`] via
//!    [`KeyExtractor::create_record_batch`] and sends it downstream.
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
//! - [`key_extractions`] — the [`KeyExtractor`] trait and implementations
//!   ([`IntKeyExtractor`], [`StringKeyExtractor`])
//! - [`hashtables`] — `BaseHashTable`, [`AggregatedTable`], [`MultiSlabTable`],
//!   and associated type machinery
//! - [`merge`] — partition-parallel merge of per-worker tables
//! - [`arena`] / [`arena_key`] — shared string storage and the [`ArenaKey`] type
//! - [`factory`] — [`GroupFactory`] for creating per-worker [`Group`] instances
//! - [`aggregations`] — value types (currently [`Count`](aggregations::Count))

mod aggregations;
pub(crate) mod arena;
mod arena_key;
mod factory;
mod key_extractions;
mod merge;

pub use factory::GroupFactory;
mod hashtables;

pub use key_extractions::{IntKeyExtractor, KeyExtractor, StringKeyExtractor};

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{AggregatedTable, MultiSlabTable};
use crate::worker::worker_waker;
use ahash::RandomState;
use arena::SharedArena;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use thiserror::Error;
use tracing::{debug, info};
pub use unary::group::arena_key::{ArenaKey, StringKey};
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

/// Per-worker GROUP BY consumer.
///
/// During the consume phase, each worker owns a `Group` that hashes incoming
/// rows and inserts them into its local [`AggregatedTable`]. When consumption
/// finishes, the accumulated tables are sent to a shared channel and the
/// `Group` transitions into a [`GroupOutputter`] for the merge phase.
pub struct Group<K: KeyExtractor> {
    group_column: usize,

    aggregated_table: AggregatedTable<K>,
    sender: mpsc::Sender<Vec<MultiSlabTable<K>>>,
    outputter: GroupOutputter<K>,
}

impl<K: KeyExtractor> Group<K> {
    pub fn new(
        shared_arena: Arc<SharedArena>,
        state: RandomState,
        injector: Arc<Injector<PartitionJob<K>>>,
        group_column: usize,
        sender: mpsc::Sender<Vec<MultiSlabTable<K>>>,
        receiver: Option<mpsc::Receiver<Vec<MultiSlabTable<K>>>>,
        partition_jobs_injected: Arc<AtomicBool>,
    ) -> Self {
        Self {
            group_column,
            outputter: GroupOutputter {
                shared_arena: shared_arena.clone(),
                injector,
                receiver,
                partition_jobs_injected,
            },
            sender,
            aggregated_table: AggregatedTable::new(state, shared_arena),
        }
    }
}

impl<K: KeyExtractor> Consumer<RecordBatch, RecordBatch> for Group<K> {
    type Outputter = GroupOutputter<K>;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        self.aggregated_table
            .merge_array(batch.column(self.group_column));
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
pub struct GroupOutputter<K: KeyExtractor> {
    shared_arena: Arc<SharedArena>,
    injector: Arc<Injector<PartitionJob<K>>>,
    receiver: Option<mpsc::Receiver<Vec<MultiSlabTable<K>>>>,
    partition_jobs_injected: Arc<AtomicBool>,
}

/// A single partition's merge work unit.
///
/// Created by the [`GroupOutputter`] that holds the receiver and pushed to a
/// shared [`Injector`] for work-stealing execution. Each job merges all
/// source tables for partition `index` into one result table and sends the
/// output as a [`RecordBatch`].
pub struct PartitionJob<K: KeyExtractor> {
    tables: Arc<Vec<MultiSlabTable<K>>>,
    index: usize,
    arena: Arc<SharedArena>,
    partition_capacity: usize,
}

unsafe impl<K: KeyExtractor> Send for PartitionJob<K> {}

impl<K: KeyExtractor> PartitionJob<K> {
    /// Merge all source tables for this partition and send the result batch.
    pub fn run<S: Sender<RecordBatch>>(self, sender: &mut S) -> Result<()> {
        debug!("Merging maps for partition {:?}...", self.index);
        let result_map = merge::merge_partition::<K>(
            self.index,
            &self.tables,
            &self.arena,
            self.partition_capacity,
        );

        if result_map.len() == 0 {
            return Ok(());
        }

        debug!("Sending record batch {:?}", self.index);
        sender.send(K::create_record_batch(result_map, &self.arena)?)?;
        Ok(())
    }
}

impl<K: KeyExtractor> Outputter<RecordBatch> for GroupOutputter<K> {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
            let tables: Vec<MultiSlabTable<K>> = rx.into_iter().flatten().collect::<Vec<_>>();
            debug!("Outputting {:?} maps", tables.len());
            let total_entries: usize = tables.iter().map(|m| m.capacity()).sum::<usize>() / 2;
            let partition_capacity = (total_entries / PARTITIONS).next_power_of_two();
            info!("Partition capacity {:?}", partition_capacity);
            let tables = Arc::new(tables);
            for i in 0..PARTITIONS {
                self.injector.push(PartitionJob {
                    tables: tables.clone(),
                    index: i,
                    arena: self.shared_arena.clone(),
                    partition_capacity,
                })
            }
            self.partition_jobs_injected.store(true, Ordering::Relaxed);

            // Wake up all workers so that they can start working on partitions
            worker_waker().notify();
        }

        let steal = self.injector.steal();
        match steal {
            Steal::Success(job) => {
                job.run(sender).map_err(unary::Error::from)?;
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
    use crate::operations::unary::group::key_extractions::IntKeyExtractor;
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    type IntExtractor = IntKeyExtractor<Int32Type>;

    fn batch_with_column(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn run_group(worker_batches: Vec<Vec<RecordBatch>>) -> CollectSender {
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
                Group::<IntExtractor>::new(
                    arena.clone(),
                    state.clone(),
                    injector.clone(),
                    0,
                    tx.clone(),
                    rx_opt.take(),
                    partition_jobs_injected.clone(),
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
}
