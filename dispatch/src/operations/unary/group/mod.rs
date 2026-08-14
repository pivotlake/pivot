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
//! 3. If the hash table exceeds its load threshold, a new table is created and
//!    subsequent rows go there. The old table is kept — its entries will be
//!    merged in phase 2. See [Why partitioned merging works](#why-partitioned-merging-works-across-different-table-sizes).
//!    The replacement is larger while a table still fits a core's private
//!    cache, and the same size once it does not, so an input of any
//!    cardinality is absorbed by a growing stack of cache-resident tables
//!    rather than by one table that outgrows the cache.
//!
//! When consumption finishes, each worker sends its tables to a shared gather
//! barrier and transitions into a [`GroupOutputter`].
//!
//! ## Phase 2: Output / Merge (parallel via work-stealing)
//!
//! The final worker to reach the gather barrier collects every worker's tables
//! and publishes [`PARTITIONS`] independent
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
//!    columns from the [`AggregationValue`] — and sends it downstream.
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
//! - [`values`] — the [`AggregationValue`] trait and its container
//!   implementations (`Compiled`, `Dynamic`), the per-op folds (`Count`, `Sum`,
//!   …), plus [`AggregationKind`]/[`AggregationSlot`]
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
    ArenaKey, HashOnlyIntKeyExtractor, IntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor,
    KeyExtractor, RowKeyExtractor, RowKeySchema, StringKeyExtractor,
};
pub(crate) use values::cast_value_column;
pub use values::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot,
    CountValidSlot, Distinct, Dynamic, F64Cell, F64Max, F64Min, F64Sum, Fold, IntCell, IntRead,
    Max, MaxSlot, Min, MinSlot, NoRead, OpTuple, Read, SharedContext, StrMax, StrMin, StrRead, Sum,
    SumSlot, U128Max, U128Min, U128Sum, WideCell, WideSum, WorkerContext,
};

use crate::gather_barrier::GatherBarrier;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregatedTable, AggregatedTableOutput, BUCKET_COUNT, DEFAULT_CAPACITY, MAX_LOAD_FACTOR,
    MergeSource, MultiSlabTable, SpillConfig,
};
use crate::worker::current_node;
use ahash::RandomState;
use arena::SharedArena;
use arrow_array::RecordBatch;
use arrow_buffer::Buffer;
use arrow_schema::{ArrowError, DataType};
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

/// A LIMIT pushed down into a grouped aggregate, so each partition emits only
/// the rows the downstream LIMIT can keep instead of every group.
///
/// Both forms are decomposable across the hash-disjoint partitions: the global
/// answer is recovered by re-applying the same LIMIT (and any ORDER BY) above.
#[derive(Clone, Copy, Debug)]
pub enum GroupLimit {
    /// `ORDER BY <slot> DESC LIMIT n`: keep this partition's top-`n` groups by
    /// `V::sort_key(value, slot)`.
    TopK { slot: usize, limit: usize },
    /// Plain `LIMIT n` with no ORDER BY: keep any `n` groups from this partition
    /// (SQL leaves which rows arbitrary, so the first `n` encountered suffice).
    First { limit: usize },
}

impl GroupLimit {
    /// The most rows this limit can let through, regardless of form — used to size
    /// the output builders.
    pub(crate) fn row_limit(self) -> usize {
        match self {
            GroupLimit::TopK { limit, .. } | GroupLimit::First { limit } => limit,
        }
    }
}

/// Minimum number of hash partitions for the output merge phase. Each
/// partition is merged independently, enabling parallel output; the actual
/// count is raised to the worker count (see [`merge_partition_floor`]) so a
/// large pool doesn't idle behind too few jobs.
const PARTITIONS: usize = 64;

/// Minimum merge input (source-table slots scanned) per merge job. Each job
/// walks its slice of every source table and pays a fixed setup cost (a slab
/// buffer, a target table, its output batches), so below this much input per
/// job, more jobs just multiply setup: a group-by with a handful of groups
/// would otherwise fan out into the full worker-count floor of near-empty
/// jobs. The cap only bites on small inputs; anything sizable saturates the
/// floor anyway.
const MIN_MERGE_INPUT_SLOTS_PER_JOB: usize = 16 * 1024;

/// Slot count each merge job's result table is sized for. The job count is
/// derived from it, so this is the one knob that decides both.
const MERGE_TARGET_SLOTS: usize = 32768;

/// Compute the merge-phase partition floor. The merge uses at least
/// [`PARTITIONS`] partitions and at least one job per contributing worker
/// (rounded up to a power of two, which the hash-top-bits partitioning
/// requires), so every core has merge work.
fn merge_partition_floor(contributing_workers: usize) -> usize {
    PARTITIONS.max(contributing_workers.next_power_of_two())
}

/// Per-worker GROUP BY consumer.
///
/// During the consume phase, each worker owns a `Group` that hashes incoming
/// rows and inserts them into its local [`AggregatedTable`]. When consumption
/// finishes, the accumulated tables are published to a shared gather barrier,
/// and the `Group` transitions into a [`GroupOutputter`] for the merge phase.
pub struct Group<K: KeyExtractor, V: AggregationValue + ?Sized> {
    key_cols: Vec<usize>,
    /// Slots drive the per-batch value reader (which column / `COUNT` vs `SUM`).
    value_slots: Vec<AggregationSlot>,
    /// The value's shared, read-side context, built once from the slots + value
    /// arena (like `key_config`); resolves string keys during merge/output.
    shared_context: V::SharedContext,
    key_config: K::Config,

    aggregated_table: AggregatedTable<K, V>,
    gather: Arc<GatherBarrier<AggregatedTableOutput<K, V>>>,
    outputter: GroupOutputter<K, V>,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Group<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_arena: Arc<SharedArena>,
        value_arena: Arc<SharedArena>,
        state: RandomState,
        injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        key_config: K::Config,
        output_limit: Option<GroupLimit>,
        count_only: bool,
        gather: Arc<GatherBarrier<AggregatedTableOutput<K, V>>>,
        partition_jobs_injected: Arc<AtomicBool>,
        spill: SpillConfig,
    ) -> Self {
        let shared_context = <V::SharedContext as SharedContext>::build(&value_slots, &value_arena);
        let value_output_types: Arc<[DataType]> =
            value_slots.iter().map(|s| s.output_type.clone()).collect();
        // The per-worker write context is spawned from the shared one (a fresh
        // `WorkerArena` for a string value, `()` for numeric).
        let aggregated_table = AggregatedTable::new(
            state,
            key_arena.clone(),
            shared_context.clone(),
            shared_context.worker(),
            spill,
        );
        Self {
            key_cols,
            value_slots,
            shared_context: shared_context.clone(),
            key_config: key_config.clone(),
            outputter: GroupOutputter {
                key_arena,
                node: current_node(),
                injectors,
                partition_jobs_injected,
                key_config,
                shared_context,
                value_output_types,
                output_limit,
                count_only,
                zero_hash_pending: false,
                output_allocator: None,
                output_accumulator: None,
            },
            gather,
            aggregated_table,
        }
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Consumer<RecordBatch, RecordBatch>
    for Group<K, V>
{
    type Outputter = GroupOutputter<K, V>;

    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        self.aggregated_table.consume_batch(
            &batch,
            &self.key_cols,
            &self.value_slots,
            &self.key_config,
            &self.shared_context,
        );
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        let tables = self.aggregated_table.flush();
        self.gather.arrive(tables, |values| {
            self.outputter.create_partition_jobs(values)
        });
        Ok(Some(self.outputter))
    }
}

/// Handles the output (merge) phase of a GROUP BY.
///
/// The last worker to reach the gather barrier collects all per-worker tables
/// and publishes [`PARTITIONS`] [`PartitionJob`]s to the shared
/// work-stealing [`Injector`]. All workers (including the one that injected)
/// then steal and execute jobs until the injector is empty.
pub struct GroupOutputter<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// String *key* storage; backs the leading key column(s) at output.
    key_arena: Arc<SharedArena>,
    /// One job queue per NUMA node. A node-level merge job reads that node's
    /// tables, so it is queued on (and preferentially run by) that node's
    /// workers; a worker falls back to other nodes' queues only when its own
    /// is empty, trading locality for tail balance.
    injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
    /// This worker's node, i.e. which of `injectors` is local to it.
    node: usize,
    partition_jobs_injected: Arc<AtomicBool>,
    key_config: K::Config,
    /// The value's shared context, threaded into each [`PartitionJob`] so the merge
    /// folds existing entries via [`AggregationValue::merge_from`] and the output
    /// resolves string extremes (it carries the value arena).
    shared_context: V::SharedContext,
    /// The declared output type of each value column, in slot order. The output
    /// phase renders each accumulator at its storage width then casts it to this
    /// type (a `COUNT` in an `i128` cell down to `Int64`, a narrow `SUM` up to
    /// `Decimal128`); a no-op when the width already matches.
    value_output_types: Arc<[DataType]>,
    output_limit: Option<GroupLimit>,
    /// Global `COUNT(DISTINCT)`: emit each partition's distinct-key count instead
    /// of its keys (a downstream `SUM` totals them).
    count_only: bool,
    /// The final gather arrival saw the hash-only extractor's out-of-band zero
    /// hash and must emit its extra count row once.
    zero_hash_pending: bool,
    /// One allocator per worker for the output columns of every partition this
    /// worker handles, so small per-partition outputs pack into shared buffers
    /// instead of each grabbing a fresh 2MB one. Created lazily on the first job.
    output_allocator: Option<SlabAllocator>,
    /// Per-worker output accumulator: packs the rows from every partition job this
    /// worker runs into full output-chunk batches, so a high partition count (the
    /// radix path) doesn't emit one tiny batch per partition. Flushed when the queue
    /// drains.
    ///
    /// `Option` because it can't exist before the worker's first job: the
    /// per-output-phase buffers it needs are computed once (by the final worker
    /// at the gather barrier) and arrive attached to each [`PartitionJob`], so a
    /// worker only learns them from the first job it steals. A worker running only
    /// count-only `COUNT(DISTINCT)` jobs never builds one.
    output_accumulator: Option<output::OutputAccumulator<K, V>>,
}

/// One (NUMA node, partition) merge work unit.
///
/// Created by the final worker to reach the gather barrier and pushed to the
/// node's [`Injector`] for work-stealing execution. Each job merges one node's
/// source tables for partition `index` into one result table. On a single-node
/// pool (`cross_node_merge` is `None`) that result is the partition's final
/// table and its rows are emitted directly. With multiple nodes, each node's
/// job merges only node-local memory and sends the resulting node table into
/// the shared [`CrossNodeMerge`]; the last job to finish receives every node's
/// table and merges them into the final one. That last merge is the only step
/// that reads another node's memory.
pub struct PartitionJob<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// One source per contributing worker on this job's node set. Jobs walk
    /// each source's partition slice at `num_partitions` granularity.
    sources: Arc<Vec<MergeSource<K::Persisted, V>>>,
    index: usize,
    /// One shared instance per partition on a multi-node hierarchical merge;
    /// `None` when this job's merge result is already final (single-node pool
    /// or direct merge).
    cross_node_merge: Option<Arc<CrossNodeMerge<K, V>>>,
    key_arena: Arc<SharedArena>,
    /// The arena's ring buffers wrapped as Arrow `Buffer`s, built once for the
    /// whole output phase and shared by every partition job. String-key output
    /// emits zero-copy views into this; cloning the `Arc` is a single bump.
    output_buffers: Arc<[Buffer]>,
    partition_capacity: usize,
    /// How many ways the key space is split for the merge.
    num_partitions: usize,
    key_config: K::Config,
    /// The value's shared context, for the partition merge's entry fold + output.
    shared_context: V::SharedContext,
    /// Declared output type per value column, in slot order; the output phase casts
    /// each finished value column to its type.
    value_output_types: Arc<[DataType]>,
    output_limit: Option<GroupLimit>,
    count_only: bool,
}

unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for PartitionJob<K, V> {}

/// One partition's pending cross-node merge: each node's job sends the node's
/// merged aggregated table; the send that completes the set hands every
/// node's table back to that caller, electing it to run the final merge
/// ([`merge::merge_node_aggregated_tables`]) and emit. Senders never block
/// and no job ever waits: the mailbox is a lock-free queue and the election
/// is one atomic countdown.
struct CrossNodeMerge<K: KeyExtractor, V: AggregationValue + ?Sized> {
    node_tables: Injector<MultiSlabTable<K::Persisted, V>>,
    /// Sends still outstanding; the sender that decrements this to zero is
    /// the receiver.
    pending_sends: AtomicUsize,
}

unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for CrossNodeMerge<K, V> {}
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Sync for CrossNodeMerge<K, V> {}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> CrossNodeMerge<K, V> {
    fn new(node_count: usize) -> Self {
        Self {
            node_tables: Injector::new(),
            pending_sends: AtomicUsize::new(node_count),
        }
    }

    /// Send this node's merged table. Exactly one call - the last - returns
    /// the full set; that caller must run the final merge. Each sender's push
    /// happens-before its countdown decrement, and the last sender's
    /// decrement observes all of them, so the drain below is guaranteed to
    /// see every node's table (and after the election nobody else touches
    /// the queue).
    fn send(
        &self,
        table: MultiSlabTable<K::Persisted, V>,
    ) -> Option<Vec<MultiSlabTable<K::Persisted, V>>> {
        self.node_tables.push(table);
        if self.pending_sends.fetch_sub(1, Ordering::AcqRel) != 1 {
            return None;
        }
        let mut all = Vec::new();
        loop {
            match self.node_tables.steal() {
                Steal::Success(table) => all.push(table),
                Steal::Empty => return Some(all),
                Steal::Retry => continue,
            }
        }
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> PartitionJob<K, V> {
    /// Merge this partition's share of every source table into one result
    /// table, then feed its rows into the worker's shared `acc` (building
    /// columns into `allocator`). Accumulating across partition jobs, rather
    /// than emitting one batch per job, keeps a high partition count from
    /// producing a tiny `RecordBatch` each time.
    pub fn run_into(
        self,
        acc: &mut Option<output::OutputAccumulator<K, V>>,
        sender: &mut dyn Sender<RecordBatch>,
        allocator: &mut SlabAllocator,
    ) -> Result<()> {
        let result_map = merge::merge_combined::<K::Stored, V>(
            self.index,
            &self.sources,
            self.partition_capacity,
            self.num_partitions,
            &self.key_arena,
            &self.shared_context,
        );
        let result_map = match &self.cross_node_merge {
            None => result_map,
            Some(cross_node) => match cross_node.send(result_map) {
                // Another node's job for this partition is still running; it
                // will receive the tables and run the final merge.
                None => return Ok(()),
                Some(node_tables) => merge::merge_node_aggregated_tables::<K::Stored, V>(
                    node_tables,
                    self.partition_capacity,
                    self.num_partitions.trailing_zeros(),
                    &self.key_arena,
                    &self.shared_context,
                ),
            },
        };
        if result_map.len() == 0 {
            return Ok(());
        }
        // Global COUNT(DISTINCT) needs only each partition's distinct-key count,
        // not the keys, so it bypasses the accumulator and emits a single row.
        if self.count_only {
            return output::emit_count(result_map.len(), sender);
        }
        let acc = acc.get_or_insert_with(|| {
            output::OutputAccumulator::new(
                allocator,
                self.output_limit,
                self.key_arena.clone(),
                self.output_buffers.clone(),
                self.key_config.clone(),
                self.shared_context.clone(),
                self.value_output_types.clone(),
            )
        });
        acc.extend_from_table(result_map, allocator, &mut *sender)
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> GroupOutputter<K, V> {
    /// Combine every worker's gathered tables, size the merge from the merged
    /// distinct estimate, and publish one [`PartitionJob`] per merge partition to
    /// the shared injector. Run exactly once, by the last worker to reach the
    /// gather barrier.
    fn create_partition_jobs(&mut self, outputs: Vec<AggregatedTableOutput<K, V>>) {
        let node_count = self.injectors.len();
        let mut sources_by_node: Vec<Vec<MergeSource<K::Persisted, V>>> =
            (0..node_count).map(|_| Vec::new()).collect();
        let mut hll = Hll::new();
        let mut contributing_workers = 0usize;
        for out in outputs {
            contributing_workers += 1;
            sources_by_node[out.node].push(out.source);
            hll.merge(&out.hll);
            self.zero_hash_pending |= out.zero_hash_seen;
        }
        let partition_floor = merge_partition_floor(contributing_workers);
        let total_in_place: usize = sources_by_node.iter().flatten().map(|s| s.len()).sum();
        // Every worker folds each table into the sketch as that table is
        // retired, so this is the distinct group count across the whole pool.
        // The entry total is not a stand-in for it: a key recurring in several
        // tables is counted once here and once per table there.
        let estimate = hll.estimate();

        // Decide how many independent merge jobs to run (`num_partitions`) and how
        // large each job's result table starts (`partition_capacity`). Every job
        // builds one hash table and probes it at random, so the win is keeping
        // that table near the core: otherwise each probe is a memory round trip.
        //
        // The job count follows from the target size rather than the other way
        // round: pick how big one result table should be, then run as many jobs
        // as it takes to keep every target that size. Splitting the key space
        // further always shrinks the targets, so there is no upper bound here
        // beyond what the input volume justifies.
        let (num_partitions, partition_capacity) = {
            let input_entries: usize = sources_by_node.iter().flatten().map(|s| s.len()).sum();
            // The count never drops below 2: the merge routes rows by their
            // hash's top `log2(partitions)` bits, and a 0-bit partition id
            // has no valid shift (and no benefit over 2 near-empty jobs).
            let volume_cap = (input_entries / MIN_MERGE_INPUT_SLOTS_PER_JOB)
                .max(2)
                .next_power_of_two();
            let groups_per_target = ((MERGE_TARGET_SLOTS as f64 * MAX_LOAD_FACTOR) as usize).max(1);
            // The floor keeps every core busy without splitting a small input
            // into near-empty jobs.
            let by_target = estimate
                .div_ceil(groups_per_target)
                .max(1)
                .next_power_of_two();
            let floor = partition_floor.min(volume_cap).max(2);
            // The per-run bucket index resolves `BUCKET_COUNT` hash ranges,
            // which caps how finely jobs can split the key space.
            let partitions = by_target.max(floor).min(BUCKET_COUNT);
            (
                partitions,
                // Start each target big enough to hold its share at
                // MAX_LOAD_FACTOR (the merge's resize threshold), so it fills
                // without resizing mid-merge.
                ((estimate as f64 / partitions as f64 / MAX_LOAD_FACTOR).ceil() as usize)
                    .next_power_of_two()
                    .max(DEFAULT_CAPACITY),
            )
        };

        // Node-hierarchical or direct merge? Hierarchical (one job per node
        // and partition, then a final merge of the per-node tables) keeps
        // each job's reads node-local, but pays an extra materialization of
        // the node tables. That trade only wins when keys repeat across the
        // input enough that per-node aggregation shrinks what the cross-node
        // merge must touch. With near-unique keys (input close to the
        // distinct estimate) the node tables would be as large as the input,
        // so merge every node's sources directly and pay the remote reads
        // once. The entry total exceeds the distinct estimate exactly when
        // keys repeat across tables and workers, which is the condition that
        // makes a per-node pass worth its extra materialization.
        let hierarchical = node_count > 1 && total_in_place > 2 * estimate;
        // Wrap the arena's ring buffers once for the whole output phase (consume
        // is done, so `next_idx` is final). Every partition job shares this one
        // `Arc<[Buffer]>` for zero-copy string output, so a batch attaches it with
        // a single refcount bump instead of re-wrapping every buffer per batch.
        // Held by the jobs (and the batches they emit), never by the arena, so
        // there is no `arena -> Buffer -> arena` cycle.
        let output_buffers: Arc<[Buffer]> = self.key_arena.to_arrow_buffers();
        // All jobs are pushed before the injected flag flips, so a drained
        // queue means a finished phase.
        let push_job = |injector: &Injector<PartitionJob<K, V>>,
                        index: usize,
                        cross_node_merge: Option<Arc<CrossNodeMerge<K, V>>>,
                        sources: Arc<Vec<MergeSource<K::Persisted, V>>>| {
            injector.push(PartitionJob {
                sources,
                index,
                cross_node_merge,
                key_arena: self.key_arena.clone(),
                output_buffers: output_buffers.clone(),
                partition_capacity,
                num_partitions,
                key_config: self.key_config.clone(),
                shared_context: self.shared_context.clone(),
                value_output_types: self.value_output_types.clone(),
                output_limit: self.output_limit,
                count_only: self.count_only,
            });
        };
        if hierarchical {
            // One job per (node, partition), queued on the owning node so the
            // bulk of every merge reads node-local memory. Each node table
            // starts at the full `partition_capacity`, not a per-node share:
            // this path is chosen exactly when keys repeat across workers,
            // and row groups are hash-assigned to nodes, so a repeating key
            // reaches every node and each node's table converges toward the
            // partition's full distinct count. A per-node share would
            // guarantee a mid-merge resize; the full size only costs
            // transient memory the final cross-node merge frees.
            let sources_by_node: Vec<_> = sources_by_node.into_iter().map(Arc::new).collect();
            for i in 0..num_partitions {
                let cross_node_merge = Arc::new(CrossNodeMerge::new(node_count));
                for (node, injector) in self.injectors.iter().enumerate() {
                    push_job(
                        injector,
                        i,
                        Some(cross_node_merge.clone()),
                        sources_by_node[node].clone(),
                    );
                }
            }
        } else {
            // One job per partition over every node's sources, spread across
            // the node queues so all workers share the load.
            let sources = Arc::new(sources_by_node.into_iter().flatten().collect::<Vec<_>>());
            for i in 0..num_partitions {
                push_job(&self.injectors[i % node_count], i, None, sources.clone());
            }
        }

        // Release pairs with the Acquire load in `output`: a worker that sees
        // the flag also sees every job pushed above it. With relaxed ordering
        // another core may observe the flag before the pushes and conclude
        // from a still-empty queue that the merge phase is over.
        self.partition_jobs_injected.store(true, Ordering::Release);
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Outputter<RecordBatch>
    for GroupOutputter<K, V>
{
    fn output(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if self.zero_hash_pending && self.count_only {
            self.zero_hash_pending = false;
            output::emit_count(1, sender).map_err(unary::Error::from)?;
        }

        // Claim a job, trying the own node's queue first (its sources are
        // node-local), then other nodes' queues so a node that finished its
        // share helps with the tail instead of idling. The plain-load
        // `is_empty` pre-checks keep the drained-queue polls (every pass
        // until the dataflow finishes) from pinning a crossbeam epoch each
        // time.
        let queue_count = self.injectors.len();
        let mut steal = Steal::Empty;
        for offset in 0..queue_count {
            let queue = &self.injectors[(self.node + offset) % queue_count];
            if queue.is_empty() {
                continue;
            }
            match queue.steal() {
                Steal::Empty => {}
                s => {
                    steal = s;
                    break;
                }
            }
        }
        match steal {
            Steal::Success(job) => {
                let allocator = self
                    .output_allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                job.run_into(&mut self.output_accumulator, sender, allocator)
                    .map_err(unary::Error::from)?;
            }
            Steal::Empty => {
                if self.partition_jobs_injected.load(Ordering::Acquire) {
                    // Queue drained: emit this worker's last partial batch.
                    if let (Some(acc), Some(allocator)) = (
                        self.output_accumulator.as_mut(),
                        self.output_allocator.as_mut(),
                    ) {
                        acc.flush(allocator, sender).map_err(unary::Error::from)?;
                    }
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
    use crate::operations::unary::group::keys::{
        IntKeyExtractor, IntStrKeyExtractor, StringKeyExtractor,
    };
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    type IntExtractor = IntKeyExtractor<Int32Type>;
    type CountValue = Compiled<(CountSlot,)>;
    type SumValue = Compiled<(SumSlot<Int32Type>,)>;

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

    /// A `Utf8View` batch whose repeated values share one view, the way a
    /// dictionary-encoded column decodes: every occurrence of a value resolves
    /// to the dictionary's own view rather than to a per-row copy of the bytes.
    fn shared_view_batch(values: &[&str]) -> RecordBatch {
        use arrow_array::builder::make_view;
        let mut blob = Vec::new();
        let mut views_by_value: Vec<(&str, u128)> = Vec::new();
        for value in values {
            if views_by_value.iter().any(|(v, _)| v == value) {
                continue;
            }
            let offset = blob.len() as u32;
            blob.extend_from_slice(value.as_bytes());
            views_by_value.push((value, make_view(value.as_bytes(), 0, offset)));
        }
        let views: Vec<u128> = values
            .iter()
            .map(|value| {
                views_by_value
                    .iter()
                    .find(|(v, _)| v == value)
                    .expect("every value was registered")
                    .1
            })
            .collect();
        use arrow_buffer::ScalarBuffer;
        let array = unsafe {
            StringViewArray::new_unchecked(
                ScalarBuffer::from(views),
                vec![Buffer::from(blob)],
                None,
            )
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            DataType::Utf8View,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(array) as ArrayRef]).unwrap()
    }

    fn count_slots() -> Vec<AggregationSlot> {
        vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )]
    }

    /// Common case: `Int32` keys, `COUNT(*)`, key column 0, no LIMIT, default spill.
    fn run_group(worker_batches: Vec<Vec<RecordBatch>>) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            SpillConfig::DEFAULT,
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

    /// Common case with a chosen [`SpillConfig`] — a small capacity makes a
    /// worker stack several tables within the test slab pool.
    fn run_group_with_spill(
        worker_batches: Vec<Vec<RecordBatch>>,
        spill: SpillConfig,
    ) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            spill,
        )
    }

    /// General harness: choose the key/value extractors, key columns, aggregates,
    /// LIMIT pushdown, and spill sizing. The common-case wrappers above cover
    /// `Int32` keys + `COUNT(*)`.
    fn run_group_full<K: KeyExtractor<Config: Default>, V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        output_limit: Option<GroupLimit>,
        spill: SpillConfig,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let key_arena = SharedArena::new(64);
        let value_arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(vec![Injector::new()]);
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let gather = Arc::new(GatherBarrier::new(worker_count));

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<K, V>::new(
                    key_arena.clone(),
                    value_arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    K::Config::default(),
                    output_limit,
                    false,
                    gather.clone(),
                    partition_jobs_injected.clone(),
                    spill,
                )
            })
            .collect();

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
    fn stacked_tables_single_worker_aggregates_counts() {
        // A small capacity stops growth after ~a couple hundred keys, so one
        // worker spreads 500 keys over a stack of same-sized tables and the
        // merge has to combine them.
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let mut values: Vec<i32> = (0..500).collect();
        values.extend(0..500);

        let sender = run_group_with_spill(vec![vec![batch_with_column(&values)]], spill);

        assert_eq!(
            group_counts(&sender),
            (0..500).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn stacked_tables_merge_across_workers() {
        // Two workers each build a stack, so every key is spread over several
        // tables per worker and the merge has to find all of them.
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let worker = || vec![batch_with_column(&(0..500).collect::<Vec<_>>())];

        let sender = run_group_with_spill(vec![worker(), worker()], spill);

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
    fn shared_views_counted_once_per_distinct_key() {
        // Repeats share a view, so they take the memo's fast path; the memo has
        // to land them on the same group the first occurrence created.
        let values: Vec<&str> = (0..600)
            .map(|i| ["alpha", "beta", "gamma"][i % 3])
            .collect();

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![vec![shared_view_batch(&values)]],
            vec![0],
            count_slots(),
            None,
            SpillConfig::DEFAULT,
        );

        assert_eq!(
            string_group_pairs(&sender),
            vec![
                ("alpha".to_string(), 200),
                ("beta".to_string(), 200),
                ("gamma".to_string(), 200)
            ]
        );
    }

    #[test]
    fn shared_views_survive_table_retirement() {
        // A tiny capacity retires tables mid-batch, which invalidates the memo;
        // a stale remembered address would fold counts into a dead table.
        let values: Vec<String> = (0..4000).map(|i| format!("key-{}", i % 900)).collect();
        let refs: Vec<&str> = values.iter().map(|v| v.as_str()).collect();

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![vec![shared_view_batch(&refs)]],
            vec![0],
            count_slots(),
            None,
            SpillConfig {
                spill_capacity: 256,
            },
        );

        let pairs = string_group_pairs(&sender);
        assert_eq!(pairs.len(), 900);
        assert_eq!(
            pairs.iter().map(|(_, count)| *count).sum::<i64>(),
            4000,
            "every row lands in exactly one group"
        );
    }

    #[test]
    fn string_keys_dedup_and_count() {
        let batches = vec![vec![string_key_batch(&["a", "b", "a", "c", "b", "a"])]];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            batches,
            vec![0],
            count_slots(),
            None,
            SpillConfig::DEFAULT,
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
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            1,
            DataType::Decimal128(38, 0),
        )];

        let sender = run_group_full::<IntExtractor, SumValue>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            SpillConfig::DEFAULT,
        );

        // A SUM renders as a Decimal128 column.
        let mut pairs: Vec<(i32, i128)> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.decimal128_column(1))
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(1, 40), (2, 20), (3, 10)]);
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
            Some(GroupLimit::TopK { slot: 0, limit: 1 }),
            SpillConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(0, 100)), "dominant group kept");
        assert!(pairs.len() <= PARTITIONS, "at most one row per partition");
        assert!(
            pairs.iter().all(|&(k, c)| k == 0 || c == 1),
            "every survivor is its partition's max"
        );
    }

    #[test]
    fn first_limit_caps_each_partition_without_sorting() {
        // A plain LIMIT (no ORDER BY) keeps *any* `limit` groups per partition.
        // 1000 distinct keys, each once: with `limit = 1` each partition emits at
        // most one of its groups, so the output is capped at one row per partition
        // and every survivor is a real (key, count == 1) group.
        let values: Vec<i32> = (0..1000).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::First { limit: 1 }),
            SpillConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(!pairs.is_empty(), "some groups kept");
        assert!(
            pairs.len() <= PARTITIONS,
            "at most `limit` rows per partition"
        );
        assert!(
            pairs.iter().all(|&(k, c)| (0..1000).contains(&k) && c == 1),
            "every survivor is a real group"
        );
    }

    // ---- Row-encoded multi-column keys (`RowKeyExtractor`) ----

    /// A mixed-key batch: an `Int64` key, a `Utf8View` key, and an `Int32` value.
    fn mixed_key_batch(ids: &[i64], names: &[&str], vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let id: ArrayRef = Arc::new(arrow_array::Int64Array::from(ids.to_vec()));
        let name: ArrayRef = Arc::new(StringViewArray::from(names.to_vec()));
        let val: ArrayRef = Arc::new(Int32Array::from(vals.to_vec()));
        RecordBatch::try_new(schema, vec![id, name, val]).unwrap()
    }

    fn run_row_key_group<V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        schema: RowKeySchema,
        value_slots: Vec<AggregationSlot>,
    ) -> CollectSender {
        run_row_key_group_radix::<V>(
            worker_batches,
            key_cols,
            schema,
            value_slots,
            SpillConfig::DEFAULT,
        )
    }

    fn run_row_key_group_radix<V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        schema: RowKeySchema,
        value_slots: Vec<AggregationSlot>,
        spill: SpillConfig,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let key_arena = SharedArena::new(64);
        let value_arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(vec![Injector::new()]);
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let gather = Arc::new(GatherBarrier::new(worker_count));
        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<RowKeyExtractor, V>::new(
                    key_arena.clone(),
                    value_arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    schema.clone(),
                    None,
                    false,
                    gather.clone(),
                    partition_jobs_injected.clone(),
                    spill,
                )
            })
            .collect();
        run_consumers(groups, worker_batches)
    }

    /// Read the `(id, name, value)` output rows of a row-key group-by, sorted.
    fn row_key_rows(sender: &CollectSender) -> Vec<(i64, String, i64)> {
        let mut rows: Vec<(i64, String, i64)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        rows
    }

    /// Read the `(id, name, sum)` output rows of a row-key group-by whose value is
    /// a `SUM` (a `Decimal128` column), sorted.
    fn row_key_sum_rows(sender: &CollectSender) -> Vec<(i64, String, i128)> {
        let mut rows: Vec<(i64, String, i128)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.decimal128_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        rows
    }

    /// `GROUP BY (Int64, Utf8View)` — each distinct tuple is one group, counted.
    #[test]
    fn row_key_mixed_int_string_counts() {
        let batch = mixed_key_batch(&[1, 1, 1, 2, 2], &["a", "b", "a", "a", "a"], &[0; 5]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let sender = run_row_key_group::<Compiled<(CountSlot,)>>(
            vec![vec![batch]],
            vec![0, 1],
            schema,
            count_slots(),
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 2),
            ]
        );
    }

    /// `IntStrKeyExtractor`: `GROUP BY (Int64, Utf8View)` through the dedicated
    /// int+string extractor instead of the row encoder. Same groups as
    /// [`row_key_mixed_int_string_counts`], but keyed on the native int beside the
    /// string's arena handle.
    #[test]
    fn int_string_mixed_counts() {
        let batch = mixed_key_batch(&[1, 1, 1, 2, 2], &["a", "b", "a", "a", "a"], &[0; 5]);
        let sender = run_group_full::<
            IntStrKeyExtractor<arrow_array::types::Int64Type>,
            Compiled<(CountSlot,)>,
        >(
            vec![vec![batch]],
            vec![0, 1],
            count_slots(),
            None,
            SpillConfig::DEFAULT,
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 2),
            ]
        );
    }

    /// `IntStrKeyExtractor` across two workers: the same `(int, string)` group is
    /// seen by both, so the merge must resolve each persisted key through the
    /// shared arena (the `IntStrResolvedKey` path) and combine. Includes a
    /// > 12-byte string so the string rides an arena blob, not the inline header.
    #[test]
    fn int_string_two_workers_merge() {
        let long = "a-string-well-over-twelve-bytes";
        let sender = run_group_full::<
            IntStrKeyExtractor<arrow_array::types::Int64Type>,
            Compiled<(CountSlot,)>,
        >(
            vec![
                vec![mixed_key_batch(&[1, 2], &[long, "a"], &[0; 2])],
                vec![mixed_key_batch(&[1, 1], &[long, "b"], &[0; 2])],
            ],
            vec![0, 1],
            count_slots(),
            None,
            SpillConfig::DEFAULT,
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, long.to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 1),
            ]
        );
    }

    /// `IntStrKeyExtractor` in string-first order: `GROUP BY (name, id)`. The
    /// output leads with the string column (`k0`), then the int (`k1`), matching
    /// the GROUP BY order, and groups identically.
    #[test]
    fn str_int_mixed_counts() {
        // (name, id) pairs: (a,1)x2, (a,2)x2, (b,2)x1.
        let batch = mixed_key_batch(&[1, 2, 1, 2, 2], &["a", "a", "a", "b", "a"], &[0; 5]);
        let sender = run_group_full::<
            IntStrKeyExtractor<arrow_array::types::Int64Type, true>,
            Compiled<(CountSlot,)>,
        >(
            vec![vec![batch]],
            vec![1, 0], // name (string, col 1) first, then id (int, col 0)
            count_slots(),
            None,
            SpillConfig::DEFAULT,
        );
        let mut rows: Vec<(String, i64, i64)> = sender
            .string_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .zip(sender.i64_column(2))
            .map(|((name, id), c)| (name, id, c))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![
                ("a".to_string(), 1, 2),
                ("a".to_string(), 2, 2),
                ("b".to_string(), 2, 1),
            ]
        );
    }

    /// Row-key SUM: each distinct `(id, name)` sums its value column. Includes a
    /// > 12-byte string to exercise the arena (non-inline) path on output.
    #[test]
    fn row_key_sums_per_group() {
        let long = "a-string-well-over-twelve-bytes";
        let batch = mixed_key_batch(&[1, 2, 1, 2], &["x", long, "x", "y"], &[10, 5, 30, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];
        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![batch]],
            vec![0, 1],
            schema,
            slots,
        );
        assert_eq!(
            row_key_sum_rows(&sender),
            vec![
                (1, "x".to_string(), 40),
                (2, long.to_string(), 5),
                (2, "y".to_string(), 7),
            ]
        );
    }

    /// Keys split across two workers must merge into the same groups.
    #[test]
    fn row_key_two_workers_merge() {
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let sender = run_row_key_group::<Compiled<(CountSlot,)>>(
            vec![
                vec![mixed_key_batch(&[1, 2], &["a", "a"], &[0; 2])],
                vec![mixed_key_batch(&[1, 1], &["a", "b"], &[0; 2])],
            ],
            vec![0, 1],
            schema,
            count_slots(),
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 1),
            ]
        );
    }

    /// `GROUP BY (Utf8View, Int64)` — a string in a *non-trailing* position, so
    /// it keeps its `u32` length prefix while the trailing int does not. A string
    /// longer than 12 bytes forces an arena (non-inline) blob, exercising
    /// prefixed-string decode where a fixed-width field follows the string.
    #[test]
    fn row_key_string_not_trailing() {
        let long = "a-string-well-over-twelve-bytes";
        // GROUP BY (name, id), summing the value column.
        let batch = mixed_key_batch(&[1, 2, 1, 2], &[long, "y", long, "y"], &[10, 5, 30, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Utf8View, DataType::Int64],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];
        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![batch]],
            vec![1, 0], // name (col 1) then id (col 0)
            schema,
            slots,
        );
        // Output columns: k0 = name (str), k1 = id (i64), agg = sum (Decimal128).
        let mut rows: Vec<(String, i64, i128)> = sender
            .string_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .zip(sender.decimal128_column(2))
            .map(|((name, id), v)| (name, id, v))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(long.to_string(), 1, 40), ("y".to_string(), 2, 12)]
        );
    }

    /// One worker, several batches: the encode scratch is reused between batches
    /// and keys (inline and arena) are accumulated across them.
    #[test]
    fn row_key_scratch_reused_across_batches() {
        let long = "a-string-well-over-twelve-bytes";
        let b1 = mixed_key_batch(&[1, 1], &["a", long], &[10, 5]);
        let b2 = mixed_key_batch(&[1, 1], &["a", long], &[3, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];

        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![b1, b2]],
            vec![0, 1],
            schema,
            slots,
        );

        assert_eq!(
            row_key_sum_rows(&sender),
            vec![(1, "a".to_string(), 13), (1, long.to_string(), 12)],
        );
    }

    // ---- Mixed string + integer aggregates via the runtime `Dynamic` ----

    /// The same `MIN(name)` (string) + `MAX(v)` (int) mix, but folded by the
    /// runtime [`Dynamic`] instead of a `Compiled` tuple, the path the planner
    /// now takes for a heterogeneous string signature. The wide (`i128`) cell
    /// holds the string slot's `ArenaKey` and the int slot's value; the int
    /// extreme stores wide but its slot declares `Int64`, so the output phase
    /// casts it back to `Int64`.
    #[test]
    fn dynamic_string_min_int_max() {
        let batch = mixed_key_batch(
            &[1, 2, 1, 2, 1],
            &["cat", "fig", "ant", "bee", "dog"],
            &[10, 7, 30, 5, 20],
        );
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View), // MIN(name) — string
            AggregationSlot::new(AggregationKind::Max, 2, DataType::Int64),    // MAX(v)    — int
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            SpillConfig::DEFAULT,
        );
        let mut rows: Vec<(i64, String, i64)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(1, "ant".to_string(), 30), (2, "bee".to_string(), 7)]
        );
    }

    /// `MIN(name)` and `MAX(name)` over the *same* string column — a string mix
    /// of opposite directions (rejected by the old `Compiled`-only routing). Each
    /// slot keeps its own `ArenaKey` cell and folds its own direction.
    #[test]
    fn dynamic_string_min_and_max() {
        let batch = mixed_key_batch(
            &[1, 2, 1, 2, 1],
            &["cat", "fig", "ant", "bee", "dog"],
            &[10, 7, 30, 5, 20],
        );
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View), // MIN(name)
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View), // MAX(name)
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            SpillConfig::DEFAULT,
        );
        let mut rows: Vec<(i64, String, String)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.string_column(2))
            .map(|((id, min), max)| (id, min, max))
            .collect();
        rows.sort();
        // id 1 -> names {cat, ant, dog}: min "ant", max "dog".
        // id 2 -> names {fig, bee}:      min "bee", max "fig".
        assert_eq!(
            rows,
            vec![
                (1, "ant".to_string(), "dog".to_string()),
                (2, "bee".to_string(), "fig".to_string()),
            ]
        );
    }

    /// Two workers each see part of every group, so the string extremes must
    /// survive the partition merge (which resolves both `ArenaKey`s through the
    /// shared value arena) and a long string (> 12 bytes, a non-inline view) must
    /// round-trip through the arena rather than the inline header.
    #[test]
    fn dynamic_string_extreme_merges_across_workers() {
        let long_a = "alpha-aardvark-antelope"; // > 12 bytes, non-inline view
        let long_z = "zeta-zebra-zephyr-zenith";
        let w0 = mixed_key_batch(&[1, 1], &["mango", long_z], &[1, 2]);
        let w1 = mixed_key_batch(&[1, 1], &[long_a, "mint"], &[3, 4]);
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View),
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View),
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![w0], vec![w1]],
            vec![0],
            slots,
            None,
            SpillConfig::DEFAULT,
        );
        let ids = sender.i64_column(0);
        let mins = sender.string_column(1);
        let maxes = sender.string_column(2);
        assert_eq!(ids, vec![1]);
        assert_eq!(mins, vec![long_a.to_string()]);
        assert_eq!(maxes, vec![long_z.to_string()]);
    }

    /// High cardinality with an aggressive `switch_threshold` a *numeric* value
    /// would cross and scatter on. A string value must stay on the in-place fold
    /// (radix is disabled for it, since scatter would eagerly persist every row's
    /// string), so this exercises in-place stack growth with string cells and
    /// confirms every group's `MIN`/`MAX(name)` is still correct.
    #[test]
    fn dynamic_string_high_cardinality_stays_correct() {
        const N: i64 = 400;
        let ids: Vec<i64> = (0..N).chain(0..N).collect();
        // Group k sees "a{k}" then "z{k}": MIN is the "a" form, MAX the "z" form.
        let lows: Vec<String> = (0..N).map(|k| format!("a{k:04}")).collect();
        let highs: Vec<String> = (0..N).map(|k| format!("z{k:04}")).collect();
        let names: Vec<&str> = lows
            .iter()
            .chain(highs.iter())
            .map(String::as_str)
            .collect();
        let vals: Vec<i32> = vec![0; (2 * N) as usize];
        let batch = mixed_key_batch(&ids, &names, &vals);

        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View),
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View),
        ];
        type Mix = Dynamic<i128>;
        // A capacity the table crosses well before N groups, so the worker
        // stacks tables and the string extremes are merged across them.
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            spill,
        );

        let mut rows: Vec<(i64, String, String)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.string_column(2))
            .map(|((id, min), max)| (id, min, max))
            .collect();
        rows.sort();
        let expected: Vec<(i64, String, String)> = (0..N)
            .map(|k| (k, format!("a{k:04}"), format!("z{k:04}")))
            .collect();
        assert_eq!(rows, expected);
    }

    // ---- LIMIT pushdown (per-worker TopK / First) ----

    #[test]
    fn top_k_limit_zero_returns_no_rows() {
        let values: Vec<i32> = (0..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK { slot: 0, limit: 0 }),
            SpillConfig::DEFAULT,
        );

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn first_limit_zero_returns_no_rows() {
        let values: Vec<i32> = (0..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::First { limit: 0 }),
            SpillConfig::DEFAULT,
        );

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn top_k_limit_exceeding_group_count_returns_all_groups() {
        // A LIMIT far larger than the group count must not pre-size a giant heap;
        // every group survives, each with its true count.
        let values: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2, 3];

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 1_000_000,
            }),
            SpillConfig::DEFAULT,
        );

        assert_eq!(
            group_counts(&sender),
            vec![(1, 2), (2, 2), (3, 2), (4, 1), (5, 1)]
        );
    }

    #[test]
    fn top_k_across_two_workers_keeps_the_global_max() {
        // Per-worker top-k: each output worker keeps its own top `limit`, so the
        // union still contains the global-maximum group for a downstream LIMIT.
        let mut dominant: Vec<i32> = vec![0; 100];
        dominant.extend(1..50);
        let other: Vec<i32> = (50..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![
                vec![batch_with_column(&dominant)],
                vec![batch_with_column(&other)],
            ],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK { slot: 0, limit: 1 }),
            SpillConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(0, 100)), "global-max group survives");
        assert!(pairs.iter().all(|&(_, c)| c >= 1), "every survivor is real");
    }

    // ---- Radix abandon route for blob (string / row) keys ----

    #[test]
    fn string_keys_switch_to_radix_and_count() {
        // Enough distinct strings to trip the small threshold drives the abandon
        // route (drain the table, clear, keep probing) rather than integer scatter;
        // every key is still counted once per occurrence across abandoned windows.
        let owned: Vec<String> = (0..500).map(|k| format!("phrase{k:04}")).collect();
        let names: Vec<&str> = owned
            .iter()
            .chain(owned.iter())
            .map(String::as_str)
            .collect();
        let spill = SpillConfig {
            spill_capacity: 256,
        };

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![vec![string_key_batch(&names)]],
            vec![0],
            count_slots(),
            None,
            spill,
        );

        let pairs = string_group_pairs(&sender);
        assert_eq!(pairs.len(), 500);
        assert!(pairs.iter().all(|(_, c)| *c == 2));
    }

    #[test]
    fn string_keys_radix_merges_across_workers() {
        // Both workers cross the threshold and abandon, so the merge must combine
        // both workers' drained partition buffers and surviving stacks per key.
        let owned: Vec<String> = (0..500).map(|k| format!("phrase{k:04}")).collect();
        let names: Vec<&str> = owned.iter().map(String::as_str).collect();
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let worker = || vec![string_key_batch(&names)];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            None,
            spill,
        );

        let pairs = string_group_pairs(&sender);
        assert_eq!(pairs.len(), 500);
        assert!(pairs.iter().all(|(_, c)| *c == 2));
    }

    #[test]
    fn row_key_switch_to_radix_and_count() {
        // A multi-column row key past the threshold takes the abandon route too
        // (RowKeyExtractor enables it): drained persisted handles must resolve back
        // through the arena at the merge and combine into the right groups.
        let ids: Vec<i64> = (0..500).chain(0..500).collect();
        let owned: Vec<String> = (0..500).map(|k| format!("name{k:04}")).collect();
        let names: Vec<&str> = owned
            .iter()
            .chain(owned.iter())
            .map(String::as_str)
            .collect();
        let vals = vec![0i32; 1000];
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let spill = SpillConfig {
            spill_capacity: 256,
        };

        let sender = run_row_key_group_radix::<CountValue>(
            vec![vec![mixed_key_batch(&ids, &names, &vals)]],
            vec![0, 1],
            schema,
            count_slots(),
            spill,
        );

        let rows = row_key_rows(&sender);
        assert_eq!(rows.len(), 500);
        assert!(rows.iter().all(|(_, _, c)| *c == 2));
    }

    #[test]
    fn mixed_switch_and_in_place_workers_merge_correctly() {
        // One worker crosses the switch threshold while another stays in-place under
        // it. The in-place worker's keys are absent from the merged HLL, so the
        // merge-size estimate accounts for them separately; correctness here proves
        // the mixed switched/in-place merge still combines every key.
        let switching: Vec<i32> = (0..500).collect();
        let in_place: Vec<i32> = (0..5).collect();
        let spill = SpillConfig {
            spill_capacity: 256,
        };

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![
                vec![batch_with_column(&switching)],
                vec![batch_with_column(&in_place)],
            ],
            vec![0],
            count_slots(),
            None,
            spill,
        );

        let counts = group_counts(&sender);
        assert_eq!(counts.len(), 500);
        assert!(
            counts.iter().all(|&(k, c)| c == if k < 5 { 2 } else { 1 }),
            "keys 0..5 seen by both workers count 2, the rest count 1"
        );
    }
}
