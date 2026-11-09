//! One partition's merge work and the state every partition job shares.

use crate::OperatorIO;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, HashTable, KeyExtractor, PartitionBuffers,
};
use crate::operations::unary::group::output::accumulator::{OutputAccumulator, emit_count};
use crate::operations::unary::group::output::merge;
use crate::operations::unary::group::output::topk_pruning::{
    TopKBounds, TopKCutoff, TopKThreshold,
};
use crate::operations::unary::group::{GroupLimit, Result};
use arrow_array::RecordBatch;
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// One node's merge sources: its switched workers' scatter buffers and its
/// workers' in-place stacks.
pub(super) type NodeSources<K, V> = (
    Vec<PartitionBuffers<<K as KeyExtractor>::Persisted, V>>,
    Vec<HashTable<<K as KeyExtractor>::Persisted, V>>,
);

/// Everything the partition jobs of one merge share, built once by
/// `create_partition_jobs`: the sources they read, the sizing decided from
/// the gathered outputs, the handles output needs, and the progress and
/// rendezvous state the jobs write through.
///
/// Jobs go through work queues every worker drains, so each job carries one
/// `Arc` to this instead of cloning every shared field: thousands of jobs
/// each bumping and dropping many refcounts is cross-node cache-line traffic,
/// and under a pruned top-k most jobs are popped only to skip themselves.
pub(super) struct SharedMergeState<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// Per node: that node's switched workers' scatter buffers and the
    /// in-place stacks of its workers (switched workers' pre-switch tables
    /// and non-switched workers' full stacks). A direct merge holds a single
    /// entry with every node's sources.
    pub(super) sources: Vec<NodeSources<K, V>>,
    /// Hierarchical merge only: every partition's cross-node rendezvous.
    pub(super) cross_node: Option<CrossNodeMerges<K, V>>,
    pub(super) key_arena: Arc<SharedArena>,
    /// The arena's ring buffers wrapped as Arrow `Buffer`s, built once for the
    /// whole output phase. String-key output emits zero-copy views into this.
    pub(super) output_buffers: Arc<[Buffer]>,
    pub(super) partition_capacity: usize,
    /// Power-of-two count of hash partitions the merge runs at, chosen in
    /// `create_partition_jobs` from the input volume and the worker count.
    /// When a worker switched to radix it never exceeds the scatter bucket
    /// count (at most `RADIX_PARTITIONS`).
    pub(super) num_partitions: usize,
    pub(super) key_config: K::Config,

    /// The value's shared context, for the partition merge's entry fold + output.
    pub(super) shared_context: V::SharedContext,
    /// Declared output type per value column, in slot order; the output phase casts
    /// each finished value column to its type.
    pub(super) value_output_types: Arc<[DataType]>,
    pub(super) output_limit: Option<GroupLimit>,
    pub(super) count_only: bool,
    /// Top-k pruning state, present only when a prunable top-k was
    /// pushed: each partition's upper bound on any single group's final value,
    /// and the per-bin bounds a job's
    /// [`TopKCutoff`](crate::operations::unary::group::output::topk_pruning::TopKCutoff)
    /// admits groups by.
    pub(super) partition_bounds: Option<Vec<u64>>,
    pub(super) topk_bounds: Option<TopKBounds>,
    /// Shared pushed-limit progress: the k-th best weight the top-k bounds
    /// are checked against, and the emitted-group count a plain LIMIT skips
    /// against.
    pub(super) limit_progress: Arc<TopKThreshold>,
    /// How many jobs have released their partition's scatter buckets. Once
    /// every job has, the shared state is dropped without walking the
    /// sources' bucket lists.
    pub(super) released_jobs: ReleasedJobs,
}

/// A count of released jobs on a cache line of its own: every job writes it
/// once, and the fields beside it are read by every job all through the merge.
#[repr(align(128))]
#[derive(Default)]
pub(super) struct ReleasedJobs(AtomicUsize);

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Drop for SharedMergeState<K, V> {
    /// Every job has released its partition's buckets by the time the last
    /// one lets go of this, unless the merge was abandoned, so the sources'
    /// bucket lists hold nothing and are let go of unvisited.
    fn drop(&mut self) {
        let job_count = self.num_partitions * self.sources.len();
        if *self.released_jobs.0.get_mut() != job_count {
            return;
        }
        for (buffers, _) in &mut self.sources {
            for worker_buffers in buffers {
                worker_buffers.discard_released_lists();
            }
        }
    }
}

// The sources are read-only for the whole merge and every other field is
// either immutable or internally synchronized, so sharing this across worker
// threads is sound.
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for SharedMergeState<K, V> {}
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Sync for SharedMergeState<K, V> {}

/// One (NUMA node, partition) merge work unit: an index into the shared
/// [`SharedMergeState`], plus which node's sources to read on a hierarchical merge.
///
/// Created by the final worker to reach the gather barrier and pushed to the
/// node's queue for work-stealing execution. Each job merges one node's
/// source tables for partition `index` into one result table. On a direct
/// merge (`SharedMergeState::cross_node` is `None`) that result is the partition's
/// final table and its rows are emitted directly. On a hierarchical merge,
/// each node's job merges only node-local memory and sends the resulting node
/// table into the partition's slots of [`CrossNodeMerges`]; the last job to finish
/// receives every node's table and merges them into the final one. That last
/// merge is the only step that reads another node's memory.
pub struct PartitionJob<K: KeyExtractor, V: AggregationValue + ?Sized> {
    pub(super) shared: Arc<SharedMergeState<K, V>>,
    pub(super) index: usize,
    /// Which entry of [`SharedMergeState::sources`] this job merges (its node on a
    /// hierarchical merge; 0 on a direct merge).
    pub(super) node: usize,
}

unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for PartitionJob<K, V> {}

/// Every partition's pending cross-node merge: each node's job sends the
/// node's merged aggregated table for its partition; the send that completes
/// a partition's set hands every node's table back to that caller, electing
/// it to run the final merge ([`merge::merge_node_aggregated_tables`]) and
/// emit. Senders never block and no job ever waits: a job writes a slot only
/// it reaches, and the election is one atomic update of the partition's set
/// of sent nodes.
///
/// The slots of every partition live in one allocation made up front, so a
/// send allocates nothing and dropping the rendezvous frees one block
/// however many partitions the merge ran.
pub(super) struct CrossNodeMerges<K: KeyExtractor, V: AggregationValue + ?Sized> {
    node_count: usize,
    /// One slot per (partition, node), a partition's slots adjacent. A slot
    /// holds a table exactly while its node's bit is set in the partition's
    /// entry of `sent_nodes`.
    node_tables: Box<[NodeTableSlot<K, V>]>,
    /// Per partition: one bit per node whose table waits in its slot.
    sent_nodes: Box<[AtomicU64]>,
}

/// Where one node's merged table for one partition waits for the others.
type NodeTableSlot<K, V> = UnsafeCell<MaybeUninit<HashTable<<K as KeyExtractor>::Persisted, V>>>;

// SAFETY: a slot is written by the one job of its (partition, node) and read
// only by the job elected through `sent_nodes`, after every write to the
// partition's slots was published by its sender's atomic update.
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for CrossNodeMerges<K, V> {}
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Sync for CrossNodeMerges<K, V> {}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> CrossNodeMerges<K, V> {
    pub(super) fn new(partition_count: usize, node_count: usize) -> Self {
        assert!(
            (1..=u64::BITS as usize).contains(&node_count),
            "a partition's sent nodes are tracked in one 64-bit set"
        );
        Self {
            node_count,
            node_tables: (0..partition_count * node_count)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            sent_nodes: (0..partition_count).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Send `node`'s merged table for `partition`. Exactly one call per
    /// partition - the last - returns the full set; that caller must run the
    /// final merge. Each sender's slot write happens-before its update of the
    /// sent set, and the last sender's update observes all of them, so the
    /// reads below are guaranteed to see every node's table (and after the
    /// election nobody else touches the partition's slots).
    fn send(
        &self,
        partition: usize,
        node: usize,
        table: HashTable<K::Persisted, V>,
    ) -> Option<Vec<HashTable<K::Persisted, V>>> {
        let slots = &self.node_tables[partition * self.node_count..][..self.node_count];
        // SAFETY: this job is the only one for (partition, node), so nothing
        // else reaches the slot until the election below publishes it.
        unsafe { (*slots[node].get()).write(table) };
        let sent_nodes = &self.sent_nodes[partition];
        let node_bit = 1u64 << node;
        let every_node = u64::MAX >> (u64::BITS as usize - self.node_count);
        if sent_nodes.fetch_or(node_bit, Ordering::AcqRel) | node_bit != every_node {
            return None;
        }
        // The slots are emptied below, so the drop must not visit them.
        sent_nodes.store(0, Ordering::Release);
        Some(
            slots
                .iter()
                // SAFETY: every node's bit was set, so every slot holds a
                // table, and each is moved out exactly once, here.
                .map(|slot| unsafe { (*slot.get()).assume_init_read() })
                .collect(),
        )
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Drop for CrossNodeMerges<K, V> {
    /// Drops the tables of partitions whose set never completed (some node's
    /// job was pruned after another's had sent). Only the sent sets are
    /// walked; the slots of every other partition hold nothing.
    fn drop(&mut self) {
        for (partition, sent_nodes) in self.sent_nodes.iter_mut().enumerate() {
            let mut waiting_nodes = *sent_nodes.get_mut();
            while waiting_nodes != 0 {
                let node = waiting_nodes.trailing_zeros() as usize;
                waiting_nodes &= waiting_nodes - 1;
                let slot = &mut self.node_tables[partition * self.node_count + node];
                // SAFETY: the node's bit is set, so its slot holds a table
                // nobody received.
                unsafe { slot.get_mut().assume_init_drop() };
            }
        }
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> PartitionJob<K, V> {
    /// Whether this partition can be skipped: a plain pushed LIMIT already has
    /// its rows, or no group in the partition can beat the top-k's k-th best.
    pub(super) fn is_pruned(&self) -> bool {
        let shared = &*self.shared;
        if let Some(GroupLimit::First { limit }) = shared.output_limit
            && shared.limit_progress.emitted_groups() >= limit
        {
            return true;
        }
        match &shared.partition_bounds {
            // No group in this partition can beat the k-th best exact group some
            // worker already holds, so merging it could not change the top-k.
            Some(bounds) => bounds[self.index] < shared.limit_progress.kth_best(),
            None => false,
        }
    }

    /// Which hashes the top-k pruning still admits for this partition, if a
    /// prunable top-k was pushed.
    pub(super) fn cutoff(&self) -> Option<TopKCutoff<'_>> {
        let shared = &*self.shared;
        let bounds = shared.partition_bounds.as_ref()?;
        debug_assert!(bounds.len() > self.index);
        let bin_bounds = shared
            .topk_bounds
            .as_ref()
            .expect("partition bounds only exist beside bin bounds");
        bin_bounds.cutoff_for_partition(
            self.index,
            shared.num_partitions,
            shared.limit_progress.kth_best(),
        )
    }

    /// Feed the partition's result, complete now, into the worker's shared
    /// `acc` (building columns into `allocator`). Accumulating across
    /// partition jobs, rather than emitting one batch per job, keeps the radix
    /// path's many small partitions from each producing a tiny `RecordBatch`.
    ///
    /// The partition's scatter buckets on this node are released first. Every
    /// row of theirs is in `result` by now, and the order matters for the
    /// free pool, which hands out the buffer freed last: a scatter buffer
    /// shared by many partitions comes free here cold, read by whichever
    /// workers merged the partitions in it, while the buffers `result` frees
    /// when it has been emitted are the ones this worker just wrote. Freed
    /// after the buckets, they are the ones the next job's table is built in,
    /// still in this core's cache.
    pub(super) fn finish(
        self,
        result: HashTable<K::Persisted, V>,
        acc: &mut Option<OutputAccumulator<K, V>>,
        sender: &mut dyn Sender<RecordBatch>,
        allocator: &mut SlabAllocator,
        io: &mut OperatorIO,
    ) -> Result<()> {
        self.release_buckets();
        let emitted = self.emit(result, acc, sender, allocator);
        self.release_shared_state(io);
        emitted
    }

    /// Let go of a pruned partition's scatter buckets without merging it.
    pub(super) fn skip(self, io: &mut OperatorIO) {
        self.release_buckets();
        self.release_shared_state(io);
    }

    /// Give up this job's hold on the shared state. The job that finishes
    /// last holds the only one left, and its worker is the one the rest of
    /// the query waits for: its last batch is still to be flushed and every
    /// stage downstream to finish. Dropping the state there, with every
    /// worker's tables and lists in it, would stall all of that, so the state
    /// is handed to the dataflow to drop once it has finished on this worker.
    fn release_shared_state(self, io: &mut OperatorIO) {
        if let Some(shared) = Arc::into_inner(self.shared) {
            io.drop_after_data_flow(shared);
        }
    }

    /// Drop the partition's scatter buckets on this node. Each job frees its
    /// own share so the teardown runs on every worker instead of on the one
    /// that drops the shared state last.
    fn release_buckets(&self) {
        let (buffers, _) = &self.shared.sources[self.node];
        for worker_buffers in buffers {
            worker_buffers.release_partition(self.index, self.shared.num_partitions);
        }
        self.shared.released_jobs.0.fetch_add(1, Ordering::Relaxed);
    }

    fn emit(
        &self,
        result_map: HashTable<K::Persisted, V>,
        acc: &mut Option<OutputAccumulator<K, V>>,
        sender: &mut dyn Sender<RecordBatch>,
        allocator: &mut SlabAllocator,
    ) -> Result<()> {
        let shared = &*self.shared;
        let result_map = match &shared.cross_node {
            None => result_map,
            Some(cross_node) => {
                match cross_node.send(self.index, self.node, result_map) {
                    // Another node's job for this partition is still running;
                    // it will receive the tables and run the final merge.
                    None => return Ok(()),
                    Some(node_tables) => merge::merge_node_aggregated_tables::<K::Stored, V>(
                        node_tables,
                        shared.partition_capacity,
                        shared.num_partitions.trailing_zeros(),
                        &shared.key_arena,
                        &shared.shared_context,
                    ),
                }
            }
        };
        if result_map.len() == 0 {
            return Ok(());
        }
        // Global COUNT(DISTINCT) needs only each partition's distinct-key count,
        // not the keys, so it bypasses the accumulator and emits a single row.
        if shared.count_only {
            return emit_count(result_map.len(), sender);
        }
        let acc = acc.get_or_insert_with(|| {
            OutputAccumulator::new(
                allocator,
                shared.output_limit,
                shared.key_arena.clone(),
                shared.output_buffers.clone(),
                shared.key_config.clone(),
                shared.shared_context.clone(),
                shared.value_output_types.clone(),
                shared.limit_progress.clone(),
                shared.partition_bounds.is_some(),
            )
        });
        acc.extend_from_table(result_map, allocator, &mut *sender)
    }
}
