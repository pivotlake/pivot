use crate::GatherBarrier;
use crate::memory::{SlabAllocator, SlabVec};
use crate::operations::channels::Sender;
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::build_rows::{self, BuildRows};
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::keys::{JoinKey, combined_key_validity};
use crate::operations::{Consumer, Outputter, unary};
use crate::waker::worker_waker;
use ahash::RandomState;
use arrow::compute::filter_record_batch;
use arrow_array::{BooleanArray, RecordBatch};
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::debug;

pub(crate) const NUM_PARTITIONS: usize = 64;
const PARTITION_SHIFT: u32 = 64 - NUM_PARTITIONS.trailing_zeros();

/// One build-side row, radix-partitioned by `hash`: the full-width key (for
/// exact match rejection of hash collisions in the probe) and the row's id
/// within this worker's stored batches, globalized with the worker's base at
/// scatter time.
#[derive(Clone, Copy)]
pub(crate) struct BuildTuple<K> {
    hash: u64,
    key: K,
    row: u32,
}

pub(crate) type PartitionBuffers<K> = Vec<SlabVec<BuildTuple<K>>>;

/// Everything one build worker publishes at the gather barrier: its
/// partitioned tuples, the stored build rows those tuples' ids point into,
/// and whether any build key was null.
pub(crate) struct BuildWorkerOutput<K: Copy> {
    tuples: PartitionBuffers<K>,
    build_row_batches: Vec<RecordBatch>,
    saw_null_key: bool,
}

pub struct JoinBuildConsumer<K: JoinKey, const BUILD_OUTER: bool> {
    key_columns: Vec<usize>,
    hash_state: RandomState,
    values: PartitionBuffers<K::Stored>,
    /// This worker's build rows, stored as the batches they arrived in. No
    /// bytes are copied and nothing leaves engine memory: the batches stay
    /// the arrays the upstream operator produced.
    build_row_batches: Vec<RecordBatch>,
    slab_allocator: SlabAllocator,
    gather: Arc<GatherBarrier<BuildWorkerOutput<K::Stored>>>,
    outputter: JoinBuilder<K::Stored, BUILD_OUTER>,
    /// Whether this worker saw a null-keyed build row. Published with its
    /// other build output and combined by the final gather arrival.
    saw_null_key: bool,
}

unsafe impl<K: JoinKey, const BUILD_OUTER: bool> Send for JoinBuildConsumer<K, BUILD_OUTER> {}

impl<K: JoinKey, const BUILD_OUTER: bool> JoinBuildConsumer<K, BUILD_OUTER> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        key_columns: Vec<usize>,
        hash_state: RandomState,
        gather: Arc<GatherBarrier<BuildWorkerOutput<K::Stored>>>,
        table: JoinTable<K::Stored>,
        injector: Arc<Injector<JoinBuildJob<K::Stored>>>,
        jobs_injected: Arc<AtomicBool>,
        build_ready: Arc<AtomicBool>,
        remaining_jobs: Arc<AtomicUsize>,
        finish_claimed: Arc<AtomicBool>,
        build_output_indices: Vec<usize>,
    ) -> Self {
        Self {
            key_columns,
            hash_state,
            values: (0..NUM_PARTITIONS).map(|_| SlabVec::new()).collect(),
            build_row_batches: Vec::new(),
            slab_allocator: SlabAllocator::new(false),
            gather,
            outputter: JoinBuilder {
                table,
                injector,
                jobs_injected,
                build_ready,
                remaining_jobs,
                finish_claimed,
                build_output_indices,
            },
            saw_null_key: false,
        }
    }
}

/// Drop rows any of whose join key columns is null: an equi-join can never
/// match them, and the probe's hot loops read key values without validity
/// checks. Only the probe side filters; the build side keeps its batches
/// whole (they become the stored build row batches) and skips null-keyed rows when
/// generating tuples.
pub(crate) fn filter_null_keys(batch: RecordBatch, key_columns: &[usize]) -> RecordBatch {
    let Some(combined_validity) = combined_key_validity(&batch, key_columns) else {
        return batch;
    };
    let mask = BooleanArray::new(combined_validity.inner().clone(), None);
    filter_record_batch(&batch, &mask).expect("null-key filter mask matches batch length")
}

/// Split a probe batch into its rows with fully non-null keys and, when any
/// exist, the null-keyed rest. A probe-side outer join keeps the rest: those
/// rows can never match, but they still reach the output null-padded.
pub(crate) fn split_null_keys(
    batch: RecordBatch,
    key_columns: &[usize],
) -> (RecordBatch, Option<RecordBatch>) {
    let Some(combined_validity) = combined_key_validity(&batch, key_columns) else {
        return (batch, None);
    };
    let keep = BooleanArray::new(combined_validity.inner().clone(), None);
    let drop = arrow::compute::not(&keep).expect("a validity mask negates");
    let kept =
        filter_record_batch(&batch, &keep).expect("null-key filter mask matches batch length");
    let dropped =
        filter_record_batch(&batch, &drop).expect("null-key filter mask matches batch length");
    (kept, (dropped.num_rows() > 0).then_some(dropped))
}

impl<K: JoinKey, const BUILD_OUTER: bool> Consumer<RecordBatch, ()>
    for JoinBuildConsumer<K, BUILD_OUTER>
{
    type Outputter = JoinBuilder<K::Stored, BUILD_OUTER>;

    fn consume(&mut self, batch: RecordBatch, _sender: &mut dyn Sender<()>) -> unary::Result<()> {
        debug!("Consuming build");
        for (first_row_id, stored) in build_rows::adopt(&mut self.build_row_batches, batch) {
            let reader = K::make_reader(&stored, &self.key_columns, &self.hash_state);
            for i in 0..stored.num_rows() {
                // A null-keyed row can never match, so it gets no tuple. It
                // stays stored, where only an outer build's unmatched pass
                // ever reaches it. A mark join's misses also turn NULL once
                // such a row exists, so remember having seen one.
                if K::is_null(&reader, i) {
                    self.saw_null_key = true;
                    continue;
                }
                let key = K::read_stored(&reader, i);
                let hash = K::hash_row(&reader, i, &self.hash_state);
                let partition = (hash >> PARTITION_SHIFT) as usize;
                self.values[partition].push(
                    &mut self.slab_allocator,
                    BuildTuple {
                        hash,
                        key,
                        row: first_row_id + i as u32,
                    },
                );
            }
        }
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        debug!("Running into outputter!");
        let output = BuildWorkerOutput {
            tuples: self.values,
            build_row_batches: self.build_row_batches,
            saw_null_key: self.saw_null_key,
        };
        let gather = self.gather.clone();
        if let Some(result) = gather.arrive(output, |worker_outputs| {
            self.outputter.initialize(worker_outputs)
        }) {
            result?;
        }
        Ok(Some(self.outputter))
    }
}

pub struct JoinBuilder<K: Copy + Send, const BUILD_OUTER: bool> {
    table: JoinTable<K>,
    injector: Arc<Injector<JoinBuildJob<K>>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
    /// Claimed (once) by the worker that observes the last job land; that
    /// worker finishes any deferred output views and then opens `build_ready`.
    finish_claimed: Arc<AtomicBool>,
    build_output_indices: Vec<usize>,
}

unsafe impl<K: Copy + Send, const BUILD_OUTER: bool> Send for JoinBuilder<K, BUILD_OUTER> {}

/// One unit of the parallel table-build phase, stolen and run by any build
/// worker between the gather barrier and the `build_ready` gate.
pub enum JoinBuildJob<K: Copy + Send> {
    /// Scatter one radix partition's tuples into the directory and arenas.
    Partition(JoinPartitionJob<K>),
    /// Fold one stored batch's mismatched variant columns to the canonical
    /// layout (see [`BuildRows::fold_columns`]). One job per batch, so a large
    /// mixed-layout build side folds across every worker instead of on one.
    FoldBatch(FoldBatchJob<K>),
}

impl<K: Copy + Send> JoinBuildJob<K> {
    fn run(self) -> unary::Result<()> {
        match self {
            JoinBuildJob::Partition(job) => {
                job.run();
                Ok(())
            }
            JoinBuildJob::FoldBatch(job) => job.run(),
        }
    }
}

pub struct FoldBatchJob<K: Copy + Send> {
    table: JoinTable<K>,
    batch_idx: usize,
    columns: Arc<[usize]>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl<K: Copy + Send> Send for FoldBatchJob<K> {}

impl<K: Copy + Send> FoldBatchJob<K> {
    fn run(self) -> unary::Result<()> {
        // Jobs fold disjoint batch indices, the same disjoint-writes protocol
        // the partition jobs use on the directory.
        let build_rows = unsafe { &mut *self.table.build_rows.get() };
        build_rows.fold_columns(self.batch_idx, &self.columns)?;
        // Release-publish this job's writes; the gate flip's Acquire load of
        // the drained counter is what makes the folded batch readable.
        self.remaining_jobs.fetch_sub(1, Ordering::Release);
        Ok(())
    }
}

pub struct JoinPartitionJob<K: Copy + Send> {
    /// This partition's tuples, one entry per build worker, paired with that
    /// worker's row id base (added to each tuple's local row id).
    tuples: Vec<(u32, SlabVec<BuildTuple<K>>)>,
    table: JoinTable<K>,
    arena_offset: usize,

    slot_start: usize,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl<K: Copy + Send> Send for JoinPartitionJob<K> {}

impl<K: Copy + Send> JoinPartitionJob<K> {
    fn run(self) {
        debug!("Running partition job");
        let directory = unsafe { &*self.table.directory.get() };
        let shift = directory.shift;
        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };

        // Pass 1: accumulate counts in upper 48 bits, OR bloom tags into
        // lower 16 bits
        for (_, worker_tuples) in &self.tuples {
            worker_tuples.for_each(|tuple| {
                let slot = (tuple.hash >> shift) as usize;
                unsafe {
                    directory.add_to_entry(slot, 1 << 16);
                    directory.or_to_entry(slot, JoinDirectory::compute_tag(tuple.hash) as u64);
                }
            });
        }

        // Pass 2: convert counts to absolute write cursors. Linear sweep
        // over this partition's directory slot range replaces each count
        // with a running arena pointer while preserving the tag bits.
        let slots_per_partition = directory.capacity() / NUM_PARTITIONS;
        let slot_end = self.slot_start + slots_per_partition;
        let mut cur = self.arena_offset as u64;

        for i in self.slot_start..slot_end {
            let entry = directory.entry(i);
            let count = entry >> 16;
            let tag = entry & 0xFFFF;
            cur += count;
            directory.set_entry(i, (cur << 16) | tag);
        }

        // Pass 3: scatter tuples into the parallel key/row arenas. For each
        // tuple read the directory to get its arena write pointer, advance the
        // cursor, and write the key and globalized row id. The element
        // `PREFETCH_AHEAD` positions ahead is used to prefetch its directory
        // slot before we reach it.
        const PREFETCH_AHEAD: usize = 64;
        for (row_base, worker_tuples) in &self.tuples {
            worker_tuples.for_each_prefetched::<PREFETCH_AHEAD>(|tuple, ahead| {
                if let Some(ahead_tuple) = ahead {
                    directory.prefetch_l2(ahead_tuple.hash);
                }
                let slot = (tuple.hash >> shift) as usize;
                let entry = directory.entry(slot).wrapping_sub(1 << 16);
                directory.set_entry(slot, entry);
                let arena_idx = (entry >> 16) as usize;
                unsafe {
                    keys.ptr_at_index(arena_idx).write(tuple.key);
                    rows.ptr_at_index(arena_idx).write(row_base + tuple.row);
                }
            });
        }

        // Release-publish this job's writes; the outputter's Acquire load of
        // the drained counter is what makes the table readable.
        self.remaining_jobs.fetch_sub(1, Ordering::Release);
    }
}

impl<K: Copy + Send, const OUTER_JOIN_BUILD_SIDE: bool> JoinBuilder<K, OUTER_JOIN_BUILD_SIDE> {
    /// Combine the gathered worker outputs, initialize the shared table, and
    /// publish its partition jobs. Called only by the final gather arrival.
    fn initialize(&mut self, mut worker_outputs: Vec<BuildWorkerOutput<K>>) -> unary::Result<()> {
        let build_saw_null_key = worker_outputs.iter().any(|output| output.saw_null_key);
        unsafe { *self.table.build_saw_null_key.get() = build_saw_null_key };

        let sizes: Vec<usize> = (0..NUM_PARTITIONS)
            .map(|partition| {
                worker_outputs
                    .iter()
                    .map(|output| output.tuples[partition].len())
                    .sum()
            })
            .collect();
        let total: usize = sizes.iter().sum();

        // GatherBarrier returns outputs in worker order. Merge the stored
        // build rows in that same order, so each worker-local tuple id gets
        // the corresponding row base during scatter. An empty merge means an
        // empty build side, and the probe then emits nothing. For a build
        // whose side is outer, anti, or semi this also allocates one matched
        // flag per row id (gaps included); null-keyed rows have no tuple and
        // remain unmatched.
        let (mut build_rows, row_bases, fold_columns) = BuildRows::new::<OUTER_JOIN_BUILD_SIDE>(
            worker_outputs
                .iter_mut()
                .map(|output| std::mem::take(&mut output.build_row_batches)),
        );
        // With every batch already in its final layout, the output views can
        // be built right here; otherwise they wait for the fold jobs pushed
        // below, and the worker that flips the gate builds them (see
        // `output`).
        let fold_jobs = if fold_columns.is_empty() {
            build_rows.finish_output_views(&self.build_output_indices)?;
            0
        } else {
            build_rows.batches.len()
        };
        unsafe { *self.table.build_rows.get() = build_rows };
        // The counter starts at NUM_PARTITIONS; fold jobs join it before
        // `jobs_injected` is set, so no worker can conclude completion early.
        self.remaining_jobs.fetch_add(fold_jobs, Ordering::Relaxed);
        let fold_columns: Arc<[usize]> = fold_columns.into();
        for batch_idx in 0..fold_jobs {
            self.injector.push(JoinBuildJob::FoldBatch(FoldBatchJob {
                table: self.table.clone(),
                batch_idx,
                columns: fold_columns.clone(),
                remaining_jobs: self.remaining_jobs.clone(),
            }));
        }

        // Pre-allocate directory and arenas.
        let dir_capacity = ((total as f64 * 1.125) as usize)
            .next_power_of_two()
            .max(NUM_PARTITIONS);
        let directory = unsafe { &mut *self.table.directory.get() };
        let mut directory_alloc = SlabAllocator::new(false);
        *directory = JoinDirectory::new(
            directory_alloc.create_multi_slab_buffer(dir_capacity + 1, true),
            dir_capacity,
        );

        // Sentinel at entry[capacity] holds the end pointer of the last slot.
        // Probe reads end_ptr(slot+1) for slot = capacity-1, which lands here.
        directory.set_entry(dir_capacity, (total as u64) << 16);

        let mut arena_alloc = SlabAllocator::new(false);
        let keys = unsafe { &mut *self.table.keys.get() };
        *keys = arena_alloc.create_multi_slab_buffer::<K>(total.max(1), false);
        let rows = unsafe { &mut *self.table.rows.get() };
        *rows = arena_alloc.create_multi_slab_buffer::<u32>(total.max(1), false);

        // Prefix sums give each partition its arena offset.
        let mut offsets = vec![0usize; NUM_PARTITIONS];
        for p in 1..NUM_PARTITIONS {
            offsets[p] = offsets[p - 1] + sizes[p - 1];
        }

        let slots_per_partition = dir_capacity / NUM_PARTITIONS;
        for (i, &arena_offset) in offsets.iter().enumerate() {
            let tuples: Vec<(u32, SlabVec<BuildTuple<K>>)> = worker_outputs
                .iter_mut()
                .zip(&row_bases)
                .map(|(output, &row_base)| (row_base, std::mem::take(&mut output.tuples[i])))
                .collect();
            self.injector
                .push(JoinBuildJob::Partition(JoinPartitionJob {
                    tuples,
                    table: self.table.clone(),
                    arena_offset,
                    slot_start: i * slots_per_partition,
                    remaining_jobs: self.remaining_jobs.clone(),
                }));
        }
        self.jobs_injected.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl<K: Copy + Send, const OUTER_JOIN_BUILD_SIDE: bool> Outputter<()>
    for JoinBuilder<K, OUTER_JOIN_BUILD_SIDE>
{
    fn output(&mut self, _sender: &mut dyn Sender<()>) -> unary::Result<bool> {
        match self.injector.steal() {
            Steal::Success(job) => {
                job.run()?;
            }
            Steal::Empty => {
                // An empty queue is not completion: a job stolen by another
                // worker may still be running, and the table is unreadable
                // until it lands. Report done only when every job has run.
                if self.jobs_injected.load(Ordering::Relaxed)
                    && self.remaining_jobs.load(Ordering::Acquire) == 0
                {
                    if self
                        .finish_claimed
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        // Output views deferred behind fold jobs are built
                        // here, by the one finishing worker, before the gate
                        // opens: the counter's Acquire load above ordered
                        // every fold before this.
                        let build_rows = unsafe { &mut *self.table.build_rows.get() };
                        if build_rows.output_batches.len() != build_rows.batches.len() {
                            build_rows.finish_output_views(&self.build_output_indices)?;
                        }
                        self.build_ready.store(true, Ordering::Release);
                        worker_waker().notify();
                    }
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}
