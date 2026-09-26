use crate::GatherBarrier;
use crate::memory::{MultiSlabBuffer, SlabAllocator, SlabVec};
use crate::operations::channels::Sender;
use crate::operations::unary::join::JoinBuildFilter;
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::build_rows::{self, BuildRows};
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::key_bitset::{PendingKeyBitset, integer_scalar};
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::row_arena::{RowArena, RowId};
use crate::operations::unary::{BatchesOutputter, InitializableOutputter, NormalizationBatches};
use crate::operations::{Consumer, Outputter, unary};
use crate::waker::worker_waker;
use ahash::RandomState;
use arrow::compute::{concat, sort_to_indices, take};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::SortOptions;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tracing::debug;

/// How many hash partitions a build is scattered in, one scatter job each:
/// at least two per worker (rounded up to a power of two, since a tuple's
/// partition is the top bits of its hash), so no worker sits idle through
/// the scatter's tail while a lone last job finishes, and never fewer than
/// 64. Every worker keeps one tuple list per partition while it consumes,
/// so finer than two per worker costs more in those lists on small builds
/// than the shorter tail returns.
pub(crate) fn partitions_for(worker_count: usize) -> usize {
    (2 * worker_count).next_power_of_two().max(64)
}

/// One build-side row, radix-partitioned by `hash`: the full-width key (for
/// exact match rejection of hash collisions in the probe) and the row's id
/// within this worker's stored batches, globalized with the worker's base at
/// scatter time.
#[derive(Clone, Copy)]
pub(crate) struct BuildTuple<K> {
    hash: u64,
    key: K,
    row: u64,
}

pub(crate) type PartitionBuffers<K> = Vec<SlabVec<BuildTuple<K>>>;

/// Everything one build worker publishes at the gather barrier: its
/// partitioned tuples, the stored build rows those tuples' ids point into,
/// whether any build key was null, and its filter-column extremes.
pub(crate) struct BuildWorkerOutput<K: Copy> {
    tuples: PartitionBuffers<K>,
    build_row_batches: Vec<RecordBatch>,
    saw_null_key: bool,
    /// One entry per build filter: this worker's key extremes over its own
    /// batches, computed at seal time on the worker that held them. The final
    /// gather arrival merges one candidate per worker instead of rescanning
    /// every build row serially while the other workers sit idle.
    filter_extremes: Vec<ColumnExtremes>,
}

/// One worker's smallest and largest non-null value of a filter column;
/// `None` when the worker held no non-null value.
pub(crate) struct ColumnExtremes {
    min: Option<ArrayRef>,
    max: Option<ArrayRef>,
}

impl<K: Copy + Send + 'static> NormalizationBatches for BuildWorkerOutput<K> {
    fn visit_batches_mut(&mut self, visit: &mut dyn FnMut(&mut RecordBatch)) {
        for batch in &mut self.build_row_batches {
            visit(batch);
        }
    }
}

pub struct JoinBuildConsumer<
    K: JoinKey,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    O = JoinBuilder<<K as JoinKey>::Stored, TRACK_MATCHED_BUILD_ROWS>,
> {
    key_columns: Vec<usize>,
    /// The build columns whose extremes this worker computes at seal time,
    /// one per build filter of the join.
    filter_columns: Vec<usize>,
    hash_state: RandomState,
    values: PartitionBuffers<K::Stored>,
    /// A tuple's partition is its hash shifted down by this: the top
    /// `log2(partitions)` bits.
    partition_shift: u32,
    /// This worker's build rows, stored as the batches they arrived in. No
    /// bytes are copied and nothing leaves engine memory: the batches stay
    /// the arrays the upstream operator produced.
    build_row_batches: Vec<RecordBatch>,
    slab_allocator: SlabAllocator,
    outputter: O,
    /// Whether this worker saw a null-keyed build row. Published with its
    /// other build output and combined by the final gather arrival.
    saw_null_key: bool,
    /// Where this worker leaves its filter key columns at seal time. The
    /// [`JoinBuilder`] that later runs on the same worker reads them back to
    /// set this worker's share of each filter's key bitset, which cannot be
    /// built before then because its bounds come from every worker's rows.
    filter_arrays: WorkerFilterArrays,
}

/// The filter key columns of one worker's stored build rows: per build
/// filter, one array per batch. The worker's consumer fills it when it
/// seals, and the [`JoinBuilder`] on the same worker reads it back to set
/// that worker's share of each key bitset. The group itself may travel
/// through a normalizer and land on another worker; this cell stays put.
pub(crate) type WorkerFilterArrays = Arc<OnceLock<Vec<Vec<ArrayRef>>>>;

/// Per build filter, the key bitset under construction, or `None` when the
/// filter's bounds or size ruled one out. Decided once by the final gather
/// arrival, before it publishes the partition jobs.
pub(crate) type PendingKeyBitsets = Arc<OnceLock<Vec<Option<Arc<PendingKeyBitset>>>>>;

unsafe impl<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool, O: Send> Send
    for JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS, O>
{
}

impl<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool, O>
    JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS, O>
{
    pub(crate) fn new(
        key_columns: Vec<usize>,
        filter_columns: Vec<usize>,
        hash_state: RandomState,
        outputter: O,
        filter_arrays: WorkerFilterArrays,
        partitions: usize,
    ) -> Self {
        Self {
            key_columns,
            filter_columns,
            hash_state,
            values: (0..partitions).map(|_| SlabVec::new()).collect(),
            // `partitions` is a power of two, so its trailing zeros are its log2.
            partition_shift: 64 - partitions.trailing_zeros(),
            build_row_batches: Vec::new(),
            slab_allocator: SlabAllocator::new(false),
            outputter,
            saw_null_key: false,
            filter_arrays,
        }
    }
}

/// The smallest (`largest` false) or largest (`largest` true) non-null value
/// of `column` across `batches`, as a one-element array. `None` when every
/// value is null or there are no rows.
fn column_extreme(batches: &[RecordBatch], column: usize, largest: bool) -> Option<ArrayRef> {
    let options = SortOptions {
        descending: largest,
        nulls_first: false,
    };
    // A partial sort of one index selects each batch's extreme in one pass;
    // the small per-batch winners are then combined the same way.
    let per_batch: Vec<ArrayRef> = batches
        .iter()
        .filter_map(|batch| {
            let values = batch.column(column);
            if values.len() == values.null_count() {
                return None;
            }
            let indices = sort_to_indices(values, Some(options), Some(1))
                .expect("join key columns support ordering");
            Some(take(values, &indices, None).expect("a sort index is in bounds"))
        })
        .collect();
    combine_extremes(per_batch, largest)
}

/// The extreme among one-element candidate arrays, itself a one-element
/// array. `None` when there are no candidates.
fn combine_extremes(candidates: Vec<ArrayRef>, largest: bool) -> Option<ArrayRef> {
    if candidates.is_empty() {
        return None;
    }
    let options = SortOptions {
        descending: largest,
        nulls_first: false,
    };
    let refs: Vec<&dyn Array> = candidates.iter().map(|array| array.as_ref()).collect();
    let combined = concat(&refs).expect("extreme candidates share one type");
    let indices = sort_to_indices(&combined, Some(options), Some(1))
        .expect("join key columns support ordering");
    Some(take(&combined, &indices, None).expect("a sort index is in bounds"))
}

impl<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool, O, Out> Consumer<RecordBatch, Out>
    for JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS, O>
where
    O: BatchesOutputter<BuildWorkerOutput<K::Stored>, Out>,
{
    type Outputter = O;

    fn consume(&mut self, batch: RecordBatch, _sender: &mut dyn Sender<Out>) -> unary::Result<()> {
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
                let partition = (hash >> self.partition_shift) as usize;
                self.values[partition].push(
                    &mut self.slab_allocator,
                    BuildTuple {
                        hash,
                        key,
                        row: first_row_id + i as u64,
                    },
                );
            }
        }
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        debug!("Running into outputter!");
        let filter_extremes = self
            .filter_columns
            .iter()
            .map(|&column| ColumnExtremes {
                min: column_extreme(&self.build_row_batches, column, false),
                max: column_extreme(&self.build_row_batches, column, true),
            })
            .collect();
        // The key columns stay with this worker for its key bitset pass;
        // the batches themselves leave with the group.
        let filter_arrays: Vec<Vec<ArrayRef>> = self
            .filter_columns
            .iter()
            .map(|&column| {
                self.build_row_batches
                    .iter()
                    .map(|batch| batch.column(column).clone())
                    .collect()
            })
            .collect();
        assert!(
            self.filter_arrays.set(filter_arrays).is_ok(),
            "a build worker seals once"
        );
        let output = BuildWorkerOutput {
            tuples: self.values,
            build_row_batches: self.build_row_batches,
            saw_null_key: self.saw_null_key,
            filter_extremes,
        };
        self.outputter.accept(output)?;
        Ok(Some(self.outputter))
    }
}

pub struct JoinBuilder<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool> {
    table: JoinTable<K>,
    injector: Arc<Injector<PartitionScatterJob<K>>>,
    /// Whether the final gather arrival has finished initializing the shared
    /// table, decided the pending key bitsets, and pushed the partition
    /// jobs. Release on that store, Acquire on every load, so the pending
    /// bitsets are visible to the workers that read them afterwards.
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
    gather: Arc<GatherBarrier<BuildWorkerOutput<K>>>,
    /// Hash partitions of the build, one scatter job each; see [`partitions_for`].
    partitions: usize,
    build_output_indices: Vec<usize>,
    /// Key filters to publish for sibling probe scans once every build row
    /// is present (the final gather arrival) — publishing any earlier would
    /// expose bounds or a key set tighter than the complete build side and
    /// prune probe rows that do match.
    build_filters: Vec<JoinBuildFilter>,
    /// This worker's filter key columns, left by its consumer at seal time.
    /// Once the final gather arrival has decided which filters get a key
    /// bitset, `output` sets the bits for every value in these arrays; the
    /// rows themselves have long since left with the group.
    filter_arrays: WorkerFilterArrays,
    pending_key_bitsets: PendingKeyBitsets,
    /// Whether this worker has set its share of the key bitsets and counted
    /// that pass toward the readiness gate.
    applied_filter_arrays: bool,
}

unsafe impl<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool> Send
    for JoinBuilder<K, TRACK_MATCHED_BUILD_ROWS>
{
}

/// Scatters the gathered tuples of one hash partition into the shared table.
pub struct PartitionScatterJob<K: Copy + Send> {
    /// This partition's tuples, one entry per build worker, paired with that
    /// worker's row id base (added to each tuple's local row id).
    tuples: Vec<(u64, SlabVec<BuildTuple<K>>)>,
    table: JoinTable<K>,
    arena_offset: usize,
    /// This partition's directory slot range: `slots_per_partition` slots
    /// from `slot_start`.
    slot_start: usize,
    slots_per_partition: usize,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl<K: Copy + Send> Send for PartitionScatterJob<K> {}

impl<K: Copy + Send> PartitionScatterJob<K> {
    fn run(self) {
        debug!("Running partition job");
        let directory = unsafe { &*self.table.directory.get() };
        let shift = directory.shift;
        let keys = unsafe { &*self.table.keys.get() };

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
        let slot_end = self.slot_start + self.slots_per_partition;
        let mut cur = self.arena_offset as u64;

        for i in self.slot_start..slot_end {
            let entry = directory.entry(i);
            let count = entry >> 16;
            let tag = entry & 0xFFFF;
            cur += count;
            directory.set_entry(i, (cur << 16) | tag);
        }

        // Pass 3 scatters into the row arena of whichever width the build
        // chose.
        match unsafe { &*self.table.rows.get() } {
            RowArena::Narrow(rows) => self.scatter(directory, keys, rows),
            RowArena::Wide(rows) => self.scatter(directory, keys, rows),
        }

        // Release-publish this job's writes; the outputter's Acquire load of
        // the drained counter is what makes the table readable.
        self.remaining_jobs.fetch_sub(1, Ordering::Release);
    }

    /// Pass 3: scatter tuples into the parallel key/row arenas. For each
    /// tuple read the directory to get its arena write pointer, advance the
    /// cursor, and write the key and globalized row id. The element
    /// `PREFETCH_AHEAD` positions ahead is used to prefetch its directory
    /// slot before we reach it.
    fn scatter<R: RowId>(
        &self,
        directory: &JoinDirectory,
        keys: &MultiSlabBuffer<K>,
        rows: &MultiSlabBuffer<R>,
    ) {
        const PREFETCH_AHEAD: usize = 64;
        let shift = directory.shift;
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
                    rows.ptr_at_index(arena_idx)
                        .write(R::from_row_id(row_base + tuple.row));
                }
            });
        }
    }
}

impl<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool>
    JoinBuilder<K, TRACK_MATCHED_BUILD_ROWS>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: JoinTable<K>,
        injector: Arc<Injector<PartitionScatterJob<K>>>,
        jobs_injected: Arc<AtomicBool>,
        build_ready: Arc<AtomicBool>,
        remaining_jobs: Arc<AtomicUsize>,
        gather: Arc<GatherBarrier<BuildWorkerOutput<K>>>,
        partitions: usize,
        build_output_indices: Vec<usize>,
        build_filters: Vec<JoinBuildFilter>,
        filter_arrays: WorkerFilterArrays,
        pending_key_bitsets: PendingKeyBitsets,
    ) -> Self {
        Self {
            table,
            injector,
            jobs_injected,
            build_ready,
            remaining_jobs,
            gather,
            partitions,
            build_output_indices,
            build_filters,
            filter_arrays,
            pending_key_bitsets,
            applied_filter_arrays: false,
        }
    }

    /// Combine the gathered worker outputs, initialize the shared table, and
    /// publish its partition jobs. Called only by the final gather arrival.
    fn initialize_groups(
        &mut self,
        mut worker_outputs: Vec<BuildWorkerOutput<K>>,
    ) -> unary::Result<()> {
        let build_saw_null_key = worker_outputs.iter().any(|output| output.saw_null_key);
        unsafe { *self.table.build_saw_null_key.get() = build_saw_null_key };

        // The filters read the stored build rows, so they run before those
        // rows are taken for the merge.
        self.publish_filter_bounds(&worker_outputs);
        let partition_sizes: Vec<usize> = (0..self.partitions)
            .map(|partition| {
                worker_outputs
                    .iter()
                    .map(|output| output.tuples[partition].len())
                    .sum()
            })
            .collect();
        let total_tuples: usize = partition_sizes.iter().sum();
        let (row_bases, row_id_space) = self.publish_build_rows(&mut worker_outputs)?;
        let directory_capacity = self.allocate_directory_and_arenas(total_tuples, row_id_space);
        self.inject_partition_jobs(
            &mut worker_outputs,
            &row_bases,
            &partition_sizes,
            directory_capacity,
        );
        self.jobs_injected.store(true, Ordering::Release);
        Ok(())
    }

    /// Every build row is present here, so each filter's key bounds are
    /// final: merge the per-worker extremes and publish them for the
    /// probe-side scans. A column with no non-null value publishes
    /// nothing, leaving those scans unpruned. Also decides, per filter,
    /// whether a key bitset gets built; every worker then sets its own
    /// share of it from `output`.
    fn publish_filter_bounds(&self, worker_outputs: &[BuildWorkerOutput<K>]) {
        let total_rows: usize = worker_outputs
            .iter()
            .flat_map(|output| output.build_row_batches.iter())
            .map(|batch| batch.num_rows())
            .sum();
        let mut pending_key_bitsets = Vec::with_capacity(self.build_filters.len());
        for (filter_index, filter) in self.build_filters.iter().enumerate() {
            let min = merge_worker_extremes(worker_outputs, filter_index, false);
            let max = merge_worker_extremes(worker_outputs, filter_index, true);
            if let Some(min) = &min {
                filter.min_slot.publish_value(min.clone());
            }
            if let Some(max) = &max {
                filter.max_slot.publish_value(max.clone());
            }
            pending_key_bitsets.push(plan_key_bitset(
                filter,
                min,
                max,
                total_rows,
                worker_outputs.len(),
            ));
        }
        assert!(
            self.pending_key_bitsets.set(pending_key_bitsets).is_ok(),
            "the final gather arrival decides the key bitsets once"
        );
    }

    /// Set this worker's share of every pending key bitset and count the
    /// pass toward the readiness gate.
    ///
    /// A key bitset has one bit per value of the build key domain, kept as
    /// an array of atomic words. The build rows are spread across the
    /// workers, so filling it is split the same way: each worker walks the
    /// key column of every batch it stored and sets the bit of each value
    /// with an atomic OR, which lets all workers write the same words at
    /// once. The pass runs alongside the partition jobs, so the filter is
    /// sealed by the time the probe opens.
    fn apply_filter_arrays(&mut self) {
        let pending_key_bitsets = self
            .pending_key_bitsets
            .get()
            .expect("the key bitsets are decided before the jobs are published");
        let filter_arrays = self
            .filter_arrays
            .get()
            .expect("a build worker seals before its builder runs");
        for (pending, arrays) in pending_key_bitsets.iter().zip(filter_arrays) {
            if let Some(pending) = pending {
                pending.apply_chunk(arrays);
            }
        }
        self.applied_filter_arrays = true;
        self.remaining_jobs.fetch_sub(1, Ordering::Release);
    }

    /// Merge the stored build rows and publish them on the shared table.
    /// GatherBarrier returns outputs in worker order, and the merge keeps
    /// that order, so each worker-local tuple id gets the corresponding row
    /// base during scatter. An empty merge means an empty build side, and
    /// the probe then emits nothing. For a build whose side is outer, anti,
    /// or semi this also allocates one matched flag per row id (gaps
    /// included); null-keyed rows have no tuple and remain unmatched.
    /// Returns the row id bases alongside the id space the merged rows
    /// occupy, which sizes the row arena's width.
    fn publish_build_rows(
        &mut self,
        worker_outputs: &mut [BuildWorkerOutput<K>],
    ) -> unary::Result<(Vec<u64>, usize)> {
        let (build_rows, row_bases) = BuildRows::new::<TRACK_MATCHED_BUILD_ROWS>(
            worker_outputs
                .iter_mut()
                .map(|output| std::mem::take(&mut output.build_row_batches)),
            &self.build_output_indices,
        )?;
        let row_id_space = build_rows::row_id_space(&build_rows.batches);
        unsafe { *self.table.build_rows.get() = build_rows };
        Ok((row_bases, row_id_space))
    }

    /// Size the directory for `total_tuples` and allocate it together with
    /// the key and row arenas the partition jobs scatter into, the row arena
    /// in the width that addresses `row_id_space`. Returns the directory
    /// capacity, which fixes each partition's slot range.
    fn allocate_directory_and_arenas(&mut self, total_tuples: usize, row_id_space: usize) -> usize {
        let directory_capacity = ((total_tuples as f64 * 1.125) as usize)
            .next_power_of_two()
            .max(self.partitions);
        let directory = unsafe { &mut *self.table.directory.get() };
        let mut directory_alloc = SlabAllocator::new(false);
        *directory = JoinDirectory::new(
            directory_alloc.create_multi_slab_buffer(directory_capacity + 1, true),
            directory_capacity,
        );

        // Sentinel at entry[capacity] holds the end pointer of the last slot.
        // Probe reads end_ptr(slot+1) for slot = capacity-1, which lands here.
        directory.set_entry(directory_capacity, (total_tuples as u64) << 16);

        let mut arena_alloc = SlabAllocator::new(false);
        let keys = unsafe { &mut *self.table.keys.get() };
        *keys = arena_alloc.create_multi_slab_buffer::<K>(total_tuples.max(1), false);
        let rows = unsafe { &mut *self.table.rows.get() };
        *rows = RowArena::allocate(&mut arena_alloc, total_tuples.max(1), row_id_space);
        directory_capacity
    }

    /// Push one scatter job per partition, each carrying every worker's
    /// tuples for it along with the partition's arena offset (the running
    /// sum of the sizes before it) and its share of directory slots.
    fn inject_partition_jobs(
        &self,
        worker_outputs: &mut [BuildWorkerOutput<K>],
        row_bases: &[u64],
        partition_sizes: &[usize],
        directory_capacity: usize,
    ) {
        let slots_per_partition = directory_capacity / self.partitions;
        let mut arena_offset = 0;
        for (partition, &size) in partition_sizes.iter().enumerate() {
            let tuples: Vec<(u64, SlabVec<BuildTuple<K>>)> = worker_outputs
                .iter_mut()
                .zip(row_bases)
                .map(|(output, &row_base)| {
                    (row_base, std::mem::take(&mut output.tuples[partition]))
                })
                .collect();
            self.injector.push(PartitionScatterJob {
                tuples,
                table: self.table.clone(),
                arena_offset,
                slot_start: partition * slots_per_partition,
                slots_per_partition,
                remaining_jobs: self.remaining_jobs.clone(),
            });
            arena_offset += size;
        }
    }
}

/// The smallest (or largest) of every worker's own extreme for one filter
/// column; `None` when no worker held a non-null value.
fn merge_worker_extremes<K: Copy>(
    worker_outputs: &[BuildWorkerOutput<K>],
    filter_index: usize,
    largest: bool,
) -> Option<ArrayRef> {
    let candidates = worker_outputs
        .iter()
        .map(|output| &output.filter_extremes[filter_index])
        .filter_map(|extremes| {
            if largest {
                &extremes.max
            } else {
                &extremes.min
            }
            .clone()
        })
        .collect();
    combine_extremes(candidates, largest)
}

/// The key bitset one filter gets: `None` unless both bounds are integers
/// and the bitset accepts that range. Every one of the `worker_count`
/// workers then contributes exactly one pass over its own rows.
fn plan_key_bitset(
    filter: &JoinBuildFilter,
    min: Option<ArrayRef>,
    max: Option<ArrayRef>,
    total_rows: usize,
    worker_count: usize,
) -> Option<Arc<PendingKeyBitset>> {
    let (min, max) = min.zip(max)?;
    let bounds = integer_scalar(&min).zip(integer_scalar(&max))?;
    PendingKeyBitset::try_new(
        min.data_type().clone(),
        bounds,
        total_rows,
        worker_count,
        filter.key_bitset_slot.clone(),
    )
}

impl<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool>
    BatchesOutputter<BuildWorkerOutput<K>, ()> for JoinBuilder<K, TRACK_MATCHED_BUILD_ROWS>
{
    fn accept(&mut self, group: BuildWorkerOutput<K>) -> unary::Result<()> {
        let gather = self.gather.clone();
        if let Some(result) = gather.arrive(group, |groups| self.initialize_groups(groups)) {
            result?;
        }
        Ok(())
    }
}

impl<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool>
    InitializableOutputter<BuildWorkerOutput<K>, ()> for JoinBuilder<K, TRACK_MATCHED_BUILD_ROWS>
{
    fn initialize(&mut self, groups: Vec<BuildWorkerOutput<K>>) -> unary::Result<()> {
        self.initialize_groups(groups)
    }
}

impl<K: Copy + Send, const TRACK_MATCHED_BUILD_ROWS: bool> Outputter<()>
    for JoinBuilder<K, TRACK_MATCHED_BUILD_ROWS>
{
    fn output(&mut self, _sender: &mut dyn Sender<()>) -> unary::Result<bool> {
        // Nothing to do until the final gather arrival has initialized the
        // table; the first thing after that is this worker's own key bitset
        // pass, which no other worker can do for it.
        if !self.applied_filter_arrays {
            if !self.jobs_injected.load(Ordering::Acquire) {
                return Ok(false);
            }
            self.apply_filter_arrays();
        }
        match self.injector.steal() {
            Steal::Success(job) => {
                job.run();
            }
            Steal::Empty => {
                // An empty queue is not completion: a job stolen by another
                // worker may still be running, or a peer may not have made
                // its bitset pass yet, and the table is unreadable until
                // every one of them lands.
                if self.remaining_jobs.load(Ordering::Acquire) == 0 {
                    if self
                        .build_ready
                        .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
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
