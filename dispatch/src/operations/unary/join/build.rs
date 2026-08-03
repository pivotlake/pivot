use crate::memory::{SlabAllocator, SlabVec};
use crate::operations::channels::Sender;
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::keys::{JoinKey, combined_key_validity};
use crate::operations::{Consumer, Outputter, unary};
use crate::waker::worker_waker;
use ahash::RandomState;
use arrow::compute::{concat_batches, filter_record_batch};
use arrow_array::{BooleanArray, RecordBatch};
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use tracing::debug;

pub(crate) const NUM_PARTITIONS: usize = 64;
const PARTITION_SHIFT: u32 = 64 - NUM_PARTITIONS.trailing_zeros();

/// One build-side row, radix-partitioned by `hash`: the full-width key (for
/// exact match rejection of hash collisions in the probe) and the row's index
/// into this worker's payload batches (globalized with the worker's base
/// offset at scatter time).
#[derive(Clone, Copy)]
pub(crate) struct BuildTuple<K> {
    hash: u64,
    key: K,
    row: u32,
}

pub(crate) type PartitionBuffers<K> = Vec<SlabVec<BuildTuple<K>>>;

/// Everything one build worker hands to the single [`JoinBuilder`] that
/// assembles the join table: which worker it was (payload row indices are
/// globalized in worker-id order), its partitioned tuples, and the payload
/// batches those tuples' row indices point into.
pub(crate) struct BuildWorkerOutput<K: Copy> {
    worker_id: usize,
    tuples: PartitionBuffers<K>,
    payload: Vec<RecordBatch>,
}

pub struct JoinBuildConsumer<K: JoinKey, const BUILD_OUTER: bool> {
    key_columns: Vec<usize>,
    worker_id: usize,
    hash_state: RandomState,
    values: PartitionBuffers<K::Stored>,
    payload: Vec<RecordBatch>,
    rows_consumed: usize,
    slab_allocator: SlabAllocator,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    sender: mpsc::Sender<BuildWorkerOutput<K::Stored>>,
    outputter: JoinBuilder<K::Stored, BUILD_OUTER>,
}

unsafe impl<K: JoinKey, const BUILD_OUTER: bool> Send for JoinBuildConsumer<K, BUILD_OUTER> {}

impl<K: JoinKey, const BUILD_OUTER: bool> JoinBuildConsumer<K, BUILD_OUTER> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        key_columns: Vec<usize>,
        worker_id: usize,
        hash_state: RandomState,
        sender: mpsc::Sender<BuildWorkerOutput<K::Stored>>,
        receiver: Option<mpsc::Receiver<BuildWorkerOutput<K::Stored>>>,
        partition_sizes: Arc<Vec<AtomicUsize>>,
        table: JoinTable<K::Stored>,
        injector: Arc<Injector<JoinPartitionJob<K::Stored>>>,
        jobs_injected: Arc<AtomicBool>,
        build_ready: Arc<AtomicBool>,
        remaining_jobs: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            key_columns,
            worker_id,
            hash_state,
            values: (0..NUM_PARTITIONS).map(|_| SlabVec::new()).collect(),
            payload: Vec::new(),
            rows_consumed: 0,
            slab_allocator: SlabAllocator::new(false),
            partition_sizes: partition_sizes.clone(),
            sender,
            outputter: JoinBuilder {
                table,
                receiver,
                partition_sizes,
                injector,
                jobs_injected,
                build_ready,
                remaining_jobs,
            },
        }
    }
}

/// Drop rows any of whose join key columns is null: an equi-join can never
/// match them, and the hot loops read key values without validity checks. A
/// build-side outer join keeps its build rows instead (they still reach the
/// output, unmatched) and skips them when generating tuples.
pub(crate) fn filter_null_keys(batch: RecordBatch, key_columns: &[usize]) -> RecordBatch {
    let Some(combined_validity) = combined_key_validity(&batch, key_columns) else {
        return batch;
    };
    let mask = BooleanArray::new(combined_validity.inner().clone(), None);
    filter_record_batch(&batch, &mask).expect("null-key filter mask matches batch length")
}

impl<K: JoinKey, const BUILD_OUTER: bool> Consumer<RecordBatch, ()>
    for JoinBuildConsumer<K, BUILD_OUTER>
{
    type Outputter = JoinBuilder<K::Stored, BUILD_OUTER>;

    fn consume(&mut self, batch: RecordBatch, _sender: &mut dyn Sender<()>) -> unary::Result<()> {
        debug!("Consuming build");
        let batch = match BUILD_OUTER {
            true => batch,
            false => filter_null_keys(batch, &self.key_columns),
        };
        let reader = K::make_reader(&batch, &self.key_columns);
        let n = batch.num_rows();
        assert!(
            self.rows_consumed + n <= u32::MAX as usize,
            "join build side exceeds u32 row indexing"
        );

        for i in 0..n {
            // Null-keyed rows are still in the payload of an outer build, and
            // reach the output through the unmatched pass rather than a tuple.
            if BUILD_OUTER && K::is_null(&reader, i) {
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
                    row: (self.rows_consumed + i) as u32,
                },
            );
        }
        drop(reader);
        self.rows_consumed += n;
        self.payload.push(batch);

        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        debug!("Running into outputter!");
        for (p, partition) in self.values.iter().enumerate() {
            self.partition_sizes[p].fetch_add(partition.len(), Ordering::Relaxed);
        }
        self.sender
            .send(BuildWorkerOutput {
                worker_id: self.worker_id,
                tuples: self.values,
                payload: self.payload,
            })
            .unwrap();
        Ok(Some(self.outputter))
    }
}

pub struct JoinBuilder<K: Copy + Send, const BUILD_OUTER: bool> {
    table: JoinTable<K>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput<K>>>,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    injector: Arc<Injector<JoinPartitionJob<K>>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl<K: Copy + Send, const BUILD_OUTER: bool> Send for JoinBuilder<K, BUILD_OUTER> {}

pub struct JoinPartitionJob<K: Copy + Send> {
    /// This partition's tuples, one entry per build worker, paired with that
    /// worker's global payload row base (added to each tuple's local row).
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
        // cursor, and write the key and globalized payload row. The element
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

impl<K: Copy + Send, const BUILD_OUTER: bool> Outputter<()> for JoinBuilder<K, BUILD_OUTER> {
    fn output(&mut self, _sender: &mut dyn Sender<()>) -> unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
            let mut worker_outputs: Vec<BuildWorkerOutput<K>> = rx.into_iter().collect();
            // Payload row indices are globalized in worker-id order, so the
            // concatenated payload batch must follow the same order.
            worker_outputs.sort_by_key(|output| output.worker_id);

            let sizes: Vec<usize> = self
                .partition_sizes
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let total: usize = sizes.iter().sum();

            // Each worker's payload rows start at the end of the previous
            // worker's; tuples carry worker-local rows and get the base added
            // during scatter.
            let mut row_bases = Vec::with_capacity(worker_outputs.len());
            let mut base = 0u32;
            for output in &worker_outputs {
                row_bases.push(base);
                let rows: usize = output.payload.iter().map(|b| b.num_rows()).sum();
                base += rows as u32;
            }

            // Concatenate every worker's payload into the single batch probe
            // rows are gathered from. No batches means an empty build side;
            // the probe then emits nothing.
            let payload_batches: Vec<RecordBatch> = worker_outputs
                .iter()
                .flat_map(|output| output.payload.iter().cloned())
                .collect();
            let build_rows = unsafe { &mut *self.table.build_rows.get() };
            *build_rows = payload_batches.first().map(|first| {
                concat_batches(&first.schema(), &payload_batches)
                    .expect("build payload batches share a schema")
            });

            // One flag per payload row, not per tuple: a null-keyed row of an
            // outer build has no tuple and stays permanently unmatched.
            if BUILD_OUTER {
                let payload_rows = build_rows.as_ref().map_or(0, |batch| batch.num_rows());
                let mut matched_alloc = SlabAllocator::new(false);
                let matched = unsafe { &mut *self.table.matched.get() };
                *matched = matched_alloc.create_multi_slab_buffer::<u8>(payload_rows.max(1), true);
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
                self.injector.push(JoinPartitionJob {
                    tuples,
                    table: self.table.clone(),
                    arena_offset,
                    slot_start: i * slots_per_partition,
                    remaining_jobs: self.remaining_jobs.clone(),
                });
            }
            self.jobs_injected.store(true, Ordering::Relaxed);
        }

        match self.injector.steal() {
            Steal::Success(job) => {
                job.run();
            }
            Steal::Empty => {
                // An empty queue is not completion: a job stolen by another
                // worker may still be running, and the table is unreadable
                // until it lands. Report done only when every job has run.
                if self.jobs_injected.load(Ordering::Relaxed)
                    && self.remaining_jobs.load(Ordering::Acquire) == 0
                {
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
