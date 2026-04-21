use std::cell::UnsafeCell;
use std::hint::black_box;
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::{mpsc, Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_deque::{Injector, Steal};
use arrow_array::{Array, Int64Array, RecordBatch};
use ahash::RandomState;
use tracing::debug;
use crate::memory::{ContiguousMultiBuffer, MultiSlabBuffer, Slab, SlabAllocator, SlabBuffer, SlabVec, BUFFER_SIZE};
use crate::operations::channels::Sender;
use crate::operations::{unary, Consumer, Outputter};
use crate::operations::unary::join::directory::{prefetch_ptr, Directory, JoinDirectory};
use crate::operations::unary::join::Value;
use crate::perf_stat::{perf_disable, perf_enable};

pub(crate) const NUM_PARTITIONS: usize = 64;
const PARTITION_SHIFT: u32 = 64 - NUM_PARTITIONS.trailing_zeros();

pub struct JoinBuildConsumer {
    key_column: usize,
    hash_state: RandomState,
    values: Vec<SlabVec<(u64, Value)>>,
    slab_allocator: SlabAllocator,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    sender: mpsc::Sender<Vec<SlabVec<(u64, Value)>>>,
    outputter: JoinBuilder,
}

unsafe impl Send for JoinBuildConsumer {}

impl JoinBuildConsumer {
    pub fn new(
        key_column: usize,
        hash_state: RandomState,
        sender: mpsc::Sender<Vec<SlabVec<(u64, Value)>>>,
        receiver: Option<mpsc::Receiver<Vec<SlabVec<(u64, Value)>>>>,
        partition_sizes: Arc<Vec<AtomicUsize>>,
        directory: Arc<UnsafeCell<JoinDirectory>>,
        arena: Arc<UnsafeCell<MultiSlabBuffer<Value>>>,
        injector: Arc<Injector<JoinPartitionJob>>,
        jobs_injected: Arc<AtomicBool>,
        gate: Arc<AtomicBool>,
        remaining_jobs: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            key_column,
            hash_state,
            values: (0..NUM_PARTITIONS).map(|_| SlabVec::new()).collect(),
            slab_allocator: SlabAllocator::new(false),
            partition_sizes: partition_sizes.clone(),
            sender,
            outputter: JoinBuilder {
                directory,
                arena,
                receiver,
                partition_sizes,
                injector,
                jobs_injected,
                gate,
                remaining_jobs,
            },
        }
    }
}

impl Consumer<RecordBatch, ()> for JoinBuildConsumer {
    type Outputter = JoinBuilder;

    fn consume<S: Sender<()>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        debug!("Consuming build");
        let col = batch
            .column(self.key_column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let n = col.len();

        for i in 0..n {
            let key = unsafe { col.value_unchecked(i) };
            let hash = self.hash_state.hash_one(key);
            let partition = (hash >> PARTITION_SHIFT) as usize;
            self.values[partition].push((hash, key as u32), &mut self.slab_allocator);
        }

        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        debug!("Running into outputter!");
        for (p, partition) in self.values.iter().enumerate() {
            self.partition_sizes[p].fetch_add(partition.len(), Ordering::Relaxed);
        }
        self.sender.send(self.values).unwrap();
        Ok(Some(self.outputter))
    }
}

pub struct JoinBuilder {
    directory: Arc<UnsafeCell<JoinDirectory>>,
    arena: Arc<UnsafeCell<MultiSlabBuffer<Value>>>,
    receiver: Option<mpsc::Receiver<Vec<SlabVec<(u64, Value)>>>>,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl Send for JoinBuilder {}

pub struct JoinPartitionJob {
    tuples: Vec<SlabVec<(u64, Value)>>,
    directory: Arc<UnsafeCell<JoinDirectory>>,
    arena: Arc<UnsafeCell<MultiSlabBuffer<Value>>>,
    arena_offset: usize,

    slot_start: usize,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl Send for JoinPartitionJob {}

impl JoinPartitionJob {
    fn run(self) {
        debug!("Running partition job");
        let join_dir = unsafe { &*self.directory.get() };
        match join_dir {
            JoinDirectory::Contiguous(dir) => self.run_with_dir(dir),
            JoinDirectory::NonContiguous(dir) => self.run_with_dir(dir),
        }
    }

    #[inline(always)]
    fn run_with_dir<B: Index<usize, Output = u64> + IndexMut<usize>>(
        &self,
        directory: &Directory<B>,
    ) {
        let shift = directory.shift;
        let arena = unsafe { &*self.arena.get() };

        // Pass 1: accumulate counts in upper 48 bits, OR bloom tags into
        // lower 16 bits
        for worker_tuples in &self.tuples {
            for &(hash, _) in worker_tuples {
                let slot = (hash >> shift) as usize;
                unsafe {
                    directory.add_to_entry(slot, 1 << 16);
                    directory.or_to_entry(slot, Directory::<B>::compute_tag(hash) as u64);
                }
            }
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

        // Pass 3: scatter values into arena in batches of COUNT:
        //   1. Consume tuples, compute slots
        //   2. Read directory to get arena write pointers, advance cursors
        //   3. Write values to the collected arena pointers
        for worker_tuples in &self.tuples {

            let mut total_left = worker_tuples.len();

            for slab in worker_tuples.slabs() {
                let ptr = slab.ptr as *mut (u64, Value);
                let slab_size = slab.size / size_of::<(u64, Value)>();

                let count = std::cmp::min(slab_size, total_left);

                for i in 0..count {
                    let (hash, value) = unsafe { *(ptr.add(i) ) };
                    let slot = (hash >> shift) as usize;
                    let entry = directory.entry(slot).wrapping_sub(1 << 16);
                    directory.set_entry(slot, entry);
                    let arena_idx = (entry >> 16) as usize;
                    unsafe { arena.ptr_at_index(arena_idx).write(value) };
                }

                total_left -= count;
            }
        }

        // Last partition job to complete opens the gate for the probe side.
        if self.remaining_jobs.fetch_sub(1, Ordering::Release) == 1 {
            self.gate.store(true, Ordering::Release);
        }
    }
}

impl Outputter<()> for JoinBuilder {
    fn output<S: Sender<()>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_enable();
        if let Some(rx) = self.receiver.take() {
            let mut all_worker_tuples: Vec<Vec<SlabVec<(u64, Value)>>> = rx.into_iter().collect();

            let sizes: Vec<usize> = self
                .partition_sizes
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let total: usize = sizes.iter().sum();

            // Pre-allocate directory and arena.
            let dir_capacity = ((total as f64 * 1.125) as usize).next_power_of_two().max(NUM_PARTITIONS);
            let directory = unsafe { &mut *self.directory.get() };
            *directory = match ContiguousMultiBuffer::<u64>::new(dir_capacity + 1) {
                Ok(buf) => JoinDirectory::Contiguous(Directory::new(buf, dir_capacity)),
                Err(()) => {
                    let mut alloc = SlabAllocator::new(false);
                    JoinDirectory::NonContiguous(Directory::new(
                        alloc.create_multi_slab_buffer(dir_capacity + 1, true),
                        dir_capacity,
                    ))
                }
            };

            // let mut alloc = SlabAllocator::new(false);
            // *directory =  JoinDirectory::NonContiguous(Directory::new(
            //     alloc.create_multi_slab_buffer(dir_capacity + 1, true),
            //     dir_capacity,
            // ));


            let arena = unsafe { &mut *self.arena.get() };
            let mut arena_alloc = SlabAllocator::new(false);
            *arena = arena_alloc.create_multi_slab_buffer::<Value>(total.max(1), false);

            // Prefix sums give each partition its arena offset.
            let mut offsets = vec![0usize; NUM_PARTITIONS];
            for p in 1..NUM_PARTITIONS {
                offsets[p] = offsets[p - 1] + sizes[p - 1];
            }

            let slots_per_partition = dir_capacity / NUM_PARTITIONS;
            for i in 0..NUM_PARTITIONS {
                let tuples: Vec<SlabVec<(u64, Value)>> = all_worker_tuples
                    .iter_mut()
                    .map(|worker| std::mem::take(&mut worker[i]))
                    .collect();
                self.injector.push(JoinPartitionJob {
                    tuples,
                    directory: self.directory.clone(),
                    arena: self.arena.clone(),
                    arena_offset: offsets[i],
                    slot_start: i * slots_per_partition,
                    gate: self.gate.clone(),
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
                if self.jobs_injected.load(Ordering::Relaxed) {
                    perf_disable();
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}
