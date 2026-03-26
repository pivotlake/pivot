use std::cell::UnsafeCell;
use std::ops::{Index, IndexMut};
use std::sync::{mpsc, Arc};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_deque::{Injector, Steal};
use arrow_array::{Array, Int64Array, RecordBatch};
use ahash::RandomState;
use tracing::debug;
use crate::memory::{ContiguousMultiBuffer, SlabAllocator, SlabVec};
use crate::operations::channels::Sender;
use crate::operations::{unary, Consumer, Outputter};
use crate::operations::unary::join::directory::{prefetch_ptr, Directory, JoinDirectory};
use crate::operations::unary::join::Value;

pub(crate) const NUM_PARTITIONS: usize = 128;
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
        arena: Arc<UnsafeCell<Vec<Value>>>,
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
    arena: Arc<UnsafeCell<Vec<Value>>>,
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
    arena: Arc<UnsafeCell<Vec<Value>>>,
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
        let arena_ptr = unsafe { (*self.arena.get()).as_mut_ptr() };

        // Pass 1: accumulate counts in upper 48 bits, OR bloom tags into
        // lower 16 bits — directly in the directory, no side allocations.
        for worker_tuples in &self.tuples {
            for &(hash, _) in worker_tuples {
                let slot = (hash >> shift) as usize;
                unsafe {
                    directory.add_to_entry(slot, 1 << 16);
                    // directory.add_to_entry(slot, Directory::compute_tag(hash) as u64);
                }
            }
        }

        // Pass 2: convert counts → absolute write cursors. Linear sweep
        // over this partition's directory slot range replaces each count
        // with a running arena pointer while preserving the tag bits.
        let slots_per_partition = directory.capacity() / NUM_PARTITIONS;
        let slot_end = self.slot_start + slots_per_partition;
        let mut cur = self.arena_offset as u64;
        for i in self.slot_start..slot_end {
            let entry = directory.entry(i);
            let count = entry >> 16;
            let tag = entry & 0xFFFF;
            unsafe { directory.set_entry(i, (cur << 16) | tag); }
            cur += count;
        }

        // Pass 3: scatter values into arena. The upper 48 bits of each
        // directory entry serve as the write cursor; advancing by 1<<16
        // leaves the tag bits untouched. After this pass the cursors have
        // become end-pointers.
        for worker_tuples in &self.tuples {

            let mut iterator = worker_tuples.into_iter();
            let mut arena_iterator = worker_tuples.into_iter();

            let mut work_iterator = worker_tuples.into_iter();

            const COUNT: usize = 128;

            loop {
                for _ in 0..COUNT {
                    if let Some((hash, _)) =iterator.next() {
                        directory.prefetch(*hash);
                    }
                }

                for _ in 0..COUNT {
                    if let Some((hash, _)) =arena_iterator.next() {
                        let slot = (hash >> shift) as usize;
                        let arena_idx = (directory.entry(slot) >> 16) as usize;
                        prefetch_ptr(unsafe { arena_ptr.add(arena_idx) } as *const u8);
                    }
                }

                let mut is_done = false;
                for _ in 0..COUNT {
                    if let Some((hash, value)) =work_iterator.next() {
                        let slot = (hash >> shift) as usize;
                        unsafe {
                            let arena_idx = (directory.entry(slot) >> 16) as usize;
                            arena_ptr.add(arena_idx).write(*value);
                            directory.add_to_entry(slot, 1 << 16);
                        }
                    } else {
                        is_done = true;
                        break;
                    }
                }

                if is_done {
                    break;
                }
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
        if let Some(rx) = self.receiver.take() {
            let mut all_worker_tuples: Vec<Vec<SlabVec<(u64, Value)>>> = rx.into_iter().collect();

            let sizes: Vec<usize> = self
                .partition_sizes
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let total: usize = sizes.iter().sum();

            // Pre-allocate directory and arena.
            let dir_capacity = ((total as f64 * 1.125) as usize).next_power_of_two().max(64);
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

            let arena = unsafe { &mut *self.arena.get() };
            arena.reserve(total);
            unsafe { arena.set_len(total) };

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
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}
