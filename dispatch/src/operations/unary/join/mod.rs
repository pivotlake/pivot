mod hashtable;

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use ahash::RandomState;
use arrow_array::{Array, ArrayAccessor, Int64Array, RecordBatch};
use crossbeam_deque::{Injector, Steal};

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::hashtable::Directory;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};

type Value = (u64, u64);

const NUM_PARTITIONS: usize = 64;
const PARTITION_SHIFT: u32 = 64 - NUM_PARTITIONS.trailing_zeros();

struct JoinBuildConsumer {
    key_column: usize,
    hash_state: RandomState,
    values: Vec<Vec<(u64, Value)>>,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    sender: mpsc::Sender<Vec<Vec<(u64, Value)>>>,
    outputter: JoinBuilder,
}

unsafe impl Send for JoinBuildConsumer {}

impl JoinBuildConsumer {
    pub fn new(
        key_column: usize,
        hash_state: RandomState,
        sender: mpsc::Sender<Vec<Vec<(u64, Value)>>>,
        receiver: Option<mpsc::Receiver<Vec<Vec<(u64, Value)>>>>,
        partition_sizes: Arc<Vec<AtomicUsize>>,
        directory: Arc<UnsafeCell<Directory>>,
        arena: Arc<UnsafeCell<Vec<Value>>>,
        injector: Arc<Injector<JoinPartitionJob>>,
        jobs_injected: Arc<AtomicBool>,
    ) -> Self {
        Self {
            key_column,
            hash_state,
            values: (0..NUM_PARTITIONS).map(|_| Vec::new()).collect(),
            partition_sizes: partition_sizes.clone(),
            sender,
            outputter: JoinBuilder {
                directory,
                arena,
                receiver,
                partition_sizes,
                injector,
                jobs_injected,
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
            self.values[partition].push((hash, (key as u64, i as u64)));
        }

        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        for (p, partition) in self.values.iter().enumerate() {
            self.partition_sizes[p].fetch_add(partition.len(), Ordering::Relaxed);
        }
        self.sender.send(self.values).unwrap();
        Ok(Some(self.outputter))
    }
}

struct JoinBuilder {
    directory: Arc<UnsafeCell<Directory>>,
    arena: Arc<UnsafeCell<Vec<Value>>>,
    receiver: Option<mpsc::Receiver<Vec<Vec<(u64, Value)>>>>,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
}

unsafe impl Send for JoinBuilder {}

struct JoinPartitionJob {
    tuples: Vec<Vec<(u64, Value)>>,
    directory: Arc<UnsafeCell<Directory>>,
    arena: Arc<UnsafeCell<Vec<Value>>>,
    arena_offset: usize,
    slot_start: usize,
}

unsafe impl Send for JoinPartitionJob {}

impl JoinPartitionJob {
    fn run(self) {
        let dir_ptr = self.directory.get();
        let shift = unsafe { (*dir_ptr).shift };
        let dir_capacity = unsafe { (*dir_ptr).entries.len() };
        let dir_entries = unsafe { (*dir_ptr).entries.as_mut_ptr() };
        let arena_ptr = unsafe { (*self.arena.get()).as_mut_ptr() };

        // Pass 1: accumulate counts in upper 48 bits, OR bloom tags into
        // lower 16 bits — directly in the directory, no side allocations.
        for worker_tuples in &self.tuples {
            for &(hash, _) in worker_tuples {
                let slot = (hash >> shift) as usize;
                unsafe {
                    *dir_entries.add(slot) += 1 << 16;
                    *dir_entries.add(slot) |= Directory::bloom(hash);
                }
            }
        }

        // Pass 2: convert counts → absolute write cursors. Linear sweep
        // over this partition's directory slot range replaces each count
        // with a running arena pointer while preserving the tag bits.
        let slots_per_partition = dir_capacity / NUM_PARTITIONS;
        let slot_end = self.slot_start + slots_per_partition;
        let mut cur = self.arena_offset as u64;
        for i in self.slot_start..slot_end {
            unsafe {
                let entry = *dir_entries.add(i);
                let count = entry >> 16;
                let tag = entry & 0xFFFF;
                dir_entries.add(i).write((cur << 16) | tag);
                cur += count;
            }
        }

        // Pass 3: scatter values into arena. The upper 48 bits of each
        // directory entry serve as the write cursor; advancing by 1<<16
        // leaves the tag bits untouched. After this pass the cursors have
        // become end-pointers.
        for worker_tuples in &self.tuples {
            for &(hash, value) in worker_tuples {
                let slot = (hash >> shift) as usize;
                unsafe {
                    let arena_idx = (*dir_entries.add(slot) >> 16) as usize;
                    arena_ptr.add(arena_idx).write(value);
                    *dir_entries.add(slot) += 1 << 16;
                }
            }
        }
    }
}

impl Outputter<()> for JoinBuilder {
    fn output<S: Sender<()>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
            let mut all_worker_tuples: Vec<Vec<Vec<(u64, Value)>>> = rx.into_iter().collect();

            let sizes: Vec<usize> = self
                .partition_sizes
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let total: usize = sizes.iter().sum();

            // Pre-allocate directory and arena.
            let dir_capacity = ((total as f64 * 1.125) as usize).next_power_of_two().max(64);
            let directory = unsafe { &mut *self.directory.get() };
            *directory = Directory::with_capacity(dir_capacity);

            let arena = unsafe { &mut *self.arena.get() };
            arena.resize(total, (0, 0));

            // Prefix sums give each partition its arena offset.
            let mut offsets = vec![0usize; NUM_PARTITIONS];
            for p in 1..NUM_PARTITIONS {
                offsets[p] = offsets[p - 1] + sizes[p - 1];
            }

            let slots_per_partition = dir_capacity / NUM_PARTITIONS;
            for i in 0..NUM_PARTITIONS {
                let tuples: Vec<Vec<(u64, Value)>> = all_worker_tuples
                    .iter_mut()
                    .map(|worker| std::mem::take(&mut worker[i]))
                    .collect();
                self.injector.push(JoinPartitionJob {
                    tuples,
                    directory: self.directory.clone(),
                    arena: self.arena.clone(),
                    arena_offset: offsets[i],
                    slot_start: i * slots_per_partition,
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
