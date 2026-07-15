use crate::operations::channels::Sender;
use crate::operations::unary::join::directory::{Directory, JoinDirectory};
use crate::operations::unary::join::{JoinArena, JoinCell};
use crate::operations::{Consumer, Outputter, unary};
use ahash::RandomState;
use arrow::compute::{concat_batches, filter_record_batch};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, RecordBatch};
use crossbeam_deque::{Injector, Steal};
use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use tracing::debug;

pub(crate) const NUM_PARTITIONS: usize = 64;
const PARTITION_SHIFT: u32 = 64 - NUM_PARTITIONS.trailing_zeros();

/// One build-side row, radix-partitioned by `hash` (the combined hash of every
/// key column, which the probe compares in full — see the module doc on
/// verification), and the row's index into this worker's payload batches
/// (globalized with the worker's base offset at scatter time).
#[derive(Clone, Copy)]
pub(crate) struct BuildTuple {
    hash: u64,
    row: u32,
}

/// Per-partition staging for one worker's build tuples. Plain heap (not ring
/// slabs): a big build's staging would otherwise compete with group-bys and
/// the caches for ring memory, and heap frees safely from any thread.
pub(crate) type PartitionBuffers = Vec<Vec<BuildTuple>>;

/// Everything one build worker hands to the single [`JoinBuilder`] that
/// assembles the join table: which worker it was (payload row indices are
/// globalized in worker-id order), its partitioned tuples, and the payload
/// batches those tuples' row indices point into.
pub(crate) struct BuildWorkerOutput {
    worker_id: usize,
    tuples: PartitionBuffers,
    payload: Vec<RecordBatch>,
}

pub struct JoinBuildConsumer {
    key_columns: Vec<usize>,
    /// Aligned with `key_columns`: `true` for a null-safe (`IS NOT DISTINCT
    /// FROM`) key, where two nulls match each other.
    null_safe: Vec<bool>,
    worker_id: usize,
    hash_state: RandomState,
    values: PartitionBuffers,
    payload: Vec<RecordBatch>,
    rows_consumed: usize,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    sender: mpsc::Sender<BuildWorkerOutput>,
    outputter: JoinBuilder,
}

unsafe impl Send for JoinBuildConsumer {}

impl JoinBuildConsumer {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        key_columns: Vec<usize>,
        null_safe: Vec<bool>,
        worker_id: usize,
        hash_state: RandomState,
        sender: mpsc::Sender<BuildWorkerOutput>,
        receiver: Option<mpsc::Receiver<BuildWorkerOutput>>,
        partition_sizes: Arc<Vec<AtomicUsize>>,
        directory: Arc<JoinCell<JoinDirectory>>,
        keys: Arc<JoinCell<JoinArena<u64>>>,
        rows: Arc<JoinCell<JoinArena<u32>>>,
        build_rows: Arc<JoinCell<Option<RecordBatch>>>,
        matched: Arc<JoinCell<Vec<std::sync::atomic::AtomicBool>>>,
        injector: Arc<Injector<JoinPartitionJob>>,
        jobs_injected: Arc<AtomicBool>,
        gate: Arc<AtomicBool>,
        remaining_jobs: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            key_columns,
            null_safe,
            worker_id,
            hash_state,
            values: (0..NUM_PARTITIONS).map(|_| Vec::new()).collect(),
            payload: Vec::new(),
            rows_consumed: 0,
            partition_sizes: partition_sizes.clone(),
            sender,
            outputter: JoinBuilder {
                directory,
                keys,
                rows,
                build_rows,
                matched,
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

/// The key columns whose comparison is strict equality (`= `), where a null
/// never matches and the row can be dropped up front.
pub(crate) fn strict_key_columns(key_columns: &[usize], null_safe: &[bool]) -> Vec<usize> {
    key_columns
        .iter()
        .zip(null_safe)
        .filter(|&(_, &null_safe)| !null_safe)
        .map(|(&column, _)| column)
        .collect()
}

/// Drop rows where any strict join-key column is null: an equi-join can never
/// match them. Null-safe (`IS NOT DISTINCT FROM`) keys keep their null rows,
/// which hash and compare through their validity bit.
pub(crate) fn filter_null_keys(batch: RecordBatch, key_columns: &[usize]) -> RecordBatch {
    let mut mask: Option<BooleanArray> = None;
    for &key_column in key_columns {
        let col = batch.column(key_column);
        if col.null_count() == 0 {
            continue;
        }
        let valid = BooleanArray::new(col.nulls().unwrap().inner().clone(), None);
        mask = Some(match mask {
            None => valid,
            Some(mask) => arrow::compute::and(&mask, &valid).expect("equal-length masks"),
        });
    }
    let Some(mask) = mask else {
        return batch;
    };
    filter_record_batch(&batch, &mask).expect("null-key filter mask matches batch length")
}

/// The combined hash of one row's key columns. A single strict column hashes
/// exactly as `RandomState::hash_one(value)` does; more columns fold in
/// through the same hasher. A null-safe key folds its validity bit and reads
/// null as 0, so two nulls hash (and later compare) equal.
#[inline(always)]
pub(crate) fn hash_key_row(
    hash_state: &RandomState,
    columns: &[&arrow_array::Int64Array],
    null_safe: &[bool],
    row: usize,
) -> u64 {
    if let ([column], [false]) = (columns, null_safe) {
        return hash_state.hash_one(unsafe { column.value_unchecked(row) });
    }
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = hash_state.build_hasher();
    for (column, &null_safe) in columns.iter().zip(null_safe) {
        if null_safe {
            let valid = column.is_valid(row);
            hasher.write_u8(valid as u8);
            hasher.write_i64(if valid {
                unsafe { column.value_unchecked(row) }
            } else {
                0
            });
        } else {
            hasher.write_i64(unsafe { column.value_unchecked(row) });
        }
    }
    hasher.finish()
}

impl Consumer<RecordBatch, ()> for JoinBuildConsumer {
    type Outputter = JoinBuilder;

    fn consume<S: Sender<()>>(&mut self, batch: RecordBatch, _sender: &mut S) -> unary::Result<()> {
        debug!("Consuming build");
        let batch = filter_null_keys(
            batch,
            &strict_key_columns(&self.key_columns, &self.null_safe),
        );
        let columns: Vec<&arrow_array::Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let n = batch.num_rows();
        assert!(
            self.rows_consumed + n <= u32::MAX as usize,
            "join build side exceeds u32 row indexing"
        );

        for i in 0..n {
            let hash = hash_key_row(&self.hash_state, &columns, &self.null_safe, i);
            let partition = (hash >> PARTITION_SHIFT) as usize;
            self.values[partition].push(BuildTuple {
                hash,
                row: (self.rows_consumed + i) as u32,
            });
        }
        self.rows_consumed += n;
        // Detach the retained payload from ring memory NOW: holding the scan's
        // ring-backed batches until the build assembles would pin a big build
        // side's entire input in the ring and starve the caches (and any
        // group-by) of evictable slots.
        self.payload
            .push(crate::operations::unary::copy_out::detach_batch(&batch)?);

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

pub struct JoinBuilder {
    directory: Arc<JoinCell<JoinDirectory>>,
    keys: Arc<JoinCell<JoinArena<u64>>>,
    rows: Arc<JoinCell<JoinArena<u32>>>,
    build_rows: Arc<JoinCell<Option<RecordBatch>>>,
    matched: Arc<JoinCell<Vec<std::sync::atomic::AtomicBool>>>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput>>,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl Send for JoinBuilder {}

pub struct JoinPartitionJob {
    /// This partition's tuples, one entry per build worker, paired with that
    /// worker's global payload row base (added to each tuple's local row).
    tuples: Vec<(u32, Vec<BuildTuple>)>,
    directory: Arc<JoinCell<JoinDirectory>>,
    keys: Arc<JoinCell<JoinArena<u64>>>,
    rows: Arc<JoinCell<JoinArena<u32>>>,
    arena_offset: usize,

    slot_start: usize,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl Send for JoinPartitionJob {}

impl JoinPartitionJob {
    fn run(self) {
        debug!("Running partition job");
        let directory = unsafe { &*self.directory.get() };
        self.run_with_dir(directory);
    }

    #[inline(always)]
    fn run_with_dir<B: Index<usize, Output = u64> + IndexMut<usize>>(
        &self,
        directory: &Directory<B>,
    ) {
        let shift = directory.shift;
        let keys = unsafe { &*self.keys.get() };
        let rows = unsafe { &*self.rows.get() };

        // Pass 1: accumulate counts in upper 48 bits, OR bloom tags into
        // lower 16 bits
        for (_, worker_tuples) in &self.tuples {
            for tuple in worker_tuples {
                let slot = (tuple.hash >> shift) as usize;
                unsafe {
                    directory.add_to_entry(slot, 1 << 16);
                    directory.or_to_entry(slot, Directory::<B>::compute_tag(tuple.hash) as u64);
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

        // Pass 3: scatter tuples into the parallel key/row arenas. For each
        // tuple read the directory to get its arena write pointer, advance the
        // cursor, and write the key and globalized payload row. The element
        // `PREFETCH_AHEAD` positions ahead is used to prefetch its directory
        // slot before we reach it.
        const PREFETCH_AHEAD: usize = 64;
        for (row_base, worker_tuples) in &self.tuples {
            for (i, tuple) in worker_tuples.iter().enumerate() {
                if let Some(ahead_tuple) = worker_tuples.get(i + PREFETCH_AHEAD) {
                    directory.prefetch_l2(ahead_tuple.hash);
                }
                let slot = (tuple.hash >> shift) as usize;
                let entry = directory.entry(slot).wrapping_sub(1 << 16);
                directory.set_entry(slot, entry);
                let arena_idx = (entry >> 16) as usize;
                unsafe {
                    keys.ptr_at_index(arena_idx).write(tuple.hash);
                    rows.ptr_at_index(arena_idx).write(row_base + tuple.row);
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
            let mut worker_outputs: Vec<BuildWorkerOutput> = rx.into_iter().collect();
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
            // rows are gathered from, then detach it onto the heap: the
            // concatenated views still share the scans' ring buffers, and the
            // shared join table can be dropped from a non-worker thread (where
            // ring memory must never be released) besides pinning ring slots
            // for the whole query. No batches means an empty build side; the
            // probe then emits nothing.
            let payload_batches: Vec<RecordBatch> = worker_outputs
                .iter()
                .flat_map(|output| output.payload.iter().cloned())
                .collect();
            // The per-batch payloads were detached to the heap at consume
            // time, so the concatenation is heap-backed too.
            let build_rows = unsafe { &mut *self.build_rows.get() };
            *build_rows = payload_batches.first().map(|first| {
                concat_batches(&first.schema(), &payload_batches)
                    .expect("build payload batches share a schema")
            });
            // One match flag per payload row, for the build-emitting
            // existence modes (unused by the others).
            let matched = unsafe { &mut *self.matched.get() };
            let payload_rows = build_rows.as_ref().map(|b| b.num_rows()).unwrap_or(0);
            *matched = (0..payload_rows)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect();

            // Pre-allocate directory and arenas (heap, see JoinArena's doc).
            let dir_capacity = ((total as f64 * 1.125) as usize)
                .next_power_of_two()
                .max(NUM_PARTITIONS);
            let directory = unsafe { &mut *self.directory.get() };
            *directory = Directory::new(vec![0u64; dir_capacity + 1], dir_capacity);

            // Sentinel at entry[capacity] holds the end pointer of the last slot.
            // Probe reads end_ptr(slot+1) for slot = capacity-1, which lands here.
            directory.set_entry(dir_capacity, (total as u64) << 16);

            let keys = unsafe { &mut *self.keys.get() };
            *keys = JoinArena::new(total.max(1));
            let rows = unsafe { &mut *self.rows.get() };
            *rows = JoinArena::new(total.max(1));

            // Prefix sums give each partition its arena offset.
            let mut offsets = vec![0usize; NUM_PARTITIONS];
            for p in 1..NUM_PARTITIONS {
                offsets[p] = offsets[p - 1] + sizes[p - 1];
            }

            let slots_per_partition = dir_capacity / NUM_PARTITIONS;
            for i in 0..NUM_PARTITIONS {
                let tuples: Vec<(u32, Vec<BuildTuple>)> = worker_outputs
                    .iter_mut()
                    .zip(&row_bases)
                    .map(|(output, &row_base)| (row_base, std::mem::take(&mut output.tuples[i])))
                    .collect();
                self.injector.push(JoinPartitionJob {
                    tuples,
                    directory: self.directory.clone(),
                    keys: self.keys.clone(),
                    rows: self.rows.clone(),
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
