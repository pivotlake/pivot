use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{mpsc, Arc};

use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;

use crate::memory::{MultiSlabBuffer, SlabVec};
use crate::operations::UnaryFactory;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::{JoinTable, Value};
use crate::operations::unary::join::build_old::{JoinBuildConsumer, JoinPartitionJob, NUM_PARTITIONS};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;


/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory {
    key_column: usize,
    hash_state: RandomState,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    directory: Arc<UnsafeCell<JoinDirectory>>,
    arena: Arc<UnsafeCell<MultiSlabBuffer<Value>>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
    sender: mpsc::Sender<Vec<SlabVec<(u64, Value)>>>,
    receiver: Option<mpsc::Receiver<Vec<SlabVec<(u64, Value)>>>>,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

unsafe impl Send for JoinBuildFactory {}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory {
    pub(crate) table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    shared_total: Arc<AtomicUsize>
}

unsafe impl Send for JoinProbeFactory {}

/// Create `worker_count` build factories and probe factories that share the
/// same [`JoinTable`] (directory + arena) and hash state.
///
/// Returns `(build_factories, probe_factories, gate)`. The gate is an
/// [`AtomicBool`] that starts `false` and is set to `true` by the last
/// [`JoinPartitionJob`] to complete. Use it with [`GatedOperator`](crate::operations::GatedOperator)
/// to hold the probe pipeline until the build finishes.
pub fn create_for_workers(
    build_key_column: usize,
    probe_key_column: usize,
    worker_count: usize,
) -> (
    impl IntoIterator<Item = JoinBuildFactory>,
    impl IntoIterator<Item = JoinProbeFactory>,
    Arc<AtomicBool>,
) {
    let hash_state = RandomState::with_seeds(0, 0, 0, 0);
    let shared_total = Arc::new(AtomicUsize::new(0));
    let partition_sizes: Arc<Vec<AtomicUsize>> =
        Arc::new((0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect());
    let directory = Arc::new(UnsafeCell::new(JoinDirectory::initial()));
    let arena: Arc<UnsafeCell<MultiSlabBuffer<Value>>> = Arc::new(UnsafeCell::new(MultiSlabBuffer::new(vec![])));
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let (tx, rx) = mpsc::channel();
    let mut rx_opt = Some(rx);

    let dir_clone = directory.clone();
    let arena_clone = arena.clone();
    let hs_clone = hash_state.clone();
    let gate_ret = gate.clone();

    let build_factories = (0..worker_count).map(move |_| JoinBuildFactory {
        key_column: build_key_column,
        hash_state: hash_state.clone(),
        partition_sizes: partition_sizes.clone(),
        directory: directory.clone(),
        arena: arena.clone(),
        injector: injector.clone(),
        jobs_injected: jobs_injected.clone(),
        sender: tx.clone(),
        receiver: rx_opt.take(),
        gate: gate.clone(),
        remaining_jobs: remaining_jobs.clone(),
    });

    let probe_factories = (0..worker_count).map(move |_| JoinProbeFactory {
        shared_total: shared_total.clone(),
        table: JoinTable {
            directory: dir_clone.clone(),
            arena: arena_clone.clone(),
        },
        hash_state: hs_clone.clone(),
        key_column: probe_key_column,
    });

    (build_factories, probe_factories, gate_ret)
}

impl UnaryFactory<RecordBatch, ()> for JoinBuildFactory {
    type Unary = PipelineBreaker<RecordBatch, (), JoinBuildConsumer>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_column,
            self.hash_state,
            self.sender,
            self.receiver,
            self.partition_sizes,
            self.directory,
            self.arena,
            self.injector,
            self.jobs_injected,
            self.gate,
            self.remaining_jobs,
        ))
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for JoinProbeFactory {
    type Unary = Probe;

    fn build_unary(self) -> Probe {
        Probe::new(self.table, self.hash_state, self.key_column, self.shared_total.clone())
    }
}
