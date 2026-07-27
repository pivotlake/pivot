use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, mpsc};

use ahash::RandomState;
use arrow_array::RecordBatch;
use arrow_array::types::ArrowPrimitiveType;
use crossbeam_deque::Injector;
use std::hash::Hash;

use crate::memory::MultiSlabBuffer;
use crate::operations::UnaryFactory;
use crate::operations::unary::join::build::{
    BuildWorkerOutput, JoinBuildConsumer, JoinPartitionJob, NUM_PARTITIONS,
};
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::{JoinCell, JoinOutputColumns, JoinTable};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;

/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory<T: ArrowPrimitiveType<Native: Hash + Eq>> {
    key_column: usize,
    worker_id: usize,
    hash_state: RandomState,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    table: JoinTable<T::Native>,
    injector: Arc<Injector<JoinPartitionJob<T::Native>>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    sender: mpsc::Sender<BuildWorkerOutput<T::Native>>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput<T::Native>>>,
    remaining_jobs: Arc<AtomicUsize>,
}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory<T: ArrowPrimitiveType<Native: Hash + Eq>> {
    pub(crate) table: JoinTable<T::Native>,
    hash_state: RandomState,
    key_column: usize,
    output_columns: Arc<JoinOutputColumns>,
}

/// Create `worker_count` build factories and probe factories that share the
/// same [`JoinTable`] (directory + key/row arenas + build payload) and hash
/// state.
///
/// Returns `(build_factories, probe_factories, build_ready)`. The build
/// outputters publish readiness only after every partition job has run; the
/// graph builder uses that flag to gate every root of the probe input.
pub fn create_for_workers<T: ArrowPrimitiveType<Native: Hash + Eq>>(
    build_key_column: usize,
    probe_key_column: usize,
    output_columns: JoinOutputColumns,
    worker_count: usize,
) -> (
    impl IntoIterator<Item = JoinBuildFactory<T>>,
    impl IntoIterator<Item = JoinProbeFactory<T>>,
    Arc<AtomicBool>,
) {
    let hash_state = RandomState::with_seeds(0, 0, 0, 0);
    let partition_sizes: Arc<Vec<AtomicUsize>> =
        Arc::new((0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect());
    let table = JoinTable {
        directory: Arc::new(JoinCell::new(JoinDirectory::initial())),
        keys: Arc::new(JoinCell::new(MultiSlabBuffer::<T::Native>::new(vec![]))),
        rows: Arc::new(JoinCell::new(MultiSlabBuffer::<u32>::new(vec![]))),
        build_rows: Arc::new(JoinCell::new(None)),
    };
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let build_ready = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let (tx, rx) = mpsc::channel();
    let mut rx_opt = Some(rx);

    let table_clone = table.clone();
    let hs_clone = hash_state.clone();
    let probe_gate = build_ready.clone();

    let build_factories = (0..worker_count).map(move |worker_id| JoinBuildFactory {
        key_column: build_key_column,
        worker_id,
        hash_state: hash_state.clone(),
        partition_sizes: partition_sizes.clone(),
        table: table.clone(),
        injector: injector.clone(),
        jobs_injected: jobs_injected.clone(),
        build_ready: build_ready.clone(),
        sender: tx.clone(),
        receiver: rx_opt.take(),
        remaining_jobs: remaining_jobs.clone(),
    });

    let output_columns = Arc::new(output_columns);
    let probe_factories = (0..worker_count).map(move |_| JoinProbeFactory {
        table: table_clone.clone(),
        hash_state: hs_clone.clone(),
        key_column: probe_key_column,
        output_columns: output_columns.clone(),
    });

    (build_factories, probe_factories, probe_gate)
}

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> UnaryFactory<RecordBatch, ()>
    for JoinBuildFactory<T>
{
    type Unary = PipelineBreaker<RecordBatch, (), JoinBuildConsumer<T>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_column,
            self.worker_id,
            self.hash_state,
            self.sender,
            self.receiver,
            self.partition_sizes,
            self.table,
            self.injector,
            self.jobs_injected,
            self.build_ready,
            self.remaining_jobs,
        ))
    }
}

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> UnaryFactory<RecordBatch, RecordBatch>
    for JoinProbeFactory<T>
{
    type Unary = Probe<T>;

    fn build_unary(self) -> Probe<T> {
        Probe::new(
            self.table,
            self.hash_state,
            self.key_column,
            self.output_columns,
        )
    }
}
