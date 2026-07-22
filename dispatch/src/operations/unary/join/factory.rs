use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, mpsc};

use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;

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
pub struct JoinBuildFactory {
    key_column: usize,
    worker_id: usize,
    hash_state: RandomState,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    directory: Arc<JoinCell<JoinDirectory>>,
    keys: Arc<JoinCell<MultiSlabBuffer<u64>>>,
    rows: Arc<JoinCell<MultiSlabBuffer<u32>>>,
    build_rows: Arc<JoinCell<Option<RecordBatch>>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    sender: mpsc::Sender<BuildWorkerOutput>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput>>,
    remaining_jobs: Arc<AtomicUsize>,
}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory {
    pub(crate) table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    use_probe_array: bool,
    output_columns: Arc<JoinOutputColumns>,
}

/// Create `worker_count` build factories and probe factories that share the
/// same [`JoinTable`] (directory + key/row arenas + build payload) and hash
/// state.
///
/// Returns `(build_factories, probe_factories, build_ready)`. The build
/// outputters publish readiness only after every partition job has run; the
/// graph builder uses that flag to gate every root of the probe input.
pub fn create_for_workers(
    build_key_column: usize,
    probe_key_column: usize,
    output_columns: JoinOutputColumns,
    worker_count: usize,
) -> (
    impl IntoIterator<Item = JoinBuildFactory>,
    impl IntoIterator<Item = JoinProbeFactory>,
    Arc<AtomicBool>,
) {
    let use_probe_array = crate::env::get_env_var_with_default("PIVOT_JOIN_PROBE_ARRAY", true);
    let hash_state = RandomState::with_seeds(0, 0, 0, 0);
    let partition_sizes: Arc<Vec<AtomicUsize>> =
        Arc::new((0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect());
    let directory = Arc::new(JoinCell::new(JoinDirectory::initial()));
    let keys = Arc::new(JoinCell::new(MultiSlabBuffer::<u64>::new(vec![])));
    let rows = Arc::new(JoinCell::new(MultiSlabBuffer::<u32>::new(vec![])));
    let build_rows = Arc::new(JoinCell::new(None));
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let build_ready = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let (tx, rx) = mpsc::channel();
    let mut rx_opt = Some(rx);

    let dir_clone = directory.clone();
    let keys_clone = keys.clone();
    let rows_clone = rows.clone();
    let build_rows_clone = build_rows.clone();
    let hs_clone = hash_state.clone();
    let probe_gate = build_ready.clone();

    let build_factories = (0..worker_count).map(move |worker_id| JoinBuildFactory {
        key_column: build_key_column,
        worker_id,
        hash_state: hash_state.clone(),
        partition_sizes: partition_sizes.clone(),
        directory: directory.clone(),
        keys: keys.clone(),
        rows: rows.clone(),
        build_rows: build_rows.clone(),
        injector: injector.clone(),
        jobs_injected: jobs_injected.clone(),
        build_ready: build_ready.clone(),
        sender: tx.clone(),
        receiver: rx_opt.take(),
        remaining_jobs: remaining_jobs.clone(),
    });

    let output_columns = Arc::new(output_columns);
    let probe_factories = (0..worker_count).map(move |_| JoinProbeFactory {
        table: JoinTable {
            directory: dir_clone.clone(),
            keys: keys_clone.clone(),
            rows: rows_clone.clone(),
            build_rows: build_rows_clone.clone(),
        },
        hash_state: hs_clone.clone(),
        key_column: probe_key_column,
        use_probe_array,
        output_columns: output_columns.clone(),
    });

    (build_factories, probe_factories, probe_gate)
}

impl UnaryFactory<RecordBatch, ()> for JoinBuildFactory {
    type Unary = PipelineBreaker<RecordBatch, (), JoinBuildConsumer>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_column,
            self.worker_id,
            self.hash_state,
            self.sender,
            self.receiver,
            self.partition_sizes,
            self.directory,
            self.keys,
            self.rows,
            self.build_rows,
            self.injector,
            self.jobs_injected,
            self.build_ready,
            self.remaining_jobs,
        ))
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for JoinProbeFactory {
    type Unary = Probe;

    fn build_unary(self) -> Probe {
        Probe::new(
            self.table,
            self.hash_state,
            self.key_column,
            self.use_probe_array,
            self.output_columns,
        )
    }
}

#[cfg(test)]
impl JoinProbeFactory {
    pub(super) fn with_probe_array(mut self, use_probe_array: bool) -> Self {
        self.use_probe_array = use_probe_array;
        self
    }
}
