use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, mpsc};

use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;

use crate::operations::UnaryFactory;
use crate::operations::unary::join::build::{
    BuildWorkerOutput, JoinBuildConsumer, JoinPartitionJob, NUM_PARTITIONS,
};
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::{JoinArena, JoinCell, JoinMode, JoinTable};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;

/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory {
    key_columns: Vec<usize>,
    null_safe: Vec<bool>,
    worker_id: usize,
    hash_state: RandomState,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    directory: Arc<JoinCell<JoinDirectory>>,
    keys: Arc<JoinCell<JoinArena<u64>>>,
    rows: Arc<JoinCell<JoinArena<u32>>>,
    build_rows: Arc<JoinCell<Option<RecordBatch>>>,
    matched: Arc<JoinCell<Vec<std::sync::atomic::AtomicBool>>>,
    injector: Arc<Injector<JoinPartitionJob>>,
    jobs_injected: Arc<AtomicBool>,
    sender: mpsc::Sender<BuildWorkerOutput>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput>>,
    gate: Arc<AtomicBool>,
    remaining_jobs: Arc<AtomicUsize>,
}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory {
    pub(crate) table: JoinTable,
    hash_state: RandomState,
    key_columns: Vec<usize>,
    build_key_columns: Vec<usize>,
    null_safe: Vec<bool>,
    mode: JoinMode,
    probe_types: Vec<arrow_schema::DataType>,
    use_probe_array: bool,
}

/// Create `worker_count` build factories and probe factories that share the
/// same [`JoinTable`] (directory + key/row arenas + build payload) and hash
/// state.
///
/// Returns `(build_factories, probe_factories, gate)`. The gate is an
/// [`AtomicBool`] that starts `false` and is set to `true` by the last
/// [`JoinPartitionJob`] to complete, signalling that the build hash table is
/// fully populated and the probe side may run.
pub fn create_for_workers(
    build_key_columns: Vec<usize>,
    probe_key_columns: Vec<usize>,
    null_safe: Vec<bool>,
    mode: JoinMode,
    probe_types: Vec<arrow_schema::DataType>,
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
    let keys = Arc::new(JoinCell::new(JoinArena::<u64>::empty()));
    let rows = Arc::new(JoinCell::new(JoinArena::<u32>::empty()));
    let build_rows = Arc::new(JoinCell::new(None));
    let matched = Arc::new(JoinCell::new(Vec::new()));
    let probes_remaining = Arc::new(AtomicUsize::new(worker_count));
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let (tx, rx) = mpsc::channel();
    let mut rx_opt = Some(rx);

    let build_keys_probe = build_key_columns.clone();
    let dir_clone = directory.clone();
    let keys_clone = keys.clone();
    let rows_clone = rows.clone();
    let build_rows_clone = build_rows.clone();
    let matched_clone = matched.clone();
    let hs_clone = hash_state.clone();
    let gate_ret = gate.clone();
    let gate_probe = gate.clone();

    let null_safe_probe = null_safe.clone();
    let build_factories = (0..worker_count).map(move |worker_id| JoinBuildFactory {
        key_columns: build_key_columns.clone(),
        null_safe: null_safe.clone(),
        worker_id,
        hash_state: hash_state.clone(),
        partition_sizes: partition_sizes.clone(),
        directory: directory.clone(),
        keys: keys.clone(),
        rows: rows.clone(),
        build_rows: build_rows.clone(),
        matched: matched.clone(),
        injector: injector.clone(),
        jobs_injected: jobs_injected.clone(),
        sender: tx.clone(),
        receiver: rx_opt.take(),
        gate: gate.clone(),
        remaining_jobs: remaining_jobs.clone(),
    });

    let probe_factories = (0..worker_count).map(move |_| JoinProbeFactory {
        table: JoinTable {
            directory: dir_clone.clone(),
            keys: keys_clone.clone(),
            rows: rows_clone.clone(),
            build_rows: build_rows_clone.clone(),
            matched: matched_clone.clone(),
            probes_remaining: probes_remaining.clone(),
            gate: gate_probe.clone(),
        },
        hash_state: hs_clone.clone(),
        key_columns: probe_key_columns.clone(),
        build_key_columns: build_keys_probe.clone(),
        null_safe: null_safe_probe.clone(),
        mode,
        probe_types: probe_types.clone(),
        use_probe_array,
    });

    (build_factories, probe_factories, gate_ret)
}

impl UnaryFactory<RecordBatch, ()> for JoinBuildFactory {
    type Unary = PipelineBreaker<RecordBatch, (), JoinBuildConsumer>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_columns,
            self.null_safe,
            self.worker_id,
            self.hash_state,
            self.sender,
            self.receiver,
            self.partition_sizes,
            self.directory,
            self.keys,
            self.rows,
            self.build_rows,
            self.matched,
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
        Probe::new(
            self.table,
            self.hash_state,
            self.key_columns,
            self.build_key_columns,
            self.null_safe,
            self.mode,
            self.probe_types,
            self.use_probe_array,
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
