use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, mpsc};

use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;

use crate::api::OperatorGraphBuilder;
use crate::api::{BuildContext, OperatorFactory};
use crate::memory::MultiSlabBuffer;
use crate::operations::UnaryFactory;
use crate::operations::channels::{ChannelFactory, Sender, StealableChannelFactory};
use crate::operations::unary::UnaryOperator;
use crate::operations::unary::join::build::{
    BuildWorkerOutput, JoinBuildConsumer, JoinPartitionJob, NUM_PARTITIONS,
};
use crate::operations::unary::join::build_rows::BuildRows;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::{JoinCell, JoinKind, JoinSpec, JoinTable, UnmatchedScan};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;

/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory<K: JoinKey, const BUILD_OUTER: bool> {
    spec: Arc<JoinSpec>,
    worker_id: usize,
    hash_state: RandomState,
    partition_sizes: Arc<Vec<AtomicUsize>>,
    table: JoinTable<K::Stored>,
    injector: Arc<Injector<JoinPartitionJob<K::Stored>>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    sender: mpsc::Sender<BuildWorkerOutput<K::Stored>>,
    receiver: Option<mpsc::Receiver<BuildWorkerOutput<K::Stored>>>,
    remaining_jobs: Arc<AtomicUsize>,
}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const SEMI: bool,
    const OUTER_JOIN_PROBE_SIDE: bool,
    const ANTI: bool,
> {
    pub(crate) table: JoinTable<K::Stored>,
    hash_state: RandomState,
    spec: Arc<JoinSpec>,
    unmatched: Arc<UnmatchedScan>,
}

/// Create `worker_count` build factories and probe factories that share the
/// same [`JoinTable`] (directory + key/row arenas + stored build rows) and hash
/// state.
///
/// Returns `(build_factories, probe_factories, build_ready)`. The build
/// outputters publish readiness only after every partition job has run; the
/// graph builder uses that flag to gate every root of the probe input.
///
/// The build phase is the same for every kind, so only the probe factories
/// carry `SEMI`, `OUTER_JOIN_PROBE_SIDE`, and `ANTI`.
pub fn create_for_workers<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const SEMI: bool,
    const OUTER_JOIN_PROBE_SIDE: bool,
    const ANTI: bool,
>(
    spec: JoinSpec,
    worker_count: usize,
) -> (
    impl IntoIterator<Item = JoinBuildFactory<K, BUILD_OUTER>>,
    impl IntoIterator<Item = JoinProbeFactory<K, BUILD_OUTER, SEMI, OUTER_JOIN_PROBE_SIDE, ANTI>>,
    Arc<AtomicBool>,
) {
    debug_assert_eq!(
        BUILD_OUTER,
        matches!(spec.kind, JoinKind::BuildOuter | JoinKind::BuildAnti)
    );
    debug_assert_eq!(
        OUTER_JOIN_PROBE_SIDE,
        matches!(spec.kind, JoinKind::ProbeOuter | JoinKind::ProbeAnti)
    );
    debug_assert_eq!(
        ANTI,
        matches!(spec.kind, JoinKind::ProbeAnti | JoinKind::BuildAnti)
    );
    debug_assert!(
        !SEMI || spec.build_output_indices.is_empty(),
        "a semi join emits no build columns"
    );
    debug_assert!(
        !(ANTI && OUTER_JOIN_PROBE_SIDE) || spec.build_output_indices.is_empty(),
        "a probe-side anti join emits no build columns"
    );
    debug_assert!(
        !(ANTI && BUILD_OUTER) || spec.probe_output_indices.is_empty(),
        "a build-side anti join emits no probe columns"
    );
    let spec = Arc::new(spec);
    let hash_state = RandomState::with_seeds(0, 0, 0, 0);
    let partition_sizes: Arc<Vec<AtomicUsize>> =
        Arc::new((0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect());
    let table = JoinTable {
        directory: Arc::new(JoinCell::new(JoinDirectory::initial())),
        keys: Arc::new(JoinCell::new(MultiSlabBuffer::<K::Stored>::new(vec![]))),
        rows: Arc::new(JoinCell::new(MultiSlabBuffer::<u32>::new(vec![]))),
        build_rows: Arc::new(JoinCell::new(BuildRows::empty())),
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
    let build_spec = spec.clone();

    let build_factories = (0..worker_count).map(move |worker_id| JoinBuildFactory {
        spec: build_spec.clone(),
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

    let unmatched = Arc::new(UnmatchedScan::new(worker_count));
    let probe_factories = (0..worker_count).map(move |_| JoinProbeFactory {
        table: table_clone.clone(),
        hash_state: hs_clone.clone(),
        spec: spec.clone(),
        unmatched: unmatched.clone(),
    });

    (build_factories, probe_factories, probe_gate)
}

impl<K: JoinKey, const BUILD_OUTER: bool> UnaryFactory<RecordBatch, ()>
    for JoinBuildFactory<K, BUILD_OUTER>
{
    type Unary = PipelineBreaker<RecordBatch, (), JoinBuildConsumer<K, BUILD_OUTER>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.spec.build_key_indices.clone(),
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
            self.spec.build_output_indices.clone(),
        ))
    }
}

impl<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const SEMI: bool,
    const OUTER_JOIN_PROBE_SIDE: bool,
    const ANTI: bool,
> UnaryFactory<RecordBatch, RecordBatch>
    for JoinProbeFactory<K, BUILD_OUTER, SEMI, OUTER_JOIN_PROBE_SIDE, ANTI>
{
    type Unary = Probe<K, BUILD_OUTER, SEMI, OUTER_JOIN_PROBE_SIDE, ANTI>;

    fn build_unary(self) -> Probe<K, BUILD_OUTER, SEMI, OUTER_JOIN_PROBE_SIDE, ANTI> {
        Probe::new(self.table, self.hash_state, self.spec, self.unmatched)
    }
}

struct DiscardSender;

impl Sender<()> for DiscardSender {
    fn send(&mut self, _item: ()) -> crate::operations::channels::Result<()> {
        Ok(())
    }
}

/// Builds the probe result path and the disconnected build path into one
/// per-worker operator graph.
///
/// Generic over the build channel so each join flavor picks its delivery: the
/// hash join steals build batches across workers, the range join funnels the
/// sorted chunks to one worker to keep their order.
pub struct JoinRecordBatchOperatorFactory<BF, PF, BC = StealableChannelFactory<RecordBatch>> {
    pub probe_head: Box<dyn OperatorFactory<RecordBatch>>,
    pub build_head: Box<dyn OperatorFactory<RecordBatch>>,
    pub build_factory: BF,
    pub probe_factory: PF,
    pub build_channel_factory: BC,
    pub probe_channel_factory: StealableChannelFactory<RecordBatch>,
    pub build_siblings_left: Arc<AtomicUsize>,
    pub probe_siblings_left: Arc<AtomicUsize>,
    pub build_ready: Arc<AtomicBool>,
}

impl<BF, PF, BC> OperatorFactory<RecordBatch> for JoinRecordBatchOperatorFactory<BF, PF, BC>
where
    BF: UnaryFactory<RecordBatch, ()>,
    PF: UnaryFactory<RecordBatch, RecordBatch>,
    BC: ChannelFactory<RecordBatch>,
{
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<RecordBatch>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let (probe_tx, probe_rx) = self.probe_channel_factory.build();
        let probe_graph = self
            .probe_head
            .build(Box::new(probe_tx), context)
            .gated_by(self.build_ready)
            .with(Box::new(UnaryOperator::new(
                self.probe_factory.build_unary(),
                probe_rx,
                sender,
                self.probe_siblings_left,
            )));

        let (build_tx, build_rx) = self.build_channel_factory.build();
        let build_graph = self
            .build_head
            .build(Box::new(build_tx), context)
            .with(Box::new(UnaryOperator::new(
                self.build_factory.build_unary(),
                build_rx,
                Box::new(DiscardSender),
                self.build_siblings_left,
            )));

        probe_graph.with_side_graph(build_graph)
    }
}
