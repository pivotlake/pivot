use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;

use crate::GatherBarrier;
use crate::api::OperatorGraphBuilder;
use crate::api::{BuildContext, OperatorFactory};
use crate::memory::MultiSlabBuffer;
use crate::operations::UnaryFactory;
use crate::operations::channels::{ChannelFactory, Sender, StealableChannelFactory};
use crate::operations::unary::UnaryOperator;
use crate::operations::unary::join::build::{
    BuildWorkerOutput, JoinBuildConsumer, JoinBuildJob, NUM_PARTITIONS,
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
    hash_state: RandomState,
    table: JoinTable<K::Stored>,
    injector: Arc<Injector<JoinBuildJob<K::Stored>>>,
    jobs_injected: Arc<AtomicBool>,
    build_ready: Arc<AtomicBool>,
    gather: Arc<GatherBarrier<BuildWorkerOutput<K::Stored>>>,
    remaining_jobs: Arc<AtomicUsize>,
    finish_claimed: Arc<AtomicBool>,
}

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
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
/// carry `STOP_AFTER_FIRST_MATCH`, `TRACK_UNMATCHED_PROBE_ROWS`,
/// `DISCARD_MATCHED_PAIRS`, and `MARK`.
pub fn create_for_workers<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
>(
    spec: JoinSpec,
    worker_count: usize,
) -> (
    impl IntoIterator<Item = JoinBuildFactory<K, BUILD_OUTER>>,
    impl IntoIterator<
        Item = JoinProbeFactory<
            K,
            BUILD_OUTER,
            STOP_AFTER_FIRST_MATCH,
            TRACK_UNMATCHED_PROBE_ROWS,
            DISCARD_MATCHED_PAIRS,
            MARK,
        >,
    >,
    Arc<AtomicBool>,
) {
    debug_assert_eq!(
        BUILD_OUTER,
        matches!(
            spec.kind,
            JoinKind::BuildOuter | JoinKind::BuildAnti | JoinKind::BuildSemi
        )
    );
    debug_assert_eq!(
        TRACK_UNMATCHED_PROBE_ROWS,
        matches!(
            spec.kind,
            JoinKind::ProbeOuter | JoinKind::ProbeAnti | JoinKind::ProbeMark
        )
    );
    debug_assert_eq!(
        DISCARD_MATCHED_PAIRS,
        matches!(
            spec.kind,
            JoinKind::ProbeAnti | JoinKind::BuildAnti | JoinKind::BuildSemi
        )
    );
    debug_assert_eq!(MARK, matches!(spec.kind, JoinKind::ProbeMark));
    debug_assert!(
        !STOP_AFTER_FIRST_MATCH || spec.build_output_indices.is_empty(),
        "the first-match path does not record build rows"
    );
    debug_assert!(
        !(DISCARD_MATCHED_PAIRS && TRACK_UNMATCHED_PROBE_ROWS)
            || spec.build_output_indices.is_empty(),
        "a probe-side anti join emits no build columns"
    );
    debug_assert!(
        !(DISCARD_MATCHED_PAIRS && BUILD_OUTER) || spec.probe_output_indices.is_empty(),
        "a build-side semi or anti join emits no probe columns"
    );
    debug_assert!(
        !MARK
            || (spec.build_output_indices.is_empty()
                && spec.residual_filters.is_none()
                && spec.probe_key_indices.len() == 1),
        "a mark join emits no build columns and marks on exactly one key, with no residual"
    );
    let spec = Arc::new(spec);
    let hash_state = RandomState::with_seeds(0, 0, 0, 0);
    let table = JoinTable {
        directory: Arc::new(JoinCell::new(JoinDirectory::initial())),
        keys: Arc::new(JoinCell::new(MultiSlabBuffer::<K::Stored>::new(vec![]))),
        rows: Arc::new(JoinCell::new(MultiSlabBuffer::<u32>::new(vec![]))),
        build_rows: Arc::new(JoinCell::new(BuildRows::empty())),
        build_saw_null_key: Arc::new(JoinCell::new(false)),
    };
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let build_ready = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let finish_claimed = Arc::new(AtomicBool::new(false));
    let gather = Arc::new(GatherBarrier::new(worker_count));

    let table_clone = table.clone();
    let hs_clone = hash_state.clone();
    let probe_gate = build_ready.clone();
    let build_spec = spec.clone();

    let build_factories = (0..worker_count).map(move |_| JoinBuildFactory {
        spec: build_spec.clone(),
        hash_state: hash_state.clone(),
        table: table.clone(),
        injector: injector.clone(),
        jobs_injected: jobs_injected.clone(),
        build_ready: build_ready.clone(),
        gather: gather.clone(),
        remaining_jobs: remaining_jobs.clone(),
        finish_claimed: finish_claimed.clone(),
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
            self.hash_state,
            self.gather,
            self.table,
            self.injector,
            self.jobs_injected,
            self.build_ready,
            self.remaining_jobs,
            self.finish_claimed,
            self.spec.build_output_indices.clone(),
        ))
    }
}

impl<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> UnaryFactory<RecordBatch, RecordBatch>
    for JoinProbeFactory<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >
{
    type Unary = Probe<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >;

    fn build_unary(
        self,
    ) -> Probe<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    > {
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
