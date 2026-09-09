use std::marker::PhantomData;
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
use crate::operations::unary::join::build::{
    BuildWorkerOutput, JoinBuildConsumer, JoinBuilder, NUM_PARTITIONS,
};
use crate::operations::unary::join::build_rows::BuildRows;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::{JoinCell, JoinKind, JoinSpec, JoinTable, UnmatchedScan};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use crate::operations::unary::{BatchesOutputter, CollectorFactory, Normalizer, UnaryOperator};

/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory<K: JoinKey, const BUILD_OUTER: bool, O> {
    key_columns: Vec<usize>,
    hash_state: RandomState,
    outputter: O,
    _key: PhantomData<fn() -> K>,
}

type DirectJoinBuildFactory<K, const BUILD_OUTER: bool> =
    JoinBuildFactory<K, BUILD_OUTER, JoinBuilder<<K as JoinKey>::Stored, BUILD_OUTER>>;
type NormalizingJoinBuildFactory<K, const BUILD_OUTER: bool> =
    JoinBuildFactory<K, BUILD_OUTER, Normalizer<BuildWorkerOutput<<K as JoinKey>::Stored>>>;
type JoinBuildCollector<K, const BUILD_OUTER: bool> = CollectorFactory<
    BuildWorkerOutput<<K as JoinKey>::Stored>,
    JoinBuilder<<K as JoinKey>::Stored, BUILD_OUTER>,
>;
type JoinProbeFactories<
    K,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> = Vec<
    JoinProbeFactory<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >,
>;
type DirectJoinFactories<
    K,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> = (
    Vec<DirectJoinBuildFactory<K, BUILD_OUTER>>,
    JoinProbeFactories<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >,
    Arc<AtomicBool>,
);
type NormalizingJoinFactories<
    K,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> = (
    Vec<NormalizingJoinBuildFactory<K, BUILD_OUTER>>,
    Vec<JoinBuildCollector<K, BUILD_OUTER>>,
    JoinProbeFactories<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >,
    Arc<AtomicBool>,
);

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
) -> DirectJoinFactories<
    K,
    BUILD_OUTER,
    STOP_AFTER_FIRST_MATCH,
    TRACK_UNMATCHED_PROBE_ROWS,
    DISCARD_MATCHED_PAIRS,
    MARK,
> {
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
        rows: Arc::new(JoinCell::new(MultiSlabBuffer::<u64>::new(vec![]))),
        build_rows: Arc::new(JoinCell::new(BuildRows::empty())),
        build_saw_null_key: Arc::new(JoinCell::new(false)),
    };
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let build_ready = Arc::new(AtomicBool::new(false));
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS));
    let gather = Arc::new(GatherBarrier::new(worker_count));

    let table_clone = table.clone();
    let hs_clone = hash_state.clone();
    let probe_gate = build_ready.clone();
    let build_factories = (0..worker_count)
        .map(|_| {
            let outputter = JoinBuilder::new(
                table.clone(),
                injector.clone(),
                jobs_injected.clone(),
                build_ready.clone(),
                remaining_jobs.clone(),
                gather.clone(),
                spec.build_output_indices.clone(),
            );
            JoinBuildFactory {
                key_columns: spec.build_key_indices.clone(),
                hash_state: hash_state.clone(),
                outputter,
                _key: PhantomData,
            }
        })
        .collect::<Vec<_>>();

    let unmatched = Arc::new(UnmatchedScan::new(worker_count));
    let probe_factories = (0..worker_count)
        .map(move |_| JoinProbeFactory {
            table: table_clone.clone(),
            hash_state: hs_clone.clone(),
            spec: spec.clone(),
            unmatched: unmatched.clone(),
        })
        .collect();

    (build_factories, probe_factories, probe_gate)
}

/// The two-breaker build path used when the build schema contains variants.
/// The first breaker pairs [`JoinBuildConsumer`] with [`Normalizer`]; the
/// second funnels the normalized groups to one [`CollectorFactory`] whose
/// outputter is the ordinary [`JoinBuilder`].
pub(crate) fn create_normalizing_for_workers<
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
>(
    spec: JoinSpec,
    worker_count: usize,
    collector_worker: usize,
) -> NormalizingJoinFactories<
    K,
    BUILD_OUTER,
    STOP_AFTER_FIRST_MATCH,
    TRACK_UNMATCHED_PROBE_ROWS,
    DISCARD_MATCHED_PAIRS,
    MARK,
> {
    let (build_factories, probe_factories, build_ready) = create_for_workers::<
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >(spec, worker_count);
    let normalizers = Normalizer::create_for_workers(worker_count);
    let mut normalized_builds = Vec::with_capacity(worker_count);
    let mut collectors = Vec::with_capacity(worker_count);
    for (worker, (build, normalizer)) in build_factories.into_iter().zip(normalizers).enumerate() {
        let JoinBuildFactory {
            key_columns,
            hash_state,
            outputter,
            _key,
        } = build;
        normalized_builds.push(JoinBuildFactory {
            key_columns,
            hash_state,
            outputter: normalizer,
            _key,
        });
        collectors.push(CollectorFactory::new(outputter, worker == collector_worker));
    }
    (normalized_builds, collectors, probe_factories, build_ready)
}

impl<K: JoinKey, const BUILD_OUTER: bool, O, Out> UnaryFactory<RecordBatch, Out>
    for JoinBuildFactory<K, BUILD_OUTER, O>
where
    O: BatchesOutputter<BuildWorkerOutput<K::Stored>, Out> + Send + 'static,
    Out: 'static,
{
    type Unary = PipelineBreaker<RecordBatch, Out, JoinBuildConsumer<K, BUILD_OUTER, O>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_columns,
            self.hash_state,
            self.outputter,
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
/// The build side is already assembled as an erased operator graph factory:
/// hash joins choose either their direct breaker or the two-breaker normalized
/// composition, while range joins retain their ordered single-worker build.
pub struct JoinRecordBatchOperatorFactory<PF> {
    pub probe_head: Box<dyn OperatorFactory<RecordBatch>>,
    pub build_graph: Box<dyn OperatorFactory<()>>,
    pub probe_factory: PF,
    pub probe_channel_factory: StealableChannelFactory<RecordBatch>,
    pub probe_siblings_left: Arc<AtomicUsize>,
    pub build_ready: Arc<AtomicBool>,
}

impl<PF> OperatorFactory<RecordBatch> for JoinRecordBatchOperatorFactory<PF>
where
    PF: UnaryFactory<RecordBatch, RecordBatch>,
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

        let build_graph = self.build_graph.build(Box::new(DiscardSender), context);

        probe_graph.with_side_graph(build_graph)
    }
}
