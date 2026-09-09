use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, OnceLock};

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
    BuildWorkerOutput, JoinBuildConsumer, JoinBuilder, NUM_PARTITIONS, PendingKeyBitsets,
    WorkerFilterArrays,
};
use crate::operations::unary::join::build_rows::BuildRows;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::probe::Probe;
use crate::operations::unary::join::row_arena::RowArena;
use crate::operations::unary::join::{JoinCell, JoinKind, JoinSpec, JoinTable, UnmatchedScan};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use crate::operations::unary::{BatchesOutputter, CollectorFactory, Normalizer, UnaryOperator};

/// Creates one [`JoinBuildConsumer`] per worker, with shared state wired up.
pub struct JoinBuildFactory<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool, O> {
    key_columns: Vec<usize>,
    filter_columns: Vec<usize>,
    hash_state: RandomState,
    outputter: O,
    /// The cell the consumer fills with its filter key columns at seal time
    /// and this worker's [`JoinBuilder`] later reads to set its share of the
    /// key bitsets. Shared between the two directly, whichever outputter
    /// sits between them, since a normalizer ships the group itself to the
    /// collector worker.
    filter_arrays: WorkerFilterArrays,
    _key: PhantomData<fn() -> K>,
}

type DirectJoinBuildFactory<K, const TRACK_MATCHED_BUILD_ROWS: bool> = JoinBuildFactory<
    K,
    TRACK_MATCHED_BUILD_ROWS,
    JoinBuilder<<K as JoinKey>::Stored, TRACK_MATCHED_BUILD_ROWS>,
>;
type NormalizingJoinBuildFactory<K, const TRACK_MATCHED_BUILD_ROWS: bool> = JoinBuildFactory<
    K,
    TRACK_MATCHED_BUILD_ROWS,
    Normalizer<BuildWorkerOutput<<K as JoinKey>::Stored>>,
>;
type JoinBuildCollector<K, const TRACK_MATCHED_BUILD_ROWS: bool> = CollectorFactory<
    BuildWorkerOutput<<K as JoinKey>::Stored>,
    JoinBuilder<<K as JoinKey>::Stored, TRACK_MATCHED_BUILD_ROWS>,
>;
type JoinProbeFactories<
    K,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
> = Vec<
    JoinProbeFactory<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >,
>;
type DirectJoinFactories<
    K,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
> = (
    Vec<DirectJoinBuildFactory<K, TRACK_MATCHED_BUILD_ROWS>>,
    JoinProbeFactories<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >,
    Arc<AtomicBool>,
);
type NormalizingJoinFactories<
    K,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
> = (
    Vec<NormalizingJoinBuildFactory<K, TRACK_MATCHED_BUILD_ROWS>>,
    Vec<JoinBuildCollector<K, TRACK_MATCHED_BUILD_ROWS>>,
    JoinProbeFactories<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >,
    Arc<AtomicBool>,
);

/// Creates one [`Probe`] per worker, all sharing the same [`JoinTable`].
pub struct JoinProbeFactory<
    K: JoinKey,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
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
/// `DISCARD_MATCHED_ROWS`, and `EMIT_MARK_COLUMN`.
pub fn create_for_workers<
    K: JoinKey,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
>(
    spec: JoinSpec,
    worker_count: usize,
) -> DirectJoinFactories<
    K,
    TRACK_MATCHED_BUILD_ROWS,
    STOP_AFTER_FIRST_MATCH,
    TRACK_UNMATCHED_PROBE_ROWS,
    DISCARD_MATCHED_ROWS,
    EMIT_MARK_COLUMN,
> {
    debug_assert_eq!(
        TRACK_MATCHED_BUILD_ROWS,
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
        DISCARD_MATCHED_ROWS,
        matches!(
            spec.kind,
            JoinKind::ProbeAnti | JoinKind::BuildAnti | JoinKind::BuildSemi
        )
    );
    debug_assert_eq!(EMIT_MARK_COLUMN, matches!(spec.kind, JoinKind::ProbeMark));
    debug_assert!(
        !STOP_AFTER_FIRST_MATCH || spec.build_output_indices.is_empty(),
        "the first-match path does not record build rows"
    );
    debug_assert!(
        !(DISCARD_MATCHED_ROWS && TRACK_UNMATCHED_PROBE_ROWS)
            || spec.build_output_indices.is_empty(),
        "a probe-side anti join emits no build columns"
    );
    debug_assert!(
        !(DISCARD_MATCHED_ROWS && TRACK_MATCHED_BUILD_ROWS) || spec.probe_output_indices.is_empty(),
        "a build-side semi or anti join emits no probe columns"
    );
    debug_assert!(
        !EMIT_MARK_COLUMN
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
        rows: Arc::new(JoinCell::new(RowArena::empty())),
        build_rows: Arc::new(JoinCell::new(BuildRows::empty())),
        build_saw_null_key: Arc::new(JoinCell::new(false)),
    };
    let injector = Arc::new(Injector::new());
    let jobs_injected = Arc::new(AtomicBool::new(false));
    let build_ready = Arc::new(AtomicBool::new(false));
    // One partition scatter job each, plus one key bitset pass per worker.
    let remaining_jobs = Arc::new(AtomicUsize::new(NUM_PARTITIONS + worker_count));
    let gather = Arc::new(GatherBarrier::new(worker_count));
    let pending_key_bitsets: PendingKeyBitsets = Arc::new(OnceLock::new());

    let table_clone = table.clone();
    let hs_clone = hash_state.clone();
    let probe_gate = build_ready.clone();
    let build_factories = (0..worker_count)
        .map(|_| {
            let filter_arrays: WorkerFilterArrays = Arc::new(OnceLock::new());
            let outputter = JoinBuilder::new(
                table.clone(),
                injector.clone(),
                jobs_injected.clone(),
                build_ready.clone(),
                remaining_jobs.clone(),
                gather.clone(),
                spec.build_output_indices.clone(),
                spec.build_filters.clone(),
                filter_arrays.clone(),
                pending_key_bitsets.clone(),
            );
            JoinBuildFactory {
                key_columns: spec.build_key_indices.clone(),
                filter_columns: spec
                    .build_filters
                    .iter()
                    .map(|filter| filter.build_column)
                    .collect(),
                hash_state: hash_state.clone(),
                outputter,
                filter_arrays,
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
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
>(
    spec: JoinSpec,
    worker_count: usize,
    collector_worker: usize,
) -> NormalizingJoinFactories<
    K,
    TRACK_MATCHED_BUILD_ROWS,
    STOP_AFTER_FIRST_MATCH,
    TRACK_UNMATCHED_PROBE_ROWS,
    DISCARD_MATCHED_ROWS,
    EMIT_MARK_COLUMN,
> {
    let (build_factories, probe_factories, build_ready) = create_for_workers::<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >(spec, worker_count);
    let normalizers = Normalizer::create_for_workers(worker_count);
    let mut normalized_builds = Vec::with_capacity(worker_count);
    let mut collectors = Vec::with_capacity(worker_count);
    for (worker, (build, normalizer)) in build_factories.into_iter().zip(normalizers).enumerate() {
        let JoinBuildFactory {
            key_columns,
            filter_columns,
            hash_state,
            outputter,
            filter_arrays,
            _key,
        } = build;
        normalized_builds.push(JoinBuildFactory {
            key_columns,
            filter_columns,
            hash_state,
            outputter: normalizer,
            filter_arrays,
            _key,
        });
        collectors.push(CollectorFactory::new(outputter, worker == collector_worker));
    }
    (normalized_builds, collectors, probe_factories, build_ready)
}

impl<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool, O, Out> UnaryFactory<RecordBatch, Out>
    for JoinBuildFactory<K, TRACK_MATCHED_BUILD_ROWS, O>
where
    O: BatchesOutputter<BuildWorkerOutput<K::Stored>, Out> + Send + 'static,
    Out: 'static,
{
    type Unary =
        PipelineBreaker<RecordBatch, Out, JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS, O>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(JoinBuildConsumer::new(
            self.key_columns,
            self.filter_columns,
            self.hash_state,
            self.outputter,
            self.filter_arrays,
        ))
    }
}

impl<
    K: JoinKey,
    const TRACK_MATCHED_BUILD_ROWS: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_ROWS: bool,
    const EMIT_MARK_COLUMN: bool,
> UnaryFactory<RecordBatch, RecordBatch>
    for JoinProbeFactory<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >
{
    type Unary = Probe<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
    >;

    fn build_unary(
        self,
    ) -> Probe<
        K,
        TRACK_MATCHED_BUILD_ROWS,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_ROWS,
        EMIT_MARK_COLUMN,
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
