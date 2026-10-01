//! Builders that assemble the Parquet read pipeline into a dispatch dataflow.
//!
//! These were inherent methods on dispatch's `OperatorSpec` / `RecordBatchOperatorSpec`
//! before the Parquet pipeline moved into `catalog`; they're now free functions
//! built on dispatch's public operator toolkit (`OperatorSpec::chain`,
//! `RootUnaryOperatorFactory`, the channel factories). The stages are: row-group
//! injection + fetching (source) → index → decompress → decode.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::RowGroupFilter;
use crate::reading::empty_projection_scan::empty_projection_scan;
use arrow_array::RecordBatch;
use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection, RECORD_BATCH_SIZE,
    RecordBatchOperatorSpec, RootUnaryOperatorFactory, UnaryOperatorFactory, return_to_worker_mpsc,
    stealable, stealable_fifo, to_single_worker_mpsc,
};

use crate::reading::decode_gate::{DecodeGate, DecodeGateFactory};
use crate::{
    CompressedPage, DecodeRange, DecoderFactory, DecompressedPage, DecompressorFactory,
    IndexerFactory, MaterializerFactory, ParquetTable, RangeCutterFactory, RowGroupBuffer,
    RowGroupFetcherFactory, RowGroupInjectorFactory, RowGroupRequest, ScanEqualityPredicate,
    ScanOrder, WorkerAllocator, pending_claim_bound,
};

/// Append the index → decompress → cut → decode stages onto a source of
/// [`RowGroupBuffer`]s, producing decoded `RecordBatch`es.
pub(crate) fn read_parquet<OF>(
    input: OperatorSpec<RowGroupBuffer, OF>,
    projection: Projection,
    batch_size: usize,
    add_row_group_metadata: bool,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    pending_row_groups: Vec<Arc<AtomicUsize>>,
    outstanding_row_groups: Arc<AtomicUsize>,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<RowGroupBuffer> + Send + 'static,
{
    let n = input.dispatcher().worker_count();
    let topology = input.dispatcher().topology();
    // Two allocators per worker: one for the dictionaries its range cutter
    // builds, which live as long as their row group, and one for the batches
    // its decoder emits, which usually die before the next one is built. Kept
    // apart, a dictionary never pins the buffer the batches cycle through, so
    // the decoder keeps rewriting the same cache-resident bytes.
    let dictionary_allocators: Vec<Arc<WorkerAllocator>> =
        (0..n).map(|_| Arc::new(WorkerAllocator::new())).collect();
    let batch_allocators: Vec<Arc<WorkerAllocator>> =
        (0..n).map(|_| Arc::new(WorkerAllocator::new())).collect();
    let decoded = input
        .chain(
            stealable::<RowGroupBuffer>(topology).into_iter().collect(),
            (0..n).map(|_| IndexerFactory::new()).collect(),
        )
        .chain(
            stealable::<CompressedPage>(topology).into_iter().collect(),
            (0..n).map(|_| DecompressorFactory::new()).collect(),
        )
        .chain(
            return_to_worker_mpsc::<DecompressedPage>(n)
                .into_iter()
                .collect(),
            pending_row_groups
                .into_iter()
                .zip(dictionary_allocators)
                .map(|(pending, allocator)| RangeCutterFactory {
                    projection: projection.clone(),
                    eq_predicates: eq_predicates.clone(),
                    pending_row_groups: pending,
                    outstanding_row_groups: outstanding_row_groups.clone(),
                    allocator,
                })
                .collect(),
        )
        // A worker takes its own row groups' ranges oldest first, so it
        // follows each row group as one stream; a peer with nothing to do
        // takes the oldest range waiting on another worker.
        .chain(
            stealable_fifo::<DecodeRange>(topology)
                .into_iter()
                .collect(),
            batch_allocators
                .into_iter()
                .map(|allocator| DecoderFactory {
                    batch_size,
                    add_row_group_metadata,
                    allocator,
                })
                .collect(),
        );
    RecordBatchOperatorSpec::from_spec(decoded)
}

/// Fetch and decode the row groups that `requests` name, each read as the
/// request selects. The decoded batches carry the metadata columns when
/// `add_row_group_metadata`, so a consumer can tell which row group and rows
/// each holds.
pub fn fetch_row_groups<OF>(
    requests: OperatorSpec<RowGroupRequest, OF>,
    table: &ParquetTable,
    projection: Projection,
    add_row_group_metadata: bool,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<RowGroupRequest> + 'static,
{
    let n = requests.dispatcher().worker_count();
    let pending_row_groups = pending_row_group_counters(n);
    let buffers = fetch_buffers(requests, &pending_row_groups, pending_claim_bound(table));
    read_parquet(
        buffers,
        projection,
        RECORD_BATCH_SIZE,
        add_row_group_metadata,
        Arc::new(Vec::new()),
        pending_row_groups,
        // Row groups are claimed by explicit request, not by the injector,
        // so this count is never consulted.
        Arc::new(AtomicUsize::new(0)),
    )
}

/// [`fetch_row_groups`] for a consumer that lays the rows out itself: every
/// requested row group is fetched without waiting on decode, then held on
/// `gate_worker` until `gate` releases it to the decode stages. The decoded
/// batches carry the metadata columns.
pub fn fetch_row_groups_gated<OF>(
    requests: OperatorSpec<RowGroupRequest, OF>,
    projection: Projection,
    gate: DecodeGate,
    gate_worker: usize,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<RowGroupRequest> + 'static,
{
    let n = requests.dispatcher().worker_count();
    let pending_row_groups = pending_row_group_counters(n);
    // The gate holds fetched row groups back from decode, so a claim bound
    // counting undecoded row groups would stop the fetch at the first few.
    let buffers = fetch_buffers(requests, &pending_row_groups, usize::MAX).chain(
        to_single_worker_mpsc::<RowGroupBuffer>(n, gate_worker)
            .into_iter()
            .collect(),
        (0..n)
            .map(|_| DecodeGateFactory::new(gate.clone()))
            .collect(),
    );
    read_parquet(
        buffers,
        projection,
        RECORD_BATCH_SIZE,
        true,
        Arc::new(Vec::new()),
        pending_row_groups,
        // Row groups are claimed by explicit request, not by the injector,
        // so this count is never consulted.
        Arc::new(AtomicUsize::new(0)),
    )
}

/// The fetch stage for explicit requests. The requests keep their order on
/// the way to the fetchers, so the row groups arrive in about the order they
/// were asked for.
fn fetch_buffers<OF>(
    requests: OperatorSpec<RowGroupRequest, OF>,
    pending_row_groups: &[Arc<AtomicUsize>],
    claim_bound: usize,
) -> OperatorSpec<RowGroupBuffer, impl OperatorFactory<RowGroupBuffer> + Send + 'static>
where
    OF: OperatorFactory<RowGroupRequest> + 'static,
{
    let topology = requests.dispatcher().topology();
    requests.chain(
        stealable_fifo::<RowGroupRequest>(topology)
            .into_iter()
            .collect(),
        pending_row_groups
            .iter()
            .map(|pending| RowGroupFetcherFactory::new(pending.clone(), claim_bound))
            .collect(),
    )
}

/// One claimed-but-not-yet-cut row-group counter per worker, shared by the
/// worker's fetcher (increments and gates claims) and its range cutter
/// (decrements as row groups are cut or pruned). See `RowGroupFetcher` for
/// why claims are bounded this way.
fn pending_row_group_counters(n: usize) -> Vec<Arc<AtomicUsize>> {
    (0..n).map(|_| Arc::new(AtomicUsize::new(0))).collect()
}

/// A [`RecordBatchOperatorSpec`] that scans a Parquet table.
pub fn table_input(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    projection: Projection,
    add_row_group_metadata: bool,
) -> RecordBatchOperatorSpec {
    table_input_with_filter_and_eq_predicates(
        dispatcher,
        table,
        projection,
        add_row_group_metadata,
        None,
        None,
        Arc::new(Vec::new()),
    )
}

/// Like [`table_input`] but with a dynamic [`RowGroupFilter`] consulted as each
/// row group is stolen (a Top-N above the scan pruning against a live predicate).
pub fn table_input_with_filter(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    projection: Projection,
    add_row_group_metadata: bool,
    filter: Option<RowGroupFilter>,
) -> RecordBatchOperatorSpec {
    table_input_with_filter_and_eq_predicates(
        dispatcher,
        table,
        projection,
        add_row_group_metadata,
        filter,
        None,
        Arc::new(Vec::new()),
    )
}

/// Like [`table_input`] but with both a dynamic [`RowGroupFilter`] and
/// pushed-down equality predicates (the decoder pruning row groups by
/// dictionary contents). Either can be inert (`None` / empty `Vec`).
pub fn table_input_with_filter_and_eq_predicates(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    projection: Projection,
    add_row_group_metadata: bool,
    filter: Option<RowGroupFilter>,
    scan_order: Option<ScanOrder>,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
) -> RecordBatchOperatorSpec {
    let n = dispatcher.worker_count();
    // A projection with no data columns can't go through the column-driven page
    // pipeline (it would fetch nothing and emit no rows), so route it to a source
    // that emits each row group's rows straight from `num_rows` (no IO, no
    // decode), appending row-group/row-index metadata only when requested. Covers
    // a late-materialized plain `LIMIT` (metadata, for the downstream
    // `Materialize`) and any other zero-column scan (a row count).
    if projection.indices().is_empty() {
        return empty_projection_scan(dispatcher, table, filter, add_row_group_metadata);
    }
    // Counts the row groups the whole scan has claimed but not yet cut into
    // decode ranges. The injector increments it on claim, the range cutters
    // decrement it, and a Top-N scan throttles its claims against it while
    // its boundary converges (see `RowGroupInjector`).
    let outstanding_row_groups = Arc::new(AtomicUsize::new(0));
    let injector = RowGroupInjectorFactory::new(
        table,
        projection.clone(),
        filter,
        scan_order,
        outstanding_row_groups.clone(),
        dispatcher.topology().node_count,
    );
    let siblings = Arc::new(AtomicUsize::new(n));
    let pending_row_groups = pending_row_group_counters(n);
    // One fetcher handles disk and HTTP row groups, bounding each medium's
    // in-flight count separately, plus its worker's undecoded-claims count.
    let claim_bound = pending_claim_bound(table);
    let factories: Vec<_> = pending_row_groups
        .iter()
        .map(|pending| {
            RootUnaryOperatorFactory::new(
                RowGroupFetcherFactory::new(pending.clone(), claim_bound),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let input = OperatorSpec::new(dispatcher.clone(), factories);
    read_parquet(
        input,
        projection,
        RECORD_BATCH_SIZE,
        add_row_group_metadata,
        eq_predicates,
        pending_row_groups,
        outstanding_row_groups,
    )
}

/// Late materialization: take an existing `RecordBatch` spec (whose rows carry
/// row-group metadata), fetch `projection` for the surviving rows, and decode.
/// Chains a materializer + fetcher onto each worker's factory, then re-enters
/// the read pipeline.
pub fn materialize(
    spec: RecordBatchOperatorSpec,
    table: Arc<ParquetTable>,
    projection: Projection,
) -> RecordBatchOperatorSpec {
    let (dispatcher, mut heads) = spec.into_parts();
    let n = dispatcher.worker_count();
    let siblings_materializer = Arc::new(AtomicUsize::new(n));
    let siblings_fetcher = Arc::new(AtomicUsize::new(n));

    let pending_row_groups = pending_row_group_counters(n);
    let claim_bound = pending_claim_bound(&table);
    // Every surviving-row batch funnels to ONE worker's materializer, so each
    // row group becomes exactly one [`RowGroupRequest`] holding all of its
    // surviving rows. The range cutter relies on that: it tracks row groups
    // by index and drops pages of a group it already cut, so a second
    // request for the same group would silently lose its rows. The funneled
    // stage only merges tiny index lists; the fetch and decode stay spread
    // over every worker through the stealable request channel. Successive
    // materializes take turns hosting the merge, so concurrent queries do not
    // all pin one worker.
    let host = dispatcher.next_worker();
    let factories: Vec<_> = to_single_worker_mpsc::<RecordBatch>(n, host)
        .into_iter()
        .zip(stealable::<RowGroupRequest>(dispatcher.topology()))
        .zip(pending_row_groups.iter())
        .map(|((rb_ch, rq_ch), pending)| {
            UnaryOperatorFactory::new(
                UnaryOperatorFactory::new(
                    heads.pop_front().unwrap(),
                    MaterializerFactory::new(projection.clone(), table.clone()),
                    rb_ch,
                    siblings_materializer.clone(),
                ),
                RowGroupFetcherFactory::new(pending.clone(), claim_bound),
                rq_ch,
                siblings_fetcher.clone(),
            )
        })
        .collect();
    let input = OperatorSpec::new(dispatcher, factories);
    read_parquet(
        input,
        projection,
        RECORD_BATCH_SIZE,
        false,
        Arc::new(Vec::new()),
        pending_row_groups,
        // The materializer path claims by explicit row-group requests, not the
        // injector, so this count is never consulted.
        Arc::new(AtomicUsize::new(0)),
    )
}
