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
use crate::types::metadata::QueryRowGroupMetadata;
use arrow_array::RecordBatch;
use dispatch::io::ReadMedium;
use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection, RECORD_BATCH_SIZE,
    RecordBatchOperatorSpec, RootUnaryOperatorFactory, UnaryOperatorFactory, return_to_worker_mpsc,
    shared_work_queue, stealable, stealable_fifo, to_single_worker_mpsc,
};

use crate::{
    CompressedPage, DecodeRange, DecoderFactory, DecompressedPage, DecompressorFactory,
    IndexerFactory, MaterializerFactory, ParquetTable, RangeCutterFactory, RowGroupBuffer,
    RowGroupFetcherFactory, RowGroupInjectorFactory, ScanEqualityPredicate, ScanOrder,
    WorkerAllocator,
};

/// Append the index → decompress → cut → decode stages onto a source of
/// [`RowGroupBuffer`]s, producing decoded `RecordBatch`es.
pub(crate) fn read_parquet<OF>(
    input: OperatorSpec<RowGroupBuffer, OF>,
    projection: Projection,
    batch_size: usize,
    add_row_group_metadata: bool,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    outstanding_row_groups: Arc<AtomicUsize>,
) -> OperatorSpec<RecordBatch, impl OperatorFactory<RecordBatch> + use<OF> + 'static>
where
    OF: OperatorFactory<RowGroupBuffer> + Send + 'static,
{
    let ranges = cut_decode_ranges(input, projection, eq_predicates, outstanding_row_groups);
    decode_ranges(ranges, batch_size, add_row_group_metadata)
}

/// Append the index → decompress → cut stages onto a source of
/// [`RowGroupBuffer`]s: each row group's pages, decompressed, as the
/// [`DecodeRange`]s its rows are decoded in.
pub(crate) fn cut_decode_ranges<OF>(
    input: OperatorSpec<RowGroupBuffer, OF>,
    projection: Projection,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    outstanding_row_groups: Arc<AtomicUsize>,
) -> OperatorSpec<DecodeRange, impl OperatorFactory<DecodeRange> + use<OF> + 'static>
where
    OF: OperatorFactory<RowGroupBuffer> + Send + 'static,
{
    let n = input.dispatcher().worker_count();
    let topology = input.dispatcher().topology();
    // One allocator per worker for the dictionaries its range cutter builds,
    // which live as long as their row group. It is apart from the one the
    // decoder's batches come from, so a dictionary never pins the buffer the
    // batches cycle through.
    let dictionary_allocators: Vec<Arc<WorkerAllocator>> =
        (0..n).map(|_| Arc::new(WorkerAllocator::new())).collect();
    input
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
            dictionary_allocators
                .into_iter()
                .map(|allocator| RangeCutterFactory {
                    projection: projection.clone(),
                    eq_predicates: eq_predicates.clone(),
                    outstanding_row_groups: outstanding_row_groups.clone(),
                    allocator,
                })
                .collect(),
        )
}

/// Append the decode stage onto a source of [`DecodeRange`]s, producing
/// decoded `RecordBatch`es.
pub(crate) fn decode_ranges<OF>(
    ranges: OperatorSpec<DecodeRange, OF>,
    batch_size: usize,
    add_row_group_metadata: bool,
) -> OperatorSpec<RecordBatch, impl OperatorFactory<RecordBatch> + use<OF> + 'static>
where
    OF: OperatorFactory<DecodeRange> + Send + 'static,
{
    let n = ranges.dispatcher().worker_count();
    let topology = ranges.dispatcher().topology();
    // One allocator per worker for the batches its decoder emits, which
    // usually die before the next one is built, so the decoder keeps
    // rewriting the same cache-resident bytes.
    let batch_allocators: Vec<Arc<WorkerAllocator>> =
        (0..n).map(|_| Arc::new(WorkerAllocator::new())).collect();
    ranges
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
        )
}

/// Fetch and decode the row groups that `requests` name. The requests keep
/// their order on the way to the fetchers, so the row groups decode in about
/// the order they were asked for, and the decoded batches carry the metadata
/// columns when `add_row_group_metadata` so a consumer can tell which row
/// group and rows each holds.
pub fn fetch_row_groups<OF>(
    requests: OperatorSpec<QueryRowGroupMetadata, OF>,
    table: &ParquetTable,
    projection: Projection,
    batch_size: usize,
    add_row_group_metadata: bool,
) -> OperatorSpec<RecordBatch, impl OperatorFactory<RecordBatch> + use<OF> + 'static>
where
    OF: OperatorFactory<QueryRowGroupMetadata> + 'static,
{
    let buffers = fetch_row_group_buffers(requests, &projection, table.get_read_medium());
    read_parquet(
        buffers,
        projection,
        batch_size,
        add_row_group_metadata,
        Arc::new(Vec::new()),
        // Row groups are claimed by explicit request, not by the injector,
        // so this count is never consulted.
        Arc::new(AtomicUsize::new(0)),
    )
}

/// The fetch half of [`fetch_row_groups`]: read the bytes of the row groups
/// that `requests` name, and stop there. The reads count against each
/// worker's read-ahead budget for `read_medium`, the medium the table is
/// read from.
pub(crate) fn fetch_row_group_buffers<OF>(
    requests: OperatorSpec<QueryRowGroupMetadata, OF>,
    projection: &Projection,
    read_medium: ReadMedium,
) -> OperatorSpec<RowGroupBuffer, impl OperatorFactory<RowGroupBuffer> + use<OF> + 'static>
where
    OF: OperatorFactory<QueryRowGroupMetadata> + 'static,
{
    let worker_count = requests.dispatcher().worker_count();
    let topology = requests.dispatcher().topology();
    let shared_projection = Arc::new(projection.clone());
    requests.chain(
        stealable_fifo::<QueryRowGroupMetadata>(topology)
            .into_iter()
            .collect(),
        // Every request arrives on the fetcher's input. One that needs a
        // read goes to the second queue, which every worker's fetcher takes
        // from; nothing is sent to the first.
        shared_work_queue::<QueryRowGroupMetadata>(worker_count)
            .into_iter()
            .zip(shared_work_queue::<QueryRowGroupMetadata>(worker_count))
            .map(|(io_row_groups_channel, uncached_row_groups_channel)| {
                RowGroupFetcherFactory::new(
                    shared_projection.clone(),
                    read_medium,
                    true,
                    io_row_groups_channel,
                    uncached_row_groups_channel,
                )
            })
            .collect(),
    )
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
    // A Top-N scan asks for no read ahead of its CPU work: a row group asked
    // for early is judged against a boundary that has not tightened yet.
    let reads_ahead_of_cpu_work = scan_order.is_none();
    let (in_memory_injector, io_row_groups_injector) =
        RowGroupInjectorFactory::create_by_expected_need(
            table,
            projection.clone(),
            filter,
            scan_order,
            outstanding_row_groups.clone(),
            dispatcher.topology().node_count,
        );
    let siblings = Arc::new(AtomicUsize::new(n));
    let shared_projection = Arc::new(projection.clone());
    let read_medium = table.get_read_medium();
    let factories: Vec<_> = shared_work_queue::<QueryRowGroupMetadata>(n)
        .into_iter()
        .map(|uncached_row_groups_channel| {
            RootUnaryOperatorFactory::new(
                RowGroupFetcherFactory::new(
                    shared_projection.clone(),
                    read_medium,
                    reads_ahead_of_cpu_work,
                    io_row_groups_injector.clone(),
                    uncached_row_groups_channel,
                ),
                in_memory_injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let input = OperatorSpec::new(dispatcher.clone(), factories);
    RecordBatchOperatorSpec::from_spec(read_parquet(
        input,
        projection,
        RECORD_BATCH_SIZE,
        add_row_group_metadata,
        eq_predicates,
        outstanding_row_groups,
    ))
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

    // Every surviving-row batch funnels to ONE worker's materializer, so each
    // row group is emitted exactly once, holding all of its surviving rows,
    // and its pages are fetched and walked once however many batches its
    // rows arrived in. The funneled stage only merges tiny
    // index lists; the fetch and decode stay spread over every worker through
    // the stealable row-group channel. Successive
    // materializes take turns hosting the merge, so concurrent queries do not
    // all pin one worker.
    let host = dispatcher.next_worker();
    let read_medium = table.get_read_medium();
    let shared_projection = Arc::new(projection.clone());
    let factories: Vec<_> = to_single_worker_mpsc::<RecordBatch>(n, host)
        .into_iter()
        .zip(stealable::<QueryRowGroupMetadata>(dispatcher.topology()))
        // The row groups the materializer expects to need a read go to one
        // queue every worker's fetcher takes from.
        .zip(shared_work_queue::<QueryRowGroupMetadata>(n))
        // The row groups a fetcher expected to be in memory and found to need a
        // read go to another queue every worker's fetcher takes from.
        .zip(shared_work_queue::<QueryRowGroupMetadata>(n))
        .map(
            |(((rb_ch, rq_ch), io_row_groups_channel), uncached_row_groups_channel)| {
                UnaryOperatorFactory::new(
                    UnaryOperatorFactory::new(
                        heads.pop_front().unwrap(),
                        MaterializerFactory::new(table.clone(), io_row_groups_channel.clone()),
                        rb_ch,
                        siblings_materializer.clone(),
                    ),
                    RowGroupFetcherFactory::new(
                        shared_projection.clone(),
                        read_medium,
                        true,
                        io_row_groups_channel,
                        uncached_row_groups_channel,
                    ),
                    rq_ch,
                    siblings_fetcher.clone(),
                )
            },
        )
        .collect();
    let input = OperatorSpec::new(dispatcher, factories);
    RecordBatchOperatorSpec::from_spec(read_parquet(
        input,
        projection,
        RECORD_BATCH_SIZE,
        false,
        Arc::new(Vec::new()),
        // The materializer path claims by explicit row-group requests, not the
        // injector, so this count is never consulted.
        Arc::new(AtomicUsize::new(0)),
    ))
}
