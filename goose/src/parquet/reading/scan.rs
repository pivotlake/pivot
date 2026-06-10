//! Builders that assemble the Parquet read pipeline into a dispatch dataflow.
//!
//! These were inherent methods on dispatch's `OperatorSpec` / `RecordBatchOperatorSpec`
//! before the Parquet pipeline moved into `goose`; they're now free functions
//! built on dispatch's public operator toolkit (`OperatorSpec::chain`,
//! `RootUnaryOperatorFactory`, the channel factories). The stages are: row-group
//! injection + fetching (source) → index → decompress → decode.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::parquet::RowGroupFilter;
use arrow_array::RecordBatch;
use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection, RECORD_BATCH_SIZE,
    RecordBatchFactoryBridge, RecordBatchOperatorSpec, RootUnaryOperatorFactory,
    UnaryOperatorFactory, return_to_worker_mpsc, stealable,
};

use crate::parquet::reading::staged::StagingDecoderFactory;
use crate::parquet::{
    CompressedPage, DecoderFactory, DecompressedPage, DecompressorFactory, IndexerFactory,
    MaterializerFactory, ParquetTable, RowGroupBuffer, RowGroupFetcherFactory,
    RowGroupInjectorFactory, RowGroupRequest, ScanEqualityPredicate, ScanOrder,
};
use planner::catalog::ScanFilter;

/// Append the index → decompress → decode stages onto a source of
/// [`RowGroupBuffer`]s, producing decoded `RecordBatch`es.
pub(crate) fn read_parquet<OF>(
    input: OperatorSpec<RowGroupBuffer, OF>,
    table: &Arc<ParquetTable>,
    projection: Projection,
    batch_size: usize,
    add_row_group_metadata: bool,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<RowGroupBuffer> + Send + 'static,
{
    let n = input.dispatcher().worker_count();
    let decoded = input
        .chain(
            stealable::<RowGroupBuffer>(n).into_iter().collect(),
            (0..n).map(|_| IndexerFactory::new()).collect(),
        )
        .chain(
            stealable::<CompressedPage>(n).into_iter().collect(),
            (0..n).map(|_| DecompressorFactory::new()).collect(),
        )
        .chain(
            return_to_worker_mpsc::<DecompressedPage>(n)
                .into_iter()
                .collect(),
            (0..n)
                .map(|_| DecoderFactory {
                    batch_size,
                    table: table.clone(),
                    projection: projection.clone(),
                    add_row_group_metadata,
                    eq_predicates: eq_predicates.clone(),
                })
                .collect(),
        );
    RecordBatchOperatorSpec::from_spec(decoded)
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
        None,
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
        None,
    )
}

/// Like [`table_input`] but with every scan-time optimization input: a dynamic
/// [`RowGroupFilter`], pushed-down equality predicates (the decoder pruning row
/// groups by dictionary contents), and a pushed [`ScanFilter`] enabling the
/// staged fetch. Any of them can be inert (`None` / empty `Vec`).
#[allow(clippy::too_many_arguments)]
pub fn table_input_with_filter_and_eq_predicates(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    projection: Projection,
    add_row_group_metadata: bool,
    filter: Option<RowGroupFilter>,
    scan_order: Option<ScanOrder>,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    scan_filter: Option<ScanFilter>,
) -> RecordBatchOperatorSpec {
    // A scan that reads no columns (`SELECT COUNT(*)`-shaped) is answered
    // from row-group metadata alone: each group contributes an empty-schema
    // batch carrying only its row count — no pages are fetched or decoded.
    if projection.column_indices.is_empty() && !add_row_group_metadata {
        return count_only_scan(dispatcher, table, filter);
    }

    // A pushed row filter over a strict subset of the projection enables the
    // staged fetch: read the filter columns first, and fetch the remaining
    // columns only for row groups where some row survives. Never taken for a
    // metadata-emitting (late-materialization) scan — that path re-reads
    // survivors itself — so a plain scan's pipeline is entirely unaffected.
    if let Some(scan_filter) = scan_filter
        && !add_row_group_metadata
        && staging_applies(&scan_filter, &projection)
    {
        return staged_table_input(
            dispatcher,
            table,
            projection,
            filter,
            scan_order,
            eq_predicates,
            scan_filter,
        );
    }

    let n = dispatcher.worker_count();
    let injector = RowGroupInjectorFactory::new(table, projection.clone(), filter, scan_order);
    let siblings = Arc::new(AtomicUsize::new(n));
    // One fetcher handles disk and HTTP row groups, bounding each medium's
    // in-flight count separately.
    let factories: Vec<_> = (0..n)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                RowGroupFetcherFactory::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let input = OperatorSpec::new(dispatcher.clone(), factories);
    read_parquet(
        input,
        table,
        projection,
        RECORD_BATCH_SIZE,
        add_row_group_metadata,
        eq_predicates,
    )
}

/// A columnless scan: stream the table's row groups and emit, per group, one
/// empty-schema `RecordBatch` whose row count comes straight from the group's
/// metadata. Pure metadata — no I/O — so a bare `COUNT(*)` costs nothing. A
/// dynamic [`RowGroupFilter`] is still consulted per group (a pruned group
/// contributes zero rows), keeping the contract of the regular scan.
fn count_only_scan(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    filter: Option<RowGroupFilter>,
) -> RecordBatchOperatorSpec {
    use arrow_array::RecordBatchOptions;
    use arrow_schema::Schema;

    let groups: Vec<_> = table.row_groups().to_vec();
    dispatch::values_input(dispatcher, groups)
        .map_each(move |rg| {
            let rows = match &filter {
                Some(keep) if !keep(rg.as_ref()) => 0,
                _ => rg.num_rows as usize,
            };
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                Vec::new(),
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .expect("an empty-schema batch with an explicit row count is always valid")
        })
        .record_batches()
}

/// Whether the staged fetch can pay off for this scan: the filter must read at
/// least one column (otherwise there is nothing to evaluate phase A on) and a
/// *strict* subset of the projection (reading every projected column in phase A
/// would already incur the full I/O). The bounds check is defensive — the
/// planner only pushes filters over projected columns.
fn staging_applies(scan_filter: &ScanFilter, projection: &Projection) -> bool {
    !scan_filter.columns.is_empty()
        && scan_filter.columns.len() < projection.column_indices.len()
        && scan_filter
            .columns
            .iter()
            .all(|&p| p < projection.column_indices.len())
}

/// The staged scan: a filtered scan that never fetches bytes it can prove it
/// doesn't need.
///
/// Phase A is an ordinary scan pipeline over only the *filter columns*
/// (injector → fetcher → indexer → decompressor), terminated by a
/// [`StagingDecoder`](crate::parquet::reading::staged::StagingDecoder) instead
/// of the regular decoder: it decodes the filter columns, evaluates the pushed
/// row filter, and emits one phase-B [`RowGroupRequest`] per row group with
/// survivors — carrying their indices — while a row group with zero survivors
/// emits nothing, so the rest of its columns are never read. Phase B is the
/// standard fetch → [`read_parquet`] pipeline over the full projection (the
/// same shape as [`materialize`]'s decode → fetch → read-again loop); the
/// filter columns it re-lists were just read in phase A, so their file-cache
/// lookups are hits and cost no new I/O.
///
/// All the regular scan inputs apply to phase A exactly as they would to an
/// unstaged scan: the dynamic [`RowGroupFilter`] and `scan_order` shape which
/// row groups are stolen and in what order, and `eq_predicates` prune by
/// dictionary contents during phase-A decode (additionally saving the pruned
/// group's phase-B fetch). Phase B needs none of them — its row groups are
/// exactly the survivors.
fn staged_table_input(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
    scan_order: Option<ScanOrder>,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    scan_filter: ScanFilter,
) -> RecordBatchOperatorSpec {
    let n = dispatcher.worker_count();
    // The filter's columns as table indices, in `scan_filter.columns` order —
    // the schema its evaluator was compiled against.
    let filter_projection = Projection::columns(
        scan_filter
            .columns
            .iter()
            .map(|&p| projection.column_indices[p]),
    );

    // Phase A: fetch + decode only the filter columns of each row group.
    let injector =
        RowGroupInjectorFactory::new(table, filter_projection.clone(), filter, scan_order);
    let siblings = Arc::new(AtomicUsize::new(n));
    let factories: Vec<_> = (0..n)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                RowGroupFetcherFactory::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let phase_b_requests = OperatorSpec::new(dispatcher.clone(), factories)
        .chain(
            stealable::<RowGroupBuffer>(n).into_iter().collect(),
            (0..n).map(|_| IndexerFactory::new()).collect(),
        )
        .chain(
            stealable::<CompressedPage>(n).into_iter().collect(),
            (0..n).map(|_| DecompressorFactory::new()).collect(),
        )
        .chain(
            // Pages return to the worker that indexed their row group, so one
            // StagingDecoder sees all of a group's pages and emits its phase-B
            // request exactly once.
            return_to_worker_mpsc::<DecompressedPage>(n)
                .into_iter()
                .collect(),
            (0..n)
                .map(|_| StagingDecoderFactory {
                    batch_size: RECORD_BATCH_SIZE,
                    table: table.clone(),
                    filter_projection: filter_projection.clone(),
                    full_projection: projection.clone(),
                    eq_predicates: eq_predicates.clone(),
                    evaluator: scan_filter.evaluator.clone(),
                })
                .collect(),
        )
        // Phase B: fetch the full projection of the surviving row groups.
        .chain(
            stealable::<RowGroupRequest>(n).into_iter().collect(),
            (0..n).map(|_| RowGroupFetcherFactory::new()).collect(),
        );
    // The survivors already passed the eq predicates in phase A, so phase B
    // decodes without them (re-checking dictionaries would prune nothing).
    read_parquet(
        phase_b_requests,
        table,
        projection,
        RECORD_BATCH_SIZE,
        false,
        Arc::new(Vec::new()),
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

    let factories: Vec<_> = stealable::<RecordBatch>(n)
        .into_iter()
        .zip(stealable::<RowGroupRequest>(n))
        .map(|(rb_ch, rq_ch)| {
            UnaryOperatorFactory::new(
                UnaryOperatorFactory::new(
                    RecordBatchFactoryBridge::new(heads.pop_front().unwrap()),
                    MaterializerFactory::new(projection.clone(), table.clone()),
                    rb_ch,
                    siblings_materializer.clone(),
                ),
                RowGroupFetcherFactory::new(),
                rq_ch,
                siblings_fetcher.clone(),
            )
        })
        .collect();
    let input = OperatorSpec::new(dispatcher, factories);
    read_parquet(
        input,
        &table,
        projection,
        RECORD_BATCH_SIZE,
        false,
        Arc::new(Vec::new()),
    )
}
