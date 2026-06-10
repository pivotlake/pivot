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

use crate::parquet::{
    CompressedPage, DecoderFactory, DecompressedPage, DecompressorFactory, IndexerFactory,
    MaterializerFactory, ParquetTable, RowGroupBuffer, RowGroupFetcherFactory,
    RowGroupInjectorFactory, RowGroupRequest, ScanEqualityPredicate, ScanOrder,
};

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
    // A scan that reads no columns (`SELECT COUNT(*)`-shaped) is answered
    // from row-group metadata alone: each group contributes an empty-schema
    // batch carrying only its row count — no pages are fetched or decoded.
    if projection.column_indices.is_empty() && !add_row_group_metadata {
        return count_only_scan(dispatcher, table, filter);
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
