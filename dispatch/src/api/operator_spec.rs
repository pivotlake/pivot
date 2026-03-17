use crate::operations::UnaryOperatorFactory;
use crate::operations::channels::{
    ReturnToWorkerMpscFactory, Sender, StealableChannelFactory, return_to_worker_mpsc, stealable,
};
use crate::operations::parquet::DecoderFactory;
use crate::operations::parquet::DecompressorFactory;
use crate::operations::parquet::types::projection::Projection;
use crate::operations::parquet::{CompressedPage, DecompressedPage, ParquetTable};
use crate::operations::parquet::{IndexerFactory, RowGroupBuffer};
use crate::{Chain, dispatcher};
use arrow_array::RecordBatch;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// Generic, strongly-typed spec used internally to build the parquet read pipeline.
///
/// Holds one factory per worker, where the factory type `OF` carries the full nested
/// generic chain (e.g. `UnaryOperatorFactory<..., UnaryOperatorFactory<..., OF>>`).
/// This is needed because the parquet pipeline flows through non-RecordBatch types
/// (`RowGroupBuffer → CompressedPage → DecompressedPage → RecordBatch`), and each
/// stage uses a different channel/sender type — including `WorkerAwareSender` which
/// requires `O: WorkerIdOutput`.
///
/// [`OperatorFactory::build`] must be generic over `S: Sender<O>` to support this,
/// which makes it **not object-safe**. This is why we can't use
/// `Box<dyn OperatorFactory<O>>` directly and need a separate object-safe
/// [`RecordBatchOperatorFactory`](super::record_batch_operator::RecordBatchOperatorFactory)
/// trait at the RecordBatch boundary.
///
/// Concretely, making `OperatorFactory<O>` object-safe would require replacing
/// `build<S: Sender<O>>` with concrete methods like `build_stealable(Rc<Worker<O>>)`
/// and `build_collect(MpscSender<O>)`. But there is no clean way to add a
/// `build_worker_aware(WorkerAwareSender<O>)` method: the `Sender` impl for
/// `WorkerAwareSender<O>` requires `O: WorkerIdOutput`, and adding that bound to the
/// trait method either forces the bound onto all users (wrong for `RecordBatch`) or
/// requires a `where` clause that breaks object safety.
pub struct OperatorSpec<O, OF: OperatorFactory<O>> {
    /// Vector of factories, one per worker. These all should build the same `Box<dyn Operator>` within
    /// different workers.
    factories: VecDeque<OF>,
    _phantom: std::marker::PhantomData<O>,
}

impl<O, OF: OperatorFactory<O>> OperatorSpec<O, OF> {
    pub fn new(factories: impl IntoIterator<Item = OF>) -> Self {
        Self {
            factories: factories.into_iter().collect(),
            _phantom: Default::default(),
        }
    }

    pub fn factories(self) -> VecDeque<OF> {
        self.factories
    }
}

type ReadParquet<OF> = UnaryOperatorFactory<
    DecompressedPage,
    RecordBatch,
    DecoderFactory,
    ReturnToWorkerMpscFactory<DecompressedPage>,
    UnaryOperatorFactory<
        CompressedPage,
        DecompressedPage,
        DecompressorFactory,
        StealableChannelFactory<CompressedPage>,
        UnaryOperatorFactory<
            RowGroupBuffer,
            CompressedPage,
            IndexerFactory,
            StealableChannelFactory<RowGroupBuffer>,
            OF,
        >,
    >,
>;

impl<OF: OperatorFactory<RowGroupBuffer>> OperatorSpec<RowGroupBuffer, OF> {
    pub fn read_parquet(
        mut self,
        table: &Arc<ParquetTable>,
        projection: Projection,
        batch_size: usize,
        add_row_group_metadata: bool,
    ) -> OperatorSpec<RecordBatch, ReadParquet<OF>> {
        let siblings_left_indexer = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let siblings_left_decompressor = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let siblings_left_drain = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(
            stealable::<RowGroupBuffer>()
                .into_iter()
                .zip(stealable::<CompressedPage>())
                .zip(return_to_worker_mpsc::<DecompressedPage>())
                .map(move |((ic, dc), drc)| {
                    UnaryOperatorFactory::new(
                        UnaryOperatorFactory::new(
                            UnaryOperatorFactory::new(
                                self.factories.pop_front().unwrap(),
                                IndexerFactory::new(),
                                ic,
                                siblings_left_indexer.clone(),
                            ),
                            DecompressorFactory::new(),
                            dc,
                            siblings_left_decompressor.clone(),
                        ),
                        DecoderFactory {
                            batch_size,
                            table: table.clone(),
                            projection: projection.clone(),
                            add_row_group_metadata,
                        },
                        drc,
                        siblings_left_drain.clone(),
                    )
                }),
        )
    }
}

/// A factory that builds one worker's operator chain, given an output sender.
///
/// Called on the worker thread during [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build).
/// The `build` method consumes the factory, creates channels between stages, and returns
/// a [`Chain`] of operators ready to execute.
///
/// `build` is generic over `S: Sender<O>` because different stages connect via different
/// sender types (work-stealing, mpsc, worker-aware). This makes the trait **not object-safe**
/// — see [`RecordBatchOperatorFactory`](super::record_batch_operator::RecordBatchOperatorFactory)
/// for the object-safe equivalent at the `RecordBatch` boundary.
pub trait OperatorFactory<O>: Send {
    /// Build the operator chain, outputting to `sender`.
    fn build<S: Sender<O> + 'static>(self: Box<Self>, sender: S) -> Chain;
}
