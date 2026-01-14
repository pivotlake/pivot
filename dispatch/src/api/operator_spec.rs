use crate::api::{DataFlowBuilder, OperatorFactory};
use crate::dispatcher;
use crate::operations::parquet::{
    DecompressorFactory, IndexerFactory, PageWithInfo, RecordBatchGeneratorFactory,
    RowGroupCompressedPage,
};
use crate::operations::{
    CountFactory, FilterFactory, GroupFactory, MaterializeJobGeneratorFactory, MaterializeRequest,
    MaterializerFactory, OrderBy, OrderByLimitFactory, ProjectFactory, ReturnToWorkerMpscFactory,
    StealableChannelFactory, UnaryOperatorFactory, mpsc_channel, return_to_worker_mpsc, stealable,
};
use crate::table::input::RowGroupBuffer;
use crate::table::{Projection, Table, TableInputFactory, TableSource};
use arrow_array::{BooleanArray, RecordBatch};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

pub struct OperatorSpec<O, OF: OperatorFactory<O>> {
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
}

type ReadParquet<OF> = UnaryOperatorFactory<
    PageWithInfo,
    RecordBatch,
    RecordBatchGeneratorFactory,
    ReturnToWorkerMpscFactory<PageWithInfo>,
    UnaryOperatorFactory<
        RowGroupCompressedPage,
        PageWithInfo,
        DecompressorFactory,
        StealableChannelFactory<RowGroupCompressedPage>,
        UnaryOperatorFactory<
            RowGroupBuffer,
            RowGroupCompressedPage,
            IndexerFactory,
            StealableChannelFactory<RowGroupBuffer>,
            OF,
        >,
    >,
>;

pub fn table_input(
    table: Arc<Table>,
    projection: Option<Projection>,
) -> OperatorSpec<RecordBatch, ReadParquet<TableInputFactory>> {
    let source = Arc::new(TableSource::from(&table));
    let input = OperatorSpec::new(
        (0..dispatcher().workers())
            .map(|_| TableInputFactory::new(source.clone(), table.clone(), projection.clone())),
    );
    input.read_parquet(projection)
}

impl<OF: OperatorFactory<RowGroupBuffer>> OperatorSpec<RowGroupBuffer, OF> {
    pub fn read_parquet(
        mut self,
        projection: Option<Projection>,
    ) -> OperatorSpec<RecordBatch, ReadParquet<OF>> {
        let siblings_left_indexer = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let siblings_left_decompressor = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let siblings_left_rbg = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(
            stealable::<RowGroupBuffer>()
                .into_iter()
                .zip(stealable::<RowGroupCompressedPage>())
                .zip(return_to_worker_mpsc::<PageWithInfo>())
                .map(move |((ic, dc), rc)| {
                    UnaryOperatorFactory::new(
                        UnaryOperatorFactory::new(
                            UnaryOperatorFactory::new(
                                self.factories.pop_front().unwrap(),
                                IndexerFactory::new(projection.clone()),
                                ic,
                                siblings_left_indexer.clone(),
                            ),
                            DecompressorFactory,
                            dc,
                            siblings_left_decompressor.clone(),
                        ),
                        RecordBatchGeneratorFactory(projection.clone()),
                        rc,
                        siblings_left_rbg.clone(),
                    )
                }),
        )
    }
}

impl<OF: OperatorFactory<RecordBatch> + 'static> OperatorSpec<RecordBatch, OF> {
    pub fn project<P: FnMut(&RecordBatch) -> RecordBatch + Send + 'static, PB: Fn() -> P>(
        mut self,
        builder: PB,
    ) -> OperatorSpec<
        RecordBatch,
        UnaryOperatorFactory<
            RecordBatch,
            RecordBatch,
            ProjectFactory<P>,
            StealableChannelFactory<RecordBatch>,
            OF,
        >,
    > {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(stealable::<RecordBatch>().into_iter().map(|c| {
            UnaryOperatorFactory::new(
                self.factories.pop_front().unwrap(),
                ProjectFactory(builder()),
                c,
                siblings_left.clone(),
            )
        }))
    }

    pub fn filter<F: FnMut(&RecordBatch) -> BooleanArray + Send + 'static, FB: Fn() -> F>(
        mut self,
        builder: FB,
    ) -> OperatorSpec<
        RecordBatch,
        UnaryOperatorFactory<
            RecordBatch,
            RecordBatch,
            FilterFactory<F>,
            StealableChannelFactory<RecordBatch>,
            OF,
        >,
    > {
        let channels = stealable::<RecordBatch>();
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(channels.into_iter().map(|c| {
            UnaryOperatorFactory::new(
                self.factories.pop_front().unwrap(),
                FilterFactory(builder()),
                c,
                siblings_left.clone(),
            )
        }))
    }

    pub fn count(
        self,
    ) -> OperatorSpec<
        RecordBatch,
        UnaryOperatorFactory<
            RecordBatch,
            RecordBatch,
            CountFactory,
            StealableChannelFactory<RecordBatch>,
            OF,
        >,
    > {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(
            stealable::<RecordBatch>()
                .into_iter()
                .zip(CountFactory::create_for_workers(dispatcher().workers()))
                .zip(self.factories)
                .map(|((channel_factory, count_factory), current)| {
                    UnaryOperatorFactory::new(
                        current,
                        count_factory,
                        channel_factory,
                        siblings_left.clone(),
                    )
                }),
        )
    }

    pub fn order_by_limit(
        self,
        order_by: Vec<OrderBy>,
        limit: usize,
    ) -> OperatorSpec<
        RecordBatch,
        UnaryOperatorFactory<
            RecordBatch,
            RecordBatch,
            OrderByLimitFactory,
            StealableChannelFactory<RecordBatch>,
            OF,
        >,
    > {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(
            stealable::<RecordBatch>()
                .into_iter()
                .zip(OrderByLimitFactory::create_for_workers(
                    order_by,
                    limit,
                    dispatcher().workers(),
                ))
                .zip(self.factories)
                .map(|((channel_factory, obl_factory), current)| {
                    UnaryOperatorFactory::new(
                        current,
                        obl_factory,
                        channel_factory,
                        siblings_left.clone(),
                    )
                }),
        )
    }

    pub fn group_by_count(
        self,
        group_column: usize,
    ) -> OperatorSpec<
        RecordBatch,
        UnaryOperatorFactory<
            RecordBatch,
            RecordBatch,
            GroupFactory,
            StealableChannelFactory<RecordBatch>,
            OF,
        >,
    > {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        OperatorSpec::new(
            stealable::<RecordBatch>()
                .into_iter()
                .zip(GroupFactory::create_for_workers(
                    group_column,
                    dispatcher().workers(),
                ))
                .zip(self.factories)
                .map(|((channel_factory, group_factory), current)| {
                    UnaryOperatorFactory::new(
                        current,
                        group_factory,
                        channel_factory,
                        siblings_left.clone(),
                    )
                }),
        )
    }

    pub fn materialize(
        mut self,
        table: Arc<Table>,
        projection: Option<Projection>,
    ) -> OperatorSpec<
        RecordBatch,
        ReadParquet<
            MaterializerFactory<
                UnaryOperatorFactory<
                    RecordBatch,
                    MaterializeRequest,
                    MaterializeJobGeneratorFactory,
                    StealableChannelFactory<RecordBatch>,
                    OF,
                >,
                StealableChannelFactory<MaterializeRequest>,
            >,
        >,
    > {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));

        let materializer_spec: OperatorSpec<RowGroupBuffer, _> = OperatorSpec::new(
            stealable::<RecordBatch>()
                .into_iter()
                .zip(stealable::<MaterializeRequest>())
                .map(|(rb_ch, mr_ch)| {
                    MaterializerFactory::new(
                        UnaryOperatorFactory::new(
                            self.factories.pop_front().unwrap(),
                            MaterializeJobGeneratorFactory::new(projection.clone(), table.clone()),
                            rb_ch,
                            siblings_left.clone(),
                        ),
                        mr_ch,
                    )
                }),
        );

        materializer_spec.read_parquet(projection)
    }

    pub fn collect(self) -> Vec<RecordBatch> {
        let (tx, rx) = mpsc_channel();
        dispatcher().push_data_flow(
            self.factories
                .into_iter()
                .map(|f| DataFlowBuilder::new(Box::new(f), tx.clone())),
        );
        drop(tx);
        let (rx, _) = rx.into_parts();
        rx.into_iter().collect()
    }
}
