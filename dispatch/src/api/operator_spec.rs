use crate::operations::channels::{
    ChannelFactory, Sender, StealableChannelFactory, mpsc_channel, stealable,
};
use crate::operations::{
    DefaultUnaryFactory, Forward, InjectorSourceFactory, MapFactory, RootUnaryOperatorFactory,
    UnaryFactory, UnaryOperatorFactory,
};
use crate::{Chain, DataFlowBuilder, DataFlowDispatcher, DataFlowHandle};
use arrow_array::RecordBatch;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

/// Generic, strongly-typed spec for building multi-stage pipelines.
///
/// Holds one factory per worker, where the factory type `OF` carries the full nested
/// generic chain (e.g. `UnaryOperatorFactory<..., UnaryOperatorFactory<..., OF>>`).
/// This is needed when a pipeline flows through non-RecordBatch types — e.g.
/// `catalog`'s Parquet reader, whose decode stages pass format-specific buffers
/// between them before producing `RecordBatch`es, where each
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
    dispatcher: DataFlowDispatcher,
    factories: VecDeque<OF>,
    _phantom: std::marker::PhantomData<O>,
}

impl<O, OF: OperatorFactory<O>> OperatorSpec<O, OF> {
    pub fn new(dispatcher: DataFlowDispatcher, factories: impl IntoIterator<Item = OF>) -> Self {
        Self {
            dispatcher,
            factories: factories.into_iter().collect(),
            _phantom: Default::default(),
        }
    }

    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    pub fn into_parts(self) -> (DataFlowDispatcher, VecDeque<OF>) {
        (self.dispatcher, self.factories)
    }

    pub fn factories(self) -> VecDeque<OF> {
        self.factories
    }
}

impl<O: Send + 'static, OF: OperatorFactory<O> + Send + 'static> OperatorSpec<O, OF> {
    pub fn execute(self) -> DataFlowHandle<O> {
        let (tx, rx) = mpsc_channel();
        let (err_tx, err_rx) = std::sync::mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waker = self.dispatcher.waker().clone();
        self.dispatcher
            .push_data_flow(self.factories.into_iter().map(|f| {
                let tx = tx.clone();
                let build = Box::new(move || Box::new(f).build(tx));
                DataFlowBuilder::new(build, cancelled.clone(), err_tx.clone())
            }));
        // Close our local copies of the senders so the channels close once
        // every worker drops theirs.
        drop(err_tx);

        let (rx, _) = rx.into_parts();
        DataFlowHandle::new(rx, err_rx, cancelled, waker)
    }

    /// Run the dataflow and drain every produced item into a `Vec`. Shortcut
    /// for `self.execute().collect()`.
    pub fn collect(self) -> crate::data_flow::Result<Vec<O>> {
        self.execute().collect()
    }

    /// Append one unary stage to the pipeline: one operator per worker, the
    /// `i`th reading through `channels[i]` and transforming with `unaries[i]`.
    /// Both vecs must have one entry per worker.
    ///
    /// This is the general stage-builder that out-of-crate pipelines (e.g.
    /// `catalog`'s Parquet reader) compose, and that [`map_each`](Self::map_each)
    /// is a special case of. Choosing
    /// the channel kind lets a stage fan out work ([`stealable`]) or pin items
    /// to a worker (`return_to_worker_mpsc`); per-worker `unaries` let a
    /// pipeline breaker give one worker a distinct role (e.g. the receiver end
    /// of a worker-0 merge). The `siblings_left` finishing counter is set up for
    /// you.
    pub fn chain<O2, UF, C>(
        self,
        channels: Vec<C>,
        unaries: Vec<UF>,
    ) -> OperatorSpec<O2, UnaryOperatorFactory<O, O2, UF, C, OF>>
    where
        O2: Send + 'static,
        UF: UnaryFactory<O, O2>,
        C: ChannelFactory<O>,
    {
        let siblings_left = Arc::new(AtomicUsize::new(self.factories.len()));
        let factories: Vec<_> = channels
            .into_iter()
            .zip(unaries)
            .zip(self.factories)
            .map(|((channel, unary), head)| {
                UnaryOperatorFactory::new(head, unary, channel, siblings_left.clone())
            })
            .collect();
        OperatorSpec::new(self.dispatcher, factories)
    }

    /// Append a parallel 1→1 map stage: every item is transformed by `f` on
    /// whatever worker handles it. The stage reads through a work-stealing
    /// channel, so items rebalance across idle workers — the basis of a
    /// perfectly parallel job pipeline (e.g. [`values_input`] → `map_each` to
    /// encode one Parquet page per item).
    ///
    /// A thin wrapper over [`chain`](Self::chain): a [`stealable`] channel and
    /// `f` cloned once per worker (so `f` must be `Clone` — capture only
    /// cheap/shared state).
    #[allow(clippy::type_complexity)] // the fully-spelled builder factory type is the point
    pub fn map_each<O2, F>(
        self,
        f: F,
    ) -> OperatorSpec<O2, UnaryOperatorFactory<O, O2, MapFactory<F>, StealableChannelFactory<O>, OF>>
    where
        O2: Send + 'static,
        F: FnMut(O) -> O2 + Clone + Send + 'static,
    {
        let worker_count = self.factories.len();
        let channels: Vec<_> = stealable::<O>(worker_count).into_iter().collect();
        let unaries: Vec<_> = (0..worker_count).map(|_| MapFactory(f.clone())).collect();
        self.chain(channels, unaries)
    }
}

/// Source: stream a fixed, in-memory set of values across the worker pool.
///
/// The items are loaded into a shared work-stealing queue and fan out across
/// all workers exactly like a table scan distributes row groups — the
/// in-memory, any-type counterpart of `table_input`.
/// Chain [`map_each`](OperatorSpec::map_each) to process each item in parallel.
#[allow(clippy::type_complexity)]
pub fn values_input<T: Send + 'static>(
    dispatcher: &DataFlowDispatcher,
    items: impl IntoIterator<Item = T>,
) -> OperatorSpec<
    T,
    RootUnaryOperatorFactory<T, T, DefaultUnaryFactory<Forward<T>>, InjectorSourceFactory<T>>,
> {
    let worker_count = dispatcher.worker_count();
    let injector = InjectorSourceFactory::new(items);
    let siblings_left = Arc::new(AtomicUsize::new(worker_count));
    let factories: Vec<_> = (0..worker_count)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<Forward<T>>::new(),
                injector.clone(),
                siblings_left.clone(),
            )
        })
        .collect();
    OperatorSpec::new(dispatcher.clone(), factories)
}

impl<OF: OperatorFactory<RecordBatch> + 'static> OperatorSpec<RecordBatch, OF> {
    /// Re-enter [`RecordBatchOperatorSpec`](super::record_batch_operator::RecordBatchOperatorSpec) so the RB-only fluent methods
    /// (`.count()`, `.order_by_limit()`, `.group_by_count()`, …) can be
    /// chained after a `.map()` whose output type is `RecordBatch`.
    pub fn record_batches(self) -> super::record_batch_operator::RecordBatchOperatorSpec {
        super::record_batch_operator::RecordBatchOperatorSpec::from_spec(self)
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
