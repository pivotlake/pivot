//! A [`RecordBatchOperatorSpec`] represents a parallel operator running over Arrow
//! [`RecordBatch`]es. A dataflow is built with it in a "fluent API" style — start
//! with `table_input`, chain operations, and call [`.collect()`](RecordBatchOperatorSpec::collect)
//! to execute:
//!
//! ```ignore
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! # let dispatch = Dispatch::spin_up(1, 32);
//! # let dispatcher = dispatch.dispatcher();
//! // SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &RecordBatch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             contains.run(col)
//!         }
//!     })
//!     .count()
//!     .collect();
//! ```
//!
//! Operations like [`.filter()`](RecordBatchOperatorSpec::filter) and
//! [`.project()`](RecordBatchOperatorSpec::project) take a **builder closure**
//! (`Fn() -> F`) that is called once per worker thread, so each worker gets its own
//! independent operator instance with private mutable state — no synchronization needed.
//!
//! Nothing executes until [`.collect()`](RecordBatchOperatorSpec::collect) is called.
//! At that point the spec is shipped to worker threads, built into operator chains,
//! and run in parallel. Results are collected into `Vec<RecordBatch>`.
//!
//! # Internals
//!
//! Factories are stored as `Box<dyn RecordBatchOperatorFactory>` — an object-safe trait
//! that wraps the generic [`OperatorFactory<O>`]. This keeps the return type of every
//! chained method as plain `RecordBatchOperatorSpec`, rather than deeply nested generics.
//! See [`RecordBatchOperatorFactory`] and [`operator_spec`](super::operator_spec) for
//! details on why this split exists.

use std::any::Any;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::{BooleanArray, RecordBatch};
use crossbeam_deque::Worker;

use crate::api::Chain;
use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::operations::channels::{
    ChannelFactory, MpscSender, Sender, StealableChannelFactory, stealable,
};
use crate::operations::{
    AggSpec, AggregateFactory, AggregationKind, AggregationSlot, Compiled, CopyOutFactory, Count,
    CountFactory, DynamicFilterSlot, FilterFactory, GroupFactory, KeyExtractor, MapFactory,
    Nullary, NullaryFactory, NullaryOperatorFactory, OrderBy, OrderByLimitFactory, UnaryFactory,
    UnaryOperator, UnaryOperatorFactory, ValueExtractor,
};
use crate::{DataFlowDispatcher, DataFlowHandle, NullaryResult, WorkStatus};
pub const RECORD_BATCH_SIZE: usize = 8192;

/// Object-safe version of [`OperatorFactory<RecordBatch>`].
///
/// [`OperatorFactory::build`] is generic over `S: Sender<O>`, which prevents it from
/// being used as a trait object. This trait replaces that single generic method with
/// two concrete methods — one per sender type used at the `RecordBatch` boundary:
///
/// - [`build_stealable`](RecordBatchOperatorFactory::build_stealable) — called when this
///   factory is the head of another stage, connected via a work-stealing channel.
/// - [`build_collect`](RecordBatchOperatorFactory::build_collect) — called for the final
///   stage, which sends results to the output mpsc channel.
///
/// A blanket impl automatically implements this for any `T: OperatorFactory<RecordBatch>`,
/// so generic factories from the parquet pipeline can be erased into
/// `Box<dyn RecordBatchOperatorFactory>` without manual wrapping.
pub trait RecordBatchOperatorFactory: Send {
    /// Build the operator chain, outputting to a work-stealing channel.
    /// Called when this factory is an intermediate stage (the head of another operator).
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain;

    /// Build the operator chain, outputting to an mpsc channel.
    /// Called for the final stage by [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build).
    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain;
}

impl<T: OperatorFactory<RecordBatch>> RecordBatchOperatorFactory for T {
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain {
        self.build(sender)
    }
    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain {
        self.build(sender)
    }
}

/// Type-erased unary operator factory for `RecordBatch -> RecordBatch` stages.
///
/// Wraps a type-erased head (`Box<dyn RecordBatchOperatorFactory>`) with a concrete
/// unary operation (e.g. filter, project, count). The `UF` type parameter is the
/// concrete `UnaryFactory` — it gets erased when this struct is boxed as
/// `Box<dyn RecordBatchOperatorFactory>`.
///
/// Created internally by `RecordBatchOperatorSpec::unary` — not constructed directly.
pub struct RecordBatchUnaryOperatorFactory<UF: UnaryFactory<RecordBatch, RecordBatch>> {
    head: Box<dyn RecordBatchOperatorFactory>,
    unary_factory: UF,
    channel_factory: StealableChannelFactory<RecordBatch>,
    siblings_left: Arc<AtomicUsize>,
}

impl<UF: UnaryFactory<RecordBatch, RecordBatch>> RecordBatchOperatorFactory
    for RecordBatchUnaryOperatorFactory<UF>
{
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = self.head.build_stealable(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = self.head.build_stealable(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }
}

/// Bridges a `Box<dyn RecordBatchOperatorFactory>` into `OperatorFactory<RecordBatch>`,
/// so the type-erased `RecordBatch` head can be slotted into any generic
/// `UnaryOperatorFactory<RecordBatch, ..., RecordBatchFactoryBridge>`.
///
/// Uses [`Any`] downcasting at build time (once per worker, not per batch) to
/// dispatch the now-erased sender to the correct concrete build method on
/// the underlying [`RecordBatchOperatorFactory`].
pub struct RecordBatchFactoryBridge(Box<dyn RecordBatchOperatorFactory>);

impl RecordBatchFactoryBridge {
    /// Wrap a type-erased `RecordBatch` factory so it can head a generic
    /// [`UnaryOperatorFactory`] (used by out-of-crate late materialization).
    pub fn new(factory: Box<dyn RecordBatchOperatorFactory>) -> Self {
        Self(factory)
    }
}

impl OperatorFactory<RecordBatch> for RecordBatchFactoryBridge {
    fn build<S: Sender<RecordBatch> + 'static>(self: Box<Self>, sender: S) -> Chain {
        let sender_any: Box<dyn Any> = Box::new(sender);
        match sender_any.downcast::<Rc<Worker<RecordBatch>>>() {
            Ok(s) => self.0.build_stealable(*s),
            Err(sender_any) => match sender_any.downcast::<MpscSender<RecordBatch>>() {
                Ok(s) => self.0.build_collect(*s),
                Err(_) => panic!("RecordBatchFactoryBridge: unsupported sender type"),
            },
        }
    }
}

/// RecordBatchOperatorSpec represents a parallel operator running over Arrow [`RecordBatch`]es.
///
/// Holds one factory per worker, stored as `Box<dyn RecordBatchOperatorFactory>` to
/// erase the nested generic types that would otherwise grow with each chained operation. Factories
/// that implement `Send` are sent to workers (instead of the actual operations/channels) as the
/// final channel/operation may not be `Send`.
///
/// # Building a query
///
/// Start from a source — `values_input`, `from_nullary`, or a source built on
/// `OperatorSpec` such as `catalog`'s `table_input` — then chain operations. Each
/// method consumes `self` and returns a new `RecordBatchOperatorSpec`:
///
/// ```ignore
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
/// # let dispatch = Dispatch::spin_up(1, 32);
/// # let dispatcher = dispatch.dispatcher();
/// let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
///     .filter(|| {
///         let mut contains = Contains::new("google");
///         move |batch: &RecordBatch| {
///             let col = batch.column(0).as_any()
///                 .downcast_ref::<StringViewArray>().unwrap();
///             contains.run(col)
///         }
///     })
///     .count()
///     .collect();
/// ```
///
/// # Closures are per-worker
///
/// Methods like [`filter`](Self::filter) and [`project`](Self::project) take a
/// **builder closure** (`Fn() -> F`) that is called once per worker to create
/// independent operator instances. This allows each worker's operator to hold mutable
/// state without synchronization:
///
/// ```ignore
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
/// # let dispatch = Dispatch::spin_up(1, 32);
/// # let dispatcher = dispatch.dispatcher();
/// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
/// spec.filter(|| {
///     // This closure is called N times (once per worker).
///     // Each worker gets its own `Contains` instance.
///     let mut contains = Contains::new("google");
///
///     // Called once per RecordBatch on this worker.
///     move |batch: &RecordBatch| {
///         let col = batch.column(0).as_any()
///             .downcast_ref::<StringViewArray>().unwrap();
///         contains.run(col)
///     }
/// })
/// # ;
/// ```
pub struct RecordBatchOperatorSpec {
    dispatcher: DataFlowDispatcher,
    factories: VecDeque<Box<dyn RecordBatchOperatorFactory>>,
}

/// Return type of [`RecordBatchOperatorSpec::map`]: a generic
/// [`OperatorSpec<T, _>`] wrapping the RB→T map stage.
type MapOperatorSpec<T, F> = OperatorSpec<
    T,
    UnaryOperatorFactory<
        RecordBatch,
        T,
        MapFactory<F>,
        StealableChannelFactory<RecordBatch>,
        RecordBatchFactoryBridge,
    >,
>;

impl RecordBatchOperatorSpec {
    /// Convert a generic [`OperatorSpec`] into a type-erased `RecordBatchOperatorSpec`.
    ///
    /// Used by builders such as `catalog`'s `table_input` / `materialize` to erase
    /// the concrete nested factory types a multi-stage pipeline produces.
    pub fn from_spec<OF: OperatorFactory<RecordBatch> + 'static>(
        spec: OperatorSpec<RecordBatch, OF>,
    ) -> Self {
        let (dispatcher, factories) = spec.into_parts();
        Self {
            dispatcher,
            factories: factories
                .into_iter()
                .map(|f| Box::new(f) as Box<dyn RecordBatchOperatorFactory>)
                .collect(),
        }
    }

    /// Decompose into the dispatcher and per-worker factories, so out-of-crate
    /// code (e.g. late materialization in `catalog`) can chain further stages.
    pub fn into_parts(
        self,
    ) -> (
        DataFlowDispatcher,
        VecDeque<Box<dyn RecordBatchOperatorFactory>>,
    ) {
        (self.dispatcher, self.factories)
    }

    /// Build a `RecordBatchOperatorSpec` from per-worker nullary factories.
    ///
    /// Each factory is wrapped in a [`NullaryOperatorFactory`] and the actual nullary is
    /// built on the worker thread. This is useful for source-like or side-effect-only
    /// operators such as DDL.
    pub fn from_nullary<NF: NullaryFactory<RecordBatch>>(
        dispatcher: &DataFlowDispatcher,
        nullary_factories: impl IntoIterator<Item = NF>,
    ) -> Self {
        Self::from_spec(OperatorSpec::new(
            dispatcher.clone(),
            nullary_factories
                .into_iter()
                .map(NullaryOperatorFactory::new),
        ))
    }

    /// A spec that yields no rows. Useful for side-effecting / DDL statements
    /// (e.g. `CREATE TABLE`) whose result set is empty.
    pub fn empty(dispatcher: &DataFlowDispatcher) -> Self {
        Self::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoRowsNullaryFactory),
        )
    }

    /// Borrow the dispatcher this spec was built against.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// Append a unary (one-in, one-out) stage to the dataflow.
    ///
    /// Takes an iterator of [`UnaryFactory`] instances (one per worker) and wraps
    /// each existing factory with a [`RecordBatchUnaryOperatorFactory`].
    fn unary<UF: UnaryFactory<RecordBatch, RecordBatch>>(
        self,
        unary_factories: impl IntoIterator<Item = UF>,
    ) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(self.worker_count()));
        let factories = stealable::<RecordBatch>(self.worker_count())
            .into_iter()
            .zip(unary_factories)
            .zip(self.factories)
            .map(|((channel_factory, unary_factory), head)| {
                Box::new(RecordBatchUnaryOperatorFactory {
                    head,
                    unary_factory,
                    channel_factory,
                    siblings_left: siblings_left.clone(),
                }) as Box<dyn RecordBatchOperatorFactory>
            })
            .collect();
        Self {
            dispatcher: self.dispatcher,
            factories,
        }
    }

    fn worker_count(&self) -> usize {
        self.factories.len()
    }

    /// Filter rows from each batch using a boolean mask.
    ///
    /// Takes an **outer builder closure** (`FB`) that is called once per worker thread
    /// during setup. The builder returns an **inner closure** (`F`) that is called once
    /// per `RecordBatch` during execution, returning a [`BooleanArray`] mask of the
    /// same length indicating which rows to keep.
    ///
    /// This two-level pattern lets each worker own private mutable state (allocated in
    /// the builder):
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// spec.filter(|| {
    ///     // Called once per worker — allocate per-worker state here.
    ///     let mut contains = Contains::new("google");
    ///
    ///     // Called once per RecordBatch on this worker.
    ///     move |batch: &RecordBatch| {
    ///         let col = batch.column(0).as_any()
    ///             .downcast_ref::<StringViewArray>().unwrap();
    ///         contains.run(col)
    ///     }
    /// })
    /// # ;
    /// ```
    pub fn filter<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(&RecordBatch) -> BooleanArray + Send + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        self.unary((0..worker_count).map(|_| FilterFactory(builder())))
    }

    /// Apply a per-batch 1→1 transform that can change the output type.
    ///
    /// Replaces both the old `project` (RB → RB) and the old `for_each`
    /// (RB → T) — one method, generic over the output type. The closure
    /// takes an owned `RecordBatch` and returns a single `T`; every `send`
    /// downstream is monomorphized over the resulting `OperatorSpec<T, _>`.
    ///
    /// Lifts out of `RecordBatchOperatorSpec` into the generic
    /// [`OperatorSpec<T, _>`]. To chain back into RB-only methods like
    /// `.count()` / `.order_by_limit()`, call
    /// [`record_batches`](OperatorSpec::record_batches) on the result when
    /// `T = RecordBatch`.
    ///
    /// Same two-level closure pattern as [`filter`](Self::filter): the outer
    /// builder closure is called once per worker.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{Int64Array, RecordBatch};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// // RB → RB (the old `project`)
    /// spec.map(|| {
    ///     let indices = vec![0];
    ///     move |batch: RecordBatch| batch.project(&indices).unwrap()
    /// })
    /// # ;
    /// ```
    pub fn map<T, F, FB>(self, builder: FB) -> MapOperatorSpec<T, F>
    where
        T: Send + 'static,
        F: FnMut(RecordBatch) -> T + Send + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        let siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let factories: Vec<_> = stealable::<RecordBatch>(worker_count)
            .into_iter()
            .zip((0..worker_count).map(|_| MapFactory(builder())))
            .zip(self.factories)
            .map(|((channel_factory, unary_factory), head)| {
                UnaryOperatorFactory::new(
                    RecordBatchFactoryBridge(head),
                    unary_factory,
                    channel_factory,
                    siblings_left.clone(),
                )
            })
            .collect();
        OperatorSpec::new(self.dispatcher, factories)
    }

    /// Sugar for [`map`](Self::map) when the output is `RecordBatch`: applies
    /// the transform and stays in `RecordBatchOperatorSpec` so you can keep
    /// chaining RB-only methods like `.count()` / `.order_by_limit()`.
    ///
    /// Equivalent to `self.map(builder).record_batches()`.
    pub fn project<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(RecordBatch) -> RecordBatch + Send + 'static,
        FB: Fn() -> F,
    {
        self.map(builder).record_batches()
    }

    /// Count the total number of rows across all batches.
    ///
    /// Each worker maintains a local count, then the workers coordinate to produce
    /// a single output batch with the total.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
    ///     .count()
    ///     .collect();
    /// // results contains a single RecordBatch with one row: the count
    /// ```
    pub fn count(self) -> Self {
        let worker_count = self.worker_count();
        self.unary(CountFactory::create_for_workers(worker_count))
    }

    /// Global aggregates (no GROUP BY): one or more `SUM`/`COUNT`/`AVG` over
    /// columns, computed in a single pass. Emits one single-row output column
    /// per spec (`Int64` for SUM/COUNT, `Float64` for AVG).
    pub fn aggregate(self, specs: Vec<AggSpec>) -> Self {
        let worker_count = self.worker_count();
        self.unary(AggregateFactory::create_for_workers(specs, worker_count))
    }

    /// Sort by the given columns and keep only the first `limit` rows.
    ///
    /// Workers independently collect their top-`limit` rows, then coordinate to
    /// produce the global top-`limit` result.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// // Top 10 rows ordered by column 0 descending
    /// spec.order_by_limit(vec![OrderBy::new(0, true, false)], 10)
    /// # ;
    /// ```
    pub fn order_by_limit(self, order_by: Vec<OrderBy>, limit: usize) -> Self {
        self.order_by_limit_offset(order_by, limit, 0, None)
    }

    /// Like [`order_by_limit`](Self::order_by_limit) but skips the first
    /// `offset` rows of the globally sorted result (SQL `LIMIT … OFFSET`), and
    /// optionally publishes the running boundary into a shared
    /// [`DynamicFilterSlot`] so sibling scans can prune row groups.
    pub fn order_by_limit_offset(
        self,
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    ) -> Self {
        let worker_count = self.worker_count();
        self.unary(OrderByLimitFactory::create_for_workers(
            order_by,
            limit,
            offset,
            worker_count,
            dynamic_filter,
        ))
    }

    /// Group by a column and count occurrences per group.
    ///
    /// The type parameter `K` selects the key extractor for the group column
    /// (e.g. [`StringKeyExtractor`](crate::operations::StringKeyExtractor) for string
    /// columns, [`IntKeyExtractor`](crate::operations::IntKeyExtractor) for integers).
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// // GROUP BY column 0 (string), COUNT(*)
    /// spec.group_by_count::<StringKeyExtractor>(0)
    /// # ;
    /// ```
    pub fn group_by_count<K: KeyExtractor>(self, group_column: usize) -> Self {
        // `COUNT(*)` is one aggregate slot whose column is unused.
        self.group_by_aggregate::<K, Compiled<(Count,)>>(
            vec![group_column],
            vec![AggregationSlot::new(AggregationKind::CountStar, 0)],
            None,
        )
    }

    /// GROUP BY one or more key columns computing one or more aggregate value
    /// slots (`COUNT(*)`/`SUM`/`COUNT(col)`) per group. `K` selects the key
    /// shape, `V` the aggregate shape (e.g. its arity).
    pub fn group_by_aggregate<K: KeyExtractor, V: ValueExtractor>(
        self,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        top_k: Option<(usize, usize)>,
    ) -> Self {
        let worker_count = self.worker_count();
        let buffers = self.dispatcher.buffers;
        self.unary(GroupFactory::<K, V>::create_for_workers(
            key_cols,
            value_slots,
            top_k,
            worker_count,
            buffers,
        ))
    }

    /// Execute the dataflow and collect all output batches.
    ///
    /// Sends one factory per worker to the dispatcher, waits for all workers to
    /// finish, and returns the collected `RecordBatch` results.
    ///
    /// This is a blocking call — it does not return until all workers have completed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32);
    /// # let dispatcher = dispatch.dispatcher();
    /// let batches: Vec<RecordBatch> = table_input(&dispatcher, &table, Projection::columns([0]), false)
    ///     .filter(|| {
    ///         let mut contains = Contains::new("google");
    ///         move |batch: &RecordBatch| {
    ///             let col = batch.column(0).as_any()
    ///                 .downcast_ref::<StringViewArray>().unwrap();
    ///             contains.run(col)
    ///         }
    ///     })
    ///     .collect()
    ///     .unwrap();
    /// ```
    /// Dispatch the pipeline to the workers and return a control handle plus
    /// a stream of the produced batches.
    ///
    /// Batches arrive **ring-backed**: their `Buffer`s point into worker
    /// `WriteBuffer`s and are only safe to handle on a thread with a matching
    /// `MemoryContext`. For the usual case prefer
    /// [`collect`](Self::collect), which inserts a
    /// `CopyOut` cap so every
    /// batch leaves the worker as plain heap-backed buffers.
    pub fn execute(self) -> DataFlowHandle<RecordBatch> {
        let factories: Vec<_> = self
            .factories
            .into_iter()
            .map(RecordBatchFactoryBridge)
            .collect();
        OperatorSpec::new(self.dispatcher, factories).execute()
    }

    /// Run the dataflow and collect every batch into a `Vec`.
    ///
    /// Appends a `CopyOut`
    /// stage before executing, so the batches you receive are plain
    /// heap-backed (safe to hold on any thread, regardless of
    /// `MemoryContext`).
    pub fn collect(self) -> crate::data_flow::Result<Vec<RecordBatch>> {
        let count = self.worker_count();
        self.unary((0..count).map(|_| CopyOutFactory))
            .execute()
            .collect()
    }
}

/// Per-worker factory for [`NoRowsNullary`], backing
/// [`RecordBatchOperatorSpec::empty`].
struct NoRowsNullaryFactory;

impl NullaryFactory<RecordBatch> for NoRowsNullaryFactory {
    type Nullary = NoRowsNullary;

    fn build_nullary(self) -> NoRowsNullary {
        NoRowsNullary { ran: false }
    }
}

/// A nullary that emits nothing and finishes immediately — the body of an empty
/// (no-row) spec.
struct NoRowsNullary {
    ran: bool,
}

impl Nullary<RecordBatch> for NoRowsNullary {
    fn run<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;
        Ok(WorkStatus::Ran)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}
