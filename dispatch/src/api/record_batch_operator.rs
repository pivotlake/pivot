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
//! # let dispatch = Dispatch::spin_up(1, 32, None);
//! # let dispatcher = dispatch.dispatcher();
//! // SELECT COUNT(*) FROM events WHERE url LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: RecordBatch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             let mask = contains.run(col);
//!             filter_record_batch(&batch, &mask).unwrap()
//!         }
//!     })
//!     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
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
use std::sync::atomic::{AtomicBool, AtomicUsize};

use arrow_array::RecordBatch;
use crossbeam_deque::Worker;

use crate::api::OperatorGraphBuilder;
use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::operations::channels::{
    ChannelFactory, MpscSender, Sender, StealableChannelFactory, stealable,
};
use crate::operations::{
    AggregateFactory, AggregationSlot, AggregationValue, CopyOutFactory, Distinct,
    DynamicFilterSlot, F64Cell, FilterFactory, GroupFactory, GroupLimit, IntCell,
    JoinOutputColumns, KeyExtractor, LimitFactory, MapFactory, NullaryFactory,
    NullaryOperatorFactory, OrderBy, OrderByLimitFactory, UnaryFactory, UnaryOperator,
    UnaryOperatorFactory, create_join_factories,
};
use crate::{DataFlowDispatcher, DataFlowHandle, DataFlowStats};
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
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> OperatorGraphBuilder;

    /// Build the operator chain, outputting to an mpsc channel.
    /// Called for the final stage by [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build).
    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> OperatorGraphBuilder;
}

impl<T: OperatorFactory<RecordBatch>> RecordBatchOperatorFactory for T {
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> OperatorGraphBuilder {
        self.build(sender)
    }
    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> OperatorGraphBuilder {
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
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> OperatorGraphBuilder {
        let (tx, rx) = self.channel_factory.build();
        let chain = self.head.build_stealable(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> OperatorGraphBuilder {
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

struct DiscardSender;

impl Sender<()> for DiscardSender {
    fn send(&mut self, _item: ()) -> crate::operations::channels::Result<()> {
        Ok(())
    }
}

/// Builds the probe result path and the disconnected build path into one
/// per-worker operator graph.
struct JoinRecordBatchOperatorFactory<BF, PF> {
    probe_head: Box<dyn RecordBatchOperatorFactory>,
    build_head: Box<dyn RecordBatchOperatorFactory>,
    build_factory: BF,
    probe_factory: PF,
    build_channel_factory: StealableChannelFactory<RecordBatch>,
    probe_channel_factory: StealableChannelFactory<RecordBatch>,
    build_siblings_left: Arc<AtomicUsize>,
    probe_siblings_left: Arc<AtomicUsize>,
    build_ready: Arc<AtomicBool>,
}

impl<BF, PF> JoinRecordBatchOperatorFactory<BF, PF>
where
    BF: UnaryFactory<RecordBatch, ()>,
    PF: UnaryFactory<RecordBatch, RecordBatch>,
{
    fn build_graph<S: Sender<RecordBatch> + 'static>(self, sender: S) -> OperatorGraphBuilder {
        let (probe_tx, probe_rx) = self.probe_channel_factory.build();
        let probe_graph = self
            .probe_head
            .build_stealable(probe_tx)
            .gated_by(self.build_ready)
            .with(Box::new(UnaryOperator::new(
                self.probe_factory.build_unary(),
                probe_rx,
                sender,
                self.probe_siblings_left,
            )));

        let (build_tx, build_rx) = self.build_channel_factory.build();
        let build_graph =
            self.build_head
                .build_stealable(build_tx)
                .with(Box::new(UnaryOperator::new(
                    self.build_factory.build_unary(),
                    build_rx,
                    DiscardSender,
                    self.build_siblings_left,
                )));

        probe_graph.with_side_graph(build_graph)
    }
}

impl<BF, PF> RecordBatchOperatorFactory for JoinRecordBatchOperatorFactory<BF, PF>
where
    BF: UnaryFactory<RecordBatch, ()>,
    PF: UnaryFactory<RecordBatch, RecordBatch>,
{
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> OperatorGraphBuilder {
        self.build_graph(sender)
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> OperatorGraphBuilder {
        self.build_graph(sender)
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
    fn build<S: Sender<RecordBatch> + 'static>(self: Box<Self>, sender: S) -> OperatorGraphBuilder {
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
/// # let dispatch = Dispatch::spin_up(1, 32, None);
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
///     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
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
/// # let dispatch = Dispatch::spin_up(1, 32, None);
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
        let factories = stealable::<RecordBatch>(self.dispatcher.topology())
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

    /// Filter rows from each batch.
    ///
    /// Takes an **outer builder closure** (`FB`) that is called once per worker thread
    /// during setup. The builder returns an **inner closure** (`F`) that is called once
    /// per `RecordBatch` during execution, returning the batch with only the rows to
    /// keep. Evaluating the condition and applying it is left to the closure, so it can
    /// narrow the batch progressively instead of always materializing a full mask.
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
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// spec.filter(|| {
    ///     // Called once per worker — allocate per-worker state here.
    ///     let mut contains = Contains::new("google");
    ///
    ///     // Called once per RecordBatch on this worker.
    ///     move |batch: RecordBatch| {
    ///         let col = batch.column(0).as_any()
    ///             .downcast_ref::<StringViewArray>().unwrap();
    ///         let mask = contains.run(col);
    ///         filter_record_batch(&batch, &mask).unwrap()
    ///     }
    /// })
    /// # ;
    /// ```
    pub fn filter<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(RecordBatch) -> RecordBatch + Send + 'static,
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
    /// `.aggregate()` / `.order_by_limit()`, call
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
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
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
        let factories: Vec<_> = stealable::<RecordBatch>(self.dispatcher.topology())
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
    /// chaining RB-only methods like `.aggregate()` / `.order_by_limit()`.
    ///
    /// Equivalent to `self.map(builder).record_batches()`.
    pub fn project<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(RecordBatch) -> RecordBatch + Send + 'static,
        FB: Fn() -> F,
    {
        self.map(builder).record_batches()
    }

    /// Global aggregates (no GROUP BY): one or more `SUM`/`COUNT` slots over
    /// columns, computed in a single pass. Emits one single-row output column
    /// per slot (`Decimal128(38, 0)` for SUM, `Int64` for COUNT). `AVG` arrives
    /// pre-lowered to a SUM slot + a COUNT slot with a downstream divide.
    pub fn aggregate<A: IntCell + F64Cell>(self, slots: Vec<AggregationSlot>) -> Self {
        let worker_count = self.worker_count();
        self.unary(AggregateFactory::<A>::create_for_workers(
            slots,
            worker_count,
        ))
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
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
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

    /// SQL `LIMIT … OFFSET …` with no ORDER BY: keep `limit` rows after skipping
    /// the first `offset`, in arbitrary (input) order.
    ///
    /// Unlike an `ORDER BY … LIMIT`, this terminates early: once `limit + offset`
    /// rows have been buffered across all workers it stops, emits, and abandons
    /// the scan feeding it (see [`LimitFactory`] and the `limit` operator module).
    /// So `SELECT * FROM huge LIMIT 1` reads only as far as the first row group,
    /// not the whole table.
    pub fn limit(self, limit: usize, offset: usize) -> Self {
        let worker_count = self.worker_count();
        self.unary(LimitFactory::create_for_workers(
            limit,
            offset,
            worker_count,
        ))
    }

    /// Global `COUNT(DISTINCT x)`: GROUP BY `key_cols` with no aggregate, emitting
    /// only each hash partition's distinct-key count (one `Int64` row) rather than
    /// the keys. Partitions are hash-disjoint, so a downstream global `SUM` over
    /// those rows yields the total — without materialising the (potentially huge)
    /// key column. Pair with a keys-only extractor (e.g. `HashOnlyIntKeyExtractor`)
    /// for an 8-byte entry.
    pub fn group_by_distinct_count<K: KeyExtractor<Config: Default>>(
        self,
        key_cols: Vec<usize>,
    ) -> Self {
        let topology = self.dispatcher.topology();
        let buffers = self.dispatcher.buffers;
        self.unary(GroupFactory::<K, Distinct>::create_for_workers(
            key_cols,
            Vec::new(),
            K::Config::default(),
            None,
            true,
            topology,
            buffers,
        ))
    }

    /// GROUP BY one or more key columns computing one or more aggregate value
    /// slots (`COUNT(*)`/`SUM`/`COUNT(col)`) per group. `K` selects the key shape,
    /// `V` the aggregate shape (e.g. its arity). `key_config` configures the key
    /// extractor (e.g. a [`RowKeySchema`](crate::RowKeySchema)); pass `()` for the
    /// extractors whose key shape is fully determined by their type.
    pub fn group_by_aggregate<K: KeyExtractor, V: AggregationValue>(
        self,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        output_limit: Option<GroupLimit>,
        key_config: K::Config,
    ) -> Self {
        let topology = self.dispatcher.topology();
        let buffers = self.dispatcher.buffers;
        self.unary(GroupFactory::<K, V>::create_for_workers(
            key_cols,
            value_slots,
            key_config,
            output_limit,
            false,
            topology,
            buffers,
        ))
    }

    /// Inner hash equi-join: build a hash table from `build`'s rows keyed on
    /// `build_key_column` (`Int64`), then probe it with `self`'s rows keyed on
    /// `probe_key_column`, emitting one output row per matching pair. Each
    /// output row is the probe columns listed in `output_columns` (in list
    /// order) followed by the listed build columns. Rows with a null key on
    /// either side never match.
    ///
    /// Both sides are roots in one dataflow. Probe input may be produced while
    /// the build runs, but the probe operator does not consume it until the
    /// completed build table is published.
    pub fn join(
        self,
        build: RecordBatchOperatorSpec,
        build_key_column: usize,
        probe_key_column: usize,
        output_columns: JoinOutputColumns,
    ) -> Self {
        let worker_count = self.worker_count();
        assert_eq!(
            worker_count,
            build.worker_count(),
            "join inputs must use the same worker count"
        );
        assert!(
            Arc::ptr_eq(self.dispatcher.waker(), build.dispatcher.waker()),
            "join inputs must use the same worker pool"
        );

        let (build_factories, probe_factories, build_ready) = create_join_factories(
            build_key_column,
            probe_key_column,
            output_columns,
            worker_count,
        );

        let (_, build_heads) = build.into_parts();
        let build_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let probe_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let build_channels = stealable::<RecordBatch>(worker_count);
        let probe_channels = stealable::<RecordBatch>(worker_count);
        let factories = self
            .factories
            .into_iter()
            .zip(build_heads)
            .zip(build_factories)
            .zip(probe_factories)
            .zip(build_channels)
            .zip(probe_channels)
            .map(
                |(
                    (
                        (((probe_head, build_head), build_factory), probe_factory),
                        build_channel_factory,
                    ),
                    probe_channel_factory,
                )| {
                    Box::new(JoinRecordBatchOperatorFactory {
                        probe_head,
                        build_head,
                        build_factory,
                        probe_factory,
                        build_channel_factory,
                        probe_channel_factory,
                        build_siblings_left: build_siblings_left.clone(),
                        probe_siblings_left: probe_siblings_left.clone(),
                        build_ready: build_ready.clone(),
                    }) as Box<dyn RecordBatchOperatorFactory>
                },
            )
            .collect();

        Self {
            dispatcher: self.dispatcher,
            factories,
        }
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
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
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

    /// Like [`execute`](Self::execute) but with per-dataflow stats collection on;
    /// read them back via [`DataFlowHandle::collect_with_stats`].
    pub fn execute_with_stats(self) -> DataFlowHandle<RecordBatch> {
        let factories: Vec<_> = self
            .factories
            .into_iter()
            .map(RecordBatchFactoryBridge)
            .collect();
        OperatorSpec::new(self.dispatcher, factories).execute_with_stats()
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

    /// Like [`collect`](Self::collect), but also returns the dataflow's IO/CPU
    /// stats folded across workers.
    pub fn collect_with_stats(self) -> crate::data_flow::Result<(Vec<RecordBatch>, DataFlowStats)> {
        let count = self.worker_count();
        self.unary((0..count).map(|_| CopyOutFactory))
            .execute_with_stats()
            .collect_with_stats()
    }

    /// Append the `CopyOut` cap (like [`collect`](Self::collect)) and launch the
    /// dataflow, returning the running [`DataFlowHandle`] instead of collecting
    /// here. Lets the caller drive collection itself and cancel mid-run via
    /// [`DataFlowHandle::cancel_token`] - the batches it yields are heap-backed,
    /// safe to hold on any thread.
    pub fn execute_copying(self) -> DataFlowHandle<RecordBatch> {
        let count = self.worker_count();
        self.unary((0..count).map(|_| CopyOutFactory)).execute()
    }
}
