//! A [`RecordBatchOperatorSpec`] represents a parallel operator running over Arrow
//! [`RecordBatch`]es. A dataflow is built with it in a "fluent API" style — start
//! with [`table_input`], chain operations, and call [`.collect()`](RecordBatchOperatorSpec::collect)
//! to execute:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! // SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'
//! let results = table_input(&table, Projection::columns([0]), false)
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
use crate::api::builder::DataFlowBuilder;
use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::dispatcher;
use crate::operations::channels::{
    ChannelFactory, MpscSender, Sender, StealableChannelFactory, mpsc_channel, stealable,
};
use crate::operations::parquet::types::projection::Projection;
use crate::operations::parquet::{
    MaterializerFactory, ParquetTable, RowGroupFetcherFactory, RowGroupInjectorFactory,
    RowGroupRequest,
};
use crate::operations::{
    CountFactory, FilterFactory, GroupFactory, KeyExtractor, NullaryFactory,
    NullaryOperatorFactory, OrderBy, OrderByLimitFactory, ProjectFactory, RootUnaryOperatorFactory,
    UnaryFactory, UnaryOperator, UnaryOperatorFactory,
};
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

/// Bridges a `Box<dyn RecordBatchOperatorFactory>` back into `OperatorFactory<RecordBatch>`.
///
/// Used by [`RecordBatchOperatorSpec::materialize`] to feed type-erased factories back
/// into the generic [`UnaryOperatorFactory`] / [`OperatorSpec::read_parquet`] pipeline.
/// Uses [`Any`] downcasting to dispatch the sender to the correct build method at runtime.
struct RecordBatchFactoryBridge(Box<dyn RecordBatchOperatorFactory>);

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
/// Start with [`table_input`] to create a spec from a parquet table, then chain
/// operations. Each method consumes `self` and returns a new `RecordBatchOperatorSpec`:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
/// let results = table_input(&table, Projection::columns([0]), false)
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
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
/// # let spec = table_input(&table, Projection::columns([0]), false);
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
    factories: VecDeque<Box<dyn RecordBatchOperatorFactory>>,
}

impl RecordBatchOperatorSpec {
    /// Convert a generic [`OperatorSpec`] into a type-erased `RecordBatchOperatorSpec`.
    ///
    /// Used internally by [`table_input`] and [`materialize`](Self::materialize) to
    /// erase the concrete factory types produced by the parquet pipeline.
    pub fn from_spec<OF: OperatorFactory<RecordBatch> + 'static>(
        spec: OperatorSpec<RecordBatch, OF>,
    ) -> Self {
        Self {
            factories: spec
                .factories()
                .into_iter()
                .map(|f| Box::new(f) as Box<dyn RecordBatchOperatorFactory>)
                .collect(),
        }
    }

    /// Build a `RecordBatchOperatorSpec` from per-worker nullary factories.
    ///
    /// Each factory is wrapped in a [`NullaryOperatorFactory`] and the actual nullary is
    /// built on the worker thread. This is useful for source-like or side-effect-only
    /// operators such as DDL.
    pub fn from_nullary<NF: NullaryFactory<RecordBatch>>(
        nullary_factories: impl IntoIterator<Item = NF>,
    ) -> Self {
        Self::from_spec(OperatorSpec::new(
            nullary_factories
                .into_iter()
                .map(NullaryOperatorFactory::new),
        ))
    }

    /// Append a unary (one-in, one-out) stage to the dataflow.
    ///
    /// Takes an iterator of [`UnaryFactory`] instances (one per worker) and wraps
    /// each existing factory with a [`RecordBatchUnaryOperatorFactory`].
    fn unary<UF: UnaryFactory<RecordBatch, RecordBatch>>(
        self,
        unary_factories: impl IntoIterator<Item = UF>,
    ) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let factories = stealable::<RecordBatch>()
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
        Self { factories }
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
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let spec = table_input(&table, Projection::columns([0]), false);
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
        self.unary((0..dispatcher().workers()).map(|_| FilterFactory(builder())))
    }

    /// Transform each batch by projecting or computing new columns.
    ///
    /// Same two-level closure pattern as [`filter`](Self::filter): the outer closure
    /// is called once per worker to set up state, and the inner closure is called
    /// once per `RecordBatch`, returning a new `RecordBatch`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let spec = table_input(&table, Projection::all(3), false);
    /// spec.project(|| {
    ///     // Called once per worker.
    ///     let indices = vec![0, 2];
    ///
    ///     // Called once per RecordBatch — keep only columns 0 and 2.
    ///     move |batch: &RecordBatch| batch.project(&indices).unwrap()
    /// })
    /// # ;
    /// ```
    pub fn project<P, PB>(self, builder: PB) -> Self
    where
        P: FnMut(&RecordBatch) -> RecordBatch + Send + 'static,
        PB: Fn() -> P,
    {
        self.unary((0..dispatcher().workers()).map(|_| ProjectFactory(builder())))
    }

    /// Count the total number of rows across all batches.
    ///
    /// Each worker maintains a local count, then the workers coordinate to produce
    /// a single output batch with the total.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// let results = table_input(&table, Projection::columns([0]), false)
    ///     .count()
    ///     .collect();
    /// // results contains a single RecordBatch with one row: the count
    /// ```
    pub fn count(self) -> Self {
        self.unary(CountFactory::create_for_workers(dispatcher().workers()))
    }

    /// Sort by the given columns and keep only the first `limit` rows.
    ///
    /// Workers independently collect their top-`limit` rows, then coordinate to
    /// produce the global top-`limit` result.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let spec = table_input(&table, Projection::columns([0]), false);
    /// // Top 10 rows ordered by column 0 descending
    /// spec.order_by_limit(vec![OrderBy::new(0, true, false)], 10)
    /// # ;
    /// ```
    pub fn order_by_limit(self, order_by: Vec<OrderBy>, limit: usize) -> Self {
        self.unary(OrderByLimitFactory::create_for_workers(
            order_by,
            limit,
            dispatcher().workers(),
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
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// # let spec = table_input(&table, Projection::columns([0]), false);
    /// // GROUP BY column 0 (string), COUNT(*)
    /// spec.group_by_count::<StringKeyExtractor>(0)
    /// # ;
    /// ```
    pub fn group_by_count<K: KeyExtractor>(self, group_column: usize) -> Self {
        self.unary(GroupFactory::<K>::create_for_workers(
            group_column,
            dispatcher().workers(),
        ))
    }

    /// Materialize additional columns from the underlying parquet table.
    ///
    /// Takes the current RecordBatch results (which may have been filtered/sorted),
    /// looks up the corresponding row groups, and reads the full `projection` from
    /// disk. This is used for late materialization: first filter on a few columns,
    /// then fetch the rest only for the surviving rows.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// table_input(&table, Projection::columns([0, 13]), true)  // read EventTime + URL
    ///     .filter(|| {
    ///         let mut contains = Contains::new("google");
    ///         move |batch: &RecordBatch| {
    ///             let col = batch.column(0).as_any()
    ///                 .downcast_ref::<StringViewArray>().unwrap();
    ///             contains.run(col)
    ///         }
    ///     })
    ///     .order_by_limit(vec![OrderBy::new(0, false, false)], 10)
    ///     .materialize(table.clone(), Projection::all(105))     // fetch all 105 columns
    ///     .collect();
    /// ```
    pub fn materialize(mut self, table: Arc<ParquetTable>, projection: Projection) -> Self {
        let siblings_left_materializer = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let siblings_left_fetcher = Arc::new(AtomicUsize::new(dispatcher().workers()));

        let materializer_spec = OperatorSpec::new(
            stealable::<RecordBatch>()
                .into_iter()
                .zip(stealable::<RowGroupRequest>())
                .map(|(rb_ch, rq_ch)| {
                    UnaryOperatorFactory::new(
                        UnaryOperatorFactory::new(
                            RecordBatchFactoryBridge(self.factories.pop_front().unwrap()),
                            MaterializerFactory::new(projection.clone(), table.clone()),
                            rb_ch,
                            siblings_left_materializer.clone(),
                        ),
                        RowGroupFetcherFactory::new(),
                        rq_ch,
                        siblings_left_fetcher.clone(),
                    )
                }),
        );

        Self::from_spec(materializer_spec.read_parquet(
            &table,
            projection,
            RECORD_BATCH_SIZE,
            false,
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
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
    /// let batches: Vec<RecordBatch> = table_input(&table, Projection::columns([0]), false)
    ///     .filter(|| {
    ///         let mut contains = Contains::new("google");
    ///         move |batch: &RecordBatch| {
    ///             let col = batch.column(0).as_any()
    ///                 .downcast_ref::<StringViewArray>().unwrap();
    ///             contains.run(col)
    ///         }
    ///     })
    ///     .collect();
    /// ```
    pub fn collect(self) -> Vec<RecordBatch> {
        let (tx, rx) = mpsc_channel();
        dispatcher().push_data_flow(
            self.factories
                .into_iter()
                .map(|f| DataFlowBuilder::new(f, tx.clone())),
        );
        drop(tx);
        let (rx, _) = rx.into_parts();
        rx.into_iter().collect::<Vec<RecordBatch>>()
    }
}

/// Create a [`RecordBatchOperatorSpec`] that reads from a parquet table.
///
/// This is the starting point for building a query. It sets up the full parquet read
/// pipeline (row group injection, fetching, indexing, decompression, decoding) and
/// erases it into a `RecordBatchOperatorSpec`.
///
/// # Arguments
///
/// * `table` — The parquet table to read from.
/// * `projection` — Which columns to read. Use [`Projection::columns`] for specific
///   column indices, [`Projection::all`] for all columns, or
///   [`Projection::from_field_names`] for columns by name.
/// * `add_row_group_metadata` — If `true`, adds row group metadata to each output batch.
///   Required when using [`materialize`](RecordBatchOperatorSpec::materialize) later.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
/// // Read columns "URL" and "EventTime" from a parquet table
/// let spec = table_input(
///     &table,
///     Projection::from_field_names(table.schema(), ["URL", "EventTime"]),
///     false,
/// );
/// let results = spec.filter(|| {
///     let mut contains = Contains::new("google");
///     move |batch: &RecordBatch| {
///         let col = batch.column(0).as_any()
///             .downcast_ref::<StringViewArray>().unwrap();
///         contains.run(col)
///     }
/// }).collect();
/// ```
pub fn table_input(
    table: &Arc<ParquetTable>,
    projection: Projection,
    add_row_group_metadata: bool,
) -> RecordBatchOperatorSpec {
    let injector = RowGroupInjectorFactory::new(table, projection.clone());
    let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));
    let input = OperatorSpec::new((0..dispatcher().workers()).map(|_| {
        RootUnaryOperatorFactory::new(
            RowGroupFetcherFactory::new(),
            injector.clone(),
            siblings_left.clone(),
        )
    }));
    RecordBatchOperatorSpec::from_spec(input.read_parquet(
        table,
        projection,
        RECORD_BATCH_SIZE,
        add_row_group_metadata,
    ))
}
