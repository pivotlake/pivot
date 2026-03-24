use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::{BooleanArray, RecordBatch};
use crate::api::builder::DataFlowBuilder;
use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::dispatcher;
use crate::operations::channels::{mpsc_channel, stealable};
use crate::operations::parquet::types::projection::Projection;
use crate::operations::parquet::{
    MaterializerFactory, ParquetTable, RowGroupFetcherFactory, RowGroupInjectorFactory,
    RowGroupRequest,
};
use crate::operations::{
    create_join_factories, BinaryFactory, ConcatFactory, CountFactory, FilterFactory, GroupFactory, KeyExtractor,
    OrderBy, OrderByLimitFactory, ProjectFactory, RootUnaryOperatorFactory,
    UnaryFactory, UnaryOperatorFactory,
};

use super::factory::{JoinOperatorFactory, RecordBatchBinaryOperatorFactory, RecordBatchFactoryBridge, RecordBatchUnaryOperatorFactory};
use super::{RecordBatchOperatorFactory, RECORD_BATCH_SIZE};

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

    /// Hash join: `self` is the probe side, `build` is the build side.
    ///
    /// Both pipelines are combined into a single dataflow. The build side constructs
    /// the hash table (output discarded), and the probe side looks up matches
    /// (output becomes this operator's output).
    pub fn join(
        self,
        build: RecordBatchOperatorSpec,
        build_key_column: usize,
        probe_key_column: usize,
    ) -> RecordBatchOperatorSpec {
        let workers = dispatcher().workers();
        let (build_factories, probe_factories, gate) =
            create_join_factories(build_key_column, probe_key_column, workers);
        let build_siblings_left = Arc::new(AtomicUsize::new(workers));
        let probe_siblings_left = Arc::new(AtomicUsize::new(workers));

        let factories = stealable::<RecordBatch>()
            .into_iter()
            .zip(stealable::<RecordBatch>())
            .zip(build_factories)
            .zip(probe_factories)
            .zip(build.factories)
            .zip(self.factories)
            .map(
                |(((((build_ch, probe_ch), bf), pf), build_head), probe_head)| {
                    Box::new(JoinOperatorFactory {
                        build_head,
                        probe_head,
                        build_factory: bf,
                        probe_factory: pf,
                        build_channel: build_ch,
                        probe_channel: probe_ch,
                        build_siblings_left: build_siblings_left.clone(),
                        probe_siblings_left: probe_siblings_left.clone(),
                        gate: gate.clone(),
                    }) as Box<dyn RecordBatchOperatorFactory>
                },
            )
            .collect();
        Self { factories }
    }

    /// Append a binary (two-in, one-out) stage to the dataflow.
    ///
    /// Combines `self` (left input) and `other` (right input) through a
    /// [`BinaryFactory`]-produced operator.
    fn binary<BF: BinaryFactory<RecordBatch, RecordBatch, RecordBatch>>(
        self,
        other: RecordBatchOperatorSpec,
        binary_factories: impl IntoIterator<Item = BF>,
    ) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher().workers()));
        let factories = stealable::<RecordBatch>()
            .into_iter()
            .zip(stealable::<RecordBatch>())
            .zip(binary_factories)
            .zip(self.factories)
            .zip(other.factories)
            .map(|((((left_ch, right_ch), bf), left_head), right_head)| {
                Box::new(RecordBatchBinaryOperatorFactory {
                    left_head,
                    right_head,
                    binary_factory: bf,
                    left_channel_factory: left_ch,
                    right_channel_factory: right_ch,
                    siblings_left: siblings_left.clone(),
                }) as Box<dyn RecordBatchOperatorFactory>
            })
            .collect();
        Self { factories }
    }

    /// Concatenate two dataflows into one.
    ///
    /// All batches from both `self` and `other` are forwarded downstream.
    /// The output order is not guaranteed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table1 = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp/a")).unwrap());
    /// # let table2 = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp/b")).unwrap());
    /// let left = table_input(&table1, Projection::columns([0]), false);
    /// let right = table_input(&table2, Projection::columns([0]), false);
    /// let results = left.concat(right).count().collect();
    /// ```
    pub fn concat(self, other: RecordBatchOperatorSpec) -> Self {
        self.binary(other, (0..dispatcher().workers()).map(|_| ConcatFactory))
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
