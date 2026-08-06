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
//! Factories are stored as `Box<dyn OperatorFactory<RecordBatch>>` — an object-safe trait
//! that wraps the generic [`OperatorFactory<O>`]. This keeps the return type of every
//! chained method as plain `RecordBatchOperatorSpec`, rather than deeply nested generics.
//! See [`OperatorFactory`] and [`operator_spec`](super::operator_spec) for
//! details on why this split exists.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use arrow_schema::DataType;

use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::operations::channels::{StealableChannelFactory, stealable};
use crate::operations::{
    AggregateFactory, AggregationSlot, AggregationValue, CopyOutFactory, CteFactory,
    CteScanFactory, Distinct, DynamicFilterSlot, DynamicRowKey, F64Cell, FilterFactory,
    GroupFactory, GroupLimit, IntCell, JoinKey, JoinKind, JoinRecordBatchOperatorFactory, JoinSpec,
    KeyExtractor, LimitFactory, MapFactory, NoOpNullaryFactory, NullaryFactory,
    NullaryOperatorFactory, OrderBy, OrderByFactory, OrderByLimitFactory, PackedKey,
    SingleColumnKey, UnaryFactory, UnaryOperatorFactory, WideCell, create_join_factories,
};
use crate::{DataFlowDispatcher, DataFlowHandle, DataFlowStats};
pub const RECORD_BATCH_SIZE: usize = 8192;

/// RecordBatchOperatorSpec represents a parallel operator running over Arrow [`RecordBatch`]es.
///
/// Holds one factory per worker, stored as `Box<dyn OperatorFactory<RecordBatch>>` to
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
    factories: VecDeque<Box<dyn OperatorFactory<RecordBatch>>>,
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
        Box<dyn OperatorFactory<RecordBatch>>,
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
                .map(|f| Box::new(f) as Box<dyn OperatorFactory<RecordBatch>>)
                .collect(),
        }
    }

    /// Decompose into the dispatcher and per-worker factories, so out-of-crate
    /// code (e.g. late materialization in `catalog`) can chain further stages.
    pub fn into_parts(
        self,
    ) -> (
        DataFlowDispatcher,
        VecDeque<Box<dyn OperatorFactory<RecordBatch>>>,
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

    /// Build a spec that emits no rows and completes immediately: each worker
    /// gets a [`NoOpNullary`](crate::operations::NoOpNullary), which has no
    /// work and finishes on its first finish pass. This is the plan for a
    /// statement that turns out to have nothing to do.
    pub fn no_op(dispatcher: &DataFlowDispatcher) -> Self {
        Self::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoOpNullaryFactory),
        )
    }

    /// One scan site of a CTE: a source over every row the CTE's definition
    /// produces, in whatever order they arrive.
    ///
    /// A site reads its own channel, so several sites each see the full result
    /// rather than dividing it between them. Pair with
    /// [`with_cte`](Self::with_cte) on a spec containing the sites, which is
    /// what supplies the rows: on its own a site produces nothing and never
    /// finishes.
    pub fn cte_scan(dispatcher: &DataFlowDispatcher, cte_index: usize) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher.worker_count()));
        let factories = stealable::<RecordBatch>(dispatcher.topology())
            .into_iter()
            .map(|channel_factory| {
                Box::new(CteScanFactory {
                    channel_factory,
                    cte_index,
                    siblings_left: siblings_left.clone(),
                }) as Box<dyn OperatorFactory<RecordBatch>>
            })
            .collect();
        Self {
            dispatcher: dispatcher.clone(),
            factories,
        }
    }

    /// Run `definition` alongside this spec, feeding its rows to each of the
    /// `sites` [`cte_scan`](Self::cte_scan) sites for `cte_index` that this spec
    /// contains.
    ///
    /// The definition runs once however many sites read it, and runs
    /// concurrently with them: a site consumes batches as they are produced
    /// rather than waiting for the whole result.
    pub fn with_cte(
        self,
        definition: RecordBatchOperatorSpec,
        cte_index: usize,
        sites: usize,
    ) -> Self {
        let (_, definition_heads) = definition.into_parts();
        assert_eq!(
            self.factories.len(),
            definition_heads.len(),
            "a CTE and its definition must use the same worker count"
        );
        let factories = self
            .factories
            .into_iter()
            .zip(definition_heads)
            .map(|(body_head, definition_head)| {
                Box::new(CteFactory {
                    definition_head,
                    body_head,
                    cte_index,
                    sites,
                }) as Box<dyn OperatorFactory<RecordBatch>>
            })
            .collect();
        Self {
            dispatcher: self.dispatcher,
            factories,
        }
    }

    /// Borrow the dispatcher this spec was built against.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// Append a unary (one-in, one-out) stage to the dataflow.
    ///
    /// Takes an iterator of [`UnaryFactory`] instances (one per worker) and wraps
    /// each existing factory with a [`UnaryOperatorFactory`].
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
                Box::new(UnaryOperatorFactory::new(
                    head,
                    unary_factory,
                    channel_factory,
                    siblings_left.clone(),
                )) as Box<dyn OperatorFactory<RecordBatch>>
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
    /// per `RecordBatch` during execution, returning the batch of surviving rows. The
    /// inner closure also receives the operator'"'"'s [`SlabAllocator`](crate::memory::SlabAllocator)
    /// so survivors can be compacted into slab-backed buffers via
    /// [`take`](crate::arrays::take::take).
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
        F: FnMut(
                &RecordBatch,
                &mut crate::memory::SlabAllocator,
                &mut Vec<u32>,
            ) -> crate::RowSelection
            + Send
            + 'static,
        FB: Fn() -> F,
    {
        self.filter_with_delivery(builder, crate::RowDelivery::Coalesced)
    }

    /// [`filter`](Self::filter), choosing when survivors reach the operator
    /// below. Use [`RowDelivery::Immediate`](crate::RowDelivery::Immediate)
    /// when that operator acts on early rows, such as a LIMIT that cancels its
    /// input or a Top-N whose boundary prunes the scan; holding rows back to
    /// fill a batch defers the decision until the filter has selected a whole
    /// batch's worth, which under a selective filter reads far more of the
    /// table than the query needs.
    pub fn filter_with_delivery<F, FB>(self, builder: FB, delivery: crate::RowDelivery) -> Self
    where
        F: FnMut(
                &RecordBatch,
                &mut crate::memory::SlabAllocator,
                &mut Vec<u32>,
            ) -> crate::RowSelection
            + Send
            + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        self.unary((0..worker_count).map(|_| FilterFactory(builder(), delivery)))
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
                    head,
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
    pub fn aggregate<A: IntCell + F64Cell + WideCell>(self, slots: Vec<AggregationSlot>) -> Self {
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

    /// SQL `ORDER BY` with no LIMIT: emit the whole input sorted by
    /// `order_by`.
    ///
    /// A parallel merge sort modeled on rayon's: each worker sorts batches as
    /// they arrive (batches already in order pass through untouched), and the
    /// workers' sorted runs merge through a tree whose merges split into
    /// stealable slices, so even the final merge runs across every worker.
    /// See the `operations::unary::order_by` module.
    pub fn order_by(self, order_by: Vec<OrderBy>) -> Self {
        let worker_count = self.worker_count();
        self.unary(OrderByFactory::create_for_workers(
            Vec::new(),
            order_by,
            worker_count,
        ))
    }

    /// Like [`order_by`](Self::order_by), grouped: rows come out clustered by
    /// their `partition_columns` tuple (partitions in tuple order, every
    /// emitted batch single-partition) and sorted by `order_by` within each
    /// partition. Sorting happens per partition, so partition columns are
    /// never compared row against row. An empty `order_by` groups without
    /// ordering inside partitions.
    pub fn order_by_per_partition(
        self,
        partition_columns: Vec<usize>,
        order_by: Vec<OrderBy>,
    ) -> Self {
        let worker_count = self.worker_count();
        self.unary(OrderByFactory::create_for_workers(
            partition_columns,
            order_by,
            worker_count,
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
    pub fn group_by_aggregate<K: KeyExtractor, V: AggregationValue + ?Sized>(
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

    /// Hash equi-join: build a hash table from `build`'s rows keyed on
    /// `spec.build_key_indices`, then probe it with `self`'s rows keyed on
    /// `spec.probe_key_indices`, emitting one output row per matching pair.
    /// Each output row is the probe columns listed in
    /// `spec.probe_output_indices` followed by `spec.build_output_indices`.
    /// Rows with a null value in any key column on either side never match.
    ///
    /// `spec.kind` picks which rows reach the output: the matching pairs alone,
    /// those plus one row per build row nothing matched (with null probe
    /// columns), or one row per probe row that matched anything at all.
    ///
    /// `key_types` are the Arrow types the key columns arrive as, one per
    /// condition (the planner casts mismatched sides of each condition to a
    /// common type first); together they pick the join's [`JoinKey`]
    /// instantiation. A single key joins on any hashable fixed-width type,
    /// which excludes floats by construction.
    ///
    /// Both sides are roots in one dataflow. Probe input may be produced while
    /// the build runs, but the probe operator does not consume it until the
    /// completed build table is published.
    pub fn join(
        self,
        build: RecordBatchOperatorSpec,
        key_types: &[DataType],
        spec: JoinSpec,
    ) -> Self {
        use arrow_array::types as t;
        use arrow_schema::TimeUnit;
        assert_eq!(
            spec.build_key_indices.len(),
            key_types.len(),
            "one build key column per key type"
        );
        assert_eq!(
            spec.probe_key_indices.len(),
            key_types.len(),
            "one probe key column per key type"
        );
        assert_eq!(
            spec.probe_fields.len(),
            spec.probe_output_indices.len(),
            "one probe field per listed probe output column"
        );
        assert_eq!(
            spec.build_fields.len(),
            spec.build_output_indices.len(),
            "one build field per listed build output column"
        );
        // Multi-key shapes: the packed-lane combos compiled so far. Any int
        // pair combo (or a higher arity) is one more arm naming its
        // PackedKey tuple; the impls already exist for every arity.
        match key_types {
            [DataType::Int32, DataType::Int32] => {
                return self.join_dispatch::<PackedKey<(t::Int32Type, t::Int32Type)>>(build, spec);
            }
            [DataType::Int32, DataType::Int64] => {
                return self.join_dispatch::<PackedKey<(t::Int32Type, t::Int64Type)>>(build, spec);
            }
            [DataType::Int64, DataType::Int32] => {
                return self.join_dispatch::<PackedKey<(t::Int64Type, t::Int32Type)>>(build, spec);
            }
            [DataType::Int64, DataType::Int64] => {
                return self.join_dispatch::<PackedKey<(t::Int64Type, t::Int64Type)>>(build, spec);
            }
            _ => {}
        }
        // Every shape without a compiled instantiation: 3+ keys, mixed types,
        // strings. Hash-stored with per-candidate verification.
        let [key_type] = key_types else {
            return self.join_dispatch::<DynamicRowKey>(build, spec);
        };
        match key_type {
            DataType::Int8 => self.join_dispatch::<SingleColumnKey<t::Int8Type>>(build, spec),
            DataType::Int16 => self.join_dispatch::<SingleColumnKey<t::Int16Type>>(build, spec),
            DataType::Int32 => self.join_dispatch::<SingleColumnKey<t::Int32Type>>(build, spec),
            DataType::Int64 => self.join_dispatch::<SingleColumnKey<t::Int64Type>>(build, spec),
            DataType::UInt8 => self.join_dispatch::<SingleColumnKey<t::UInt8Type>>(build, spec),
            DataType::UInt16 => self.join_dispatch::<SingleColumnKey<t::UInt16Type>>(build, spec),
            DataType::UInt32 => self.join_dispatch::<SingleColumnKey<t::UInt32Type>>(build, spec),
            DataType::UInt64 => self.join_dispatch::<SingleColumnKey<t::UInt64Type>>(build, spec),
            DataType::Date32 => self.join_dispatch::<SingleColumnKey<t::Date32Type>>(build, spec),
            DataType::Date64 => self.join_dispatch::<SingleColumnKey<t::Date64Type>>(build, spec),
            DataType::Timestamp(TimeUnit::Second, _) => {
                self.join_dispatch::<SingleColumnKey<t::TimestampSecondType>>(build, spec)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                self.join_dispatch::<SingleColumnKey<t::TimestampMillisecondType>>(build, spec)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                self.join_dispatch::<SingleColumnKey<t::TimestampMicrosecondType>>(build, spec)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                self.join_dispatch::<SingleColumnKey<t::TimestampNanosecondType>>(build, spec)
            }
            DataType::Decimal64(_, _) => {
                self.join_dispatch::<SingleColumnKey<t::Decimal64Type>>(build, spec)
            }
            DataType::Decimal128(_, _) => {
                self.join_dispatch::<SingleColumnKey<t::Decimal128Type>>(build, spec)
            }
            // A single-column key of any other type (a string) also takes the
            // dynamic shape.
            _ => self.join_dispatch::<DynamicRowKey>(build, spec),
        }
    }

    /// Pick the kind's instantiation, so the probe's match loop carries no
    /// runtime test for it.
    fn join_dispatch<K: JoinKey>(self, build: RecordBatchOperatorSpec, spec: JoinSpec) -> Self {
        match spec.kind {
            JoinKind::Inner => self.join_typed::<K, false, false>(build, spec),
            JoinKind::BuildOuter => self.join_typed::<K, true, false>(build, spec),
            // With a residual predicate the first key match may not be a real
            // match, so the semi join runs on the pair-recording instantiation
            // and drops duplicate probe rows at drain time instead of exiting
            // the match loop early.
            JoinKind::ProbeSemi if spec.residual_filters.is_some() => {
                self.join_typed::<K, false, false>(build, spec)
            }
            JoinKind::ProbeSemi => self.join_typed::<K, false, true>(build, spec),
        }
    }

    fn join_typed<K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool>(
        self,
        build: RecordBatchOperatorSpec,
        spec: JoinSpec,
    ) -> Self {
        let worker_count = self.worker_count();
        assert_eq!(
            worker_count,
            build.worker_count(),
            "join inputs must use the same worker count"
        );
        assert!(
            self.dispatcher
                .waker_set
                .wakes_same_pool(&build.dispatcher.waker_set),
            "join inputs must use the same worker pool"
        );

        let (build_factories, probe_factories, build_ready) =
            create_join_factories::<K, BUILD_OUTER, SEMI>(spec, worker_count);

        let (_, build_heads) = build.into_parts();
        let build_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let probe_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let build_channels = stealable::<RecordBatch>(self.dispatcher.topology());
        let probe_channels = stealable::<RecordBatch>(self.dispatcher.topology());
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
                    }) as Box<dyn OperatorFactory<RecordBatch>>
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
        let factories: Vec<_> = self.factories.into_iter().collect();
        OperatorSpec::new(self.dispatcher, factories).execute()
    }

    /// Like [`execute`](Self::execute) but with per-dataflow stats collection on;
    /// read them back via [`DataFlowHandle::collect_with_stats`].
    pub fn execute_with_stats(self) -> DataFlowHandle<RecordBatch> {
        let factories: Vec<_> = self.factories.into_iter().collect();
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
