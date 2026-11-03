//! API for building parallel RecordBatch query dataflows.
//!
//! # Usage
//!
//! Build a query by chaining operations on [`RecordBatchOperatorSpec`]:
//!
//! ```ignore
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
//! # let dispatch = Dispatch::spin_up(1, 32, None);
//! let results = table_input(dispatch.dispatcher(), &table, Projection::columns([0]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             collect_selected_indices(&contains.run(col), &mut scratch.positions);
//!             Ok(RowSelection::Indices)
//!         }
//!     })
//!     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
//!     .collect();
//! ```
//!
//! Each method appends a stage to the dataflow and returns `Self`, so the full query
//! type is always just `RecordBatchOperatorSpec` — no nested generics leak out.
//!
//! # Internal architecture
//!
//! A query is a blueprint: a maker of one factory per worker thread. Nothing runs until
//! [`RecordBatchOperatorSpec::collect`] is called. The lifecycle is:
//!
//! 1. **Describe the stages** — Each chained method (`.filter(...)`, `.aggregate(...)`, etc.)
//!    sets up what its stage's workers share (a channel's deques, a builder closure, a
//!    group's arenas) and wraps the previous stages' maker in a new one, which makes a
//!    worker's [`UnaryOperatorFactory`](crate::UnaryOperatorFactory) around that worker's
//!    head. The makers are stored as `Box<dyn WorkerMaker<Box<dyn OperatorFactory<RecordBatch>>>>`
//!    to erase the nested generic types. The thread compiling the query does a fixed
//!    amount of work per stage, however many workers the pool has.
//!
//! 2. **Publish to workers** — `collect()` puts the maker and the output `MpscSender<RecordBatch>`
//!    in one launch, published once to the pool's log; every worker takes it. Building
//!    is deferred because the build step creates things which may not be `Send` such as `Rc<Worker<T>>`
//!    which live on the worker thread.
//!
//! 3. **Build on worker** — The worker calls [`DataFlowBuilder::build`], which makes its
//!    factory and calls `build(sender)` on it. This recursively builds the full operator
//!    chain: each factory creates a channel, builds its head with the channel's sender,
//!    then appends its own operator reading from the channel's receiver.
//!
//! 4. **Execute** — The worker runs the resulting [`DataFlow`](crate::data_flow::DataFlow),
//!    driving data through the operator chain.
//!
//! # Key types
//!
//! - [`RecordBatchOperatorSpec`] — The user-facing query builder.
//!   Holds the maker of each worker's `Box<dyn OperatorFactory<RecordBatch>>`.
//!
//! - [`WorkerMaker`] — Makes one worker's piece of a stage on that worker's thread;
//!   [`per_worker`] makes one from a closure, [`WorkerSlots`] hands out pieces made
//!   ahead of time, and [`split`] shares one making between two pieces.
//!
//! - [`OperatorFactory<O>`](OperatorFactory) — Factory trait with
//!   `build(self: Box<Self>, sender: Box<dyn Sender<O>>)`. The sender is a trait
//!   object, so the trait is object-safe and a factory can be boxed at any stage.
//!   Also carries pipelines that flow through non-RecordBatch intermediate types —
//!   e.g. the multi-stage decode in `catalog`'s Parquet reader.
//!
//! - [`OperatorSpec<O, OF>`](OperatorSpec) — Generic spec holding a maker of `OF`. Can be used for
//!   any operator that does not expose `RecordBatch`. Used to build multi-stage
//!   pipelines (e.g. `catalog`'s Parquet reader) before erasing into
//!   `RecordBatchOperatorSpec` via `from_spec`.
//!
//! - [`OperatorGraphBuilder`] — Accumulates operators and edges during the build step,
//!   then converts them to a `DataFlow`.
//!
//! - [`DataFlowBuilder`] — One worker's view of a published launch. The worker calls
//!   `.build()` on it to make its factory and produce its `DataFlow`.

mod build_context;
pub use build_context::BuildContext;

mod builder;
pub use builder::{DataFlowBuilder, OperatorGraphBuilder};

mod worker_maker;
pub use worker_maker::{
    FnMaker, HandOut, IntoWorkerMaker, SplitHalf, WorkerMaker, WorkerSlots, per_worker, split,
};

pub(crate) mod launch;

mod operator_spec;
pub use operator_spec::{OperatorFactory, OperatorSpec};

mod data_flow_handle;
mod record_batch_operator;
pub use data_flow_handle::{CancelToken, DataFlowHandle};

pub use operator_spec::{channel_input, values_input};
pub use record_batch_operator::{OutputBatch, RECORD_BATCH_SIZE, RecordBatchOperatorSpec};
