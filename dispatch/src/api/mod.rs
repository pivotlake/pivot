//! API for building parallel RecordBatch query dataflows.
//!
//! # Usage
//!
//! Build a query by chaining operations on [`RecordBatchOperatorSpec`]:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! # let dispatch = Dispatch::spin_up(1, 32);
//! let results = table_input(dispatch.dispatcher(), &table, Projection::columns([0]), false)
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
//! Each method appends a stage to the dataflow and returns `Self`, so the full query
//! type is always just `RecordBatchOperatorSpec` — no nested generics leak out.
//!
//! # Internal architecture
//!
//! A query is a blueprint: one factory per worker thread. Nothing runs until
//! [`RecordBatchOperatorSpec::collect`] is called. The lifecycle is:
//!
//! 1. **Build factories** — Each chained method (`.filter(...)`, `.count()`, etc.) wraps
//!    the previous factories in a new layer of [`RecordBatchUnaryOperatorFactory`], producing
//!    already one factory per worker.
//!    The factories are stored as `Box<dyn RecordBatchOperatorFactory>`
//!    to erase the nested generic types.
//!
//! 2. **Ship to workers** — `collect()` pairs each factory with an `MpscSender<RecordBatch>`
//!    inside a [`DataFlowBuilder`] and sends it to the corresponding worker thread. Building
//!    is deferred because the build step creates things which may not be `Send` such as `Rc<Worker<T>>`
//!    which live on the worker thread.
//!
//! 3. **Build on worker** — The worker calls [`DataFlowBuilder::build`], which calls
//!    `build_collect(sender)` on the factory. This recursively builds the full operator
//!    chain: each factory creates a channel, builds its head with the channel's sender,
//!    then appends its own operator reading from the channel's receiver.
//!
//! 4. **Execute** — The worker runs the resulting [`DataFlow`](crate::data_flow::DataFlow),
//!    driving data through the operator chain.
//!
//! # Key types
//!
//! - [`RecordBatchOperatorFactory`] — Object-safe trait that erases
//!   `OperatorFactory<RecordBatch>`. Has two methods (`build_stealable`, `build_collect`)
//!   for the two sender types used at the RecordBatch boundary.
//!
//! - [`RecordBatchOperatorSpec`] — The user-facing query builder.
//!   Holds `VecDeque<Box<dyn RecordBatchOperatorFactory>>` (one per worker).
//!
//! - [`OperatorFactory<O>`](OperatorFactory) — Generic factory trait with
//!   `build<S: Sender<O>>`. Not object-safe (generic method), but used internally for
//!   the parquet pipeline where data flows through non-RecordBatch types
//!   (`RowGroupBuffer → CompressedPage → DecompressedPage → RecordBatch`).
//!
//! - [`OperatorSpec<O, OF>`](OperatorSpec) — Generic spec holding `VecDeque<OF>`. Can be used for
//!   any operator that does not expose `RecordBatch`
//!   Only used internally by `table_input` and `read_parquet` to build the parquet
//!   read stages before erasing into `RecordBatchOperatorSpec` via `from_spec`.
//!
//! - [`Chain`] — Accumulates `Box<dyn Operator>` during the build step, then converts
//!   to a `DataFlow`.
//!
//! - [`DataFlowBuilder`] — Pairs a `Box<dyn RecordBatchOperatorFactory>` with the output
//!   `MpscSender`. Sent to a worker thread, which calls `.build()` to produce a `DataFlow`.

mod builder;
pub use builder::{Chain, DataFlowBuilder};

mod operator_spec;
pub use operator_spec::{OperatorFactory, OperatorSpec};

mod data_flow_handle;
mod record_batch_operator;
pub use data_flow_handle::{CancelToken, DataFlowHandle};

pub use operator_spec::values_input;
pub use record_batch_operator::{
    RECORD_BATCH_SIZE, RecordBatchFactoryBridge, RecordBatchOperatorFactory,
    RecordBatchOperatorSpec, RecordBatchUnaryOperatorFactory, table_input, table_input_with_filter,
    table_input_with_filter_and_eq_predicates,
};
