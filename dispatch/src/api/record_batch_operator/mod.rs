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

use std::rc::Rc;

use arrow_array::RecordBatch;
use crossbeam_deque::Worker;

use crate::api::Chain;
use crate::api::operator_spec::OperatorFactory;
use crate::operations::channels::MpscSender;

mod factory;
pub use factory::{RecordBatchBinaryOperatorFactory, RecordBatchUnaryOperatorFactory};

mod spec;
pub use spec::{RecordBatchOperatorSpec, table_input};

pub const RECORD_BATCH_SIZE: usize = 2048;

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
