use super::record_batch_operator::RecordBatchOperatorFactory;
use crate::Identifier;
use crate::data_flow::DataFlow;
use crate::operations::Operator;
use crate::operations::channels::MpscSender;
use ahash::HashMap;
use arrow_array::RecordBatch;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, mpsc};
use thiserror::Error;

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

fn next_dataflow_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    PanicOnBuild(String),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// A linear sequence of operators built during the factory `build` step on a worker thread.
///
/// Factories produce a `Chain` by recursively building their head, then appending their own
/// operator via [`Chain::with`]. Once complete, [`Chain::into_data_flow`] converts it into
/// a `DataFlow` that the worker can execute.
pub struct Chain {
    operators: Vec<Box<dyn Operator>>,
}

impl Chain {
    /// Start a new chain with a root operator (one that has no upstream head).
    pub fn root(operator: Box<dyn Operator>) -> Self {
        Self {
            operators: vec![operator],
        }
    }

    /// Append an operator to the end of the chain.
    pub fn with(mut self, operator: Box<dyn Operator>) -> Self {
        self.operators.push(operator);
        self
    }

    /// Convert this chain into an executable `DataFlow`.
    pub fn into_data_flow(
        self,
        cancelled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<crate::data_flow::Error>,
    ) -> DataFlow {
        let map: HashMap<Identifier, Vec<Identifier>> = (0..self.operators.len() - 1)
            .map(|i| (i, vec![i + 1]))
            .collect();
        DataFlow::new(next_dataflow_id(), cancelled, err_tx, self.operators, map)
    }
}

/// A factory + output sender pair, ready to be sent to a worker thread.
///
/// Created by [`RecordBatchOperatorSpec::collect`](super::record_batch_operator::RecordBatchOperatorSpec::collect),
/// one per worker. The worker calls [`build`](DataFlowBuilder::build) to produce a `DataFlow`,
/// which triggers the recursive factory build chain on the worker thread.
pub struct DataFlowBuilder {
    /// A flag shared across all workers (and the DataFlowHandle) on whether to cancel this query
    cancelled: Arc<AtomicBool>,
    /// A sender for errors that may occur during running
    err_tx: mpsc::Sender<crate::data_flow::Error>,
    /// The last operator in the dataflow
    tail: Box<dyn RecordBatchOperatorFactory>,
    /// A sender for record batches that result from this dataflow (say a queries response)
    output_tx: MpscSender<RecordBatch>,
}

impl DataFlowBuilder {
    pub fn new(
        cancelled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<crate::data_flow::Error>,
        tail: Box<dyn RecordBatchOperatorFactory>,
        output_tx: MpscSender<RecordBatch>,
    ) -> Self {
        Self {
            cancelled,
            err_tx,
            tail,
            output_tx,
        }
    }

    /// Build the full operator chain and convert it into an executable `DataFlow`.
    /// Called on the worker thread.
    pub fn build(self) -> Result<DataFlow> {
        let chain = catch_unwind(AssertUnwindSafe(|| self.tail.build_collect(self.output_tx)))
            .map_err(|e| {
                let msg = e
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                Error::PanicOnBuild(msg.to_string())
            })?;

        Ok(chain.into_data_flow(self.cancelled, self.err_tx))
    }
}
