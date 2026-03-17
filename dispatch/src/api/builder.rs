use super::record_batch_operator::RecordBatchOperatorFactory;
use crate::Identifier;
use crate::data_flow::DataFlow;
use crate::operations::Operator;
use crate::operations::channels::MpscSender;
use ahash::HashMap;
use arrow_array::RecordBatch;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

fn next_dataflow_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

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
    pub fn into_data_flow(self) -> DataFlow {
        let map: HashMap<Identifier, Vec<Identifier>> = (0..self.operators.len() - 1)
            .map(|i| (i, vec![i + 1]))
            .collect();
        DataFlow::new(next_dataflow_id(), self.operators, map)
    }
}

/// A factory + output sender pair, ready to be sent to a worker thread.
///
/// Created by [`RecordBatchOperatorSpec::collect`](super::record_batch_operator::RecordBatchOperatorSpec::collect),
/// one per worker. The worker calls [`build`](DataFlowBuilder::build) to produce a `DataFlow`,
/// which triggers the recursive factory build chain on the worker thread.
pub struct DataFlowBuilder {
    tail: Box<dyn RecordBatchOperatorFactory>,
    output_tx: MpscSender<RecordBatch>,
}

impl DataFlowBuilder {
    pub fn new(
        tail: Box<dyn RecordBatchOperatorFactory>,
        output_tx: MpscSender<RecordBatch>,
    ) -> Self {
        Self { tail, output_tx }
    }

    /// Build the full operator chain and convert it into an executable `DataFlow`.
    /// Called on the worker thread.
    pub fn build(self) -> DataFlow {
        self.tail.build_collect(self.output_tx).into_data_flow()
    }
}
