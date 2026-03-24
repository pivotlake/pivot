use super::record_batch_operator::RecordBatchOperatorFactory;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ahash::HashMap;
use arrow_array::RecordBatch;

use crate::Identifier;
use crate::data_flow::DataFlow;
use crate::operations::{GatedOperator, Operator};
use crate::operations::channels::MpscSender;

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

fn next_dataflow_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

/// A sequence of operators built during the factory `build` step on a worker thread.
///
/// Factories produce a `Chain` by recursively building their head, then appending their own
/// operator via [`Chain::with`]. Once complete, [`Chain::into_data_flow`] converts it into
/// a `DataFlow` that the worker can execute.
///
/// For binary operators, [`Chain::merge`] combines two chains with a binary operator
/// node that reads from both tails.
pub struct Chain {
    operators: Vec<Box<dyn Operator>>,
    edges: HashMap<Identifier, Vec<Identifier>>,
}

impl Chain {
    /// Start a new chain with a root operator (one that has no upstream head).
    pub fn root(operator: Box<dyn Operator>) -> Self {
        Self {
            operators: vec![operator],
            edges: HashMap::default(),
        }
    }

    /// Append an operator to the end of the chain.
    pub fn with(mut self, operator: Box<dyn Operator>) -> Self {
        let prev = self.operators.len() - 1;
        let new = self.operators.len();
        self.edges.entry(prev).or_default().push(new);
        self.operators.push(operator);
        self
    }

    /// Merge two chains with a binary operator that reads from both tails.
    ///
    /// The resulting chain has the binary operator as its new tail, so
    /// subsequent [`with`](Self::with) calls append after it.
    pub fn merge(left: Chain, right: Chain, binary_op: Box<dyn Operator>) -> Self {
        let left_len = left.operators.len();
        let left_tail = left_len - 1;

        // Start with left edges, then shift and add right edges
        let mut edges = left.edges;
        for (k, v) in right.edges {
            edges.insert(k + left_len, v.into_iter().map(|i| i + left_len).collect());
        }

        let right_tail = left_len + right.operators.len() - 1;
        let binary_idx = left_len + right.operators.len();

        // Both tails feed into the binary operator
        edges.entry(left_tail).or_default().push(binary_idx);
        edges.entry(right_tail).or_default().push(binary_idx);

        let mut operators = left.operators;
        operators.extend(right.operators);
        operators.push(binary_op);

        Self { operators, edges }
    }

    /// Combine two independent chains into a single operator graph.
    ///
    /// No edges are created between them — coordination happens through shared state
    /// (e.g. a gate atomic). Both chains' operators run in the same `DataFlow`.
    pub fn parallel(left: Chain, right: Chain) -> Self {
        let left_len = left.operators.len();
        let mut edges = left.edges;
        for (k, v) in right.edges {
            edges.insert(k + left_len, v.into_iter().map(|i| i + left_len).collect());
        }
        let mut operators = left.operators;
        operators.extend(right.operators);
        Self { operators, edges }
    }

    /// Wrap all root operators (those with no parents) in a [`GatedOperator`].
    ///
    /// While the gate is closed, the roots produce no work, no IO, and refuse to
    /// finish — effectively freezing the entire sub-pipeline they feed.
    pub fn gate_roots(self, gate: Arc<AtomicBool>) -> Self {
        let mut has_parent = vec![false; self.operators.len()];
        for children in self.edges.values() {
            for &child in children {
                has_parent[child] = true;
            }
        }

        let operators = self
            .operators
            .into_iter()
            .enumerate()
            .map(|(i, op)| -> Box<dyn Operator> {
                if has_parent[i] {
                    op
                } else {
                    Box::new(GatedOperator::new(op, gate.clone()))
                }
            })
            .collect();

        Self {
            operators,
            edges: self.edges,
        }
    }

    /// Convert this chain into an executable `DataFlow`.
    pub fn into_data_flow(self) -> DataFlow {
        DataFlow::new(next_dataflow_id(), self.operators, self.edges)
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
