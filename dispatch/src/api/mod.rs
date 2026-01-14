pub mod operator_spec;

use crate::data_flow::DataFlow;
use crate::identified::Identifier;
use crate::operations::{MpscSender, Operator, Sender};
use ahash::HashMap;
use arrow_array::RecordBatch;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

pub fn next_dataflow_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

pub trait OperatorFactory<O>: Send {
    fn build<S: Sender<O> + 'static>(self: Box<Self>, sender: S) -> Chain;
}

pub struct Chain {
    operators: Vec<Box<dyn Operator>>,
}

impl Chain {
    pub fn root(operator: Box<dyn Operator>) -> Self {
        Self {
            operators: vec![operator],
        }
    }

    pub fn with(mut self, operator: Box<dyn Operator>) -> Self {
        self.operators.push(operator);
        self
    }

    pub fn into_data_flow(self) -> DataFlow {
        let map: HashMap<Identifier, Vec<Identifier>> = (0..self.operators.len() - 1)
            .map(|i| (i, vec![i + 1]))
            .collect();
        DataFlow::new(next_dataflow_id(), self.operators, map)
    }
}

pub trait BuildDataFlow: Send {
    fn build(self: Box<Self>, sender: MpscSender<RecordBatch>) -> DataFlow;
}

impl<T: OperatorFactory<RecordBatch>> BuildDataFlow for T {
    fn build(self: Box<Self>, sender: MpscSender<RecordBatch>) -> DataFlow {
        let chain = OperatorFactory::build(self, sender);
        chain.into_data_flow()
    }
}

pub struct DataFlowBuilder {
    tail: Box<dyn BuildDataFlow>,
    output_tx: MpscSender<RecordBatch>,
}

impl DataFlowBuilder {
    pub fn new(tail: Box<dyn BuildDataFlow>, output_tx: MpscSender<RecordBatch>) -> Self {
        Self { tail, output_tx }
    }
    pub fn build(self) -> DataFlow {
        self.tail.build(self.output_tx)
    }
}
