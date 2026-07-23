use std::sync::mpsc::{self, Receiver};

use arrow_array::RecordBatch;

use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::OrderBy;
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use crate::operations::unary::window::WindowRowNumber;

/// Creates one [`WindowRowNumber`] operator per worker sharing a single mpsc
/// channel: every worker forwards its rows to the one collector (the factory
/// that took the receiver), which does the global sort + numbering.
pub struct WindowRowNumberFactory {
    order_by: Vec<OrderBy>,
    partition_key_count: usize,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl WindowRowNumberFactory {
    pub fn create_for_workers(
        order_by: Vec<OrderBy>,
        partition_key_count: usize,
        worker_count: usize,
    ) -> impl IntoIterator<Item = WindowRowNumberFactory> {
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);
        (0..worker_count).map(move |_| WindowRowNumberFactory {
            order_by: order_by.clone(),
            partition_key_count,
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for WindowRowNumberFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, WindowRowNumber>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, WindowRowNumber> {
        PipelineBreaker::Consuming(WindowRowNumber::new(
            self.order_by,
            self.partition_key_count,
            self.sender,
            self.receiver.take(),
        ))
    }
}
