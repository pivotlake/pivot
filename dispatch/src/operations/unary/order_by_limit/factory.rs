use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::{OrderBy, OrderByLimit};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use arrow_array::RecordBatch;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;

pub struct OrderByLimitFactory {
    order_by: Vec<OrderBy>,
    limit: usize,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl OrderByLimitFactory {
    pub fn create_for_workers(
        order_by: Vec<OrderBy>,
        limit: usize,
        worker_count: usize,
    ) -> impl IntoIterator<Item = OrderByLimitFactory> {
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| OrderByLimitFactory {
            order_by: order_by.clone(),
            limit,
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for OrderByLimitFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, OrderByLimit>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, OrderByLimit> {
        PipelineBreaker::Consuming(OrderByLimit::new(
            self.order_by,
            self.limit,
            self.sender,
            self.receiver.take(),
        ))
    }
}
