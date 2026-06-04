use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::{DynamicFilterSlot, OrderBy, OrderByLimit};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use arrow_array::RecordBatch;
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;

/// Creates one [`OrderByLimit`] operator per worker with shared channel wiring.
///
/// The first factory receives the channel receiver; the rest get `None`.
/// All share a sender so per-worker top-k results flow to a single collector.
/// When a dynamic-filter slot is present, every worker shares it and publishes
/// its running boundary into it.
pub struct OrderByLimitFactory {
    order_by: Vec<OrderBy>,
    limit: usize,
    offset: usize,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
    dynamic_filter: Option<Arc<DynamicFilterSlot>>,
}

impl OrderByLimitFactory {
    /// Create `worker_count` factories sharing a single mpsc channel and an
    /// optional shared dynamic-filter slot.
    pub fn create_for_workers(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        worker_count: usize,
        dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    ) -> impl IntoIterator<Item = OrderByLimitFactory> {
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| OrderByLimitFactory {
            order_by: order_by.clone(),
            limit,
            offset,
            sender: tx.clone(),
            receiver: rx_opt.take(),
            dynamic_filter: dynamic_filter.clone(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for OrderByLimitFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, OrderByLimit>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, OrderByLimit> {
        PipelineBreaker::Consuming(OrderByLimit::new(
            self.order_by,
            self.limit,
            self.offset,
            self.sender,
            self.receiver.take(),
            self.dynamic_filter,
        ))
    }
}
