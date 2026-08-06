use crate::GatherBarrier;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::{DynamicFilterSlot, OrderBy, OrderByLimit};
use arrow_array::RecordBatch;
use std::sync::Arc;

/// Creates one [`OrderByLimit`] operator per worker with shared gathering state.
///
/// Per-worker top-k results flow into a shared gather barrier.
/// When a dynamic-filter slot is present, every worker shares it and publishes
/// its running boundary into it.
pub struct OrderByLimitFactory {
    order_by: Vec<OrderBy>,
    limit: usize,
    offset: usize,
    gather: Arc<GatherBarrier<Option<RecordBatch>>>,
    dynamic_filter: Option<Arc<DynamicFilterSlot>>,
}

impl OrderByLimitFactory {
    /// Create `worker_count` factories sharing a gather barrier and an
    /// optional shared dynamic-filter slot.
    pub fn create_for_workers(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        worker_count: usize,
        dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    ) -> impl IntoIterator<Item = OrderByLimitFactory> {
        let gather = Arc::new(GatherBarrier::new(worker_count));

        (0..worker_count).map(move |_| OrderByLimitFactory {
            order_by: order_by.clone(),
            limit,
            offset,
            gather: gather.clone(),
            dynamic_filter: dynamic_filter.clone(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for OrderByLimitFactory {
    type Unary = OrderByLimit;

    fn build_unary(self) -> OrderByLimit {
        OrderByLimit::new(
            self.order_by,
            self.limit,
            self.offset,
            self.gather,
            self.dynamic_filter,
        )
    }
}
