use crate::GatherBarrier;
use crate::boundary_slot::BoundarySlot;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::{OrderBy, OrderByLimit};
use arrow_array::{ArrayRef, RecordBatch};
use std::sync::{Arc, Mutex};

/// Creates one [`OrderByLimit`] operator per worker with shared gathering state.
///
/// Per-worker top-k results flow into a shared gather barrier.
/// When a boundary slot is present, every worker shares it (and one key
/// window to pool keys in) and publishes its running boundary into it.
pub struct OrderByLimitFactory {
    order_by: Vec<OrderBy>,
    limit: usize,
    offset: usize,
    gather: Arc<GatherBarrier<Option<RecordBatch>>>,
    boundary_slot: Option<Arc<BoundarySlot>>,
    key_window: Arc<Mutex<Option<ArrayRef>>>,
}

impl OrderByLimitFactory {
    /// Create `worker_count` factories sharing a gather barrier, an optional
    /// boundary slot, and the key window its workers pool keys in.
    pub fn create_for_workers(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        worker_count: usize,
        boundary_slot: Option<Arc<BoundarySlot>>,
    ) -> impl IntoIterator<Item = OrderByLimitFactory> {
        let gather = Arc::new(GatherBarrier::new(worker_count));
        let key_window = Arc::new(Mutex::new(None));

        (0..worker_count).map(move |_| OrderByLimitFactory {
            order_by: order_by.clone(),
            limit,
            offset,
            gather: gather.clone(),
            boundary_slot: boundary_slot.clone(),
            key_window: key_window.clone(),
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
            self.boundary_slot,
            self.key_window,
        )
    }
}
