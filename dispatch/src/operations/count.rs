use crate::operations::{Identifier, Operation};
use arrow_array::RecordBatch;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

pub struct Count {
    id: Identifier,
    internal_count: usize,
    shared_count: Arc<AtomicUsize>,
    barrier: Arc<Barrier>,
}

impl Count {
    pub fn new(id: Identifier, shared_count: Arc<AtomicUsize>, barrier: Arc<Barrier>) -> Self {
        Self {
            id,
            internal_count: 0,
            shared_count,
            barrier,
        }
    }
}
impl Operation for Count {
    fn id(&self) -> Identifier {
        self.id
    }

    fn consume_output_batch(&mut self) -> Option<RecordBatch> {
        None
    }

    fn run(&mut self, batch: &RecordBatch) {
        self.internal_count += batch.num_rows();
    }

    fn finish(&mut self) -> Option<RecordBatch> {
        self.shared_count
            .fetch_add(self.internal_count, Ordering::Relaxed);
        if self.barrier.wait().is_leader() {
            println!(
                "THE RESULT IS {:?}",
                self.shared_count.load(Ordering::Relaxed)
            );
        }
        None
    }
}
