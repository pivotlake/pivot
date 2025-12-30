use crate::Output;
use crate::input::Input;
use arrow_array::RecordBatch;
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A `MemorySource` is a shared in-memory source of RecordBatches, which can be popped out of it
/// using the `MemoryInput` and written to using the `MemoryOutput`.
///
/// The `MemorySource` can be used to connect one pipeline to another (one using
/// `MemoryOutput` and the other using `MemoryInput`)
pub struct MemorySource {
    input_finished: AtomicBool,
    record_batches: Injector<RecordBatch>,
}

impl Default for MemorySource {
    fn default() -> Self {
        Self::new()
    }
}

impl MemorySource {
    pub fn new() -> Self {
        Self {
            input_finished: AtomicBool::new(false),
            record_batches: Default::default(),
        }
    }
    fn set_input_finished(&self) {
        self.input_finished.store(true, Ordering::Relaxed);
    }

    pub fn is_empty(&self) -> bool {
        self.record_batches.is_empty()
    }

    fn push_record_batch(&self, batch: RecordBatch) {
        self.record_batches.push(batch);
    }

    fn pop_record_batch(&self) -> Option<RecordBatch> {
        loop {
            return match self.record_batches.steal() {
                Steal::Empty => None,
                Steal::Success(s) => Some(s),
                Steal::Retry => {
                    continue;
                }
            };
        }
    }
}

pub struct MemoryOutput {
    source: Arc<MemorySource>,
    active: Arc<AtomicUsize>,
}

impl MemoryOutput {
    pub fn new(source: Arc<MemorySource>) -> Self {
        Self {
            source,
            active: Arc::new(AtomicUsize::new(1)),
        }
    }
}

impl Clone for MemoryOutput {
    fn clone(&self) -> Self {
        self.active.fetch_add(1, Ordering::Relaxed);
        Self {
            source: self.source.clone(),
            active: self.active.clone(),
        }
    }
}

impl Output for MemoryOutput {
    fn write(&mut self, batch: RecordBatch) {
        self.source.push_record_batch(batch);
    }

    fn finish(&mut self) {
        let val = self.active.fetch_sub(1, Ordering::Relaxed);
        if val == 1 {
            self.source.set_input_finished();
        }
    }
}

pub struct MemoryInput {
    source: Arc<MemorySource>,
}

impl MemoryInput {
    pub fn new(source: Arc<MemorySource>) -> Self {
        Self { source }
    }
}

impl Input for MemoryInput {
    fn source_finished(&self) -> bool {
        self.source.input_finished.load(Ordering::Relaxed) && self.source.is_empty()
    }

    fn poll_record_batch(&self) -> Option<RecordBatch> {
        self.source.pop_record_batch()
    }
}
