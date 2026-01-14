use crate::input::Input;
use crate::operations::Output;
use arrow_array::RecordBatch;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use tracing::debug;

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
}

pub struct MemoryOutput {
    source: Arc<MemorySource>,
    active: Arc<AtomicUsize>,
    notify: Arc<mpsc::Sender<()>>,
}

impl MemoryOutput {
    pub fn new(
        active: Arc<AtomicUsize>,
        source: Arc<MemorySource>,
        notify: Arc<mpsc::Sender<()>>,
    ) -> Self {
        Self {
            source,
            active,
            notify,
        }
    }
}

impl Clone for MemoryOutput {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            active: self.active.clone(),
            notify: self.notify.clone(),
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
            debug!("Setting notify to end!");
            let _ = self.notify.send(());
        }
    }
}

impl Input for Arc<MemorySource> {
    fn source_finished(&self) -> bool {
        self.input_finished.load(Ordering::Relaxed) && self.is_empty()
    }

    fn poll_record_batch(&self) -> Option<RecordBatch> {
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
