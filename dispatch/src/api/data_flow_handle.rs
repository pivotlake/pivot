use crate::worker::worker_waker;
use arrow_array::RecordBatch;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

/// A handle to a running dataflow, allowing the user to interact with a current running dataflow in
/// dispatch - receiving record batches (from the receiver)
/// as they arrive, check for errors, and cancel the query across workers.
pub struct DataFlowHandle {
    /// A receiver from the last operator in the real Dataflow. All workers have a producer to send
    /// to here
    record_batch_rx: mpsc::Receiver<RecordBatch>,
    /// A receiver for errors that occurred during running.
    err_rx: mpsc::Receiver<crate::data_flow::Error>,
    /// A flag that is by default off - it can be turned on to cause all workers to cancel the
    /// running query
    cancelled: Arc<AtomicBool>,
}

impl DataFlowHandle {
    pub fn new(
        record_batch_rx: mpsc::Receiver<RecordBatch>,
        err_rx: mpsc::Receiver<crate::data_flow::Error>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            record_batch_rx,
            err_rx,
            cancelled,
        }
    }

    /// Collect all record batches from the running dataflow across all workers
    pub fn collect(mut self) -> crate::data_flow::Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        for batch_res in &mut self {
            batches.push(batch_res?);
        }
        // The record batch channel closes when all workers drop their senders
        // (e.g. on panic or completion). A worker that panicked/erred may have queued
        // the error before dropping the sender, so check err_rx after the channel has closed.
        if let Ok(e) = self.err_rx.try_recv() {
            return Err(e);
        }
        Ok(batches)
    }

    /// Cancel the current running dataflow. This does NOT wait for all workers to finish running
    /// it; for that collect must be called.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        worker_waker().notify();
    }

    /// A cheap, `Clone + Send + Sync` handle that can fire cancellation from a
    /// thread that doesn't own the [`DataFlowHandle`] — e.g. from an async
    /// task driving `next()` on a separate blocking thread.
    pub fn cancel_token(&self) -> CancelToken {
        CancelToken {
            cancelled: self.cancelled.clone(),
        }
    }
}

/// A detached cancellation handle for a running dataflow.
///
/// Obtain one with [`DataFlowHandle::cancel_token`]. Cloning is cheap (it's an
/// `Arc<AtomicBool>` under the hood), so it can be moved into a Drop guard,
/// stored in a registry keyed by connection id, etc.
#[derive(Clone)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        worker_waker().notify();
    }
}

impl Iterator for DataFlowHandle {
    type Item = crate::data_flow::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(e) = self.err_rx.try_recv() {
            Some(Err(e))
        } else {
            self.record_batch_rx.recv().ok().map(Ok)
        }
    }
}
