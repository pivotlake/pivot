use crate::worker::WorkerWaker;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use crate::operations::channels::{MpscReceiver, Receiver};

/// A handle to a running dataflow that produces items of type `T`.
///
/// Generic over the output type so the same handle shape works for record
/// batches, future DDL responses, materialize-to-disk completions, etc.
/// Exposes the universal control surface — cancel, cancel_token — plus
/// iteration and collection of the output stream.
///
/// For record-batch dataflows you usually don't construct one directly; go
/// through [`RecordBatchOperatorSpec::collect`](crate::api::RecordBatchOperatorSpec::collect),
/// which inserts a `LocalCollect` cap so the batches you receive are plain
/// heap-backed (safe to hold on any thread, regardless of `MemoryContext`).
pub struct DataFlowHandle<T> {
    rx: mpsc::Receiver<T>,
    /// Receives errors that a worker queues before dropping its err-sender.
    err_rx: mpsc::Receiver<crate::data_flow::Error>,
    /// Process-wide cancel flag, checked by every worker on each iteration.
    cancelled: Arc<AtomicBool>,
    /// Shared with the [`DataFlowDispatcher`] so that cancel callers (which
    /// may not be on a worker thread) can still wake any parked worker.
    waker: Arc<WorkerWaker>,
}

impl<T> DataFlowHandle<T> {
    pub fn new(
        rx: mpsc::Receiver<T>,
        err_rx: mpsc::Receiver<crate::data_flow::Error>,
        cancelled: Arc<AtomicBool>,
        waker: Arc<WorkerWaker>,
    ) -> Self {
        Self { rx, err_rx, cancelled, waker }
    }

    /// Cancel the running dataflow. Doesn't block — workers exit on the next
    /// iteration of their event loop.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.waker.notify();
    }

    /// A cheap, `Clone + Send + Sync` handle that can fire cancellation from
    /// another thread.
    pub fn cancel_token(&self) -> CancelToken {
        CancelToken {
            cancelled: self.cancelled.clone(),
            waker: self.waker.clone(),
        }
    }


    /// Block until every worker drops its output sender, then return all
    /// items in order, or the first worker error if any occurred.
    pub fn collect(mut self) -> crate::data_flow::Result<Vec<T>> {
        let mut items = Vec::new();
        for item in &mut self {
            items.push(item?);
        }
        // The output channel closes when every worker drops its sender (on
        // panic or completion). A worker that errored may have queued the
        // error before dropping, so check err_rx after the channel closes.
        if let Ok(e) = self.err_rx.try_recv() {
            return Err(e);
        }
        Ok(items)
    }
}

impl<T> Iterator for DataFlowHandle<T> {
    type Item = crate::data_flow::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(e) = self.err_rx.try_recv() {
            Some(Err(e))
        } else {
            self.rx.recv().ok().map(Ok)
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
    waker: Arc<WorkerWaker>,
}

impl CancelToken {
    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.waker.notify();
    }
}
