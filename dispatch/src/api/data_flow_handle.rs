use crate::stats::DataFlowStats;
use crate::worker::WorkerWaker;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

/// A handle to a running dataflow that produces items of type `T`.
///
/// Generic over the output type so the same handle shape works for record
/// batches, future DDL responses, materialize-to-disk completions, etc.
/// Exposes the universal control surface — cancel, cancel_token — plus
/// iteration and collection of the output stream.
pub struct DataFlowHandle<T> {
    rx: mpsc::Receiver<T>,
    /// Receives errors that a worker queues before dropping its err-sender.
    err_rx: mpsc::Receiver<crate::data_flow::Error>,
    /// Receives each worker's stats tally (empty unless the query opted in).
    stats_rx: mpsc::Receiver<DataFlowStats>,
    /// Process-wide cancel flag, checked by every worker on each iteration.
    cancelled: Arc<AtomicBool>,
    /// Shared with the [`DataFlowDispatcher`](crate::DataFlowDispatcher) so that cancel callers (which
    /// may not be on a worker thread) can still wake any parked worker.
    waker: Arc<WorkerWaker>,
}

impl<T> DataFlowHandle<T> {
    pub fn new(
        rx: mpsc::Receiver<T>,
        err_rx: mpsc::Receiver<crate::data_flow::Error>,
        stats_rx: mpsc::Receiver<DataFlowStats>,
        cancelled: Arc<AtomicBool>,
        waker: Arc<WorkerWaker>,
    ) -> Self {
        Self {
            rx,
            err_rx,
            stats_rx,
            cancelled,
            waker,
        }
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
    pub fn collect(self) -> crate::data_flow::Result<Vec<T>> {
        Ok(self.collect_with_stats()?.0)
    }

    /// Like [`collect`](Self::collect), but also returns the dataflow's stats
    /// folded across every worker. The tally is all zeros unless the dataflow
    /// was launched with [`execute_with_stats`](crate::OperatorSpec::execute_with_stats).
    /// By the time the output channel has closed, every worker that finished has
    /// already shipped its tally, so the stats channel is fully drained here.
    pub fn collect_with_stats(mut self) -> crate::data_flow::Result<(Vec<T>, DataFlowStats)> {
        let mut items = Vec::new();
        let mut error = None;
        // Drain to channel close (every worker has dropped its sender, finished
        // or cancelled) so all stats have been reported before we fold them. Keep
        // the first error rather than returning on it, so a failed query's IO is
        // still tallied.
        for item in &mut self {
            match item {
                Ok(value) => items.push(value),
                Err(e) if error.is_none() => error = Some(e),
                Err(_) => {}
            }
        }
        let mut stats = DataFlowStats::default();
        while let Ok(worker_stats) = self.stats_rx.try_recv() {
            stats.merge(&worker_stats);
        }
        match error {
            // The success path reports via the server's stats NOTICE; a failed
            // query has no such path, so log what IO it did before failing.
            Some(e) => {
                stats.log_failed_query();
                Err(e)
            }
            None => Ok((items, stats)),
        }
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
