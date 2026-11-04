//! A dataflow handed to the pool: published once, and built by every worker
//! from the same shared description.
//!
//! The thread executing a spec publishes one [`WorkerLaunch`] to the pool's
//! [`LaunchLog`] and wakes the workers. Each worker takes every launch it has
//! not seen, makes its own operator chain from the launch's maker, and runs
//! it. So handing a dataflow to 192 workers costs the publishing thread one
//! log entry, not 192 sends; the per-worker work happens on the workers.
//!
//! The launch also owns the dataflow's one output sender. Every worker's
//! chain sends through a handle on the launch, and the output channel closes
//! when the last of those handles goes, which is when the last worker drops
//! its chain, exactly as it did when each worker held a sender of its own.

use crate::api::builder::{self, OperatorGraphBuilder};
use crate::api::{BuildContext, OperatorFactory, WorkerMaker};
use crate::data_flow::DataFlow;
use crate::operations::channels::{MpscSender, Sender};
use crate::stats::DataFlowStats;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

/// What every worker's chain of one dataflow shares.
pub(crate) struct LaunchShared {
    pub(crate) err_tx: mpsc::Sender<crate::data_flow::Error>,
    pub(crate) stats_tx: mpsc::Sender<DataFlowStats>,
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) collect_stats: bool,
    /// Whether the dataflow runs exclusively while a `perf` capture is in
    /// progress.
    #[cfg(feature = "perf")]
    pub(crate) profiled: bool,
}

/// A dataflow as every worker sees it: how to make its own chain, and what
/// every chain shares.
pub(crate) trait WorkerLaunch: Send + Sync {
    /// Make and build this worker's operator chain, outputting to the
    /// dataflow's output channel.
    fn build_graph(
        self: Arc<Self>,
        worker_idx: usize,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder;

    fn shared(&self) -> &LaunchShared;
}

impl dyn WorkerLaunch {
    /// Build this worker's operator graph and turn it into its executable
    /// `DataFlow`. Called on the worker thread.
    pub(crate) fn build_for_worker(
        self: Arc<Self>,
        worker_idx: usize,
    ) -> builder::Result<DataFlow> {
        let mut context = BuildContext::default();
        let graph = match catch_unwind(AssertUnwindSafe(|| {
            self.clone().build_graph(worker_idx, &mut context)
        })) {
            Ok(graph) => graph,
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                // A build failure must fail the whole query the way a runtime
                // failure does (see `DataFlow::bail_and_cancel`): peers that
                // built their part of the graph hold output senders until the
                // query is cancelled, and the collector blocks until every
                // sender drops, so swallowing the error here hangs the caller.
                let shared = self.shared();
                if shared
                    .err_tx
                    .send(crate::data_flow::Error::Panic(message.to_string()))
                    .is_err()
                {
                    tracing::warn!("Unable to send build error...");
                }
                shared.cancelled.store(true, Ordering::Relaxed);
                crate::waker::waker_set().notify_all();
                return Err(builder::Error::PanicOnBuild(message.to_string()));
            }
        };

        let shared = self.shared();
        let data_flow = graph.into_data_flow(
            shared.cancelled.clone(),
            shared.err_tx.clone(),
            shared.stats_tx.clone(),
            shared.collect_stats,
        );
        #[cfg(feature = "perf")]
        let data_flow = data_flow.with_profiling(shared.profiled);
        Ok(data_flow)
    }
}

/// The launch of a spec whose per-worker chains a maker of `OF` produces.
pub(crate) struct Launch<O, OF> {
    pub(crate) chain: Box<dyn WorkerMaker<OF>>,
    /// The dataflow's one output sender; every worker's chain sends through
    /// a handle on this launch.
    pub(crate) output: MpscSender<O>,
    pub(crate) shared: LaunchShared,
}

/// One worker's handle on the launch's output sender.
struct LaunchOutput<O, OF> {
    launch: Arc<Launch<O, OF>>,
}

impl<O: Send + 'static, OF: Send + 'static> Sender<O> for LaunchOutput<O, OF> {
    fn send(&mut self, item: O) -> crate::operations::channels::Result<()> {
        self.launch.output.send_ref(item)
    }
}

impl<O: Send + 'static, OF: OperatorFactory<O> + Send + 'static> WorkerLaunch for Launch<O, OF> {
    fn build_graph(
        self: Arc<Self>,
        worker_idx: usize,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let factory = self.chain.for_worker(worker_idx);
        let output = Box::new(LaunchOutput { launch: self });
        Box::new(factory).build(output, context)
    }

    fn shared(&self) -> &LaunchShared {
        &self.shared
    }
}

struct LogEntry {
    seq: u64,
    launch: Arc<dyn WorkerLaunch>,
    /// How many workers have taken this entry; it leaves the log once every
    /// worker has, so the log never keeps a finished dataflow's output
    /// channel open.
    taken: usize,
}

/// The pool's launches, in publication order, each kept until every worker
/// has taken it.
pub(crate) struct LaunchLog {
    entries: Mutex<VecDeque<LogEntry>>,
    /// The sequence number of the latest entry, written under the lock and
    /// read without it, so a worker with nothing new to take skips the lock.
    published: AtomicU64,
    worker_count: usize,
}

impl LaunchLog {
    pub(crate) fn new(worker_count: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            published: AtomicU64::new(0),
            worker_count,
        }
    }

    pub(crate) fn publish(&self, launch: Arc<dyn WorkerLaunch>) {
        let mut entries = self
            .entries
            .lock()
            .expect("the launch log is never poisoned");
        let seq = self.published.load(Ordering::Relaxed) + 1;
        entries.push_back(LogEntry {
            seq,
            launch,
            taken: 0,
        });
        self.published.store(seq, Ordering::Release);
    }

    /// Every launch published since the one `cursor` names, oldest first,
    /// advancing `cursor` past them.
    pub(crate) fn take_new(&self, cursor: &mut u64) -> Vec<Arc<dyn WorkerLaunch>> {
        if self.published.load(Ordering::Acquire) == *cursor {
            return Vec::new();
        }
        let mut entries = self
            .entries
            .lock()
            .expect("the launch log is never poisoned");
        let mut taken = Vec::new();
        for entry in entries.iter_mut() {
            if entry.seq > *cursor {
                taken.push(entry.launch.clone());
                entry.taken += 1;
                *cursor = entry.seq;
            }
        }
        while entries
            .front()
            .is_some_and(|entry| entry.taken == self.worker_count)
        {
            entries.pop_front();
        }
        taken
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Inert;

    impl WorkerLaunch for Inert {
        fn build_graph(self: Arc<Self>, _: usize, _: &mut BuildContext) -> OperatorGraphBuilder {
            unreachable!("the log test never builds")
        }
        fn shared(&self) -> &LaunchShared {
            unreachable!("the log test never builds")
        }
    }

    #[test]
    fn each_worker_takes_every_launch_once_in_order() {
        let log = LaunchLog::new(2);
        log.publish(Arc::new(Inert));
        log.publish(Arc::new(Inert));
        let mut cursor = 0;

        let first_pass = log.take_new(&mut cursor).len();
        let second_pass = log.take_new(&mut cursor).len();

        assert_eq!((first_pass, second_pass, cursor), (2, 0, 2));
    }

    #[test]
    fn a_launch_leaves_the_log_once_every_worker_took_it() {
        let log = LaunchLog::new(2);
        let launch: Arc<dyn WorkerLaunch> = Arc::new(Inert);
        log.publish(launch.clone());
        let (mut first, mut second) = (0, 0);

        log.take_new(&mut first);
        let held_by_one = Arc::strong_count(&launch);
        log.take_new(&mut second);
        let held_by_none = Arc::strong_count(&launch);

        assert_eq!((held_by_one, held_by_none), (2, 1));
    }
}
