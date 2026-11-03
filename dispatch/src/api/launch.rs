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

use crate::api::{BuildContext, OperatorFactory, OperatorGraphBuilder, WorkerMaker};
use crate::operations::channels::{MpscSender, Sender};
use crate::stats::DataFlowStats;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

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
    fn cancelled(&self) -> &Arc<AtomicBool>;
    fn err_tx(&self) -> &mpsc::Sender<crate::data_flow::Error>;
    fn stats_tx(&self) -> &mpsc::Sender<DataFlowStats>;
    fn collect_stats(&self) -> bool;
    #[cfg(feature = "perf")]
    fn profiled(&self) -> bool;
}

/// The launch of a spec whose per-worker chains a maker of `OF` produces.
pub(crate) struct Launch<O, OF> {
    make: Box<dyn WorkerMaker<OF>>,
    /// The dataflow's one output sender; every worker's chain sends through
    /// a handle on this launch.
    output: MpscSender<O>,
    err_tx: mpsc::Sender<crate::data_flow::Error>,
    stats_tx: mpsc::Sender<DataFlowStats>,
    cancelled: Arc<AtomicBool>,
    collect_stats: bool,
    #[cfg(feature = "perf")]
    profiled: bool,
}

impl<O, OF> Launch<O, OF> {
    pub(crate) fn new(
        make: Box<dyn WorkerMaker<OF>>,
        output: MpscSender<O>,
        err_tx: mpsc::Sender<crate::data_flow::Error>,
        stats_tx: mpsc::Sender<DataFlowStats>,
        cancelled: Arc<AtomicBool>,
        collect_stats: bool,
        #[cfg(feature = "perf")] profiled: bool,
    ) -> Self {
        Self {
            make,
            output,
            err_tx,
            stats_tx,
            cancelled,
            collect_stats,
            #[cfg(feature = "perf")]
            profiled,
        }
    }
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
        let factory = self.make.make(worker_idx);
        let output = Box::new(LaunchOutput { launch: self });
        Box::new(factory).build(output, context)
    }

    fn cancelled(&self) -> &Arc<AtomicBool> {
        &self.cancelled
    }

    fn err_tx(&self) -> &mpsc::Sender<crate::data_flow::Error> {
        &self.err_tx
    }

    fn stats_tx(&self) -> &mpsc::Sender<DataFlowStats> {
        &self.stats_tx
    }

    fn collect_stats(&self) -> bool {
        self.collect_stats
    }

    #[cfg(feature = "perf")]
    fn profiled(&self) -> bool {
        self.profiled
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
        fn cancelled(&self) -> &Arc<AtomicBool> {
            unreachable!()
        }
        fn err_tx(&self) -> &mpsc::Sender<crate::data_flow::Error> {
            unreachable!()
        }
        fn stats_tx(&self) -> &mpsc::Sender<DataFlowStats> {
            unreachable!()
        }
        fn collect_stats(&self) -> bool {
            false
        }
        #[cfg(feature = "perf")]
        fn profiled(&self) -> bool {
            false
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
