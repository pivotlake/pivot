//! A dataflow handed to the pool: published once, and built by every worker
//! from the same shared description.
//!
//! The thread executing a spec sends one shared [`WorkerLaunch`] to every
//! worker's own inbox and wakes the workers. Each worker takes the launches
//! from its inbox, makes its own operator chain from the launch's maker, and
//! runs it. Every worker reads only its own inbox, so the pickup never
//! contends on anything shared: a lock every worker takes on the same wake
//! would be handed from worker to worker, and the last in line would start
//! its chain hundreds of microseconds late.
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
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

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
