//! A DataFlow is a graph of operators that a single worker executes.
//!
//! Built from an [`OperatorGraphBuilder`](crate::api::OperatorGraphBuilder) during
//! [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build)
//! on the worker thread. Operators are connected by channels created during the
//! build step; a graph may contain multiple roots, such as a join's build and probe paths.
//!
//! The worker drives execution by calling methods on the `DataFlow`:
//! - [`run_ready_cpu_work`](DataFlow::run_ready_cpu_work) — traverse leaf-to-root,
//!   running the first operator that has work ready. Leaf-to-root (downstream first)
//!   keeps data hot in cache — we process what was just produced before moving upstream.
//! - [`try_stealing_work`](DataFlow::try_stealing_work) — traverse root-to-leaf,
//!   attempting to steal from peer workers' channels. Root-to-leaf (upstream first)
//!   means the stealing worker picks up data early in the dataflow, to not interrupt current
//!   hot-in-cache processing
//! - [`get_next_fs_request`](DataFlow::get_next_fs_request) /
//!   [`get_next_http_request`](DataFlow::get_next_http_request) — collect pending IO
//!   requests from operators (e.g. parquet page reads, or HTTP range reads).
//! - [`process_fs_read`](DataFlow::process_fs_read) /
//!   [`process_fs_write`](DataFlow::process_fs_write),
//!   [`process_http_get_response`](DataFlow::process_http_get_response), and
//!   [`process_http_upload_response`](DataFlow::process_http_upload_response) —
//!   deliver completed IO to the operator that requested it.
//! - [`maybe_finish`](DataFlow::maybe_finish) — check if all operators have completed.

use crate::Identifier;
use crate::io::{
    DataFlowRequest, FsReadRequest, FsRequest, FsWriteRequest, HttpGetRequest, HttpRequest,
    HttpUploadRequest,
};
use crate::operations::{FinishStatus, Operator};
use crate::stats::{DataFlowStats, StatsCollector};
use crate::waker::waker_set;
use ahash::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::ControlFlow;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;
use std::{panic, result};
use thiserror::Error;
use tracing::{debug, error, warn};

/// One node of the operator graph. Owns its operator; edges are tracked
/// in the parent `DataFlow` as index lists.
struct OperatorNode {
    id: Identifier,
    /// The node's operator, or `None` once it has finished (or been abandoned).
    /// Dropping it at that moment frees what it held right away: the batches
    /// buffered in its channels, its hash tables, and the downstream sender it
    /// owns. Releasing that sender is also how a stage's completion becomes
    /// observable to whoever holds the other end.
    ///
    /// Retiring the node in place, rather than restructuring the graph, keeps
    /// every edge index stable: traversals skip a retired node and move on, and
    /// a late IO completion routed to one lands nowhere.
    operator: Option<Box<dyn Operator>>,
    /// One-shot publication gates. Open gates are removed after an Acquire
    /// observation, leaving no atomic loads on this node's steady-state path.
    gates: Vec<Arc<AtomicBool>>,
}

impl OperatorNode {
    /// Whether this node has retired, which is exactly "it no longer holds an
    /// operator".
    #[inline]
    fn is_finished(&self) -> bool {
        self.operator.is_none()
    }

    /// Run `f` against this node's operator, or return `None` when the node has
    /// nothing to run because it retired or is still gated.
    #[inline]
    fn run<T>(
        &mut self,
        f: &mut impl FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    ) -> Result<Option<ControlFlow<T>>> {
        if self.is_gated() {
            return Ok(None);
        }
        let id = self.id;
        match self.operator.as_deref_mut() {
            Some(operator) => f(id, operator).map(Some),
            None => Ok(None),
        }
    }

    #[inline]
    fn is_gated(&mut self) -> bool {
        let gated_before = self.gates.len();
        self.gates.retain(|gate| !gate.load(Ordering::Acquire));
        // A gate opened since the last look. While gated, this operator was
        // skipped by every IO collection pass, so IO it staged may sit behind
        // a cleared pending-IO flag; re-flag so the next pass walks to it.
        if self.gates.len() != gated_before {
            crate::io::note_pending_io();
        }
        !self.gates.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Cannot find operation {0}")]
    CannotFindOperation(Identifier),
    #[error(transparent)]
    Operation(#[from] crate::operations::Error),
    #[error(transparent)]
    IORequester(#[from] crate::io::IORequesterError),
    #[error("{0}")]
    Panic(String),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// Whether an operator did useful work in this step.
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum WorkStatus {
    /// No work was available or ready.
    Pending,
    /// The operator consumed or produced data.
    Ran,
}

/// Directed graph of operators with precomputed roots and leaves.
///
/// Each node has at most one downstream subscriber (`edges`) and zero or more
/// upstream publishers (`back_edges`). Supports two traversal orders:
/// - **Backwards** (leaf-to-root): for running CPU work and collecting IO requests.
/// - **Forwards** (root-to-leaf): for stealing and finishing.
struct OperatorGraph {
    operators: Vec<OperatorNode>,
    leafs: Vec<usize>,
    roots: Vec<usize>,
    /// Subscriber index for each node, or `None` if the node is a leaf.
    edges: Vec<Option<usize>>,
    /// Publisher indices for each node.
    back_edges: Vec<Vec<usize>>,
}

impl OperatorGraph {
    fn from_edges(
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscriber: HashMap<Identifier, Identifier>,
        mut gates: HashMap<Identifier, Vec<Arc<AtomicBool>>>,
    ) -> Self {
        debug!(
            "Building from publisher to subscriber {:?}",
            publisher_to_subscriber
        );

        let n = operators.len();
        let mut edges: Vec<Option<usize>> = vec![None; n];
        let mut back_edges: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (&publisher, &sub) in &publisher_to_subscriber {
            edges[publisher] = Some(sub);
            back_edges[sub].push(publisher);
        }

        let operators: Vec<OperatorNode> = operators
            .into_iter()
            .enumerate()
            .map(|(id, operator)| OperatorNode {
                id,
                operator: Some(operator),
                gates: gates.remove(&id).unwrap_or_default(),
            })
            .collect();

        let roots = (0..n).filter(|&i| back_edges[i].is_empty()).collect();
        let leafs = (0..n).filter(|&i| edges[i].is_none()).collect();

        Self {
            operators,
            roots,
            leafs,
            edges,
            back_edges,
        }
    }

    /// Walk leaf-to-root via back-edges, running the closure against every node
    /// that still has work to give: a retired or gated node is stepped over, but
    /// the walk carries on through it to its publishers.
    pub fn traverse_backwards<
        T,
        F: FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    >(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut stack: Vec<usize> = self.leafs.clone();
        while let Some(idx) = stack.pop() {
            if let Some(res) = self.operators[idx].run(&mut f)? {
                if matches!(res, ControlFlow::Break(..)) {
                    return Ok(res);
                }
            }
            stack.extend(self.back_edges[idx].iter().copied());
        }
        Ok(ControlFlow::Continue(()))
    }

    /// The root-to-leaf twin of [`traverse_backwards`](Self::traverse_backwards).
    pub fn traverse_forwards<
        T,
        F: FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    >(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut stack: Vec<usize> = self.roots.clone();
        while let Some(idx) = stack.pop() {
            if let Some(res) = self.operators[idx].run(&mut f)? {
                if matches!(res, ControlFlow::Break(..)) {
                    return Ok(res);
                }
            }
            if let Some(child) = self.edges[idx] {
                stack.push(child);
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Advance finishing independently from every root. A pending node blocks
    /// only its own downstream path, not sibling roots.
    fn try_finish(&mut self) -> Result<FinishStatus> {
        let mut working = false;

        for root in self.roots.clone() {
            let mut next = Some(root);
            while let Some(idx) = next {
                // A converging node cannot finish before all of its publishers.
                if self.back_edges[idx]
                    .iter()
                    .any(|&publisher| !self.operators[publisher].is_finished())
                {
                    break;
                }

                let node = &mut self.operators[idx];
                if node.is_gated() {
                    break;
                }
                if let Some(operator) = node.operator.as_deref_mut() {
                    match operator.try_finish()? {
                        // Retiring the node drops the operator, and with it the
                        // sender it holds, so a stage that watches the other end
                        // of that channel learns the stage is over.
                        FinishStatus::Done => node.operator = None,
                        FinishStatus::Working => working = true,
                        FinishStatus::Pending => {}
                    }
                }

                if !node.is_finished() {
                    break;
                }
                next = self.edges[idx];
            }
        }

        Ok(if self.operators.iter().all(OperatorNode::is_finished) {
            FinishStatus::Done
        } else if working {
            FinishStatus::Working
        } else {
            FinishStatus::Pending
        })
    }

    /// Abandon every operator transitively *upstream* of `node` (its publishers,
    /// their publishers, and so on) by retiring each of them. The node itself and
    /// everything downstream are left running. Used by a satisfied `LIMIT` to
    /// stop and free the scan feeding it.
    ///
    /// This walks `back_edges`. Because every node has at most one downstream
    /// subscriber, every ancestor reached this way can feed downstream only
    /// through `node`, even when the dataflow contains multiple roots.
    fn abandon_ancestors(&mut self, node: usize) {
        let mut stack: Vec<usize> = self.back_edges[node].clone();
        while let Some(idx) = stack.pop() {
            if self.operators[idx].is_finished() {
                // Guards against re-visiting on a DAG and against re-abandoning
                // across repeated calls; on a chain it simply never triggers.
                continue;
            }
            self.operators[idx].operator = None;
            stack.extend(self.back_edges[idx].iter().copied());
        }
    }
}

/// A graph of operators executed by a single worker.
///
/// Operators are connected by channels and arranged in an [`OperatorGraph`].
/// The worker drives execution by repeatedly calling [`run_ready_cpu_work`](Self::run_ready_cpu_work),
/// [`try_stealing_work`](Self::try_stealing_work), and IO methods.
pub struct DataFlow {
    id: Identifier,
    cancelled: Arc<AtomicBool>,
    err_tx: mpsc::Sender<Error>,
    graph: OperatorGraph,
    /// Operators that can ask to abandon their upstream, paired with the flag
    /// they raise to do so (see [`Operator::upstream_cancel_flag`]). Collected
    /// once at construction so the hot loop only polls the (usually empty) list,
    /// never every operator. An entry is dropped once it has fired.
    upstream_cancellers: Vec<(usize, Arc<AtomicBool>)>,
    /// Collects this worker's execution stats for the dataflow. Inert unless the
    /// query opted in; driven by the worker through [`stats`](Self::stats).
    stats: StatsCollector,
    /// Whether this dataflow is being profiled. While any profiled dataflow is
    /// live the worker runs *only* profiled dataflows (see
    /// [`crate::profiler`]), so a `perf` capture isn't polluted by other work
    /// sharing the pool. Set when the launching dispatcher was marked (see
    /// [`DataFlowDispatcher::with_profiling`](crate::DataFlowDispatcher::with_profiling)).
    #[cfg(feature = "perf")]
    profiled: bool,
}

impl Debug for DataFlow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Dataflow");
        debug_struct.field("id", &self.id).finish()
    }
}

#[cfg(feature = "perf")]
impl Drop for DataFlow {
    fn drop(&mut self) {
        // Release this dataflow's hold on exclusive execution once it finishes
        // (or is cancelled/dropped), paired with the bump in `with_profiling`.
        if self.profiled && crate::profiler::PROFILED_FLOWS.fetch_sub(1, Ordering::Relaxed) == 1 {
            // Exclusive mode just lifted. A worker that parked while holding only
            // non-profiled (paused) dataflows got no channel send to wake it, so
            // notify every node unconditionally or it sleeps until the next
            // unrelated wake.
            waker_set().notify_all();
        }
    }
}

impl DataFlow {
    #[allow(clippy::too_many_arguments)] // wired straight from the builder; each is needed
    pub fn new(
        id: Identifier,
        canceled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<Error>,
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscriber: HashMap<Identifier, Identifier>,
        gates: HashMap<Identifier, Vec<Arc<AtomicBool>>>,
        stats_tx: mpsc::Sender<DataFlowStats>,
        collect_stats: bool,
    ) -> Self {
        let graph = OperatorGraph::from_edges(operators, publisher_to_subscriber, gates);
        let upstream_cancellers = graph
            .operators
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                node.operator
                    .as_ref()?
                    .upstream_cancel_flag()
                    .map(|flag| (idx, flag))
            })
            .collect();
        Self {
            id,
            graph,
            upstream_cancellers,
            cancelled: canceled,
            err_tx,
            stats: StatsCollector::new(stats_tx, collect_stats),
            #[cfg(feature = "perf")]
            profiled: false,
        }
    }
    pub fn id(&self) -> Identifier {
        self.id
    }

    /// Mark this dataflow profiled. Bumps the global live-profiled count (paired
    /// with the decrement in [`Drop`]) so the worker switches to running only
    /// profiled dataflows while this one exists. Call exactly once per dataflow
    /// (it is not idempotent): a second `true` call leaks a count that never
    /// drops, wedging exclusive mode on. `pub(crate)`: only the builder uses it.
    #[cfg(feature = "perf")]
    pub(crate) fn with_profiling(mut self, profiled: bool) -> Self {
        if profiled {
            crate::profiler::PROFILED_FLOWS.fetch_add(1, Ordering::Relaxed);
        }
        self.profiled = profiled;
        self
    }

    /// Whether this dataflow is profiled (and so runs exclusively).
    #[cfg(feature = "perf")]
    pub fn is_profiled(&self) -> bool {
        self.profiled
    }

    /// This dataflow's stats collector. The worker records IO/CPU work against it
    /// (`flow.stats().record_*`) and `report`s it when the dataflow finishes.
    pub fn stats(&mut self) -> &mut StatsCollector {
        &mut self.stats
    }

    /// Is this dataflow cancelled? This is an AtomicBool that can be set from other workers or from
    /// outside dispatch
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Bail on running the current dataflow because of an error. This will set `canceled` to true
    /// (cancelling it for all workers) and send the error out. Called both from inside
    /// [`try_run`](Self::try_run) and directly by the worker when work it submitted on the
    /// dataflow's behalf (e.g. a remote read) fails terminally.
    pub fn bail_and_cancel(&self, err: Error) {
        error!("DataFlow {:?} failed: {err}", self.id());
        if self.err_tx.send(err).is_err() {
            warn!("Unable to send error...");
        }
        // Cancellation must reach workers parked on any node.
        self.cancelled.store(true, Ordering::Relaxed);
        waker_set().notify_all();
    }

    /// Try running a function within the dataflow- this will gracefully catch any errors/panics and
    /// send them via `err_tx`, along with setting `cancelled` to true across all workers (thus
    /// causing the dataflow to stop processing).
    fn try_run(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) {
        self.try_run_or((), f);
    }

    /// Similar to the above function, `try_run_or` will run catch any errors and handle them
    /// appropriately, while also returning a default value if there is indeed an error
    fn try_run_or<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {
        // Every operator step funnels through here, so it's the one place that
        // can see the dataflow's CPU work (the worker only sees opaque method
        // calls). `then` skips the clock reads when stats are off.
        let started = self.stats.enabled().then(Instant::now);
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| f(self)));
        if let Some(started) = started {
            self.stats.record_cpu(started.elapsed());
        }
        match outcome {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                self.bail_and_cancel(e);
                default
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                self.bail_and_cancel(Error::Panic(msg.to_string()));
                default
            }
        }
    }

    /// Drive every operator's finish (root-to-leaf) and report the dataflow's
    /// aggregate [`FinishStatus`]: [`Done`](FinishStatus::Done) when all
    /// operators have finished (the worker drops the dataflow),
    /// [`Working`](FinishStatus::Working) when one is still producing output
    /// (the worker must keep driving rather than park), or
    /// [`Pending`](FinishStatus::Pending) when stalled on input/siblings (the
    /// worker may park).
    ///
    /// An operator that reports [`FinishStatus::Done`] is dropped there and then,
    /// so subsequent passes step over its node.
    pub fn maybe_finish(&mut self) -> FinishStatus {
        self.try_run_or(FinishStatus::Pending, |d| d.graph.try_finish())
    }

    /// Honour any pending upstream-cancellation request: for each operator that
    /// has raised its [`upstream_cancel_flag`](Operator::upstream_cancel_flag)
    /// (e.g. a satisfied `LIMIT`), abandon everything upstream of it. Operators
    /// downstream are left running.
    ///
    /// Called once per worker-loop iteration. When no operator can cancel
    /// upstream (the common case) `upstream_cancellers` is empty and this is a
    /// single `is_empty` check.
    pub fn cancel_upstream_if_requested(&mut self) {
        if self.upstream_cancellers.is_empty() {
            return;
        }
        // Retain only the cancellers that haven't fired yet; abandon the
        // ancestors of the ones that have, so each request is acted on once.
        let graph = &mut self.graph;
        self.upstream_cancellers.retain(|(node, flag)| {
            if flag.load(Ordering::Relaxed) {
                graph.abandon_ancestors(*node);
                false
            } else {
                true
            }
        });
    }

    /// Notify the operator that requested it that one of its filesystem reads
    /// has landed (already committed into the cache slot by the requester).
    pub fn process_fs_read(&mut self, node_id: Identifier, request: FsReadRequest) {
        self.try_run(|d| {
            if let Some(operator) = d.graph.operators[node_id].operator.as_deref_mut() {
                operator.process_fs_read_response(request)?;
            }
            Ok(())
        });
    }

    /// Notify the operator that requested it that one of its filesystem writes
    /// has completed.
    pub fn process_fs_write(&mut self, node_id: Identifier, request: FsWriteRequest) {
        self.try_run(|d| {
            if let Some(operator) = d.graph.operators[node_id].operator.as_deref_mut() {
                operator.process_fs_write_response(request)?;
            }
            Ok(())
        });
    }

    /// Notify the operator that requested it that one of its HTTP GETs has
    /// landed (already committed into the cache slot by the requester).
    pub fn process_http_get_response(&mut self, node_id: Identifier, request: HttpGetRequest) {
        self.try_run(|d| {
            if let Some(operator) = d.graph.operators[node_id].operator.as_deref_mut() {
                operator.process_http_get_response(request)?;
            }
            Ok(())
        });
    }

    /// Notify the operator that requested it that one of its HTTP uploads has
    /// completed.
    pub fn process_http_upload_response(
        &mut self,
        node_id: Identifier,
        request: HttpUploadRequest,
    ) {
        self.try_run(|d| {
            if let Some(operator) = d.graph.operators[node_id].operator.as_deref_mut() {
                operator.process_http_upload_response(request)?;
            }
            Ok(())
        });
    }

    /// Run one unit of CPU work, traversing leaf-to-root (downstream first for cache locality).
    /// Returns [`WorkStatus::Ran`] if any operator did work.
    pub fn run_ready_cpu_work(&mut self) -> WorkStatus {
        self.try_run_or(WorkStatus::Ran, |d| {
            d.graph
                .traverse_backwards(|_, operator| match operator.run_cpu_work()? {
                    WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                    WorkStatus::Ran => Ok(ControlFlow::Break(())),
                })
                .map(|c| match c {
                    ControlFlow::Continue(_) => WorkStatus::Pending,
                    ControlFlow::Break(_) => WorkStatus::Ran,
                })
        })
    }

    /// Attempt to steal work from peer workers, traversing root-to-leaf (upstream first
    /// so the original worker's downstream data stays hot).
    pub fn try_stealing_work(&mut self) -> WorkStatus {
        self.try_run_or(WorkStatus::Ran, |d| {
            d.graph
                .traverse_forwards(|_, operator| match operator.try_steal_work()? {
                    WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                    WorkStatus::Ran => Ok(ControlFlow::Break(())),
                })
                .map(|c| match c {
                    ControlFlow::Continue(_) => WorkStatus::Pending,
                    ControlFlow::Break(_) => WorkStatus::Ran,
                })
        })
    }

    /// Collect pending filesystem read requests from operators (leaf-to-root).
    /// Returns the first batch found, or `None` if no operator needs disk IO.
    pub fn get_next_fs_request(&mut self) -> Option<Vec<DataFlowRequest<FsRequest>>> {
        self.try_run_or(None, |d| {
            d.graph
                .traverse_backwards(|id, operator| {
                    let requests = operator.next_fs_requests()?;
                    if !requests.is_empty() {
                        Ok(ControlFlow::Break(
                            requests
                                .into_iter()
                                .map(|r| DataFlowRequest::new(d.id, id, r))
                                .collect(),
                        ))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })
                .map(|c| c.break_value())
        })
    }

    /// Collect pending HTTP requests from operators (leaf-to-root). Mirrors
    /// [`get_next_fs_request`](Self::get_next_fs_request).
    pub fn get_next_http_request(&mut self) -> Option<Vec<DataFlowRequest<HttpRequest>>> {
        self.try_run_or(None, |d| {
            d.graph
                .traverse_backwards(|id, operator| {
                    let requests = operator.next_http_requests()?;
                    if !requests.is_empty() {
                        Ok(ControlFlow::Break(
                            requests
                                .into_iter()
                                .map(|r| DataFlowRequest::new(d.id, id, r))
                                .collect(),
                        ))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })
                .map(|c| c.break_value())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A test operator that always reports it `Ran` and never finishes on its
    /// own, so a node still holding one is visibly "live".
    struct LiveOperator;
    impl Operator for LiveOperator {
        fn run_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
            Ok(WorkStatus::Ran)
        }
        fn next_fs_requests(&mut self) -> crate::operations::Result<Vec<FsRequest>> {
            Ok(vec![])
        }
        fn process_fs_read_response(
            &mut self,
            _request: FsReadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_get_response(
            &mut self,
            _request: HttpGetRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(FinishStatus::Pending)
        }
    }

    /// Reports a fixed [`FinishStatus`], and optionally raises a flag when it is
    /// dropped so a test can watch for the node letting go of it.
    struct FinishOperator {
        status: FinishStatus,
        dropped: Option<Arc<AtomicBool>>,
    }

    impl FinishOperator {
        fn new(status: FinishStatus) -> Self {
            Self {
                status,
                dropped: None,
            }
        }

        fn signalling_drop(status: FinishStatus, dropped: Arc<AtomicBool>) -> Self {
            Self {
                status,
                dropped: Some(dropped),
            }
        }
    }

    impl Drop for FinishOperator {
        fn drop(&mut self) {
            if let Some(dropped) = &self.dropped {
                dropped.store(true, Ordering::Release);
            }
        }
    }

    impl Operator for FinishOperator {
        fn run_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
            Ok(WorkStatus::Pending)
        }
        fn next_fs_requests(&mut self) -> crate::operations::Result<Vec<FsRequest>> {
            Ok(vec![])
        }
        fn process_fs_read_response(
            &mut self,
            _request: FsReadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_get_response(
            &mut self,
            _request: HttpGetRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(self.status)
        }
    }

    struct CountingOperator(Arc<AtomicUsize>);
    impl Operator for CountingOperator {
        fn run_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(WorkStatus::Ran)
        }
        fn next_fs_requests(&mut self) -> crate::operations::Result<Vec<FsRequest>> {
            Ok(vec![])
        }
        fn process_fs_read_response(
            &mut self,
            _request: FsReadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_get_response(
            &mut self,
            _request: HttpGetRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(FinishStatus::Pending)
        }
    }

    /// Build a linear chain `0 -> 1 -> ... -> n-1` of [`LiveOperator`]s, where
    /// node 0 is the source and node n-1 is the leaf.
    fn live_chain(n: usize) -> OperatorGraph {
        let operators: Vec<Box<dyn Operator>> = (0..n)
            .map(|_| Box::new(LiveOperator) as Box<dyn Operator>)
            .collect();
        let publisher_to_subscriber: HashMap<Identifier, Identifier> =
            (0..n - 1).map(|i| (i, i + 1)).collect();
        OperatorGraph::from_edges(operators, publisher_to_subscriber, HashMap::default())
    }

    /// An abandoned node has given up its operator; a node still holding a
    /// [`LiveOperator`] has not.
    fn is_abandoned(graph: &OperatorGraph, idx: usize) -> bool {
        graph.operators[idx].is_finished()
    }

    #[test]
    fn a_node_lets_go_of_its_operator_once_it_finishes() {
        let dropped = Arc::new(AtomicBool::new(false));
        let operators: Vec<Box<dyn Operator>> = vec![Box::new(FinishOperator::signalling_drop(
            FinishStatus::Done,
            dropped.clone(),
        ))];
        let mut graph =
            OperatorGraph::from_edges(operators, HashMap::default(), HashMap::default());

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Done);

        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn finishing_pending_root_does_not_block_other_roots() {
        let operators: Vec<Box<dyn Operator>> = vec![
            Box::new(FinishOperator::new(FinishStatus::Pending)),
            Box::new(FinishOperator::new(FinishStatus::Done)),
        ];
        let mut graph =
            OperatorGraph::from_edges(operators, HashMap::default(), HashMap::default());

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        assert!(!graph.operators[0].is_finished());
        assert!(graph.operators[1].is_finished());
    }

    #[test]
    fn closed_gate_blocks_cpu_work_until_published() {
        let runs = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(AtomicBool::new(false));
        let mut gates = HashMap::default();
        gates.insert(0, vec![gate.clone()]);
        let (err_tx, _err_rx) = mpsc::channel();
        let (stats_tx, _stats_rx) = mpsc::channel();
        let mut flow = DataFlow::new(
            0,
            Arc::new(AtomicBool::new(false)),
            err_tx,
            vec![Box::new(CountingOperator(runs.clone()))],
            HashMap::default(),
            gates,
            stats_tx,
            false,
        );

        assert!(matches!(flow.run_ready_cpu_work(), WorkStatus::Pending));
        assert_eq!(runs.load(Ordering::Relaxed), 0);

        gate.store(true, Ordering::Release);
        assert!(matches!(flow.run_ready_cpu_work(), WorkStatus::Ran));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(flow.graph.operators[0].gates.is_empty());
    }

    #[test]
    fn root_waits_for_all_gates_before_finishing() {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut gates = HashMap::default();
        gates.insert(0, vec![first.clone(), second.clone()]);
        let operators: Vec<Box<dyn Operator>> =
            vec![Box::new(FinishOperator::new(FinishStatus::Done))];
        let mut graph = OperatorGraph::from_edges(operators, HashMap::default(), gates);

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        first.store(true, Ordering::Release);
        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        second.store(true, Ordering::Release);
        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Done);
    }

    #[test]
    fn abandon_ancestors_tears_down_upstream_only() {
        // Chain: 0 (source) -> 1 -> 2 (limit) -> 3 (downstream, e.g. GROUP BY).
        let mut graph = live_chain(4);

        graph.abandon_ancestors(2);

        // Everything upstream of the limit is abandoned and freed.
        assert!(is_abandoned(&graph, 0));
        assert!(is_abandoned(&graph, 1));
        // The limit and everything downstream keep running.
        assert!(!is_abandoned(&graph, 2));
        assert!(!is_abandoned(&graph, 3));
    }

    #[test]
    fn abandon_ancestors_of_leaf_spares_nothing_downstream() {
        // The whole chain is upstream of the leaf, so all but the leaf go.
        let mut graph = live_chain(3);

        graph.abandon_ancestors(2);

        assert!(is_abandoned(&graph, 0));
        assert!(is_abandoned(&graph, 1));
        assert!(!is_abandoned(&graph, 2));
    }
}
