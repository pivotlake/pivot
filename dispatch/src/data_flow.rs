//! A DataFlow is a graph of operators that a single worker executes.
//!
//! Built from a [`Chain`](crate::api::Chain) during [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build)
//! on the worker thread. The operators are arranged in a directed graph (currently
//! always a linear chain), connected by channels created during the build step.
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
//! - [`process_io`](DataFlow::process_io) — deliver a completed IO buffer to the
//!   operator that requested it.
//! - [`maybe_finish`](DataFlow::maybe_finish) — check if all operators have completed.

use crate::Identifier;
use crate::io::{DataFlowRequest, IORequest};
use crate::operations::Operator;
use crate::worker::worker_waker;
use ahash::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::ControlFlow;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::{panic, result};
use thiserror::Error;
use tracing::{debug, error, warn};

/// One node of the operator graph. Owns its operator; edges are tracked
/// in the parent `DataFlow` as index lists.
struct OperatorNode {
    id: Identifier,
    operator: Box<dyn Operator>,
    /// turns true once `try_finish` returns `true`. Future traversals skip
    /// this node so each operator's try_finish isn't called after it reports
    /// done.
    finished: bool,
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
                operator,
                finished: false,
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

    /// Walk leaf-to-root via back-edges. The closure receives `&mut OperatorNode`
    /// so it can drive the operator and read/update `finished` or other properties.
    pub fn traverse_backwards<T, F: FnMut(&mut OperatorNode) -> Result<ControlFlow<T>>>(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut stack: Vec<usize> = self.leafs.clone();
        while let Some(idx) = stack.pop() {
            let res = f(&mut self.operators[idx])?;
            if matches!(res, ControlFlow::Break(..)) {
                return Ok(res);
            }
            stack.extend(self.back_edges[idx].iter().copied());
        }
        Ok(ControlFlow::Continue(()))
    }

    pub fn traverse_forwards<T, F: FnMut(&mut OperatorNode) -> Result<ControlFlow<T>>>(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut stack: Vec<usize> = self.roots.clone();
        while let Some(idx) = stack.pop() {
            let res = f(&mut self.operators[idx])?;
            if matches!(res, ControlFlow::Break(..)) {
                return Ok(res);
            }
            if let Some(child) = self.edges[idx] {
                stack.push(child);
            }
        }
        Ok(ControlFlow::Continue(()))
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
}

impl Debug for DataFlow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Dataflow");
        debug_struct.field("id", &self.id).finish()
    }
}

impl DataFlow {
    pub fn new(
        id: Identifier,
        canceled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<Error>,
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscriber: HashMap<Identifier, Identifier>,
    ) -> Self {
        Self {
            id,
            graph: OperatorGraph::from_edges(operators, publisher_to_subscriber),
            cancelled: canceled,
            err_tx,
        }
    }
    pub fn id(&self) -> Identifier {
        self.id
    }

    /// Is this dataflow cancelled? This is an AtomicBool that can be set from other workers or from
    /// outside dispatch
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Bail on running the current dataflow because of an error. This will set `canceled` to true
    /// (cancelling it for all workers) and send the error out
    fn bail_and_cancel(&self, err: Error) {
        error!("DataFlow {:?} failed: {err}", self.id());
        if self.err_tx.send(err).is_err() {
            warn!("Unable to send error...");
        }
        self.cancelled.store(true, Ordering::Relaxed);
        worker_waker().notify();
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
        match panic::catch_unwind(AssertUnwindSafe(|| f(self))) {
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

    /// Try to finish all operators (root-to-leaf). Returns `true` if every operator
    /// has completed, meaning this dataflow can be removed from the worker.
    ///
    /// Operators that have already reported `try_finish == true` are latched
    /// via `OperatorNode::finished` and skipped on subsequent passes, so each
    /// operator's `try_finish` is invoked at most once after it reports done.
    pub fn maybe_finish(&mut self) -> bool {
        self.try_run_or(false, |d| {
            d.graph
                .traverse_forwards(|node| {
                    if !node.finished {
                        node.finished = node.operator.try_finish()?;
                    }
                    Ok(if node.finished {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    })
                })
                .map(|c| matches!(c, ControlFlow::Continue(..)))
        })
    }

    /// Notify the operator that requested it that one of its reads has landed
    /// (already committed into the cache slot by the requester).
    pub fn process_io(&mut self, node_id: Identifier, request: IORequest) {
        self.try_run(|d| {
            d.graph.operators[node_id]
                .operator
                .process_io_response(request)?;
            Ok(())
        });
    }

    /// Run one unit of CPU work, traversing leaf-to-root (downstream first for cache locality).
    /// Returns [`WorkStatus::Ran`] if any operator did work.
    pub fn run_ready_cpu_work(&mut self) -> WorkStatus {
        self.try_run_or(WorkStatus::Ran, |d| {
            d.graph
                .traverse_backwards(|op| match op.operator.run_cpu_work()? {
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
                .traverse_forwards(|op| match op.operator.try_steal_work()? {
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
    /// The kind-specific [`FsRequest`](crate::io::FsRequest)s convert into the
    /// transport-tagged [`IORequest`] here.
    pub fn get_next_fs_request(&mut self) -> Option<Vec<DataFlowRequest>> {
        self.try_run_or(None, |d| {
            d.graph
                .traverse_backwards(|op| {
                    let requests = op.operator.next_fs_requests()?;
                    if !requests.is_empty() {
                        Ok(ControlFlow::Break(
                            requests
                                .into_iter()
                                .map(|r| DataFlowRequest::new(d.id, op.id, r.into()))
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
    /// [`get_next_fs_request`](Self::get_next_fs_request); the kind-specific
    /// [`HttpRequest`](crate::io::HttpRequest)s convert into [`IORequest`] here.
    pub fn get_next_http_request(&mut self) -> Option<Vec<DataFlowRequest>> {
        self.try_run_or(None, |d| {
            d.graph
                .traverse_backwards(|op| {
                    let requests = op.operator.next_http_requests()?;
                    if !requests.is_empty() {
                        Ok(ControlFlow::Break(
                            requests
                                .into_iter()
                                .map(|r| DataFlowRequest::new(d.id, op.id, r.into()))
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
