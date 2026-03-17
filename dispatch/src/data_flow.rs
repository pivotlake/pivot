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
//! - [`try_stealing_cpu_work`](DataFlow::try_stealing_cpu_work) — traverse root-to-leaf,
//!   attempting to steal from peer workers' channels. Root-to-leaf (upstream first)
//!   means the stealing worker picks up data early in the dataflow, giving the original
//!   worker's downstream cache lines time to cool before being touched.
//! - [`get_next_io_request`](DataFlow::get_next_io_request) — collect pending IO
//!   requests from operators (e.g. parquet page reads).
//! - [`process_io`](DataFlow::process_io) — deliver a completed IO buffer to the
//!   operator that requested it.
//! - [`maybe_finish`](DataFlow::maybe_finish) — check if all operators have completed.

use crate::Identifier;
use crate::io::{DataFlowRequest, IORequest};
use crate::memory::ReadBuffer;
use crate::operations::Operator;
use ahash::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::ControlFlow;
use std::result;
use thiserror::Error;
use tracing::debug;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Cannot find operation {0}")]
    CannotFindOperation(Identifier),
    #[error("{0}")]
    Operation(#[from] crate::operations::Error),
    #[error("{0}")]
    IORequester(#[from] crate::io::IORequesterError),
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

/// Directed graph of operators with precomputed roots, leaves, and edges.
///
/// Supports two traversal orders:
/// - **Backwards** (leaf-to-root): for running CPU work and collecting IO requests.
/// - **Forwards** (root-to-leaf): for stealing and finishing.
struct OperatorGraph {
    operators: Vec<Box<dyn Operator>>,

    leafs: Vec<usize>,
    roots: Vec<usize>,

    edges: Vec<Vec<usize>>,
    back_edges: Vec<usize>,
}

impl OperatorGraph {
    fn from_edges(
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
    ) -> Self {
        debug!(
            "Building from publishers to subscribers {:?}",
            publisher_to_subscribers
        );
        let n = operators.len();

        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut back_edges: Vec<usize> = vec![usize::MAX; n]; // MAX = no parent

        for (&publisher, subscribers) in &publisher_to_subscribers {
            edges[publisher] = subscribers.clone();
            for &sub in subscribers {
                back_edges[sub] = publisher;
            }
        }

        let roots = (0..n).filter(|&i| back_edges[i] == usize::MAX).collect();
        let leafs = (0..n).filter(|&i| edges[i].is_empty()).collect();

        Self {
            operators,
            leafs,
            roots,
            edges,
            back_edges,
        }
    }

    pub fn traverse_backwards<
        T,
        F: FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    >(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        for leaf_idx in &self.leafs {
            let mut current_idx = Some(*leaf_idx);

            while let Some(idx) = current_idx {
                let res = f(idx, self.operators[idx].as_mut())?;
                if matches!(res, ControlFlow::Break(..)) {
                    return Ok(res);
                }

                let next_idx = self.back_edges[idx];
                if next_idx == usize::MAX {
                    break;
                }
                current_idx = next_idx.into();
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    pub fn traverse_forwards<T, F: FnMut(usize, &mut dyn Operator) -> Result<ControlFlow<T>>>(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut stack = self.roots.clone();
        while let Some(idx) = stack.pop() {
            let res = f(idx, self.operators[idx].as_mut())?;
            if matches!(res, ControlFlow::Break(..)) {
                return Ok(res);
            }

            for &child_idx in &self.edges[idx] {
                stack.push(child_idx);
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// A graph of operators executed by a single worker.
///
/// Operators are connected by channels and arranged in an [`OperatorGraph`].
/// The worker drives execution by repeatedly calling [`run_ready_cpu_work`](Self::run_ready_cpu_work),
/// [`try_stealing_cpu_work`](Self::try_stealing_cpu_work), and IO methods.
pub struct DataFlow {
    id: Identifier,
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
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
    ) -> Self {
        Self {
            id,
            graph: OperatorGraph::from_edges(operators, publisher_to_subscribers),
        }
    }
    pub fn id(&self) -> Identifier {
        self.id
    }

    /// Try to finish all operators (root-to-leaf). Returns `true` if every operator
    /// has completed, meaning this dataflow can be removed from the worker.
    pub fn maybe_finish(&mut self) -> Result<bool> {
        self.graph
            .traverse_forwards(|_s, b| {
                Ok(if b.try_finish()? {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                })
            })
            .map(|c| matches!(c, ControlFlow::Continue(..)))
    }

    /// Deliver a completed IO buffer to the operator that requested it.
    pub fn process_io(
        &mut self,
        node_id: Identifier,
        request: IORequest,
        buffer: ReadBuffer,
    ) -> Result<()> {
        let op = self.graph.operators.get_mut(node_id).unwrap();
        op.process_disk_response(buffer, request)?;
        Ok(())
    }

    /// Run one unit of CPU work, traversing leaf-to-root (downstream first for cache locality).
    /// Returns [`WorkStatus::Ran`] if any operator did work.
    pub fn run_ready_cpu_work(&mut self) -> Result<WorkStatus> {
        self.graph
            .traverse_backwards(|_, op| match op.run_cpu_work()? {
                WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                WorkStatus::Ran => Ok(ControlFlow::Break(())),
            })
            .map(|c| match c {
                ControlFlow::Continue(_) => WorkStatus::Pending,
                ControlFlow::Break(_) => WorkStatus::Ran,
            })
    }

    /// Attempt to steal work from peer workers, traversing root-to-leaf (upstream first
    /// so the original worker's downstream data stays hot).
    pub fn try_stealing_cpu_work(&mut self) -> Result<WorkStatus> {
        self.graph
            .traverse_forwards(|_id, op| match op.try_steal_cpu_work()? {
                WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                WorkStatus::Ran => Ok(ControlFlow::Break(())),
            })
            .map(|c| match c {
                ControlFlow::Continue(_) => WorkStatus::Pending,
                ControlFlow::Break(_) => WorkStatus::Ran,
            })
    }

    /// Collect pending IO requests from operators (leaf-to-root).
    /// Returns the first batch of requests found, or `None` if no operator needs IO.
    pub fn get_next_io_request(&mut self) -> Result<Option<Vec<DataFlowRequest>>> {
        self.graph
            .traverse_backwards(|id, op| {
                let requests = op.next_io_requests()?;
                if !requests.is_empty() {
                    Ok(ControlFlow::Break(
                        requests
                            .into_iter()
                            .map(|r| DataFlowRequest::new(self.id, id, r))
                            .collect(),
                    ))
                } else {
                    Ok(ControlFlow::Continue(()))
                }
            })
            .map(|c| c.break_value())
    }
}
