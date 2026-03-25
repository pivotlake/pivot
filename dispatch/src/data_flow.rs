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
use std::collections::VecDeque;
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
///
/// Binary operators create DAG structure (multiple parents per node). Both
/// traversals use visited tracking to handle this correctly.
struct OperatorGraph {
    operators: Vec<Box<dyn Operator>>,

    leafs: Vec<usize>,
    roots: Vec<usize>,

    edges: Vec<Vec<usize>>,
    /// Each node can have multiple parents (e.g. a binary operator has two).
    back_edges: Vec<Vec<usize>>,
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
        let mut back_edges: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (&publisher, subscribers) in &publisher_to_subscribers {
            edges[publisher] = subscribers.clone();
            for &sub in subscribers {
                back_edges[sub].push(publisher);
            }
        }

        let roots = (0..n).filter(|&i| back_edges[i].is_empty()).collect();
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
        let mut visited = vec![false; self.operators.len()];
        let mut stack: Vec<usize> = self.leafs.clone();
        while let Some(idx) = stack.pop() {
            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            let res = f(idx, self.operators[idx].as_mut())?;
            if matches!(res, ControlFlow::Break(..)) {
                return Ok(res);
            }

            for &parent in &self.back_edges[idx] {
                stack.push(parent);
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Try to finish all operators. Unlike [`traverse_forwards`](Self::traverse_forwards),
    /// this does not short-circuit: an unfinished operator blocks its children but not
    /// unrelated branches. Returns `true` only if every operator finished.
    pub fn try_finish_all(&mut self) -> Result<bool> {
        let mut remaining_parents: Vec<usize> = self.back_edges.iter().map(|p| p.len()).collect();
        let mut queue: VecDeque<usize> = self.roots.iter().copied().collect();
        let mut all_finished = true;

        while let Some(idx) = queue.pop_front() {
            let finished = self.operators[idx].try_finish()?;
            if !finished {
                all_finished = false;
                continue; // don't enqueue children
            }

            for &child_idx in &self.edges[idx] {
                remaining_parents[child_idx] -= 1;
                if remaining_parents[child_idx] == 0 {
                    queue.push_back(child_idx);
                }
            }
        }
        Ok(all_finished)
    }

    /// Traverse root-to-leaf in topological order (Kahn's algorithm).
    ///
    /// A node is only visited after **all** its parents have been visited.
    /// This guarantees that upstream operators finish before downstream ones,
    /// which is critical for binary operators that have two parent chains.
    pub fn traverse_forwards<T, F: FnMut(usize, &mut dyn Operator) -> Result<ControlFlow<T>>>(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        let mut remaining_parents: Vec<usize> = self.back_edges.iter().map(|p| p.len()).collect();
        let mut queue: VecDeque<usize> = self.roots.iter().copied().collect();

        while let Some(idx) = queue.pop_front() {
            let res = f(idx, self.operators[idx].as_mut())?;
            if matches!(res, ControlFlow::Break(..)) {
                return Ok(res);
            }

            for &child_idx in &self.edges[idx] {
                remaining_parents[child_idx] -= 1;
                if remaining_parents[child_idx] == 0 {
                    queue.push_back(child_idx);
                }
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
    ///
    /// Unlike other traversals, this does **not** short-circuit on a single unfinished
    /// operator. Independent branches (e.g. build vs probe in a join) must all make
    /// progress even if one branch isn't done yet. An operator that returns `false`
    /// still prevents its children from being visited (topological guarantee), but
    /// unrelated branches continue normally.
    pub fn maybe_finish(&mut self) -> Result<bool> {
        self.graph.try_finish_all()
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
    pub fn try_stealing_work(&mut self) -> Result<WorkStatus> {
        self.graph
            .traverse_forwards(|_id, op| match op.try_steal_work()? {
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
