use crate::identified::Identifier;
use crate::io::{IORequest, PipelineRequest};
use crate::operations::Operator;
use ahash::HashMap;
use bytes::Bytes;
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

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum WorkStatus {
    Pending,
    Ran,
}

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

    pub fn roots(&self) -> impl Iterator<Item = &Box<dyn Operator>> {
        self.roots.iter().map(|i| &self.operators[*i])
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

/// A `Pipeline` is the main unit of work of a `Worker`. A Pipeline consists of multiple operations
/// chained together (their connections are described through `publishers_to_subscribers`), and is
/// meant to be executed by a single worker (and thus a core).
pub struct DataFlow {
    id: Identifier,
    graph: OperatorGraph,
}

impl Debug for DataFlow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Pipeline");
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

    pub fn maybe_finish(&mut self) -> Result<bool> {
        self.graph
            .traverse_forwards(|_s, b| {
                // TODO: maybe we should replace with CompeletedNode so we don't call it every time?
                Ok(if b.try_finish()? {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                })
            })
            .map(|c| matches!(c, ControlFlow::Continue(..)))
    }

    pub fn process_io(
        &mut self,
        node_id: Identifier,
        request: IORequest,
        buffer: Bytes,
    ) -> Result<()> {
        let op = self.graph.operators.get_mut(node_id).unwrap();
        op.process_disk_response(buffer, request)?;
        Ok(())
    }

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

    pub fn get_next_io_request(&mut self) -> Result<Option<PipelineRequest>> {
        self.graph
            .traverse_backwards(|id, op| {
                if let Some(r) = op.next_io_request()? {
                    Ok(ControlFlow::Break(PipelineRequest::new(self.id, id, r)))
                } else {
                    Ok(ControlFlow::Continue(()))
                }
            })
            .map(|c| c.break_value())
    }
}
