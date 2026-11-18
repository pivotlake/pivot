//! Scratch state shared by the factories building one worker's operator graph.

use crate::Identifier;
use crate::waker::{WakerSet, waker_set};
use crate::worker::{WORKER_IDX, current_node};
use ahash::HashMap;
use arrow_array::RecordBatch;
use crossbeam_deque::Worker;
use std::rc::Rc;

/// State that outlives a single factory's `build` but not the graph build as a
/// whole, passed down the recursive build chain.
///
/// A chain is built by calls that hand values *downstream only*: a stage creates
/// the channel it reads from and passes the sending end to the chain below it. A
/// CTE needs the reverse. Each of its scan sites creates the channel it will
/// read from, and the chain that produces the CTE's rows, built once by an
/// ancestor of every site, needs all of their sending ends together. Sites leave
/// them here on the way past, and the ancestor collects them.
///
/// Created on the worker thread and dropped when the build ends, so it can hold
/// the `Rc`-shared sending end of a work-stealing channel: neither `Send`, nor
/// creatable before the build.
///
/// It also carries the id of the dataflow being built. The id is decided before
/// any operator is constructed, so an operator that has to register itself with
/// per-worker state (for example to be found again by dataflow) can do so while
/// it is being built, instead of waiting for the finished `DataFlow` to tell it.
///
/// Last, it tells the worker its place in the dataflow. A dataflow numbers
/// the workers that run it from zero, and the nodes they are on likewise, and
/// everything it shares between them (channels, per-node queues, barriers) is
/// laid out by those numbers. They are the pool's own numbers only for a
/// dataflow that runs on the whole pool, so an operator takes them from here
/// and not from the worker thread.
pub struct BuildContext {
    dataflow_id: Identifier,
    /// `None` for an operator built outside a dispatched dataflow, whose
    /// place is the thread's place in the pool.
    placement: Option<Placement>,
    cte_senders: HashMap<usize, Vec<Rc<Worker<RecordBatch>>>>,
}

/// The place of one worker among the workers that run a dataflow.
#[derive(Clone)]
pub struct Placement {
    /// The worker's index among the dataflow's workers.
    pub worker: usize,
    /// The node the worker is on, among the nodes the dataflow runs on.
    pub node: usize,
    /// The wakers of the dataflow's workers, which take the dataflow's
    /// worker and node numbers.
    pub wakers: WakerSet,
}

impl BuildContext {
    /// Start a build of an operator outside a dispatched dataflow, on a
    /// thread set up as a worker of a pool. The thread's place in that pool
    /// is its place in the build.
    pub fn new(dataflow_id: Identifier) -> Self {
        Self {
            dataflow_id,
            placement: None,
            cte_senders: HashMap::default(),
        }
    }

    /// Start the build of `dataflow_id` on the worker that has `placement`
    /// in it.
    pub fn placed(dataflow_id: Identifier, placement: Placement) -> Self {
        Self {
            dataflow_id,
            placement: Some(placement),
            cte_senders: HashMap::default(),
        }
    }

    /// This worker's index among the dataflow's workers.
    pub fn worker_idx(&self) -> usize {
        match &self.placement {
            Some(placement) => placement.worker,
            None => WORKER_IDX.get(),
        }
    }

    /// The node this worker is on, among the nodes the dataflow runs on.
    pub fn node(&self) -> usize {
        match &self.placement {
            Some(placement) => placement.node,
            None => current_node(),
        }
    }

    /// The wakers of the dataflow's workers, which take the dataflow's
    /// worker and node numbers.
    pub fn wakers(&self) -> WakerSet {
        match &self.placement {
            Some(placement) => placement.wakers.clone(),
            None => waker_set().clone(),
        }
    }

    /// The id the `DataFlow` produced by this build will have.
    pub fn dataflow_id(&self) -> Identifier {
        self.dataflow_id
    }

    /// Leave one scan site's sending end for CTE `id` to pick up.
    pub fn deposit_cte_sender(&mut self, id: usize, sender: Rc<Worker<RecordBatch>>) {
        self.cte_senders.entry(id).or_default().push(sender);
    }

    /// Take every sending end left for CTE `id`, in the order the sites were
    /// built. Taking them empties the entry, so a second CTE with the same id
    /// (which cannot happen: ids are minted per compile) would find nothing.
    pub fn take_cte_senders(&mut self, id: usize) -> Vec<Rc<Worker<RecordBatch>>> {
        self.cte_senders.remove(&id).unwrap_or_default()
    }
}
