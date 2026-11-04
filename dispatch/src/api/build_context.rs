//! Scratch state shared by the factories building one worker's operator graph.

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
#[derive(Default)]
pub struct BuildContext {
    cte_senders: HashMap<usize, Vec<Rc<Worker<RecordBatch>>>>,
}

impl BuildContext {
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
