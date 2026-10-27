//! Lock-free summing of every worker's bin totals into the pool-wide totals.

use crate::operations::unary::group::output::topk_pruning::HashBinTotals;
use crossbeam_deque::{Injector, Steal};

/// The pool-wide bin totals under construction: every worker adds its own
/// totals here when it finishes consuming, and the gather barrier's final
/// arrival takes the sum.
///
/// Summing is spread over the workers on purpose. Done by the final arrival
/// alone it would be `workers x HASH_BINS` adds on one thread, the serial
/// tail of every large top-k query on a big machine. Instead each node keeps
/// a lock-free queue of arrays waiting to be summed. An arriving worker
/// merges into its own every array earlier arrivals left in its node's
/// queue, then queues the result. Nothing ever waits: workers arriving
/// together each take a different array and merge in parallel, the queue
/// drains as a tree, and the final arrival's sum is left with the few
/// arrays queued after the last steal. Workers finish consuming within a
/// fraction of a millisecond of each other, so a worker taking only one
/// array would leave half of them for that serial final sum. Per-node
/// queues keep every merge in node-local memory; the final sum is the one
/// step that reads across nodes.
pub(crate) struct SharedBinTotals {
    /// Per node: arrays waiting to be merged into the pool-wide sum.
    pending: Vec<Injector<HashBinTotals>>,
}

impl SharedBinTotals {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            pending: (0..node_count).map(|_| Injector::new()).collect(),
        }
    }

    /// Adds one worker's totals: merges in every array earlier arrivals
    /// left queued, then queues the result.
    pub(crate) fn add(&self, node: usize, mut totals: HashBinTotals) {
        let pending = &self.pending[node];
        loop {
            match pending.steal() {
                Steal::Success(other) => totals.merge(&other),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        pending.push(totals);
    }

    /// Sums every node's queued arrays into the pool-wide totals. Called
    /// once, by the gather barrier's final arrival, after every worker's add.
    /// `None` when no worker added.
    pub(crate) fn take_sum(&self) -> Option<HashBinTotals> {
        let mut sum: Option<HashBinTotals> = None;
        for pending in &self.pending {
            loop {
                match pending.steal() {
                    Steal::Success(totals) => match &mut sum {
                        Some(sum) => sum.merge(&totals),
                        empty => *empty = Some(totals),
                    },
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        sum
    }
}
