//! What the merge coordinator needs from every worker's consume phase, summed
//! across the pool as the workers arrive rather than by the last arrival alone.

use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::output::topk_pruning::HashBinTotals;
use crossbeam_deque::{Injector, Steal};

/// One worker's merge-sizing summary, or the sum of several workers' once
/// merged: everything `create_partition_jobs` reads per worker that is not a
/// table or buffer it hands to the merge jobs.
///
/// Reading these straight off 192 workers' outputs on one thread costs a
/// cross-core transfer per cache line (a 4 KB sketch and a 512 KB bin array
/// per worker), so they are folded into each other on the way in instead.
pub(crate) struct WorkerSummary {
    /// Distinct-count sketch over the rows of workers that switched to radix.
    pub hll: Hll,
    /// Groups held in place by workers that never switched: their exact
    /// distinct counts, which the sketch does not cover.
    pub non_switched_groups: usize,
    /// The top-k bin totals of the workers that built them.
    pub bin_totals: Option<HashBinTotals>,
    /// Whether every group the summary covers is in `bin_totals`. A worker
    /// with groups but no totals (too small to bin) makes the bounds
    /// undercount, which disables pruning.
    pub every_group_binned: bool,
    /// Raw-scattered rows per scatter bucket of the workers that fell back to
    /// raw scatter, to widen the bounds by (see
    /// [`HashBinTotals::add_raw_bucket_rows`]).
    pub raw_scatter_rows: Option<Vec<u64>>,
}

impl WorkerSummary {
    pub(crate) fn merge(&mut self, other: WorkerSummary) {
        self.hll.merge(&other.hll);
        self.non_switched_groups += other.non_switched_groups;
        self.every_group_binned &= other.every_group_binned;
        if let Some(theirs) = other.bin_totals {
            if let Some(mine) = &mut self.bin_totals {
                mine.merge(&theirs);
            } else {
                self.bin_totals = Some(theirs);
            }
        }
        if let Some(theirs) = other.raw_scatter_rows {
            if let Some(mine) = &mut self.raw_scatter_rows {
                for (bucket, rows) in mine.iter_mut().zip(theirs) {
                    *bucket += rows;
                }
            } else {
                self.raw_scatter_rows = Some(theirs);
            }
        }
    }
}

/// The pool-wide summary under construction: every worker adds its own when
/// it finishes consuming, and the gather barrier's final arrival takes the
/// sum.
///
/// Summing is spread over the workers on purpose. Done by the final arrival
/// alone it would be `workers x` (sketch + bin array) of cross-core reads on
/// one thread, the serial tail of every grouped query on a big machine.
/// Instead each node keeps a lock-free queue of summaries waiting to be
/// merged. An arriving worker merges into its own every summary earlier
/// arrivals left in its node's queue, then queues the result. Nothing ever
/// waits: workers arriving together each take a different summary and merge
/// in parallel, the queue drains as a tree, and the final arrival's sum is
/// left with the few summaries queued after the last steal. Per-node queues
/// keep every merge in node-local memory; the final sum is the one step that
/// reads across nodes.
pub(crate) struct SharedSummaries {
    /// Per node: summaries waiting to be merged into the pool-wide sum.
    pending: Vec<Injector<WorkerSummary>>,
}

impl SharedSummaries {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            pending: (0..node_count).map(|_| Injector::new()).collect(),
        }
    }

    /// Adds one worker's summary: merges in every summary earlier arrivals
    /// left queued, then queues the result.
    pub(crate) fn add(&self, node: usize, mut summary: WorkerSummary) {
        let pending = &self.pending[node];
        loop {
            match pending.steal() {
                Steal::Success(other) => summary.merge(other),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        pending.push(summary);
    }

    /// Sums every node's queued summaries into the pool-wide one. Called
    /// once, by the gather barrier's final arrival, after every worker's
    /// add. `None` when no worker added.
    pub(crate) fn take_sum(&self) -> Option<WorkerSummary> {
        let mut sum: Option<WorkerSummary> = None;
        for pending in &self.pending {
            loop {
                match pending.steal() {
                    Steal::Success(summary) => match &mut sum {
                        Some(sum) => sum.merge(summary),
                        empty => *empty = Some(summary),
                    },
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        sum
    }
}
