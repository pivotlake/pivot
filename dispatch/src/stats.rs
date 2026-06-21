//! Per-dataflow execution counters.
//!
//! A [`DataFlow`](crate::data_flow::DataFlow) collects these only when a query
//! opts in — every worker that runs a copy of the dataflow keeps its own tally
//! and ships it back when the copy finishes, and the
//! [`DataFlowHandle`](crate::DataFlowHandle) folds them into one total. They show
//! where a query's time went: how many remote/disk reads it issued and how much
//! CPU it burned (vs. time parked waiting on IO). Collection is gated so it costs
//! nothing when off — see [`DataFlow`](crate::data_flow::DataFlow).

use std::time::Duration;

/// Execution counters for one dataflow, summed across the workers that ran it.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DataFlowStats {
    /// HTTP range reads issued (remote object reads).
    pub http_requests: u64,
    /// Filesystem block reads issued (local object reads).
    pub disk_requests: u64,
    /// CPU time spent in this dataflow's operators (decode, group-by, …), summed
    /// across workers. Excludes time the workers spent parked waiting on IO, so a
    /// query whose wall time dwarfs its `cpu` was IO-bound.
    pub cpu: Duration,
}

impl DataFlowStats {
    /// Fold one worker's tally into the running total.
    pub fn merge(&mut self, other: &Self) {
        self.http_requests += other.http_requests;
        self.disk_requests += other.disk_requests;
        self.cpu += other.cpu;
    }
}
