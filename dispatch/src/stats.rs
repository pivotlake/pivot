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
///
/// All three of `cpu`, `http_time`, and `disk_time` are *sums* (over workers,
/// and — for the IO times — over in-flight reads), so they can each exceed the
/// query's wall time. The shape they reveal is the point: a high `*_time`
/// relative to `cpu` means the dataflow spent its time waiting on reads, and
/// `*_time / *_requests` is the average read latency.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DataFlowStats {
    /// HTTP range reads issued (remote object reads).
    pub http_requests: u64,
    /// Wall time those HTTP reads were in flight, summed over every read.
    pub http_time: Duration,
    /// Filesystem block reads issued (local object reads).
    pub disk_requests: u64,
    /// Wall time those disk reads were in flight, summed over every read.
    pub disk_time: Duration,
    /// CPU time spent in this dataflow's operators (decode, group-by, …), summed
    /// across workers. Excludes time parked waiting on IO.
    pub cpu: Duration,
}

impl DataFlowStats {
    /// Fold one worker's tally into the running total.
    pub fn merge(&mut self, other: &Self) {
        self.http_requests += other.http_requests;
        self.http_time += other.http_time;
        self.disk_requests += other.disk_requests;
        self.disk_time += other.disk_time;
        self.cpu += other.cpu;
    }
}
