//! Per-dataflow execution counters and the collector that accumulates them.
//!
//! A [`DataFlow`](crate::data_flow::DataFlow) owns a [`StatsCollector`] and
//! exposes it via `stats()`; the worker drives it (`flow.stats().record_*`) as it
//! issues reads, sees them complete, and runs operator work, then ships the tally
//! with [`report`](StatsCollector::report) when the dataflow finishes. The
//! [`DataFlowHandle`](crate::DataFlowHandle) folds every worker's. Collection is
//! gated so it costs nothing — no clock reads, no counting — when a query didn't
//! opt in.

use crate::io::{DataFlowRequest, ReadyBytesLen};
use std::sync::mpsc;
use std::time::{Duration, Instant};

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
    /// Bytes read over those HTTP reads, summed over every read.
    pub http_bytes: u64,
    /// Wall time those HTTP reads were in flight, summed over every read.
    pub http_time: Duration,
    /// Filesystem block reads issued (local object reads).
    pub disk_requests: u64,
    /// Bytes read over those disk reads, summed over every read.
    pub disk_bytes: u64,
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
        self.http_bytes += other.http_bytes;
        self.http_time += other.http_time;
        self.disk_requests += other.disk_requests;
        self.disk_bytes += other.disk_bytes;
        self.disk_time += other.disk_time;
        self.cpu += other.cpu;
    }
}

/// Accumulates one worker's [`DataFlowStats`] for a dataflow, or nothing when the
/// query didn't ask for stats — in which case every `record_*` is a no-op and no
/// clock is read. Owned by the [`DataFlow`](crate::data_flow::DataFlow) and driven
/// by the worker through `flow.stats()`.
pub struct StatsCollector {
    /// `Some` only when stats are on.
    stats: Option<DataFlowStats>,
    /// Where [`report`](Self::report) ships the tally for the handle to fold.
    tx: mpsc::Sender<DataFlowStats>,
}

impl StatsCollector {
    pub(crate) fn new(tx: mpsc::Sender<DataFlowStats>, enabled: bool) -> Self {
        Self {
            stats: enabled.then(DataFlowStats::default),
            tx,
        }
    }

    /// Whether collection is on, so a caller can skip a clock read it would only
    /// hand to a no-op.
    pub fn enabled(&self) -> bool {
        self.stats.is_some()
    }

    /// Count `requests` as HTTP reads issued, total their bytes, and stamp each
    /// with the issue time, so its completion can be billed by
    /// [`record_http_time`](Self::record_http_time).
    pub fn record_issued_http<R: ReadyBytesLen>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        self.record_issued(requests, |s| &mut s.http_requests, |s| &mut s.http_bytes);
    }

    /// As [`record_issued_http`](Self::record_issued_http), for disk reads.
    pub fn record_issued_disk<R: ReadyBytesLen>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        self.record_issued(requests, |s| &mut s.disk_requests, |s| &mut s.disk_bytes);
    }

    /// Bill a completed HTTP read's in-flight time, given the `submitted_at` stamp
    /// [`record_issued_http`](Self::record_issued_http) left on it.
    pub fn record_http_time(&mut self, submitted_at: Option<Instant>) {
        self.record_io_time(submitted_at, |s| &mut s.http_time);
    }

    /// As [`record_http_time`](Self::record_http_time), for a disk read.
    pub fn record_disk_time(&mut self, submitted_at: Option<Instant>) {
        self.record_io_time(submitted_at, |s| &mut s.disk_time);
    }

    /// Add `elapsed` to the dataflow's operator CPU time.
    pub fn record_cpu(&mut self, elapsed: Duration) {
        if let Some(stats) = &mut self.stats {
            stats.cpu += elapsed;
        }
    }

    /// Ship this worker's tally for the handle to fold (a no-op when off).
    pub fn report(&self) {
        if let Some(stats) = self.stats {
            let _ = self.tx.send(stats);
        }
    }

    fn record_issued<R: ReadyBytesLen>(
        &mut self,
        requests: &mut [DataFlowRequest<R>],
        count: impl FnOnce(&mut DataFlowStats) -> &mut u64,
        bytes: impl FnOnce(&mut DataFlowStats) -> &mut u64,
    ) {
        if let Some(stats) = &mut self.stats {
            *count(stats) += requests.len() as u64;
            *bytes(stats) += requests
                .iter()
                .map(|r| r.request.ready_bytes_len())
                .sum::<u64>();
            let now = Instant::now();
            for request in requests.iter_mut() {
                request.submitted_at = Some(now);
            }
        }
    }

    fn record_io_time(
        &mut self,
        submitted_at: Option<Instant>,
        field: impl FnOnce(&mut DataFlowStats) -> &mut Duration,
    ) {
        if let (Some(stats), Some(submitted_at)) = (&mut self.stats, submitted_at) {
            *field(stats) += submitted_at.elapsed();
        }
    }
}
