//! Per-dataflow execution counters and the collector that accumulates them.
//!
//! A [`DataFlow`](crate::data_flow::DataFlow) owns a [`StatsCollector`] and
//! exposes it via `stats()`; the worker drives it (`flow.stats().record_*`) as it
//! issues reads, sees them complete, and runs operator work, then ships the tally
//! with [`report`](StatsCollector::report) when the dataflow finishes. The
//! [`DataFlowHandle`](crate::DataFlowHandle) folds every worker's. Collection is
//! gated so it costs nothing — no clock reads, no counting — when a query didn't
//! opt in.

use crate::io::{DataFlowRequest, ReadyBytesLen, RemoteReadSplit, RemoteReadTime};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Execution counters for one dataflow, summed across the workers that ran it.
///
/// The `cpu` and `*_time` figures are sums (over workers, and for the IO times
/// over in-flight reads), so they can each exceed the query's wall time. The
/// shape they reveal is the point: a high `*_time` relative to `cpu` means the
/// dataflow spent its time waiting on reads, and `*_time / *_requests` is the
/// average read latency. A remote read split across tiers counts each piece in
/// its own tier, so the http and disk-cache counters never hide each other.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DataFlowStats {
    /// HTTP fetches issued, one per hole piece of a remote read. A read split
    /// across the cache and the network counts a fetch here and a cache read under
    /// `disk_cache_requests`. Pairs with `http_time` for an average fetch latency.
    pub http_requests: u64,
    /// Bytes fetched over the network (the hole pieces), summed over every fetch.
    pub http_bytes: u64,
    /// Wall time the network fetches were in flight, summed over every fetch.
    pub http_time: Duration,
    /// Cache-file reads issued, one per resident piece of a remote read (a hit
    /// serving a read or part of one). Pairs with `disk_cache_time`.
    pub disk_cache_requests: u64,
    /// Bytes served from the on-disk cache file (the resident pieces), summed over
    /// every cache read.
    pub disk_cache_bytes: u64,
    /// Wall time the cache-file reads were in flight, summed over every cache read.
    pub disk_cache_time: Duration,
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
        self.disk_cache_requests += other.disk_cache_requests;
        self.disk_cache_bytes += other.disk_cache_bytes;
        self.disk_cache_time += other.disk_cache_time;
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

    /// Stamp each read with its issue time so its completion can bill the read's
    /// in-flight latency. Shared by the local-filesystem and remote submit paths.
    pub fn stamp_issued<R>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        if self.stats.is_some() {
            let now = Instant::now();
            for request in requests.iter_mut() {
                request.submitted_at = Some(now);
            }
        }
    }

    /// Count `requests` as local-filesystem reads issued, total their bytes, and
    /// stamp them for [`record_disk_time`](Self::record_disk_time).
    pub fn record_issued_disk<R: ReadyBytesLen>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        if let Some(stats) = &mut self.stats {
            stats.disk_requests += requests.len() as u64;
            stats.disk_bytes += requests
                .iter()
                .map(|r| r.request.ready_bytes_len())
                .sum::<u64>();
        }
        self.stamp_issued(requests);
    }

    /// Count one remote read's per-piece split: each piece adds a request and its
    /// bytes under the tier that served it (the on-disk cache or the network).
    /// Unlike the disk path this can't count at stamp time, since how a read
    /// splits across the tiers is known only once the cache is consulted.
    pub fn record_issued_remote(&mut self, split: RemoteReadSplit) {
        if let Some(stats) = &mut self.stats {
            stats.disk_cache_requests += split.disk_cache_requests;
            stats.disk_cache_bytes += split.disk_cache_bytes;
            stats.http_requests += split.http_requests;
            stats.http_bytes += split.http_bytes;
        }
    }

    /// Bill a completed remote read's in-flight time, each piece's wait charged to
    /// the tier that served it (see [`RemoteReadTime`]).
    pub fn record_remote_time(&mut self, time: RemoteReadTime) {
        if let Some(stats) = &mut self.stats {
            stats.http_time += time.http;
            stats.disk_cache_time += time.disk_cache;
        }
    }

    /// Bill a completed local-filesystem read's in-flight time, from the
    /// `submitted_at` stamp [`record_issued_disk`](Self::record_issued_disk) left.
    pub fn record_disk_time(&mut self, submitted_at: Option<Instant>) {
        if let (Some(stats), Some(submitted_at)) = (&mut self.stats, submitted_at) {
            stats.disk_time += submitted_at.elapsed();
        }
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

}
