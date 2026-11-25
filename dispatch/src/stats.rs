//! Execution counters and the collector that accumulates them for a dataflow.
//!
//! A [`DataFlow`](crate::data_flow::DataFlow) owns a [`StatsCollector`] and
//! exposes it via `stats()`; the worker drives it (`flow.stats().record_*`) as it
//! issues reads, sees them complete, and runs operator work, then ships the tally
//! with [`report`](StatsCollector::report) when the dataflow finishes. The
//! [`DataFlowHandle`](crate::DataFlowHandle) folds every worker's. Collection is
//! gated so it costs nothing — no clock reads, no counting — when a query didn't
//! opt in.
//!
//! A query can also ask for live stats (see [`LiveStatsSubscription`]). Its
//! subscriber first receives the dataflow's operators and how they feed each
//! other, and then, while the dataflow runs, a [`WorkerLiveStats`] message from
//! each worker: the counters of every operator on that worker so far, and
//! whatever [`CustomStat`]s the operators reported, each through its own
//! [`CustomStats`], since the worker's last message.

use crate::Identifier;
use crate::io::{DataFlowRequest, FsRequest, ReadyBytesLen, RemoteReadTime, RemoteSplit};
use crate::worker::WORKER_IDX;
use std::sync::{Arc, Mutex, Once, mpsc};
use std::time::{Duration, Instant};

/// CPU time and IO counters of some work: of one operator on one worker, or,
/// folded together, of a whole dataflow across the workers that ran it.
///
/// The `cpu` and `*_time` figures are sums (over workers, and for the IO times
/// over in-flight operations), so they can each exceed the query's wall time. The
/// shape they reveal is the point: a high `*_time` relative to `cpu` means the
/// work spent its time waiting on I/O. A remote read split across tiers counts
/// each piece in its own tier, so the HTTP and disk-cache counters never hide each
/// other; GET/upload and read/write latency also stay in separate time buckets.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct CpuAndIoCounters {
    /// HTTP operations issued: uploads, plus one fetch per hole piece of a
    /// remote read. A read split across the cache and the network counts a fetch
    /// here and a cache read under `disk_cache_requests`.
    pub http_requests: u64,
    /// Bytes transferred over HTTP, summed over fetches and uploads.
    pub http_bytes: u64,
    /// Wall time HTTP GETs were in flight.
    pub http_get_time: Duration,
    /// HTTP fetches issued again because their connection stalled. A count of
    /// these well above zero means packets are being lost on the way in or the
    /// store is slow to answer.
    pub http_reissues: u64,
    /// Wall time HTTP uploads were in flight.
    pub http_upload_time: Duration,
    /// Cache-file reads issued, one per resident piece of a remote read (a hit
    /// serving a read or part of one). Pairs with `disk_cache_time`.
    pub disk_cache_requests: u64,
    /// Bytes served from the on-disk cache file (the resident pieces), summed over
    /// every cache read.
    pub disk_cache_bytes: u64,
    /// Wall time the cache-file reads were in flight, summed over every cache read.
    pub disk_cache_time: Duration,
    /// Filesystem operations issued (local object reads and writes).
    pub disk_requests: u64,
    /// Bytes transferred by those filesystem operations.
    pub disk_bytes: u64,
    /// Wall time local-filesystem reads were in flight.
    pub disk_read_time: Duration,
    /// Wall time local-filesystem writes were in flight.
    pub disk_write_time: Duration,
    /// CPU time spent running operators (decode, group-by, …). Excludes time
    /// parked waiting on IO.
    pub cpu: Duration,
}

impl CpuAndIoCounters {
    /// Add `other`'s counters to these: one operator's into its dataflow's,
    /// or one worker's into the total over workers.
    pub fn merge(&mut self, other: &Self) {
        self.http_requests += other.http_requests;
        self.http_bytes += other.http_bytes;
        self.http_get_time += other.http_get_time;
        self.http_reissues += other.http_reissues;
        self.http_upload_time += other.http_upload_time;
        self.disk_cache_requests += other.disk_cache_requests;
        self.disk_cache_bytes += other.disk_cache_bytes;
        self.disk_cache_time += other.disk_cache_time;
        self.disk_requests += other.disk_requests;
        self.disk_bytes += other.disk_bytes;
        self.disk_read_time += other.disk_read_time;
        self.disk_write_time += other.disk_write_time;
        self.cpu += other.cpu;
    }

    /// Log the IO tally at WARN when any operation happened, for a path that has no
    /// other reporting: a failed query never reaches the server's stats NOTICE,
    /// so this keeps a slow or failed scan diagnosable. A no-op when the query did
    /// no IO (or did not opt into stats, leaving every counter zero).
    pub fn log_failed_query(&self) {
        if self.http_requests + self.disk_requests + self.disk_cache_requests == 0 {
            return;
        }
        tracing::warn!(
            http_requests = self.http_requests,
            http_mib = self.http_bytes >> 20,
            http_get_ms = self.http_get_time.as_millis() as u64,
            http_reissues = self.http_reissues,
            http_upload_ms = self.http_upload_time.as_millis() as u64,
            disk_cache_requests = self.disk_cache_requests,
            disk_cache_mib = self.disk_cache_bytes >> 20,
            disk_requests = self.disk_requests,
            disk_mib = self.disk_bytes >> 20,
            disk_read_ms = self.disk_read_time.as_millis() as u64,
            disk_write_ms = self.disk_write_time.as_millis() as u64,
            cpu_ms = self.cpu.as_millis() as u64,
            "query failed; IO completed before the failure"
        );
    }
}

/// Something an operator reports about its own work, beyond the counters
/// dispatch keeps for every operator: for example a scan reporting each row
/// group it read or skipped. `name` says what happened and `value` holds the
/// details, in whatever shape the operator chooses.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomStat {
    pub name: &'static str,
    pub value: serde_json::Value,
}

/// Where one operator's custom stats are reported, and kept until a
/// live-stats message carries them.
///
/// Whoever builds the operator creates one with
/// [`DataFlowDispatcher::create_custom_stats`](crate::DataFlowDispatcher::create_custom_stats),
/// gives the operator a clone, and may give clones to code that works on the
/// operator's behalf where the operator itself is not running: for example
/// the code that hands a scan its row groups and skips some of them. The
/// operator names its `CustomStats` once (see
/// [`Operator::custom_stats`](crate::Operator::custom_stats)), and the
/// live-stats messages of the worker running it, the last one included,
/// carry what was reported since the previous message.
///
/// For a query without live stats it keeps nothing, and reporting costs one
/// branch.
#[derive(Clone)]
pub struct CustomStats {
    /// The stats no message has carried yet. `None` when the query has no
    /// live stats.
    unreported: Option<Arc<Mutex<Vec<CustomStat>>>>,
}

impl CustomStats {
    pub(crate) fn new(live_stats_enabled: bool) -> Self {
        Self {
            unreported: live_stats_enabled.then(Arc::default),
        }
    }

    /// A `CustomStats` that keeps nothing, for an operator built outside any
    /// query.
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Report one stat: `name` says what happened and `build_value` gives the
    /// details. `build_value` runs only when the query has live stats.
    pub fn report(&self, name: &'static str, build_value: impl FnOnce() -> serde_json::Value) {
        if let Some(unreported) = &self.unreported {
            // Built before the lock is taken, so a `build_value` that panics
            // leaves the stats already reported usable.
            let value = build_value();
            unreported.lock().unwrap().push(CustomStat { name, value });
        }
    }

    /// Hand over the stats reported since the last call, leaving none behind.
    fn take_unreported(&self) -> Vec<CustomStat> {
        match &self.unreported {
            Some(unreported) => std::mem::take(&mut *unreported.lock().unwrap()),
            None => Vec::new(),
        }
    }
}

/// Asks for a query's live stats: `sender` receives each dataflow's
/// [`LiveStatsMessage`]s, where every worker sends its stats at most once per
/// `interval` while the dataflow runs, and once more when it ends.
#[derive(Clone, Debug)]
pub struct LiveStatsSubscription {
    pub interval: Duration,
    pub sender: mpsc::Sender<LiveStatsMessage>,
}

/// What a [`LiveStatsSubscription`] receives about one dataflow.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveStatsMessage {
    /// The first message, sent once before any worker's stats: the
    /// dataflow's operators, in the order of the operator graph, which is the
    /// same on every worker. A [`WorkerLiveStats`] lists its operators in
    /// this order too.
    Operators(Vec<OperatorDescription>),
    /// One worker's stats so far.
    Worker(WorkerLiveStats),
}

/// One operator of a dataflow, as [`LiveStatsMessage::Operators`] describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorDescription {
    /// The operator's type name without its module path and generics, for
    /// example `RowGroupFetcher`.
    pub name: &'static str,
    /// The indices of the operators whose output this operator reads.
    pub inputs: Vec<Identifier>,
}

/// One worker's stats for one dataflow, as of the moment the message was
/// sent. Counters are totals since the dataflow started on that worker, so a
/// later message from the same worker replaces an earlier one. Custom stats
/// are not repeated: each is sent once, in the first message after the
/// operator reported it.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerLiveStats {
    pub worker_idx: usize,
    /// Whether this is the worker's last message for the dataflow.
    pub finished: bool,
    /// Time the worker spent in this dataflow that belongs to no single
    /// operator, such as asking operators that had nothing to do.
    pub unattributed_cpu: Duration,
    /// One entry per operator of the dataflow, in the order
    /// [`LiveStatsMessage::Operators`] lists them.
    pub operators: Vec<OperatorLiveStats>,
}

/// One operator's part of a [`WorkerLiveStats`] message.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorLiveStats {
    /// CPU time and IO of this operator alone, on this worker.
    pub counters: CpuAndIoCounters,
    /// What the operator reported since this worker's previous message.
    pub custom_stats: Vec<CustomStat>,
}

impl LiveStatsMessage {
    /// This message as JSON, the form a client receives it in. Its `type` is
    /// `operators` or `worker`. Durations are whole microseconds, in the
    /// fields whose names end in `_us`.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            LiveStatsMessage::Operators(operators) => {
                let operators: Vec<serde_json::Value> = operators
                    .iter()
                    .enumerate()
                    .map(|(operator_idx, operator)| {
                        serde_json::json!({
                            "operator": operator_idx,
                            "name": operator.name,
                            "inputs": operator.inputs,
                        })
                    })
                    .collect();
                serde_json::json!({ "type": "operators", "operators": operators })
            }
            LiveStatsMessage::Worker(worker) => {
                let operators: Vec<serde_json::Value> = worker
                    .operators
                    .iter()
                    .enumerate()
                    .map(|(operator_idx, operator)| {
                        let counters = &operator.counters;
                        let custom_stats: Vec<serde_json::Value> = operator
                            .custom_stats
                            .iter()
                            .map(|stat| {
                                serde_json::json!({ "name": stat.name, "value": stat.value })
                            })
                            .collect();
                        serde_json::json!({
                            "operator": operator_idx,
                            "cpu_us": counters.cpu.as_micros() as u64,
                            "disk_requests": counters.disk_requests,
                            "disk_bytes": counters.disk_bytes,
                            "disk_read_us": counters.disk_read_time.as_micros() as u64,
                            "disk_write_us": counters.disk_write_time.as_micros() as u64,
                            "http_requests": counters.http_requests,
                            "http_bytes": counters.http_bytes,
                            "http_get_us": counters.http_get_time.as_micros() as u64,
                            "http_reissues": counters.http_reissues,
                            "http_upload_us": counters.http_upload_time.as_micros() as u64,
                            "disk_cache_requests": counters.disk_cache_requests,
                            "disk_cache_bytes": counters.disk_cache_bytes,
                            "disk_cache_us": counters.disk_cache_time.as_micros() as u64,
                            "custom_stats": custom_stats,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "type": "worker",
                    "worker": worker.worker_idx,
                    "finished": worker.finished,
                    "unattributed_cpu_us": worker.unattributed_cpu.as_micros() as u64,
                    "operators": operators,
                })
            }
        }
    }
}

/// The per-worker state for sending one dataflow's live stats.
struct LiveStatsPublisher {
    subscription: LiveStatsSubscription,
    /// The [`CustomStats`] of each operator that names one, with that
    /// operator's index.
    custom_stats_of_operators: Vec<(Identifier, CustomStats)>,
    /// When the worker last sent a message or found it had nothing to send.
    last_published: Instant,
    /// Whether any operator's counters changed since the last message. A
    /// worker with nothing new sends nothing.
    has_unpublished_changes: bool,
}

/// The counters a [`StatsCollector`] keeps while collection is on.
struct OperatorTallies {
    /// One entry per operator of the dataflow, indexed by operator index.
    per_operator: Vec<CpuAndIoCounters>,
    /// CPU time of dataflow steps that was not billed to an operator.
    unattributed_cpu: Duration,
    /// When the time not yet billed to anyone started: the start of the
    /// current dataflow step, or the last billing within it.
    cpu_lap_started: Instant,
}

impl OperatorTallies {
    fn new(operator_count: usize) -> Self {
        Self {
            per_operator: vec![CpuAndIoCounters::default(); operator_count],
            unattributed_cpu: Duration::ZERO,
            cpu_lap_started: Instant::now(),
        }
    }
}

/// Accumulates one worker's stats for a dataflow, or nothing when the query
/// asked for neither totals nor live stats — in which case every `record_*`
/// is a no-op and no clock is read. Owned by the
/// [`DataFlow`](crate::data_flow::DataFlow) and driven by the worker through
/// `flow.stats()`.
///
/// Counters are kept per operator. [`report`](Self::report) folds them into
/// the dataflow's totals; live stats send them unfolded.
pub struct StatsCollector {
    /// `Some` only when totals or live stats are on.
    tallies: Option<OperatorTallies>,
    /// Whether [`report`](Self::report) ships the folded totals.
    reports_totals: bool,
    /// Where [`report`](Self::report) ships the totals for the handle to fold.
    tx: mpsc::Sender<CpuAndIoCounters>,
    /// `Some` only when the query subscribed to live stats.
    live: Option<LiveStatsPublisher>,
}

impl StatsCollector {
    /// A collector for a dataflow of `operator_count` operators, which keeps
    /// counters only when `reports_totals` is set.
    pub(crate) fn new(
        tx: mpsc::Sender<CpuAndIoCounters>,
        reports_totals: bool,
        operator_count: usize,
    ) -> Self {
        Self {
            tallies: reports_totals.then(|| OperatorTallies::new(operator_count)),
            reports_totals,
            tx,
            live: None,
        }
    }

    /// Build a no-op collector for standalone test and benchmark drivers.
    #[cfg(any(test, feature = "test-util"))]
    pub fn disabled() -> Self {
        let (tx, _rx) = mpsc::channel();
        Self::new(tx, false, 0)
    }

    /// Start sending this dataflow's live stats to `subscription`.
    ///
    /// `operators` describes each operator, in operator-index order, with its
    /// full type name; the message carries the names without module path and
    /// generics. The first worker of the dataflow to get here sends them as
    /// [`LiveStatsMessage::Operators`], and `operators_sent`, which the
    /// dataflow's workers share, holds the others back until it has, so that
    /// message comes before any worker's stats.
    ///
    /// `custom_stats_of_operators` holds the [`CustomStats`] of each operator
    /// that names one, with that operator's index.
    pub(crate) fn enable_live_stats(
        &mut self,
        subscription: LiveStatsSubscription,
        operators_sent: &Once,
        operators: Vec<OperatorDescription>,
        custom_stats_of_operators: Vec<(Identifier, CustomStats)>,
    ) {
        if self.tallies.is_none() {
            self.tallies = Some(OperatorTallies::new(operators.len()));
        }
        operators_sent.call_once(|| {
            let operators = operators
                .into_iter()
                .map(|operator| {
                    let without_generics = operator
                        .name
                        .split_once('<')
                        .map_or(operator.name, |(before_generics, _)| before_generics);
                    OperatorDescription {
                        name: without_generics
                            .rsplit_once("::")
                            .map_or(without_generics, |(_, last_segment)| last_segment),
                        inputs: operator.inputs,
                    }
                })
                .collect();
            // A subscriber that stopped listening is not an error: the query
            // runs on.
            let _ = subscription
                .sender
                .send(LiveStatsMessage::Operators(operators));
        });
        self.live = Some(LiveStatsPublisher {
            subscription,
            custom_stats_of_operators,
            last_published: Instant::now(),
            has_unpublished_changes: false,
        });
    }

    /// Whether collection is on, so a caller can skip a clock read it would only
    /// hand to a no-op.
    pub fn enabled(&self) -> bool {
        self.tallies.is_some()
    }

    /// The counters of operator `operator_idx`, or `None` when collection is
    /// off. Asking for them marks the worker's live stats as changed.
    fn counters_of(&mut self, operator_idx: Identifier) -> Option<&mut CpuAndIoCounters> {
        let tallies = self.tallies.as_mut()?;
        if let Some(live) = &mut self.live {
            live.has_unpublished_changes = true;
        }
        Some(&mut tallies.per_operator[operator_idx])
    }

    /// Stamp each read with its issue time so its completion can bill the read's
    /// in-flight latency. Shared by the local-filesystem and remote submit paths.
    pub fn stamp_issued<R>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        if self.tallies.is_some() {
            let now = Instant::now();
            for request in requests.iter_mut() {
                request.submitted_at = Some(now);
            }
        }
    }

    /// Count local-filesystem operations against the operators that issued
    /// them, total their bytes, and stamp them so their completion can bill
    /// the matching read or write time.
    pub fn record_issued_disk(&mut self, requests: &mut [DataFlowRequest<FsRequest>]) {
        for request in requests.iter() {
            if let Some(counters) = self.counters_of(request.operator_idx) {
                counters.disk_requests += 1;
                counters.disk_bytes += request.request.ready_bytes_len();
            }
        }
        self.stamp_issued(requests);
    }

    /// Count a remote operation of operator `operator_idx` against the
    /// transports that serve it. Unlike the disk path this is known only
    /// after the requester resolves a GET against the cache or accepts an
    /// upload.
    pub fn record_issued_remote(&mut self, operator_idx: Identifier, split: RemoteSplit) {
        if let Some(counters) = self.counters_of(operator_idx) {
            counters.disk_cache_requests += split.disk_cache_requests;
            counters.disk_cache_bytes += split.disk_cache_bytes;
            counters.http_requests += split.http_requests;
            counters.http_bytes += split.http_bytes;
        }
    }

    /// Bill a completed remote read's in-flight time to operator
    /// `operator_idx`, each piece's wait charged to the tier that served it
    /// (see [`RemoteReadTime`]).
    pub fn record_http_get_time(&mut self, operator_idx: Identifier, time: RemoteReadTime) {
        if let Some(counters) = self.counters_of(operator_idx) {
            counters.http_get_time += time.http;
            counters.http_reissues += time.http_reissues;
            counters.disk_cache_time += time.disk_cache;
        }
    }

    /// Bill one completed HTTP upload of operator `operator_idx` from
    /// submission through response drain.
    pub fn record_http_upload_time(
        &mut self,
        operator_idx: Identifier,
        submitted_at: Option<Instant>,
    ) {
        self.record_elapsed(operator_idx, submitted_at, |counters| {
            &mut counters.http_upload_time
        });
    }

    /// Bill a completed local-filesystem read's in-flight time to operator
    /// `operator_idx`.
    pub fn record_disk_read_time(
        &mut self,
        operator_idx: Identifier,
        submitted_at: Option<Instant>,
    ) {
        self.record_elapsed(operator_idx, submitted_at, |counters| {
            &mut counters.disk_read_time
        });
    }

    /// Bill a completed local-filesystem write's in-flight time to operator
    /// `operator_idx`.
    pub fn record_disk_write_time(
        &mut self,
        operator_idx: Identifier,
        submitted_at: Option<Instant>,
    ) {
        self.record_elapsed(operator_idx, submitted_at, |counters| {
            &mut counters.disk_write_time
        });
    }

    /// Add the time since `submitted_at` to the duration `bucket` picks out
    /// of operator `operator_idx`'s counters. An operation issued while
    /// collection was off carries no `submitted_at` and bills nothing.
    fn record_elapsed(
        &mut self,
        operator_idx: Identifier,
        submitted_at: Option<Instant>,
        bucket: fn(&mut CpuAndIoCounters) -> &mut Duration,
    ) {
        if let (Some(counters), Some(submitted_at)) = (self.counters_of(operator_idx), submitted_at)
        {
            *bucket(counters) += submitted_at.elapsed();
        }
    }

    /// Start timing one dataflow step. The time from here on is billed by
    /// [`bill_cpu_lap_to_operator`](Self::bill_cpu_lap_to_operator) and
    /// [`finish_cpu_lap`](Self::finish_cpu_lap).
    pub(crate) fn start_cpu_lap(&mut self) {
        if let Some(tallies) = &mut self.tallies {
            tallies.cpu_lap_started = Instant::now();
        }
    }

    /// Bill the time since the step started, or since the last billing within
    /// it, to operator `operator_idx`. Called right after an operator did
    /// work. Only live stats break CPU time down by operator, so without them
    /// this reads no clock and the whole step is billed when it finishes.
    pub(crate) fn bill_cpu_lap_to_operator(&mut self, operator_idx: Identifier) {
        let (Some(tallies), Some(live)) = (&mut self.tallies, &mut self.live) else {
            return;
        };
        let now = Instant::now();
        tallies.per_operator[operator_idx].cpu += now - tallies.cpu_lap_started;
        tallies.cpu_lap_started = now;
        live.has_unpublished_changes = true;
    }

    /// End the dataflow step started by [`start_cpu_lap`](Self::start_cpu_lap):
    /// bill the time no operator was billed for to the dataflow itself, and
    /// send the worker's live stats if they are due.
    pub(crate) fn finish_cpu_lap(&mut self) {
        let Some(tallies) = &mut self.tallies else {
            return;
        };
        let now = Instant::now();
        tallies.unattributed_cpu += now - tallies.cpu_lap_started;
        tallies.cpu_lap_started = now;
        self.publish_due_live_stats();
    }

    /// Send the worker's live stats if the subscription's interval has passed
    /// since it last sent them or found nothing to send. Called at the end of
    /// every dataflow step, and by the worker before it goes idle, so a
    /// worker never sends more than one message per interval.
    ///
    /// A worker that goes idle sooner than that keeps what it did last until
    /// it next wakes.
    pub fn publish_due_live_stats(&mut self) {
        let is_due = self
            .live
            .as_ref()
            .is_some_and(|live| live.last_published.elapsed() >= live.subscription.interval);
        if is_due {
            self.publish_live_stats(false);
        }
    }

    /// Send one [`WorkerLiveStats`] message holding every operator's counters
    /// so far and the custom stats the operators reported since the previous
    /// message. A message that is not the worker's last is sent only when
    /// something changed since the previous one. Does nothing without a
    /// subscription. A subscriber that stopped listening is not an error: the
    /// query runs on.
    fn publish_live_stats(&mut self, finished: bool) {
        let (Some(tallies), Some(live)) = (&self.tallies, &mut self.live) else {
            return;
        };
        live.last_published = Instant::now();
        let unreported_custom_stats: Vec<(Identifier, Vec<CustomStat>)> = live
            .custom_stats_of_operators
            .iter()
            .map(|(operator_idx, custom_stats)| (*operator_idx, custom_stats.take_unreported()))
            .filter(|(_, unreported)| !unreported.is_empty())
            .collect();
        if !finished && !live.has_unpublished_changes && unreported_custom_stats.is_empty() {
            return;
        }
        let mut operators: Vec<OperatorLiveStats> = tallies
            .per_operator
            .iter()
            .map(|counters| OperatorLiveStats {
                counters: *counters,
                custom_stats: Vec::new(),
            })
            .collect();
        for (operator_idx, unreported) in unreported_custom_stats {
            operators[operator_idx].custom_stats = unreported;
        }
        let _ = live
            .subscription
            .sender
            .send(LiveStatsMessage::Worker(WorkerLiveStats {
                worker_idx: WORKER_IDX.get(),
                finished,
                unattributed_cpu: tallies.unattributed_cpu,
                operators,
            }));
        live.has_unpublished_changes = false;
    }

    /// Ship this worker's final numbers when its dataflow ends: the last
    /// live-stats message, and the totals folded over every operator for the
    /// handle to fold across workers (each a no-op when off).
    pub fn report(&mut self) {
        self.publish_live_stats(true);
        if let (true, Some(tallies)) = (self.reports_totals, &self.tallies) {
            let mut totals = CpuAndIoCounters::default();
            for counters in &tallies.per_operator {
                totals.merge(counters);
            }
            totals.cpu += tallies.unattributed_cpu;
            let _ = self.tx.send(totals);
        }
    }
}
