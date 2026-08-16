//! Per-core worker threads that execute dataflows.
//!
//! Each worker is pinned to a CPU core and runs an event loop that:
//! - Receives [`DataFlowBuilder`]s from the [`Dispatch`](crate::Dispatch) via a channel
//! - Builds them into [`DataFlow`]s (creating operators, channels, etc. on the worker thread)
//! - Drives execution by interleaving CPU work, IO, and work stealing
//!
//! Workers are the execution backbone of the system. A query built with
//! [`RecordBatchOperatorSpec`](crate::api::RecordBatchOperatorSpec)
//! doesn't run until [`.collect()`](crate::api::RecordBatchOperatorSpec::collect)
//! sends one `DataFlowBuilder` to each worker. The worker then:
//!
//! 1. Calls [`DataFlowBuilder::build`] to construct the operator chain — this happens
//!    here because some resources (`Rc<Worker<T>>`, thread-local buffers) are not `Send`
//!    and must be created on the thread that uses them.
//! 2. Runs the resulting `DataFlow` through its event loop, which prioritizes keeping IO
//!    saturated (to hide disk latency), then runs CPU work, then attempts to steal work
//!    from sibling workers when idle.
//!
//! Workers coordinate with each other only at the operator level — e.g. a `Count` operator
//! uses shared atomics so the last worker to finish can emit the total. The worker itself
//! has no direct knowledge of other workers; synchronization is encapsulated in the
//! operators.

use crate::Identifier;
use crate::api::DataFlowBuilder;
use crate::data_flow::{DataFlow, WorkStatus};
use crate::io::request_tracker::{ReadKey, RegistrationCaps, RequestTracker};
use crate::io::{Completion, DiskCache, FailedRequest, IORequester};
use crate::memory::{MemoryContextFactory, init_memory_context, memory_ctx};
use crate::operations::FinishStatus;
use crate::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};
use core_affinity::CoreId;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Spin budget while a **dataflow is in flight** but this worker momentarily has
/// nothing to do (waiting on a pipeline-stage barrier / for a sibling to produce
/// stealable work). A parked worker pays a condvar/futex wakeup (tens of µs plus
/// OS scheduling) every stage transition; with many workers and little data per
/// stage that latency dominates small-query time. The core is idle at a barrier
/// regardless, so spinning here costs nothing useful and lets the worker resume
/// in nanoseconds when the next `notify` lands. Generous enough to bridge the
/// µs-scale gaps between a small query's stages; a longer real stall still falls
/// through to a park. Large queries keep workers busy and rarely reach this path.
const IN_FLIGHT_SPIN_LIMIT: u32 = 120_000;

/// Scheduled-but-not-landed disk read blocks per worker before the
/// `register_pending_io` stage stops registering further filesystem requests.
/// Sized to the ring: one full submission batch can be outstanding without
/// overflowing the completion queue (see the tracker's submission batching), so
/// registration keeps the ring fed without piling up pinned cache slots beyond
/// what the ring can absorb.
const MAX_DISK_READS_IN_FLIGHT: usize = crate::io::RING_SIZE as usize;

/// Per-worker remote in-flight ceiling, tunable via `PIVOT_HTTP_INFLIGHT`. Caps
/// both the registration of staged HTTP read requests and the submission of
/// HTTP operations (reads and uploads) onto the ring.
///
/// Remote objects sit behind ~tens-of-ms RTTs, so a deep read-ahead is what
/// hides the latency: stopping at the first outstanding read would serialise a
/// scan to one read at a time per worker — catastrophic over a table of many
/// small files, where the whole query becomes round-trip bound.
fn http_inflight_target() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| crate::env::get_env_var_with_default("PIVOT_HTTP_INFLIGHT", 100))
}

use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, instrument, warn};

thread_local! {
    /// This worker's global index, dense across all NUMA node groups
    /// (`0..NUM_WORKERS`, node 0's workers first).
    pub static WORKER_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Total workers across all node groups.
    pub static NUM_WORKERS: Cell<usize> = const { Cell::new(usize::MAX) };
    /// The NUMA node group this worker belongs to.
    static NODE_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// The node group of the current worker thread.
pub fn current_node() -> usize {
    NODE_IDX.get()
}

#[cfg(any(test, feature = "test-util"))]
pub(crate) fn set_current_node(node: usize) {
    NODE_IDX.set(node);
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Pipeline(#[from] crate::data_flow::Error),
    #[error("{0}")]
    IORequester(#[from] crate::io::IORequesterError),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// A Worker is spun up per CPU core. The worker's main entry-point, `create`, spins up a
/// thread with affinity to a CPU which continuously requests work from the dispatcher and does it.
///
/// The main idea of a Worker is to keep everything possible "local" to it, to prevent
/// context-switching/CPU cache-invalidation. Workers are grouped by NUMA node (see
/// [`Dispatch::spin_up`](crate::Dispatch::spin_up)); every dataflow runs on all workers,
/// but each worker allocates only node-local ring memory and steals work only from
/// same-node siblings. We try to make our physical CPU cores first class citizens.
///
/// The Worker receives dataflows from the Dispatcher and runs them- the logic is outlined is as
/// follows:
///
/// 1. Process any outstanding IO completions — we want to always have IO running in background, so
///    we clear any pending if possible
/// 2. Register pending IO - drain the requests operators staged in their `OperatorIO`s into the
///    tracker, then submit the resulting operations onto the ring
/// 3. Run any CPU work available - this will run a single operation if available
/// 4. Try running trough all dataflows and finishing then
///
/// If no work was done (no CPU work or finish work), then
/// - Wait on existing IO - if none exists, then:
/// - Try stealing work - if none exists, then:
/// - Sleep!
///
/// And forever back to 1!
pub struct Worker {
    /// The identifier of the worker (every worker has a unique ID)
    id: Identifier,
    should_exit: Arc<AtomicBool>,
    /// A requester (through a single per-core uring) for all IO — both disk reads
    /// and HTTP(S) reads of remote regions share the one ring.
    io: IORequester,
    /// This worker's tracker for cache-backed reads: the `register_pending_io`
    /// stage drains operators' staged requests into it, it resolves them
    /// against the caches and dedups the physical reads, and completions are
    /// routed through it back to the operators that staged them.
    tracker: RequestTracker,
    /// A queue of dataflows for the worker to work on. The worker can work on many dataflows
    /// simultaneously
    data_flow_queue: Receiver<DataFlowBuilder>,
    /// A mapping of dataflows currently running
    data_flows: HashMap<Identifier, DataFlow>,
    /// Did the worker do work on last iteration? This is used to decide whether the worker should
    /// do background tasks, just as cleaning
    did_work_last_iteration: bool,
    /// Shared park/notify object. Keeping an owned `Arc` here means the
    /// [`init_worker_waker`] raw pointer stays valid for the lifetime of the
    /// worker thread.
    waker: Arc<WorkerWaker>,
    /// Wake-count snapshot used to decide whether the next park can sleep.
    /// Initialised once at worker startup. Iterations that do work do not
    /// refresh it — any `notify` they missed will simply make the next park
    /// observe a mismatch and return immediately, so no wake is lost.
    last_seen_wake_count: u64,
    /// This worker's index within its node group, which is also its park
    /// slot in the waker.
    node_local_idx: usize,
    /// Last delegated-broadcast epoch this worker has fanned out (see
    /// [`WorkerWaker::finish_delegated_wake`]).
    last_seen_broadcast: u64,
}

impl Worker {
    /// Spawn a worker thread pinned to `core`. Blocks on `ready_barrier` before
    /// entering the event loop, so all workers start roughly together.
    ///
    /// The worker checks the shutdown flag set by [`Dispatch::exit`](crate::Dispatch::exit) on every
    /// iteration and returns from `run` once it flips to `true`; panics
    /// inside the event loop propagate normally and surface through the
    /// returned [`JoinHandle`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        idx: usize,
        num_workers: usize,
        node: usize,
        core: CoreId,
        should_exit: Arc<AtomicBool>,
        memory_context_factory: MemoryContextFactory,
        disk_cache: Option<Arc<DiskCache>>,
        receiver: Receiver<DataFlowBuilder>,
        ready_barrier: Arc<Barrier>,
        waker: Arc<WorkerWaker>,
        waker_set: WakerSet,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            // Worker startup (io_uring + memory-context setup) can fail. Every worker
            // must reach `ready_barrier` or `Dispatch::spin_up` deadlocks on it forever,
            // so run startup under `catch_unwind`: on failure we still trip the barrier
            // below (letting spin_up return its handles) and then re-raise the panic.
            let started = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                WORKER_IDX.set(idx);
                NUM_WORKERS.set(num_workers);
                NODE_IDX.set(node);
                // Pin to this worker's core BEFORE touching any memory, so everything
                // this worker first-faults - its node-local ring (via `prefault_buffers`),
                // free pools, and io_uring buffers - lands on this core's NUMA node. Pages
                // are placed where first written under the default policy, so prefaulting
                // before pinning would scatter the ring across nodes and defeat the
                // per-node split.
                core_affinity::set_for_current(core);
                // Register this worker's OS tid so the server can scope a
                // `perf record -t` to the worker pool.
                #[cfg(feature = "perf")]
                crate::profiler::register_worker_tid();
                let node_local_idx = idx % waker_set.workers_per_node();
                waker.register(node_local_idx);
                let last_seen_wake_count = waker.wake_count();
                let last_seen_broadcast = waker.broadcast_epoch();
                debug!("Initializing worker waker {:?}", idx);
                init_worker_waker(&waker);
                init_waker_set(waker_set);
                // Give this worker thread a handle to the shared disk cache so
                // `drop_cache()` can clear it; the requester takes ownership.
                crate::io::disk_cache::install_worker_disk_cache(disk_cache.clone());
                let worker = Self {
                    io: IORequester::new(disk_cache),
                    tracker: RequestTracker::default(),
                    id: core.id,
                    data_flows: HashMap::new(),
                    data_flow_queue: receiver,
                    did_work_last_iteration: false,
                    should_exit,
                    waker,
                    last_seen_wake_count,
                    node_local_idx,
                    last_seen_broadcast,
                };
                debug!("Initializing memory context for worker {:?}", idx);
                init_memory_context(memory_context_factory.create_memory_ctx());
                debug!("Pre-faulting for worker {:?}", idx);
                memory_ctx().prefault_buffers();
                worker
            }));
            debug!("Waiting for barrier for worker {:?}", idx);
            ready_barrier.wait();
            let mut worker = match started {
                Ok(worker) => worker,
                // Re-raise now that spin_up has been released by the barrier
                Err(payload) => std::panic::resume_unwind(payload),
            };
            debug!("Starting worker {:?}", idx);
            worker.run().expect("Worker failed!");
        })
    }

    /// Whether the worker should skip `flow` this pass: a `perf` capture is in
    /// progress and `flow` is not the dataflow being profiled, so it pauses while
    /// the profiled dataflow runs alone (see [`crate::profiler`]).
    #[cfg(feature = "perf")]
    fn paused_for_profiling(flow: &DataFlow) -> bool {
        crate::profiler::profiling_active() && !flow.is_profiled()
    }

    /// Drain the IO operators staged in their
    /// [`OperatorIO`](crate::io::OperatorIO)s into this worker's tracker, one
    /// dataflow at a time: reads register against the caches (until the
    /// per-medium in-flight budget is spent), writes and uploads queue for
    /// submission. Requests whose ranges were all cache-resident complete and
    /// are delivered during the pass. Returns whether every dataflow's staged
    /// IO was fully drained.
    fn register_pending_io(&mut self) -> bool {
        let caps = RegistrationCaps {
            disk_reads: MAX_DISK_READS_IN_FLIGHT,
            http_reads: http_inflight_target(),
        };
        let mut clean = true;
        for flow in self.data_flows.values_mut() {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                clean = false;
                continue;
            }
            let pass = flow.register_pending_io(&mut self.tracker, &caps);
            clean &= pass.clean;
            // A cache-served delivery is work: the operator consumed its
            // response (and possibly emitted downstream), so don't park.
            self.did_work_last_iteration |= pass.delivered;
        }
        clean
    }

    /// Submit the tracker's queued operations onto the ring: filesystem
    /// reads/writes while the disk queue is free (the tracker batches them so
    /// the ring's completion queue can't overflow), remote operations up to the
    /// in-flight ceiling - remote objects sit behind ~tens-of-ms RTTs, so that
    /// depth of read-ahead is what hides the latency. Each operation is charged
    /// to the stats of the dataflow that staged it (for a deduped read, the one
    /// whose request first scheduled it).
    ///
    /// Reads whose dataflow has since been cancelled are submitted regardless:
    /// the read may have waiters in other dataflows, and its cache slot is
    /// already allocated and pinned either way.
    fn submit_staged_requests(&mut self) -> Result<()> {
        while !self.io.has_file_pending() {
            let batch = self.tracker.take_fs_submissions();
            if batch.is_empty() {
                break;
            }
            for mut request in batch {
                if let Some(flow) = self.data_flows.get_mut(&request.data_flow_id) {
                    flow.stats()
                        .record_issued_disk(std::slice::from_mut(&mut request));
                }
                self.io.request(request)?;
            }
        }
        while self.io.http_in_flight() < http_inflight_target() {
            let batch = self.tracker.take_http_submissions();
            if batch.is_empty() {
                break;
            }
            for mut request in batch {
                if let Some(flow) = self.data_flows.get_mut(&request.data_flow_id) {
                    flow.stats()
                        .stamp_issued(std::slice::from_mut(&mut request));
                }
                let data_flow_id = request.data_flow_id;
                let read = match &request.request {
                    crate::io::HttpRequest::Get(get) => Some(ReadKey::of_http_get(get)),
                    crate::io::HttpRequest::Upload(_) => None,
                };
                // Submitting an HTTP operation can fail (socket exhaustion,
                // TLS setup). The operation never went out, so fail it where
                // it's accounted: a read through the tracker (cancelling every
                // dataflow that waited on it), an upload against its issuing
                // dataflow alone. Never propagated - that would panic the
                // worker and take the whole server down.
                match self.io.request_http(request) {
                    Ok(split) => {
                        if let Some(flow) = self.data_flows.get_mut(&data_flow_id) {
                            flow.stats().record_issued_remote(split);
                        }
                    }
                    Err(e) => match read {
                        Some(read) => self.fail_tracker_read(&read, &e),
                        None => {
                            if let Some(flow) = self.data_flows.get_mut(&data_flow_id) {
                                flow.bail_and_cancel(e.into());
                            }
                        }
                    },
                }
            }
        }
        Ok(())
    }

    /// A tracker-managed read failed terminally: cancel every dataflow that had
    /// a request waiting on it.
    fn fail_tracker_read(&mut self, read: &ReadKey, error: &crate::io::IORequesterError) {
        let message = error.to_string();
        for data_flow_id in self.tracker.fail(read) {
            if let Some(flow) = self.data_flows.get_mut(&data_flow_id) {
                let error =
                    crate::io::IORequesterError::from(std::io::Error::other(message.clone()));
                flow.bail_and_cancel(error.into());
            }
        }
    }

    /// Hand every request that `read`'s landing completed back to its operator.
    /// A request's dataflow may already be gone (finished or cancelled); its
    /// response is dropped, and the cache pins it holds release with it.
    fn deliver_completed_requests(&mut self, read: ReadKey) {
        for delivery in self.tracker.complete(&read) {
            if let Some(flow) = self.data_flows.get_mut(&delivery.data_flow_id) {
                flow.process_io_response(delivery.operator_idx, delivery.completed);
            }
        }
    }

    /// Run one unit of CPU work from the first dataflow that has work ready.
    fn step_run_ready_cpu_work(&mut self) {
        // While a `perf` capture is in progress, run only the profiled dataflow
        // so the profile isn't polluted by other queries / ingest on the pool.
        for flow in self.data_flows.values_mut() {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                continue;
            }
            if let WorkStatus::Ran = flow.run_ready_cpu_work() {
                self.did_work_last_iteration = true;
            }
        }
    }

    /// Deliver completed IO back to where it's tracked: reads resolve through
    /// this worker's tracker (which knows every request waiting on each read),
    /// writes and uploads go straight to the operator that issued them. By the
    /// time we're here each read's bytes are already committed to its cache
    /// slot.
    fn process_io_completions(&mut self) -> Result<()> {
        for completion in self.io.completions()? {
            match completion {
                Ok(Completion::FsRead(r)) => {
                    // The read's issuing dataflow may already be gone (a
                    // finished or cancelled query leaves its read-ahead reads
                    // in flight); its stats are simply lost, but the tracker
                    // delivery below still serves any other waiting dataflow.
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_disk_read_time(r.submitted_at);
                    }
                    self.deliver_completed_requests(ReadKey::of_fs(&r.request));
                }
                Ok(Completion::HttpGet(r, time)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_http_get_time(time);
                    }
                    self.deliver_completed_requests(ReadKey::of_http_get(&r.request));
                }
                // Writes and uploads are operator-owned; a missing dataflow
                // means the query is gone and the completion is dropped.
                Ok(Completion::FsWrite(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_disk_write_time(r.submitted_at);
                        data_flow.process_fs_write(r.operator_idx, r.request);
                    }
                }
                Ok(Completion::HttpUpload(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_http_upload_time(r.submitted_at);
                        data_flow.process_http_upload_response(r.operator_idx, r.request);
                    }
                }
                Err(failed) => match failed.request {
                    // A failed read may have requests waiting on it across
                    // several dataflows; the tracker knows them all.
                    FailedRequest::FsRead(read) => {
                        self.fail_tracker_read(&ReadKey::of_fs(&read), &failed.error)
                    }
                    FailedRequest::HttpGet(get) => {
                        self.fail_tracker_read(&ReadKey::of_http_get(&get), &failed.error)
                    }
                    // A failed write/upload cancels just the owning dataflow —
                    // its query errors out to the client while the worker and
                    // every other query keep running. The dataflow may already
                    // be gone if the query was cancelled meanwhile.
                    FailedRequest::FsWrite | FailedRequest::HttpUpload => {
                        if let Some(data_flow) = self.data_flows.get_mut(&failed.data_flow_id) {
                            data_flow.bail_and_cancel(failed.error.into());
                        }
                    }
                },
            }
        }
        Ok(())
    }

    /// Check each dataflow for completion and remove finished ones.
    fn try_finishing_dataflows(&mut self) {
        let mut to_remove = Vec::new();
        for (id, flow) in &mut self.data_flows {
            match flow.maybe_finish() {
                FinishStatus::Done => to_remove.push(*id),
                // An operator is still producing output: count it as work so we
                // re-drive it via `run_cpu_work` next iteration instead of
                // parking one stage short.
                FinishStatus::Working => self.did_work_last_iteration = true,
                FinishStatus::Pending => {}
            }
        }

        for id in to_remove {
            debug!("Finished data flow {:?}", id);
            if let Some(mut flow) = self.data_flows.remove(&id) {
                flow.stats().report();
            }
        }
    }

    /// Attempt to steal work from sibling workers' channels when this worker is idle.
    fn try_steal_work(&mut self) {
        for flow in self.data_flows.values_mut() {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                continue;
            }
            if let WorkStatus::Ran = flow.try_stealing_work() {
                self.did_work_last_iteration = true;
            }
        }
    }

    /// Called when no other work is available; we either park or clean a dirty buffer.
    /// We clean dirty buffers when we have nothing else to do to help future execution.
    ///
    /// Parking compares against `last_seen_wake_count` (captured under the waker's
    /// mutex at the previous park): if it has not advanced, we wait on the condvar;
    /// if it has, we return immediately to retry. The new count is captured
    /// atomically with the wait and stored back into `last_seen_wake_count` for the
    /// next park.
    fn clear_dirty_buffer_or_park(&mut self) {
        // Only clean dirty buffers between queries — while a dataflow is
        // running we don't want to spend filler-time on buffer cleanup that
        // could otherwise be CPU available for stealing work from sibling workers.
        //
        // Ideally, we would do cleanup also when you have dataflows above a certain
        // threshold of dirty buffers - but we don't have that implemented yet.
        if self.data_flows.is_empty()
            && let Some(b) = memory_ctx().pop_dirty_buffer()
        {
            b.zero_out();

            // Bound work per pass: zero one buffer, then hand
            // control back to the main loop so a newly-arrived
            // dataflow / new stealable work isn't starved behind
            // a long cleanup run.
            return;
        }

        // Spin-before-park, but only while a dataflow is in flight. Parking on
        // the condvar costs a wakeup (tens of µs + OS scheduling) on the next
        // `notify`; for a small multi-stage query that per-stage wakeup latency
        // dominates, so a worker idle at a stage barrier polls the lock-free
        // wake count and resumes in nanoseconds when the next notify lands (the
        // core is idle at the barrier anyway). With no dataflow running we're
        // genuinely idle between queries — park immediately rather than burn CPU.
        if !self.data_flows.is_empty() {
            for _ in 0..IN_FLIGHT_SPIN_LIMIT {
                let now = self.waker.wake_count();
                if now != self.last_seen_wake_count {
                    self.last_seen_wake_count = now;
                    return;
                }
                std::hint::spin_loop();
            }
        }

        self.last_seen_wake_count = self
            .waker
            .wait_if_unchanged(self.last_seen_wake_count, self.node_local_idx);
        self.waker
            .finish_delegated_wake(&mut self.last_seen_broadcast);
    }

    fn clear_cancelled_dataflows(&mut self) {
        self.data_flows.retain(|_, d| {
            if d.cancelled() {
                // Ship this worker's partial tally before dropping the flow, so a
                // failed or cancelled query's IO stats still reach the handle (a
                // finished flow reports in `try_finishing_dataflows`).
                d.stats().report();
                false
            } else {
                true
            }
        })
    }

    /// Let any operator that's done with its upstream (e.g. a satisfied `LIMIT`)
    /// abandon the scan feeding it, freeing its in-flight buffers and stopping
    /// further reads. Distinct from [`clear_cancelled_dataflows`](Self::clear_cancelled_dataflows):
    /// that tears down a whole query, this prunes only one query's upstream while
    /// the rest keeps running.
    fn cancel_upstream_in_dataflows(&mut self) {
        for flow in self.data_flows.values_mut() {
            flow.cancel_upstream_if_requested();
        }
    }

    fn try_receiving_new_dataflow(&mut self) {
        // Drain everything queued. Taking one builder per pass loses dataflows:
        // two dispatches can land while this worker is parked, their notifies
        // coalescing into one wake-count advance. The woken pass would take only
        // the first builder, finish it without setting `did_work`, and re-park
        // with the count already current, stranding the second dataflow's
        // builder in the queue forever (its query hangs; sibling workers spin
        // or park waiting for this worker's operator to ever exist).
        while let Ok(builder) = self.data_flow_queue.try_recv() {
            debug!("Received data flow...");
            match builder.build() {
                Ok(data_flow) => {
                    self.data_flows.insert(data_flow.id(), data_flow);
                }
                Err(e) => {
                    warn!("Failed to build dataflow {:?}", e)
                }
            }
        }
    }

    #[instrument(skip(self), fields(worker_id = %self.id))]
    pub fn run(&mut self) -> Result<()> {
        loop {
            // Cooperative shutdown: [`crate::Dispatch::exit`] flips this flag
            // and we exit cleanly so the thread can be reaped.
            if self.should_exit.load(Ordering::Relaxed) {
                debug!("Worker {} exiting via shutdown flag", self.id);
                return Ok(());
            }
            self.did_work_last_iteration = false;

            self.try_receiving_new_dataflow();

            self.clear_cancelled_dataflows();

            // Prune any abandoned upstream before submitting IO, so a satisfied
            // LIMIT's scan issues no further reads.
            self.cancel_upstream_in_dataflows();

            self.process_io_completions()?;
            // Walk the dataflows for staged IO only while some operator has
            // flagged it; on an IO-free hot path the walks are the whole cost.
            // Clearing only after a fully clean pass keeps IO that was left
            // staged by a spent registration budget flagged for the next pass.
            if crate::io::has_pending_io() && self.register_pending_io() {
                crate::io::clear_pending_io();
            }
            // Operations the register stage queued (and any leftovers a busy
            // ring or full HTTP window deferred) go out now.
            if self.tracker.has_requests_to_submit() {
                self.submit_staged_requests()?;
            }

            self.step_run_ready_cpu_work();

            self.try_finishing_dataflows();

            if !self.did_work_last_iteration {
                // One ring serves both disk and HTTP, so a single wait wakes on
                // either kind of completion — no dual-ring coordination needed.
                if self.io.has_pending() {
                    debug!("Waiting for IO...");
                    self.io.wait()?;
                    continue;
                }

                self.try_steal_work();

                if !self.did_work_last_iteration {
                    self.clear_dirty_buffer_or_park();
                }
            }
        }
    }
}
