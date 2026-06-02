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
use crate::io::IORequester;
use crate::memory::{MemoryContextFactory, init_memory_context, memory_ctx};
use core_affinity::CoreId;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread::JoinHandle;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

thread_local! {
    pub static WORKER_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
    pub static NUM_WORKERS: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Worker-thread-only handle to the shared [`WorkerWaker`].
    ///
    /// Set once by [`Worker::create`] before the event loop starts, so that
    /// worker-side code (channel sends, the worker's own park path) can call
    /// [`worker_waker`] without threading an extra parameter through every
    /// call site. Non-worker threads — the embedding server, cancellation
    /// handles created off-worker — must instead reach the same instance
    /// through their owned `Arc<WorkerWaker>` (e.g.
    /// [`crate::DataFlowDispatcher::waker`]).
    static WORKER_WAKER: Cell<*const WorkerWaker> = const { Cell::new(std::ptr::null()) };
}

/// Wake mechanism shared by all workers. Holds a monotonic `wake_count` and a
/// `Condvar`. Anyone who generates work that idle workers should pick up
/// (a new dataflow, a sibling-counter transitioning to 0) calls [`notify`](WorkerWaker::notify),
/// which bumps the count and wakes every waiting worker.
///
/// Each worker remembers the count it observed at the end of its previous
/// park. When it next finds no work and tries to sleep, [`wait_if_unchanged`](WorkerWaker::wait_if_unchanged)
/// parks on the condvar only if the count still matches — if it has advanced,
/// a `notify` arrived during the work pass and the worker returns immediately
/// to retry.
/// State protected by the waker's mutex: the wake counter and the number of
/// currently parked workers.
struct WakerState {
    /// Monotonic counter (wrapping) bumped on every [`WorkerWaker::notify`].
    wake_count: u64,
    /// How many workers are currently parked on the condvar. Used by
    /// `notify` to skip the `notify_all` syscall when nobody is waiting.
    sleepers: usize,
}

pub struct WorkerWaker {
    state: Mutex<WakerState>,
    /// Workers wait here when idle; `notify` wakes all of them.
    cond: Condvar,
}

impl WorkerWaker {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(WakerState {
                wake_count: 0,
                sleepers: 0,
            }),
            cond: Condvar::new(),
        }
    }

    /// Bump the wake count and wake all waiting workers. Skips the `notify_all`
    /// call when no workers are currently parked, avoiding the syscall on
    /// hot send paths that nobody is waiting on.
    pub fn notify(&self) {
        let mut state = self.state.lock().unwrap();
        state.wake_count = state.wake_count.wrapping_add(1);
        if state.sleepers > 0 {
            self.cond.notify_all();
        }
    }

    /// Current wake count. Workers snapshot this before doing a pass of work.
    pub fn wake_count(&self) -> u64 {
        self.state.lock().unwrap().wake_count
    }

    /// If the wake count still matches `last_seen`, block on the condvar until
    /// a `notify` advances it. If it has already advanced, return immediately
    /// so the worker re-runs its loop. Returns the wake count observed under
    /// the lock after the wait, suitable for use as the next iteration's
    /// `last_seen` — captured atomically with the wait so a concurrent notify
    /// cannot be lost between sleep cycles.
    pub fn wait_if_unchanged(&self, last_seen: u64) -> u64 {
        let mut state = self.state.lock().unwrap();
        if state.wake_count == last_seen {
            state.sleepers += 1;
            state = self.cond.wait(state).unwrap();
            state.sleepers -= 1;
        }
        state.wake_count
    }
}

/// Install this thread's view of the shared [`WorkerWaker`].
///
/// Called by [`Worker::create`] before the event loop starts (and by
/// [`install_test_worker_waker`] from test setup). The `Arc` is kept alive
/// by the [`Worker`] itself / by [`crate::DataFlowDispatcher`] / by the
/// (leaked) test waker, so the raw pointer cached here is valid for the
/// lifetime of the thread.
pub fn init_worker_waker(waker: &Arc<WorkerWaker>) {
    WORKER_WAKER.set(Arc::as_ptr(waker));
}

/// Return the shared [`WorkerWaker`] for code running on a worker thread.
///
/// Expected to always be set on threads that drive dataflow work; mirrors
/// [`crate::memory::memory_ctx`] in that the caller is trusted to have
/// installed one via [`init_worker_waker`] (workers do this in
/// [`Worker::create`]; tests do it via [`install_test_worker_waker`]).
pub fn worker_waker() -> &'static WorkerWaker {
    unsafe { &*WORKER_WAKER.get() }
}

/// Install a leaked [`WorkerWaker`] on the current test thread.
///
/// Operator code unconditionally calls `worker_waker().notify()` on send
/// paths; tests that drive operators directly (without spinning up a real
/// [`crate::Dispatch`]) need a waker installed first or the TLS pointer is
/// null. The waker is leaked because the pointer is cached in TLS for the
/// lifetime of the test thread — the binary tears down right after.
#[cfg(test)]
pub(crate) fn install_test_worker_waker() {
    let waker = Arc::new(WorkerWaker::new());
    init_worker_waker(&waker);
    std::mem::forget(waker);
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
/// context-switching/CPU cache-invalidation and in the future allow NUMA optimizations etc. We try
/// to make our physical CPU cores first class citizens.
///
/// The Worker receives dataflows from the Dispatcher and runs them- the logic is outlined is as
/// follows:
///
/// 1. Process any outstanding IO completions — we want to always have IO running in background, so
///    we clear any pending if possible
/// 2. Saturate IO - if we have space now, submit new IO
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
    /// A requester (through uring) for any IO necessary
    io: IORequester,
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
    /// Captured under the waker's mutex at the moment of the previous park
    /// (initialised once at worker startup). Iterations that do work do not
    /// refresh it — any `notify` they missed will simply make the next park
    /// observe a mismatch and return immediately, so no wake is lost.
    last_seen_wake_count: u64,
}

impl Worker {
    /// Spawn a worker thread pinned to `core`. Blocks on `ready_barrier` before
    /// entering the event loop, so all workers start roughly together.
    ///
    /// The worker checks the process-wide [`crate::EXIT`] flag on every
    /// iteration and returns from `run` once it flips to `true`; panics
    /// inside the event loop propagate normally and surface through the
    /// returned [`JoinHandle`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        idx: usize,
        num_workers: usize,
        core: CoreId,
        should_exit: Arc<AtomicBool>,
        memory_context_factory: MemoryContextFactory,
        receiver: Receiver<DataFlowBuilder>,
        ready_barrier: Arc<Barrier>,
        waker: Arc<WorkerWaker>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            WORKER_IDX.set(idx);
            NUM_WORKERS.set(num_workers);
            let last_seen_wake_count = waker.wake_count();
            debug!("Initializing worker waker {:?}", idx);
            init_worker_waker(&waker);
            let mut worker = Self {
                io: IORequester::new(),
                id: core.id,
                data_flows: HashMap::new(),
                data_flow_queue: receiver,
                did_work_last_iteration: false,
                should_exit,
                waker,
                last_seen_wake_count,
            };
            debug!("Initializing memory context for worker {:?}", idx);
            init_memory_context(memory_context_factory.create_memory_ctx());
            debug!("Pre-faulting for worker {:?}", idx);
            memory_ctx().prefault_buffers();
            core_affinity::set_for_current(core);
            debug!("Waiting for barrier for worker {:?}", idx);
            ready_barrier.wait();
            debug!("Starting worker {:?}", idx);
            worker.run().expect("Worker failed!");
        })
    }

    /// Submit IO requests from dataflows until the IO queue is full or no more requests remain.
    fn saturate_io(&mut self) -> Result<()> {
        for flow in self.data_flows.values_mut() {
            if self.io.has_pending() {
                break;
            }

            while let Some(requests) = flow.get_next_io_request() {
                for r in requests {
                    self.io.request(r)?;
                }

                if self.io.has_pending() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Run one unit of CPU work from the first dataflow that has work ready.
    fn step_run_ready_cpu_work(&mut self) {
        for flow in self.data_flows.values_mut() {
            if let WorkStatus::Ran = flow.run_ready_cpu_work() {
                self.did_work_last_iteration = true;
            }
        }
    }

    /// Deliver completed IO buffers back to the operators that requested them.
    fn process_io_completions(&mut self) -> Result<()> {
        for request in self.io.completions()? {
            let data_flow = self.data_flows.get_mut(&request.data_flow_id).unwrap();
            data_flow.process_io(request.operator_idx, request.request);
        }
        Ok(())
    }

    /// Check each dataflow for completion and remove finished ones.
    fn try_finishing_dataflows(&mut self) {
        let mut to_remove = Vec::new();
        for (id, flow) in &mut self.data_flows {
            if flow.maybe_finish() {
                // If nobody broke out- everybody is finished!
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            info!("Finished data flow {:?}", id);
            self.data_flows.remove(&id);
        }
    }

    /// Attempt to steal work from sibling workers' channels when this worker is idle.
    fn try_steal_work(&mut self) {
        for flow in self.data_flows.values_mut() {
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

        self.last_seen_wake_count = self.waker.wait_if_unchanged(self.last_seen_wake_count);
    }

    fn clear_cancelled_dataflows(&mut self) {
        self.data_flows.retain(|_, d| !d.cancelled())
    }

    fn try_receiving_new_dataflow(&mut self) {
        if let Ok(builder) = self.data_flow_queue.try_recv() {
            info!("Received data flow...");
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

            self.process_io_completions()?;
            self.saturate_io()?;

            self.step_run_ready_cpu_work();

            self.try_finishing_dataflows();

            if !self.did_work_last_iteration {
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
