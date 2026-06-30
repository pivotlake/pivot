//! Dispatch is a crate designed for high-throughput, low-latency, never-crashing (always-running),
//! parallel data flow execution.
//!
//! The basic idea is to have a Worker per core (thread per core) which runs an event loop,
//! prioritizing local work but also capable of stealing work from sibling workers.
//! We want to enjoy both worlds - both cache locality (and in the future NUMA awareness)
//! that often eludes work-stealing libraries like tokio, while still having the parallelism of a
//! work-stealing library/library that allows parallel operation.
//!
//! Each worker is meant to run in large part "alone" on dataflows — its operations are all
//! parallelism-aware, and workers synchronize with each other only at the deepest level
//! (a running `Operator`).
//!
//! The main execution is done through sending a [`DataFlowBuilder`] to each worker. For a
//! given query, every worker receives the full dataflow (source through output), with each
//! stage able to synchronize across workers. For example, in a `Table -> Filter -> Aggregate`
//! query, each worker filters its own chunk of data and maintains a local count, then the
//! aggregate operators coordinate to produce the final result.
//!
//! To work with the dispatch library, build and execute dataflows through the `api` module.
//!
//! The architecture is partially based on the paper: <https://db.in.tum.de/~leis/papers/morsels.pdf>.
//!
//!
//! # Example
//!
//! ```ignore
//! # use std::sync::Arc;
//! # use arrow_array::StringViewArray;
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let dispatch = Dispatch::spin_up(1, 32, None);
//! # let dispatcher = dispatch.dispatcher();
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! // SELECT COUNT(*) FROM events WHERE url LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([13]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &arrow_array::RecordBatch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             contains.run(col)
//!         }
//!     })
//!     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
//!     .collect();
//! ```
//!

// Internal engine crate: a handful of public-facing items document their
// behaviour by linking to the private traits they're built on (e.g. a
// `KeyExtractor` impl links to the trait). That's intentional here — we're not
// a published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

use core_affinity::CoreId;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Sender as StdSender, channel};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use tikv_jemallocator::Jemalloc;
use tracing::info;

// Engine infrastructure exposed as public API so the Parquet reader (which now
// lives in `catalog`, not here) can build on it: memory/IO/array-builder
// primitives and worker identity.
pub mod arrays;
pub mod env;
pub mod io;
pub mod memory;
pub mod worker;

mod api;
mod data_flow;
mod functions;
mod numa;
mod operations;
#[cfg(feature = "perf")]
mod profiler;
mod scan;
mod stats;

use crate::operations::nullary::OneShotNullaryFactory;
use crate::worker::{Worker, WorkerWaker};
pub use api::*;
pub use data_flow::{Error as DataFlowError, WorkStatus};
pub use functions::*;
pub use io::{FsRequest, HttpRequest};
pub use memory::BUFFER_SIZE;
pub use memory::ReadBuffer;
pub use memory::{MemoryContextFactory, init_memory_context, memory_ctx};
pub use operations::channels::{MpscSender, Sender};
pub use operations::nullary::Result as NullaryResult;
#[cfg(feature = "perf")]
pub use profiler::worker_tids;
pub use scan::{Projection, ROW_GROUP_IDX_FIELD, ROW_IDX_FIELD, trailing_metadata_columns};
pub use stats::DataFlowStats;

pub use operations::channels::{
    ChannelFactory, FanInChannelFactory, MpscReceiver, Receiver, ReturnToWorkerMpscFactory,
    RootChannelFactory, StealableChannelFactory, WorkerAwareSender, WorkerIdOutput, fan_in,
    mpsc_channel, return_to_worker_mpsc, stealable,
};
#[cfg(any(test, feature = "test-util"))]
pub use operations::unary::test_utils;
pub use operations::unary::{Error as UnaryError, Result as UnaryResult};
pub use operations::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot, Distinct,
    Dynamic, DynamicFilterSlot, Fold, GroupLimit, HashOnlyIntKeyExtractor, IntKeyExtractor,
    IntPairKeyExtractor, IntRead, IntStrKeyExtractor, Max, MaxSlot, Min, MinSlot, NoRead, Nullary,
    NullaryFactory, NullaryOperatorFactory, Numeric, OpTuple, Operator, OrderBy, Read,
    Result as OperatorResult, RowKeyExtractor, RowKeySchema, StrMax, StrMin, StrRead,
    StringKeyExtractor, Sum, SumSlot, WideSum,
};
pub use operations::{
    Consumer, DefaultUnaryFactory, MapFactory, Outputter, PipelineBreaker,
    RootUnaryOperatorFactory, Unary, UnaryFactory, UnaryOperator, UnaryOperatorFactory,
};

#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8] = b"percpu_arena:percpu,oversize_threshold:0,\
muzzy_decay_ms:5000,dirty_decay_ms:10000,\
lg_extent_max_active_fit:8,background_thread:true\0";

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// Our default identifier across the system is a usize. To denote this (instead of simply having a
/// usize which could also be a counter etc) we have an system-wide alias
pub type Identifier = usize;

/// A helper object to shut-down a running dispatch. This does NOT collect the worker threads, but
/// wakes all workers up and turns on the flag to notify them to shut down. They will exit on
/// their next iteration.
pub struct Shutdown {
    flag: Arc<AtomicBool>,
    /// One waker per NUMA node group; shutdown wakes every group.
    wakers: Vec<Arc<WorkerWaker>>,
}

impl Shutdown {
    pub fn shutdown(&self) {
        self.flag.store(true, Ordering::Relaxed);
        // Workers parked on a waker won't observe `exit_flag` until someone
        // wakes them. Notify every group's waker so every parked worker returns
        // from `wait_if_unchanged` and sees the flag on its next loop iteration.
        for waker in &self.wakers {
            waker.notify();
        }
    }
}

/// One NUMA node's worker group: the channels feeding its workers, the waker that
/// parks/unparks them, and a count of dataflows currently dispatched to it (used to
/// pick the least-loaded node). Every worker in a group shares a node-local memory
/// ring, so a dataflow dispatched here keeps all its CPU and memory on one node.
#[derive(Clone)]
struct DispatchHandle {
    senders: Vec<StdSender<DataFlowBuilder>>,
    waker: Arc<WorkerWaker>,
    in_flight: Arc<AtomicUsize>,
}

/// Decrements a node group's in-flight count when the dataflow's handle is dropped.
/// Held by [`DataFlowHandle`](crate::DataFlowHandle) so a node's load reflects only
/// dataflows whose results are still being consumed.
pub struct InFlightGuard {
    in_flight: Arc<AtomicUsize>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// What a successful dispatch hands back: the chosen node's waker (so the consumer can
/// wake those workers to cancel) and a guard that releases the node's load on drop.
pub struct Dispatched {
    pub(crate) waker: Arc<WorkerWaker>,
    pub(crate) guard: InFlightGuard,
}

#[derive(Clone)]
pub struct DataFlowDispatcher {
    /// One worker group per NUMA node. A dataflow is dispatched to exactly one.
    handles: Vec<DispatchHandle>,
    /// Workers in each node group (equal across groups), and the size every
    /// dataflow's operator chain is built for.
    workers_per_node: usize,
    /// Ring slots per node group, used to size group-by working memory.
    buffers: usize,
    /// When set, dispatch is forced to this node group instead of the least-loaded
    /// one. Set on a per-call clone (see [`pin_to_node`](Self::pin_to_node)) so maintenance
    /// fan-outs can target every node in turn; `None` on the shared handle.
    pinned_node: Option<usize>,
    /// Whether every dataflow launched through this handle is marked profiled.
    /// Set only on a per-query clone (see [`with_profiling`](Self::with_profiling))
    /// so profiling (and the exclusive execution it triggers) is scoped to one
    /// query's dataflows; `false` on the shared handle.
    #[cfg(feature = "perf")]
    profiled: bool,
}

#[cfg(feature = "perf")]
impl DataFlowDispatcher {
    /// Return a clone of this dispatcher that marks every dataflow it launches as
    /// profiled. Clone the shared handle, call this, and compile/execute through
    /// the result so only that query's dataflows are profiled (and run
    /// exclusively while they do).
    pub fn with_profiling(mut self, profiled: bool) -> Self {
        self.profiled = profiled;
        self
    }

    /// Whether this handle marks its dataflows profiled.
    pub(crate) fn profiled(&self) -> bool {
        self.profiled
    }
}

impl DataFlowDispatcher {
    /// Index of the node group to dispatch the next dataflow to. Always the first
    /// NUMA node (node 0) so a dataflow runs entirely on one socket with node-local
    /// memory — the fastest, most stable config for memory-bound work on a
    /// multi-socket box. `pinned_node` still overrides, which is how
    /// [`run_on_workers`](Self::run_on_workers) reaches every node in turn.
    fn select_target_node(&self) -> usize {
        self.pinned_node.unwrap_or(0)
    }

    /// Dispatch one pre-built `DataFlow` bundle to a single node group: each of that
    /// node's workers receives exactly one builder. Returns the node's waker and an
    /// in-flight guard for the resulting [`DataFlowHandle`](crate::DataFlowHandle).
    pub fn push_data_flow(
        &self,
        builders: impl IntoIterator<Item = DataFlowBuilder>,
    ) -> Dispatched {
        let handle = &self.handles[self.select_target_node()];
        for (sender, builder) in handle.senders.iter().zip(builders) {
            sender.send(builder).unwrap();
        }
        handle.in_flight.fetch_add(1, Ordering::Relaxed);
        // Wake idle workers so they pick up the new dataflow without
        // waiting out their park.
        handle.waker.notify();
        Dispatched {
            waker: handle.waker.clone(),
            guard: InFlightGuard {
                in_flight: handle.in_flight.clone(),
            },
        }
    }

    /// A clone of this dispatcher that forces dispatch to `node` rather than the
    /// least-loaded one. Used by [`run_on_workers`](Self::run_on_workers) to reach
    /// every node in turn, and to pin one-shot setup to a known node.
    fn pin_to_node(&self, node: usize) -> Self {
        let mut dispatcher = self.clone();
        dispatcher.pinned_node = Some(node);
        dispatcher
    }

    /// Workers in one node group: the size every dataflow's operator chain is built
    /// for, since a dataflow runs on exactly one group.
    pub fn worker_count(&self) -> usize {
        self.workers_per_node
    }

    /// Total workers across all node groups.
    pub fn total_worker_count(&self) -> usize {
        self.workers_per_node * self.handles.len()
    }

    /// Ship a `FnOnce() -> T` to worker 0 and return its result.
    ///
    /// Useful for one-shot setup work that needs a `MemoryContext` to run
    /// (e.g. `ParquetTable::from_directory`, which touches the compressed cache)
    /// from a thread that doesn't have one. Builds a single-element
    /// `OperatorSpec` whose nullary fires once, sends one item, and finishes.
    pub fn run_on_worker<T, F>(&self, f: F) -> crate::data_flow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let spec = OperatorSpec::new(
            self.pin_to_node(0),
            std::iter::once(NullaryOperatorFactory::new(OneShotNullaryFactory::new(f))),
        );
        let mut results = spec.collect()?;
        Ok(results
            .pop()
            .expect("OneShotNullary should have produced exactly one result"))
    }

    /// Run `f` once on **every** worker thread and block until all have
    /// finished. The fan-out sibling of [`run_on_worker`](Self::run_on_worker):
    /// builds one [`OneShotNullary`](crate::operations::nullary::OneShotNullary)
    /// per worker (each gets its own clone of `f`), so `f` executes on a thread
    /// that has a live `MemoryContext` — e.g. to touch per-worker ring/free-pool
    /// state such as `memory_ctx().zero_dirty_buffers()`.
    pub fn run_on_workers<F>(&self, f: F)
    where
        F: Fn() + Clone + Send + 'static,
    {
        // Per-worker setup must touch every physical worker, so dispatch a one-shot
        // dataflow to each node group in turn (a single dispatch only reaches one node).
        for node in 0..self.handles.len() {
            let factories = (0..self.workers_per_node).map(|_| {
                let f = f.clone();
                NullaryOperatorFactory::new(OneShotNullaryFactory::new(f))
            });
            OperatorSpec::new(self.pin_to_node(node), factories)
                .collect()
                .expect("run_on_workers dataflow failed");
        }
    }
}

/// Owns the worker threads and exposes a [`DataFlowDispatcher`] for sending work to them.
///
/// Created via [`Dispatch::spin_up`], which spawns one `Worker` per CPU core (pinned
/// to its core, connected via an mpsc channel for receiving [`DataFlowBuilder`]s) and
/// then blocks until all workers have completed their startup. Borrow the inner
/// dispatcher with [`dispatcher`](Self::dispatcher) and shut everything down with
/// [`exit`](Self::exit).
///
/// `Dispatch` does not actively schedule — [`push_data_flow`](DataFlowDispatcher::push_data_flow)
/// just sends one `DataFlowBuilder` to each worker of the chosen node group. The workers
/// themselves drive execution in their own event loops.
pub struct Dispatch {
    dataflow_dispatcher: DataFlowDispatcher,
    handles: Vec<JoinHandle<()>>,
    shutdown: Shutdown,
}

impl Dispatch {
    /// Spawn up to `worker_count` worker threads and return a `Dispatch` that owns them.
    ///
    /// Cores are grouped by NUMA node; `worker_count` is split evenly across nodes (each
    /// group is the same size, leftover cores dropped). Each node group gets its own
    /// memory ring of `buffers / node_count` 2MB slots, first-touched by that node's
    /// workers so it stays node-local, plus its own caches and waker. A dataflow is later
    /// dispatched to exactly one group, so it never spans nodes. `disk_cache` is the
    /// optional disk cache for remote reads, shared by every worker's requester (`None`
    /// disables it). Blocks until every worker has finished pre-faulting.
    pub fn spin_up(
        worker_count: usize,
        buffers: usize,
        disk_cache: Option<Arc<crate::io::DiskCache>>,
    ) -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        let groups = numa::balance_worker_groups(numa::group_cores_by_node(cores), worker_count);
        Self::spin_up_groups(groups, buffers, disk_cache)
    }

    /// Spawn one self-contained worker group per entry in `core_groups` (every group must
    /// be the same length). The total ring budget `buffers` is divided evenly across the
    /// groups. Shared by [`spin_up`](Self::spin_up) and tests that inject a synthetic
    /// multi-node topology on a single-node machine.
    fn spin_up_groups(
        core_groups: Vec<Vec<CoreId>>,
        buffers: usize,
        disk_cache: Option<Arc<crate::io::DiskCache>>,
    ) -> Self {
        // Empty groups would size the ready barrier for one phantom worker and then
        // spawn none, deadlocking the barrier wait below. Fail loudly instead.
        assert!(
            !core_groups.is_empty(),
            "no cores available to spin up workers"
        );
        let workers_per_node = core_groups[0].len();
        let node_count = core_groups.len();
        debug_assert!(
            core_groups.iter().all(|group| group.len() == workers_per_node),
            "node groups must be equal-sized"
        );
        let total_workers = workers_per_node * node_count;
        let per_node_buffers = (buffers / node_count).max(1);
        info!("Starting {total_workers} workers across {node_count} node group(s)...");

        let barrier = Arc::new(Barrier::new(total_workers + 1));
        let should_exit = Arc::new(AtomicBool::new(false));

        let mut threads = vec![];
        let mut handles = vec![];
        for group in core_groups {
            let waker = Arc::new(WorkerWaker::new());
            let mut factories =
                MemoryContextFactory::create_many(workers_per_node, per_node_buffers);
            let mut senders = vec![];
            for (node_local_idx, core) in group.into_iter().enumerate() {
                let (tx, rx) = channel();
                senders.push(tx);
                threads.push(Worker::create(
                    node_local_idx,
                    workers_per_node,
                    core,
                    should_exit.clone(),
                    factories.pop().unwrap(),
                    disk_cache.clone(),
                    rx,
                    barrier.clone(),
                    waker.clone(),
                ));
            }
            handles.push(DispatchHandle {
                senders,
                waker,
                in_flight: Arc::new(AtomicUsize::new(0)),
            });
        }
        barrier.wait();
        info!("All workers have begun...");

        let wakers = handles.iter().map(|h| h.waker.clone()).collect();
        Dispatch {
            dataflow_dispatcher: DataFlowDispatcher {
                handles,
                workers_per_node,
                buffers: per_node_buffers,
                pinned_node: None,
                #[cfg(feature = "perf")]
                profiled: false,
            },
            handles: threads,
            shutdown: Shutdown {
                flag: should_exit,
                wakers,
            },
        }
    }

    /// Total number of active worker threads across all node groups.
    pub fn workers(&self) -> usize {
        self.handles.len()
    }

    /// Signal every worker to stop and join their threads.
    pub fn exit(self) {
        info!("Shutting down!");
        self.shutdown.shutdown();
        // Drop our copies of the senders so the worker channels close.
        drop(self.dataflow_dispatcher);
        for handle in self.handles {
            handle.join().unwrap();
        }
        info!("Shut down...");
    }

    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dataflow_dispatcher
    }

    pub fn into_parts(self) -> (Vec<JoinHandle<()>>, Shutdown) {
        (self.handles, self.shutdown)
    }
}

#[cfg(test)]
mod tests {
    //! Node-group dispatch: a dataflow stays on one group, maintenance fan-out
    //! reaches every group, and a group's load clears once its dataflow is done.
    //! Each worker's ring pointer identifies its group (one ring per group), so a
    //! synthetic two-group topology is testable on a single-node machine.

    use super::*;
    use crate::operations::nullary::OneShotNullaryFactory;
    use std::collections::HashSet;
    use std::sync::Mutex;

    /// Build `node_count` groups of `workers_per_node` workers from real core ids,
    /// duplicating the first core so the test runs on any machine.
    fn synthetic_groups(node_count: usize, workers_per_node: usize) -> Vec<Vec<CoreId>> {
        let core = core_affinity::get_core_ids().unwrap()[0];
        (0..node_count).map(|_| vec![core; workers_per_node]).collect()
    }

    /// Run a one-shot-per-worker dataflow that returns each worker's ring pointer.
    fn ring_pointers_for_one_dataflow(dispatcher: &DataFlowDispatcher) -> Vec<usize> {
        let factories = (0..dispatcher.worker_count()).map(|_| {
            NullaryOperatorFactory::new(OneShotNullaryFactory::new(|| {
                memory_ctx().ring() as *const _ as usize
            }))
        });
        OperatorSpec::new(dispatcher.clone(), factories)
            .collect()
            .unwrap()
    }

    #[test]
    fn a_dataflow_runs_on_a_single_node_group() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        let rings = ring_pointers_for_one_dataflow(dispatch.dispatcher());

        assert_eq!(rings.len(), 2, "ran on one group's workers");
        assert_eq!(
            rings.iter().collect::<HashSet<_>>().len(),
            1,
            "every worker shared one ring, so the dataflow stayed on one node"
        );
        dispatch.exit();
    }

    #[test]
    fn run_on_workers_reaches_every_node_group() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);
        let seen = Arc::new(Mutex::new(HashSet::new()));

        let collector = seen.clone();
        dispatch.dispatcher().run_on_workers(move || {
            collector
                .lock()
                .unwrap()
                .insert(memory_ctx().ring() as *const _ as usize);
        });

        assert_eq!(seen.lock().unwrap().len(), 2, "touched both node groups");
        dispatch.exit();
    }

    #[test]
    fn equal_load_selects_the_lowest_node_index() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        let node = dispatch.dispatcher().select_target_node();

        assert_eq!(node, 0);
        dispatch.exit();
    }

    #[test]
    fn node_load_returns_to_zero_after_a_dataflow_completes() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        ring_pointers_for_one_dataflow(dispatch.dispatcher());

        for handle in &dispatch.dataflow_dispatcher.handles {
            assert_eq!(handle.in_flight.load(Ordering::Relaxed), 0);
        }
        dispatch.exit();
    }
}
