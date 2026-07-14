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
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::worker::{WakerSet, Worker, WorkerWaker};
pub use api::*;
pub use data_flow::{Error as DataFlowError, WorkStatus};
pub use functions::*;
pub use io::{FsRequest, HttpRequest};
pub use memory::BUFFER_SIZE;
pub use memory::ReadBuffer;
pub use memory::{MemoryContextFactory, init_memory_context, memory_ctx};
pub use numa::Topology;
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
    Dynamic, DynamicFilterSlot, Fold, GroupLimit, HashOnlyIntKeyExtractor, IntCell,
    IntKeyExtractor, IntPairKeyExtractor, IntRead, IntStrKeyExtractor, Max, MaxSlot, Min, MinSlot,
    NoRead, Nullary, NullaryFactory, NullaryOperatorFactory, OpTuple, Operator, OrderBy, Read,
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

/// What a successful dispatch hands back: the wakers of every node the dataflow
/// runs on, so cancellation (which may fire from a non-worker thread) can wake
/// parked workers on any node.
pub struct Dispatched {
    pub(crate) wakers: WakerSet,
}

#[derive(Clone)]
pub struct DataFlowDispatcher {
    /// One channel per worker, in global worker order (node 0's workers first).
    senders: Vec<StdSender<DataFlowBuilder>>,
    /// The worker/node shape, mirrored by every per-worker structure a
    /// dataflow builds (channels, sibling counters).
    topology: numa::Topology,
    /// One waker per node group, for dispatch-time and cancellation wakes.
    waker_set: WakerSet,
    /// Total ring slots across all nodes, used to size group-by working memory.
    buffers: usize,
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
    /// Dispatch one pre-built `DataFlow` bundle across the node groups.
    /// Builder `i` goes to global worker `i` (node `i / workers_per_node`). A
    /// full bundle has one builder per worker; a shorter one (e.g.
    /// [`run_on_worker`](Self::run_on_worker)'s single builder) reaches only
    /// the first workers, which is consistent with the dense `0..builder_count`
    /// worker indexing its channels are built for.
    pub fn push_data_flow(
        &self,
        builders: impl IntoIterator<Item = DataFlowBuilder>,
    ) -> Dispatched {
        for (worker, builder) in builders.into_iter().enumerate() {
            self.senders[worker].send(builder).unwrap();
        }
        // Wake idle workers on every node so they pick up the new dataflow
        // without waiting out their park.
        self.waker_set.notify_all();
        Dispatched {
            wakers: self.waker_set.clone(),
        }
    }

    /// The total worker count across all node groups. Every dataflow's
    /// operator chain is built at this size, since a dataflow runs on every
    /// worker.
    pub fn worker_count(&self) -> usize {
        self.topology.total_workers()
    }

    /// The worker/node shape of the pool, for code that partitions per-worker
    /// state by node (e.g. work-stealing channels, scan queues).
    pub fn topology(&self) -> numa::Topology {
        self.topology
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
            self.clone(),
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
        let factories = (0..self.worker_count()).map(|_| {
            let f = f.clone();
            NullaryOperatorFactory::new(OneShotNullaryFactory::new(f))
        });
        OperatorSpec::new(self.clone(), factories)
            .collect()
            .expect("run_on_workers dataflow failed");
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
/// just sends one `DataFlowBuilder` to each worker. The workers themselves drive
/// execution in their own event loops.
pub struct Dispatch {
    dataflow_dispatcher: DataFlowDispatcher,
    handles: Vec<JoinHandle<()>>,
    shutdown: Shutdown,
}

impl Dispatch {
    /// Spawn up to `worker_count` worker threads and return a `Dispatch` that owns them.
    ///
    /// Cores are grouped by NUMA node; `worker_count` is split evenly across nodes (each
    /// group is the same size, leftover cores dropped). All workers share one memory ring
    /// of `buffers` 2MB slots and one set of caches, but the ring is regioned per node:
    /// each node's workers first-touch, acquire, and evict only their own region, so
    /// every allocation is node-local while cached data stays visible pool-wide (see
    /// [`memory::RingLayout`]). Every dataflow runs on all workers. `disk_cache` is the
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

    /// Spawn one worker group per entry in `core_groups` (every group must be the
    /// same length). The total ring budget `buffers` is divided evenly across the
    /// groups' regions. Shared by [`spin_up`](Self::spin_up) and tests that inject
    /// a synthetic multi-node topology on a single-node machine.
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
        let topology = numa::Topology {
            workers_per_node: core_groups[0].len(),
            node_count: core_groups.len(),
        };
        debug_assert!(
            core_groups
                .iter()
                .all(|group| group.len() == topology.workers_per_node),
            "node groups must be equal-sized"
        );
        let total_workers = topology.total_workers();
        let layout = memory::RingLayout::new(topology, (buffers / topology.node_count).max(1));
        info!(
            "Starting {total_workers} workers across {} node group(s)...",
            topology.node_count
        );

        let barrier = Arc::new(Barrier::new(total_workers + 1));
        let should_exit = Arc::new(AtomicBool::new(false));
        let mut memory_factories = MemoryContextFactory::create_for_layout(layout).into_iter();
        let node_wakers: Vec<Arc<WorkerWaker>> = (0..topology.node_count)
            .map(|_| Arc::new(WorkerWaker::new()))
            .collect();
        let waker_set = WakerSet::new(node_wakers.clone(), topology.workers_per_node);

        let mut threads = vec![];
        let mut senders = vec![];
        for (node, group) in core_groups.into_iter().enumerate() {
            for (node_local_idx, core) in group.into_iter().enumerate() {
                let (tx, rx) = channel();
                senders.push(tx);
                threads.push(Worker::create(
                    node * topology.workers_per_node + node_local_idx,
                    total_workers,
                    node,
                    core,
                    should_exit.clone(),
                    memory_factories.next().unwrap(),
                    disk_cache.clone(),
                    rx,
                    barrier.clone(),
                    node_wakers[node].clone(),
                    waker_set.clone(),
                ));
            }
        }
        barrier.wait();
        info!("All workers have begun...");

        Dispatch {
            dataflow_dispatcher: DataFlowDispatcher {
                senders,
                topology,
                waker_set,
                buffers: layout.total_slots(),
                #[cfg(feature = "perf")]
                profiled: false,
            },
            handles: threads,
            shutdown: Shutdown {
                flag: should_exit,
                wakers: node_wakers,
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
    //! These tests cover node-group dispatch: a dataflow spans every group,
    //! workers carry global indices and node identities, and all groups share
    //! one ring. A synthetic two-group topology makes this testable on a
    //! single-node machine.

    use super::*;
    use crate::operations::nullary::OneShotNullaryFactory;
    use std::collections::HashSet;
    use std::sync::Mutex;

    /// Build `node_count` groups of `workers_per_node` workers from real core ids,
    /// duplicating the first core so the test runs on any machine.
    fn synthetic_groups(node_count: usize, workers_per_node: usize) -> Vec<Vec<CoreId>> {
        let core = core_affinity::get_core_ids().unwrap()[0];
        (0..node_count)
            .map(|_| vec![core; workers_per_node])
            .collect()
    }

    /// Run a one-shot-per-worker dataflow collecting `f()` from every worker.
    fn collect_from_each_worker<T: Send + 'static>(
        dispatcher: &DataFlowDispatcher,
        f: impl Fn() -> T + Clone + Send + 'static,
    ) -> Vec<T> {
        let factories = (0..dispatcher.worker_count())
            .map(|_| NullaryOperatorFactory::new(OneShotNullaryFactory::new(f.clone())));
        OperatorSpec::new(dispatcher.clone(), factories)
            .collect()
            .unwrap()
    }

    #[test]
    fn a_dataflow_spans_every_node_group() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        let nodes = collect_from_each_worker(dispatch.dispatcher(), crate::worker::current_node);

        assert_eq!(nodes.len(), 4, "ran on every worker of every group");
        assert_eq!(
            nodes.into_iter().collect::<HashSet<_>>(),
            HashSet::from([0, 1]),
            "workers of both node groups took part"
        );
        dispatch.exit();
    }

    #[test]
    fn all_node_groups_share_one_ring() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        let rings = collect_from_each_worker(dispatch.dispatcher(), || {
            memory_ctx().ring() as *const _ as usize
        });

        assert_eq!(rings.into_iter().collect::<HashSet<_>>().len(), 1);
        dispatch.exit();
    }

    #[test]
    fn workers_carry_dense_global_indices() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);

        let mut indices =
            collect_from_each_worker(dispatch.dispatcher(), || crate::worker::WORKER_IDX.get());

        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        dispatch.exit();
    }

    #[test]
    fn group_by_merges_correctly_across_node_groups() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Int32Type, Int64Type};
        use arrow_array::{ArrayRef, Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        // Enough ring slots per node region for the group-by's arenas, slab
        // tables, and merge fragments.
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 256, None);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        // Every key appears once per batch, so each of the 8 batches contributes
        // to every group and the per-node merge fragments must combine across
        // nodes to produce the right counts.
        let batches: Vec<RecordBatch> = (0..8)
            .map(|_| {
                let keys: ArrayRef = Arc::new(Int32Array::from((0..100).collect::<Vec<_>>()));
                RecordBatch::try_new(schema.clone(), vec![keys]).unwrap()
            })
            .collect();

        let results = values_input(dispatch.dispatcher(), batches)
            .record_batches()
            .group_by_aggregate::<IntKeyExtractor<Int32Type>, Compiled<(CountSlot,)>>(
                vec![0],
                vec![AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    arrow_schema::DataType::Int64,
                )],
                None,
                (),
            )
            .collect()
            .unwrap();

        let mut counts: Vec<(i32, i64)> = results
            .iter()
            .flat_map(|batch| {
                let keys = batch.column(0).as_primitive::<Int32Type>();
                let values = batch.column(1).as_primitive::<Int64Type>();
                (0..batch.num_rows())
                    .map(|i| (keys.value(i), values.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect();
        counts.sort();
        assert_eq!(counts.len(), 100, "one row per group");
        assert!(
            counts.iter().all(|&(_, c)| c == 8),
            "every key seen 8 times"
        );
        dispatch.exit();
    }

    #[test]
    fn run_on_workers_reaches_every_worker() {
        let dispatch = Dispatch::spin_up_groups(synthetic_groups(2, 2), 16, None);
        let seen = Arc::new(Mutex::new(HashSet::new()));

        let collector = seen.clone();
        dispatch.dispatcher().run_on_workers(move || {
            collector
                .lock()
                .unwrap()
                .insert(crate::worker::WORKER_IDX.get());
        });

        assert_eq!(seen.lock().unwrap().len(), 4, "touched every worker");
        dispatch.exit();
    }
}
