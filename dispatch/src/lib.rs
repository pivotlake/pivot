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
//! # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
//! // SELECT COUNT(*) FROM events WHERE url LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([13]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &arrow_array::RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             collect_selected_indices(&contains.run(col), &mut scratch.positions);
//!             Ok(RowSelection::Indices)
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
pub mod cpu_features;
pub mod env;
pub mod gather_barrier;
pub mod io;
pub mod memory;
pub mod waker;
pub mod worker;

mod api;
mod boundary_slot;
mod data_flow;
mod functions;
mod graph;
mod numa;
mod operations;
#[cfg(feature = "perf")]
mod profiler;
mod scan;
mod stats;

use crate::waker::{WakerSet, WorkerWaker};
use crate::worker::Worker;
pub use api::*;
pub use boundary_slot::BoundarySlot;
pub use cpu_features::missing_cpu_features;
pub use data_flow::{Error as DataFlowError, WorkStatus};
pub use functions::*;
pub use gather_barrier::GatherBarrier;
#[cfg(any(test, feature = "test-util"))]
pub use io::TestOperatorIO;
pub use io::{
    ASSUMED_NETWORK_BITS_PER_SECOND, FileRange, FsRequest, HttpRequest, OperatorIO, ReadData,
    ReadRequestId, ReadResponse, WriteRequestId, limit_network_receive_rate,
};
pub use memory::BUFFER_SIZE;
pub use memory::ReadBuffer;
pub use memory::Spiller;
pub use memory::{MemoryBlockState, MemoryBlockStatus, block_size_bytes};
pub use memory::{MemoryContextFactory, init_memory_context, memory_ctx};
pub use memory::{SpillDirectory, SpillDirectoryError};
pub use numa::{Topology, default_worker_count, dominant_node};
pub use operations::channels::{MpscSender, Sender};
pub use operations::nullary::Result as NullaryResult;
pub use operations::unary::filter::{
    RowDelivery, RowSelection, SelectionScratch, collect_selected_indices,
};
pub use operations::{
    Built, BuiltNullary, BuiltUnary, JoinBuildFilter, JoinKind, JoinResidualFn, JoinResidualSpec,
    JoinSpec, KeyBitset, KeyBitsetSlot, RangeCompare, RangeJoinSpec,
};
#[cfg(feature = "perf")]
pub use profiler::worker_tids;
pub use scan::{
    Projection, ROW_GROUP_IDX_FIELD, ROW_IDX_FIELD, VariantExtract, trailing_metadata_columns,
};
pub use stats::DataFlowStats;
#[cfg(feature = "test-util")]
pub use stats::StatsCollector;

pub use operations::channels::{
    ChannelFactory, FanInChannelFactory, MpscReceiver, NodeIdOutput, NodeWorkQueueChannelFactory,
    Receiver, ReturnToWorkerMpscFactory, RootChannelFactory, SharedWorkQueueChannelFactory,
    SharedWorkQueueReceiver, SharedWorkQueueSender, StealableChannelFactory, WorkerAwareSender,
    WorkerIdOutput, fan_in, mpsc_channel, node_work_queue, return_to_worker_mpsc,
    shared_work_queue, stealable, stealable_fifo, to_single_worker_mpsc,
};
#[cfg(any(test, feature = "test-util"))]
pub use operations::unary::test_utils;
pub use operations::unary::{Error as UnaryError, Result as UnaryResult, copy_out};
pub use operations::unary::{
    KWayMergePlan, KWayMergeTask, LocatedBatch, MergeRun, MergedOutput, batch_sort_indices,
    count_rows_not_after, is_batch_sorted,
};
pub use operations::unary::{WindowFunction, WindowRowLimit, WindowSpec};
pub use operations::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot,
    CountValidSlot, Distinct, Dynamic, Fold, GroupLimit, HashOnlyIntKeyExtractor, IntCell,
    IntKeyExtractor, IntPairKeyExtractor, IntRead, IntStrKeyExtractor, Max, MaxSlot, Min, MinSlot,
    NoRead, Nullary, NullaryFactory, NullaryOperatorFactory, OneShotNullaryFactory, OpTuple,
    Operator, OrderBy, Read, Result as OperatorResult, RowKeyExtractor, RowKeySchema, StrMax,
    StrMin, StrRead, StringKeyExtractor, Sum, SumSlot, WideSum,
};
pub use operations::{
    ChannelInputFull, ChannelInputSender, Consumer, DefaultUnaryFactory, MapFactory, OutputStatus,
    Outputter, PipelineBreaker, RootUnaryOperatorFactory, Unary, UnaryFactory, UnaryOperator,
    UnaryOperatorFactory,
};

// `max_background_threads` is capped well below its default (one per core):
// jemalloc's purger threads are unpinned, and on a box fully occupied by
// pinned workers every wakeup of an unpinned thread preempts a worker.
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8] = b"percpu_arena:percpu,oversize_threshold:0,\
muzzy_decay_ms:5000,dirty_decay_ms:10000,\
lg_extent_max_active_fit:8,background_thread:true,max_background_threads:2\0";

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
    /// Every node group's waker; shutdown wakes them all.
    wakers: WakerSet,
}

impl Shutdown {
    pub fn shutdown(&self) {
        self.flag.store(true, Ordering::Relaxed);
        // Workers parked on a waker won't observe `exit_flag` until someone
        // wakes them. Notify every group's waker so every parked worker returns
        // from `wait_if_unchanged` and sees the flag on its next loop iteration.
        self.wakers.notify_all();
    }
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
    /// Shared by every clone so arbitrary single-worker stages rotate across
    /// this pool rather than accumulating on its first worker.
    next_worker: Arc<AtomicUsize>,
    /// The pool's shared exit flag, the same `Arc<AtomicBool>` the [`Shutdown`]
    /// handle flips. Lets long-lived background work owned elsewhere (per-table
    /// refresh and compaction loops) observe that the pool is tearing down and
    /// stop, without the pool having to hand out its `Shutdown`.
    shutdown_flag: Arc<AtomicBool>,
    /// Shared by every clone so successive single-node handles (see
    /// [`on_next_node`](Self::on_next_node)) take the pool's nodes in turn.
    next_node: Arc<AtomicUsize>,
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
    /// The wakers of the pool's workers, for waking one of them from
    /// outside its own operators.
    pub fn waker_set(&self) -> WakerSet {
        self.waker_set.clone()
    }

    /// Dispatch one pre-built `DataFlow` bundle to this dispatcher's workers.
    /// Builder `i` goes to worker `i` of them (on their node
    /// `i / workers_per_node`). A full bundle has one builder per worker; a
    /// shorter one reaches only the first workers, consistent with the dense
    /// `0..builder_count` indexing its channels were built for. Returns the
    /// wakers of those workers, so cancellation can wake the parked ones.
    pub fn push_data_flow(&self, builders: impl IntoIterator<Item = DataFlowBuilder>) -> WakerSet {
        for (worker, builder) in builders.into_iter().enumerate() {
            self.senders[worker].send(builder).unwrap();
        }
        // Wake the idle workers so they pick up the new dataflow without
        // waiting out their park.
        self.waker_set.notify_all_delegated();
        self.waker_set.clone()
    }

    /// How many workers this dispatcher reaches: every worker of the pool,
    /// or those of one node (see [`on_next_node`](Self::on_next_node)). A
    /// dataflow's operator chain is built at this size, since the dataflow
    /// runs on every one of them.
    pub fn worker_count(&self) -> usize {
        self.topology.total_workers()
    }

    /// Choose the next worker for a stage that needs one arbitrary host.
    pub fn next_worker(&self) -> usize {
        self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_count()
    }

    /// Return a dispatcher over the workers of one NUMA node alone: a
    /// dataflow launched through it is built and run only by them, so the
    /// memory it touches stays on the node. Successive calls, on any clone,
    /// take the pool's nodes in turn. For long maintenance work such as a
    /// compaction, which then leaves the other nodes' workers and memory to
    /// the queries running beside it.
    ///
    /// To the dataflows launched through it the node is the whole pool: its
    /// workers are numbered from zero, on one node. A worker learns its
    /// place in the dataflow it is building from the
    /// [`BuildContext`](crate::api::BuildContext).
    pub fn on_next_node(&self) -> Self {
        let node = self.next_node.fetch_add(1, Ordering::Relaxed) % self.topology.node_count;
        let node_workers = self
            .topology
            .node_siblings(node * self.topology.workers_per_node);
        Self {
            senders: self.senders[node_workers].to_vec(),
            topology: numa::Topology::single_node(self.topology.workers_per_node),
            waker_set: self.waker_set.of_node(node),
            buffers: self.buffers / self.topology.node_count,
            ..self.clone()
        }
    }

    /// The worker/node shape of the pool, for code that partitions per-worker
    /// state by node (e.g. work-stealing channels, scan queues).
    pub fn topology(&self) -> numa::Topology {
        self.topology
    }

    /// Whether the worker pool is shutting down: the shared exit flag has been
    /// flipped (by [`Shutdown::shutdown`]). Background loops that outlive a
    /// single query poll this to know when to stop.
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_flag.load(Ordering::Relaxed)
    }

    /// Ship a `FnOnce() -> T` to one worker and return its result.
    ///
    /// Useful for one-shot setup work that needs a `MemoryContext` to run
    /// (e.g. `ParquetTable::from_files`, which touches the compressed cache)
    /// from a thread that doesn't have one. Successive calls rotate across the
    /// worker pool.
    pub fn run_on_worker<T, F>(&self, f: F) -> crate::data_flow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let worker_count = self.worker_count();
        let target = self.next_worker();
        let mut task = Some(f);
        let factories = (0..worker_count).map(|worker| {
            let task = if worker == target { task.take() } else { None };
            NullaryOperatorFactory::new(OneShotNullaryFactory::new(move || task.map(|task| task())))
        });
        let spec = OperatorSpec::new(self.clone(), factories);
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
    pub fn run_on_workers<T, F>(&self, f: F) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn() -> T + Clone + Send + 'static,
    {
        let factories = (0..self.worker_count()).map(|_| {
            let f = f.clone();
            NullaryOperatorFactory::new(OneShotNullaryFactory::new(move || Some(f())))
        });
        OperatorSpec::new(self.clone(), factories)
            .collect()
            .expect("run_on_workers dataflow failed")
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
    ///
    /// The pool has no spill directory, so it spills nothing: a dataflow that needs
    /// more memory than the ring has fails. See
    /// [`spin_up_with_spill_directory`](Self::spin_up_with_spill_directory).
    pub fn spin_up(
        worker_count: usize,
        buffers: usize,
        disk_cache: Option<Arc<crate::io::DiskCache>>,
    ) -> Self {
        Self::spin_up_with_spill_directory(worker_count, buffers, disk_cache, None)
    }

    /// [`spin_up`](Self::spin_up), with the directory the workers spill to when the
    /// ring runs short: operators that can are asked to write rows out to files
    /// under it and give the slots back, so a dataflow can hold more than fits in
    /// memory. `None` leaves the pool without one, and it never spills.
    pub fn spin_up_with_spill_directory(
        worker_count: usize,
        buffers: usize,
        disk_cache: Option<Arc<crate::io::DiskCache>>,
        spill_directory: Option<SpillDirectory>,
    ) -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        let groups = numa::balance_worker_groups(numa::group_cores_by_node(cores), worker_count);
        Self::spin_up_groups(groups, buffers, disk_cache, spill_directory)
    }

    /// Spawn `node_count` node groups of `workers_per_node` workers each,
    /// whatever nodes the machine has, for tests of work that is laid out by
    /// node. Every worker is pinned to the machine's first core.
    #[cfg(any(test, feature = "test-util"))]
    pub fn spin_up_synthetic_nodes(
        node_count: usize,
        workers_per_node: usize,
        buffers: usize,
    ) -> Self {
        let core = core_affinity::get_core_ids().unwrap()[0];
        let core_groups = vec![vec![core; workers_per_node]; node_count];
        Self::spin_up_groups(core_groups, buffers, None, None)
    }

    /// Spawn one worker group per entry in `core_groups` (every group must be the
    /// same length). The total ring budget `buffers` is divided evenly across the
    /// groups' regions. Shared by the public constructors and tests that inject
    /// a synthetic multi-node topology on a single-node machine.
    fn spin_up_groups(
        core_groups: Vec<Vec<CoreId>>,
        buffers: usize,
        disk_cache: Option<Arc<crate::io::DiskCache>>,
        spill_directory: Option<SpillDirectory>,
    ) -> Self {
        // A CPU missing an extension this binary was compiled for faults with
        // SIGILL inside whichever vectorised loop reaches the instruction first,
        // with no message. Fail here, before any worker runs one.
        let missing = cpu_features::missing_cpu_features();
        assert!(
            missing.is_empty(),
            "this CPU does not implement instruction-set extensions this binary was compiled \
             to use: {}. Run it on a CPU that has them, or rebuild with a lower floor (the \
             `[target.aarch64-unknown-linux-gnu]` rustflags in `.cargo/config.toml`)",
            missing.join(", ")
        );

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
            .map(|_| Arc::new(WorkerWaker::new(topology.workers_per_node)))
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
                    spill_directory.clone(),
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
                waker_set: waker_set.clone(),
                buffers: layout.total_slots(),
                next_worker: Arc::new(AtomicUsize::new(0)),
                shutdown_flag: should_exit.clone(),
                next_node: Arc::new(AtomicUsize::new(0)),
                #[cfg(feature = "perf")]
                profiled: false,
            },
            handles: threads,
            shutdown: Shutdown {
                flag: should_exit,
                wakers: waker_set,
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
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[test]
    fn a_dataflow_spans_every_node_group() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);

        let nodes = dispatch
            .dispatcher()
            .run_on_workers(crate::worker::current_node);

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
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);

        let rings = dispatch
            .dispatcher()
            .run_on_workers(|| memory_ctx().ring() as *const _ as usize);

        assert_eq!(rings.into_iter().collect::<HashSet<_>>().len(), 1);
        dispatch.exit();
    }

    #[test]
    fn workers_carry_dense_global_indices() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);

        let mut indices = dispatch
            .dispatcher()
            .run_on_workers(|| crate::worker::WORKER_IDX.get());

        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        dispatch.exit();
    }

    #[test]
    fn single_worker_hosts_rotate_across_dispatcher_clones() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(1, 3, 16);
        let first_handle = dispatch.dispatcher().clone();
        let second_handle = first_handle.clone();

        assert_eq!(first_handle.next_worker(), 0);
        assert_eq!(second_handle.next_worker(), 1);
        assert_eq!(first_handle.next_worker(), 2);
        assert_eq!(second_handle.next_worker(), 0);

        dispatch.exit();
    }

    /// The nodes whose workers handled the items of a dataflow launched
    /// through `dispatcher`.
    fn collect_working_nodes(dispatcher: &DataFlowDispatcher) -> HashSet<usize> {
        values_input(dispatcher, 0..1000)
            .map_each(|_: i32| Ok(crate::worker::current_node()))
            .collect()
            .unwrap()
            .into_iter()
            .collect()
    }

    #[test]
    fn confined_dataflows_each_work_on_one_node_and_take_the_nodes_in_turn() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);
        let dispatcher = dispatch.dispatcher();

        let nodes_of_each: Vec<HashSet<usize>> = (0..3)
            .map(|_| collect_working_nodes(&dispatcher.on_next_node()))
            .collect();

        assert_eq!(
            nodes_of_each,
            vec![HashSet::from([0]), HashSet::from([1]), HashSet::from([0])]
        );
        dispatch.exit();
    }

    #[test]
    fn a_single_node_handle_reaches_only_the_workers_of_its_node() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);
        let dispatcher = dispatch.dispatcher();
        let _on_first_node = dispatcher.on_next_node();

        let mut workers = dispatcher
            .on_next_node()
            .run_on_workers(|| crate::worker::WORKER_IDX.get());

        workers.sort();
        assert_eq!(workers, vec![2, 3]);
        dispatch.exit();
    }

    #[test]
    fn a_message_for_a_worker_of_a_single_node_dataflow_reaches_that_worker() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);
        let dispatcher = dispatch.dispatcher();
        let _on_first_node = dispatcher.on_next_node();
        let on_second_node = dispatcher.on_next_node();

        let receivers = values_input(&on_second_node, 0..100)
            .chain(
                to_single_worker_mpsc::<i32>(2, 1).into_iter().collect(),
                (0..2)
                    .map(|_| MapFactory(|_: i32| Ok(crate::worker::WORKER_IDX.get())))
                    .collect(),
            )
            .collect()
            .unwrap();

        assert_eq!(receivers, vec![3; 100]);
        dispatch.exit();
    }

    /// Count the rows of each of 100 keys, every one of which is in each of
    /// 8 batches, in a dataflow launched through `dispatcher`. Returns each
    /// key with its count, in key order.
    fn count_rows_per_key(dispatcher: &DataFlowDispatcher) -> Vec<(i32, i64)> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Int32Type, Int64Type};
        use arrow_array::{ArrayRef, Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batches: Vec<RecordBatch> = (0..8)
            .map(|_| {
                let keys: ArrayRef = Arc::new(Int32Array::from((0..100).collect::<Vec<_>>()));
                RecordBatch::try_new(schema.clone(), vec![keys]).unwrap()
            })
            .collect();

        let results = values_input(dispatcher, batches)
            .record_batches()
            .group_by_aggregate::<IntKeyExtractor<Int32Type>, Compiled<(CountSlot,), u8>>(
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
        counts
    }

    #[test]
    fn group_by_merges_correctly_across_node_groups() {
        // Enough ring slots per node region for the group-by's arenas, slab
        // tables, and merge targets. Each batch holds every key, so the
        // per-node aggregated tables must merge across nodes to produce the
        // right counts.
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 256);

        let counts = count_rows_per_key(dispatch.dispatcher());

        assert_eq!(counts, (0..100).map(|key| (key, 8)).collect::<Vec<_>>());
        dispatch.exit();
    }

    #[test]
    fn a_group_by_on_the_second_node_counts_every_row() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 512);
        let dispatcher = dispatch.dispatcher();
        let _on_first_node = dispatcher.on_next_node();

        let counts = count_rows_per_key(&dispatcher.on_next_node());

        assert_eq!(counts, (0..100).map(|key| (key, 8)).collect::<Vec<_>>());
        dispatch.exit();
    }

    #[test]
    fn an_order_by_on_the_second_node_sorts_every_row() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        use arrow_array::{ArrayRef, Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 512);
        let dispatcher = dispatch.dispatcher();
        let _on_first_node = dispatcher.on_next_node();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batches: Vec<RecordBatch> = (0..8)
            .map(|batch| {
                let keys: ArrayRef = Arc::new(Int32Array::from_iter_values(
                    (0..100).map(|i| 8 * i + batch),
                ));
                RecordBatch::try_new(schema.clone(), vec![keys]).unwrap()
            })
            .collect();

        let sorted: Vec<i32> = values_input(&dispatcher.on_next_node(), batches)
            .record_batches()
            .order_by(vec![OrderBy::new(0, false, false)])
            .collect()
            .unwrap()
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();

        assert_eq!(sorted, (0..800).collect::<Vec<_>>());
        dispatch.exit();
    }

    #[test]
    fn channel_input_streams_items_fed_while_the_flow_runs() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);
        let claimed = Arc::new((Mutex::new(()), std::sync::Condvar::new()));
        let on_claim = {
            let claimed = claimed.clone();
            Box::new(move || claimed.1.notify_all()) as Box<dyn Fn() + Send + Sync>
        };
        let (sender, spec) = channel_input::<i64>(dispatch.dispatcher(), 4, on_claim);

        let producer = std::thread::spawn(move || {
            for mut item in 0..1000i64 {
                loop {
                    match sender.try_send(item) {
                        Ok(()) => break,
                        Err(ChannelInputFull(returned)) => {
                            item = returned;
                            let guard = claimed.0.lock().unwrap();
                            // Timed wait: a claim may land between the failed
                            // send and this park, and the missed notify must
                            // not strand the producer.
                            drop(
                                claimed
                                    .1
                                    .wait_timeout(guard, std::time::Duration::from_millis(10)),
                            );
                        }
                    }
                }
            }
            sender.close();
        });
        let mut results = spec.map_each(|item| Ok(item * 2)).collect().unwrap();
        producer.join().unwrap();

        results.sort();
        assert_eq!(results, (0..1000i64).map(|i| i * 2).collect::<Vec<_>>());
        dispatch.exit();
    }

    #[test]
    fn run_on_workers_reaches_every_worker() {
        let dispatch = Dispatch::spin_up_synthetic_nodes(2, 2, 16);
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
