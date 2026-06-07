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
//! stage able to synchronize across workers. For example, in a `Table -> Filter -> Count`
//! query, each worker filters its own chunk of data and maintains a local count, then the
//! count operators coordinate to produce the final result.
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
//! # let dispatch = Dispatch::spin_up(1, 32);
//! # let dispatcher = dispatch.dispatcher();
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! // SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([13]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &arrow_array::RecordBatch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             contains.run(col)
//!         }
//!     })
//!     .count()
//!     .collect();
//! ```
//!

// Internal engine crate: a handful of public-facing items document their
// behaviour by linking to the private traits they're built on (e.g. a
// `KeyExtractor` impl links to the trait). That's intentional here — we're not
// a published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

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
mod operations;
mod scan;

use crate::operations::nullary::OneShotNullaryFactory;
use crate::worker::{Worker, WorkerWaker};
pub use api::*;
pub use data_flow::{Error as DataFlowError, WorkStatus};
pub use functions::*;
pub use io::IORequest;
pub use memory::BUFFER_SIZE;
pub use memory::ReadBuffer;
pub use memory::{MemoryContextFactory, init_memory_context, memory_ctx};
pub use operations::channels::{MpscSender, Sender};
pub use operations::nullary::Result as NullaryResult;
pub use scan::Projection;

pub use operations::channels::{
    ChannelFactory, FanInChannelFactory, MpscReceiver, Receiver, ReturnToWorkerMpscFactory,
    RootChannelFactory, StealableChannelFactory, WorkerAwareSender, WorkerIdOutput, fan_in,
    mpsc_channel, return_to_worker_mpsc, stealable,
};
#[cfg(any(test, feature = "test-util"))]
pub use operations::unary::test_utils;
pub use operations::unary::{Error as UnaryError, Result as UnaryResult};
pub use operations::{
    AggKind, AggSpec, AggregationKind, AggregationRowValueExtractor, AggregationSlot, Compiled,
    Count, DynamicFilterSlot, IntKeyExtractor, IntPairKeyExtractor, Nullary, NullaryFactory,
    NullaryOperatorFactory, Operator, OrderBy, Result as OperatorResult, StringKeyExtractor, Sum,
    ValueExtractor,
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
    waker: Arc<WorkerWaker>,
}

impl Shutdown {
    pub fn shutdown(&self) {
        self.flag.store(true, Ordering::Relaxed);
        // Workers parked on the waker won't observe `exit_flag` until someone
        // wakes them. Notify so every parked worker returns from
        // `wait_if_unchanged` and sees the flag on its next loop iteration.
        self.waker.notify();
    }
}

#[derive(Clone)]
pub struct DataFlowDispatcher {
    senders: Vec<StdSender<DataFlowBuilder>>,
    buffers: usize,
    /// Shared [`WorkerWaker`] handed to every worker on startup. Non-worker
    /// threads (e.g. the embedding server) can call `waker().notify()` to
    /// kick idle workers out of their park; worker threads access the same
    /// instance through their thread-local [`crate::worker::worker_waker`].
    waker: Arc<WorkerWaker>,
}

impl DataFlowDispatcher {
    /// Send each worker their pre-built `DataFlow` bundle. Each worker
    /// receives exactly one builder with their specific resources.
    pub fn push_data_flow(&self, builders: impl IntoIterator<Item = DataFlowBuilder>) {
        for (sender, builder) in self.senders.iter().zip(builders) {
            sender.send(builder).unwrap();
        }
        // Wake idle workers so they pick up the new dataflow without
        // waiting out their park.
        self.waker.notify();
    }

    /// Borrow the shared waker so callers outside a worker thread (e.g.
    /// cancellation handles, the embedding server's shutdown path) can wake
    /// parked workers without going through the worker-thread TLS.
    pub fn waker(&self) -> &Arc<WorkerWaker> {
        &self.waker
    }

    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }

    /// Ship a `FnOnce() -> T` to worker 0 and return its result.
    ///
    /// Useful for one-shot setup work that needs a `MemoryContext` to run
    /// (e.g. `ParquetTable::from_directory`, which touches the file cache)
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

/// Decide which cores to run workers on and how large the ring may be.
///
/// On a single-NUMA-node machine: the first `worker_count` cores and the
/// requested `buffers`, unchanged.
///
/// On a **multi-NUMA-node** machine (e.g. a 2-socket box) automatically confine
/// workers to the first node's CPUs and cap the ring to 4/5 of that node's RAM.
/// Combined with pinning-before-prefault in [`Worker::create`], the ring is
/// first-touched (and thus placed) on that one node — keeping memory-bound queries
/// off the slow, erratic cross-socket path. This relies on affinity + first-touch
/// rather than a hard `mbind`, which can trip a lost-wakeup hang.
///
/// Set `PIVOT_NUMA_CONFINE=0` to opt out and use every core.
fn select_cores_and_buffers(
    cores: Vec<core_affinity::CoreId>,
    worker_count: usize,
    buffers: usize,
) -> (Vec<core_affinity::CoreId>, usize) {
    let confine = std::env::var("PIVOT_NUMA_CONFINE").map(|v| v != "0").unwrap_or(true);
    if confine && numa_node_count() > 1 {
        let allowed = read_id_list("/sys/devices/system/node/node0/cpulist");
        if !allowed.is_empty() {
            let node_cores: Vec<core_affinity::CoreId> =
                cores.into_iter().filter(|c| allowed.contains(&c.id)).collect();
            let buffers = buffers.min(node0_ring_cap_buffers());
            info!(
                "multi-NUMA box detected: confining workers to node0 ({} cores), ring capped to {} buffers",
                node_cores.len(),
                buffers
            );
            return (node_cores, buffers);
        }
    }
    (cores.into_iter().take(worker_count).collect(), buffers)
}

/// Number of online NUMA nodes (from `/sys/devices/system/node/online`, e.g.
/// `0-1`). Returns 1 when the topology can't be read.
fn numa_node_count() -> usize {
    read_id_list("/sys/devices/system/node/online").len().max(1)
}

/// Parse a Linux sysfs id-list file (e.g. `0-95` or `0-3,8-11`) into the set of
/// ids it names. Empty set if the file can't be read.
fn read_id_list(path: &str) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    let Ok(s) = std::fs::read_to_string(path) else {
        return set;
    };
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    set.extend(a..=b);
                }
            }
            None => {
                if let Ok(a) = part.trim().parse::<usize>() {
                    set.insert(a);
                }
            }
        }
    }
    set
}

/// Cap the ring at 4/5 of node0's RAM (in [`BUFFER_SIZE`] slots), from
/// `/sys/devices/system/node/node0/meminfo` (`Node 0 MemTotal: <kB> kB`).
/// Falls back to no cap (`usize::MAX`) if unreadable.
fn node0_ring_cap_buffers() -> usize {
    let Ok(s) = std::fs::read_to_string("/sys/devices/system/node/node0/meminfo") else {
        return usize::MAX;
    };
    for line in s.lines() {
        if line.contains("MemTotal:") {
            // ".. MemTotal:   198045696 kB" -> number is the second-to-last token
            if let Some(kb) = line
                .split_whitespace()
                .rev()
                .nth(1)
                .and_then(|v| v.parse::<usize>().ok())
            {
                // kB -> bytes (*1024), take 4/5, then convert to BUFFER_SIZE slots.
                return kb.saturating_mul(1024) / 5 * 4 / BUFFER_SIZE;
            }
        }
    }
    usize::MAX
}

impl Dispatch {
    /// Spawn `worker_count` worker threads (capped by available cores) and return a
    /// `Dispatch` that owns them. `buffers` sets the size of the shared ring (in 2MB
    /// slots) used for all worker memory contexts. Blocks until every worker has
    /// finished pre-faulting and is ready for work.
    pub fn spin_up(worker_count: usize, buffers: usize) -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        info!("Setting up io...");

        let (cores, buffers) = select_cores_and_buffers(cores, worker_count, buffers);
        // Number of workers actually spawned (may be < requested when confined to
        // a single NUMA node); used for prefault striding and `NUM_WORKERS`.
        let worker_count = cores.len();
        let mut threads = vec![];
        info!("Creating memory context ({})...", cores.len());
        let barrier = Arc::new(Barrier::new(cores.len() + 1));
        let mut senders = vec![];
        let mut memory_context_factories = MemoryContextFactory::create_many(cores.len(), buffers);
        let should_exit = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(WorkerWaker::new());
        info!("Starting workers...");
        for (i, core) in cores.into_iter().enumerate() {
            let (tx, rx) = channel();
            senders.push(tx);
            threads.push(Worker::create(
                i,
                worker_count,
                core,
                should_exit.clone(),
                memory_context_factories.pop().unwrap(),
                rx,
                barrier.clone(),
                waker.clone(),
            ));
        }
        barrier.wait();
        info!("All workers have begun...");

        Dispatch {
            dataflow_dispatcher: DataFlowDispatcher {
                senders,
                buffers,
                waker: waker.clone(),
            },
            handles: threads,
            shutdown: Shutdown {
                flag: should_exit,
                waker,
            },
        }
    }

    /// Number of active worker threads (one per core).
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
