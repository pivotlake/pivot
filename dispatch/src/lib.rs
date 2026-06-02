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
//! ```no_run
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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender as StdSender, channel};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use tikv_jemallocator::Jemalloc;
use tracing::info;

mod env;

mod api;
mod data_flow;
mod functions;
mod io;
mod memory;
mod operations;
mod record_batch_metadata;
mod worker;

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
pub use operations::parquet::types::metadata::{
    ColumnChunkMeta, ColumnStatistics, RowGroupMetadata,
};
pub use operations::parquet::types::projection::Projection;
pub use operations::parquet::types::table::{Error as ParquetTableError, ParquetTable};
pub use operations::{
    AggKind, AggSpec, GroupAggKind, GroupAggSlot, IntKeyExtractor, IntPairAggExtractor, Nullary,
    NullaryFactory, NullaryOperatorFactory, Operator, OrderBy, Result as OperatorResult,
    StringKeyExtractor,
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
/// Created via [`Dispatch::spin_up`], which spawns one [`Worker`] per CPU core (pinned
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
    /// Spawn `worker_count` worker threads (capped by available cores) and return a
    /// `Dispatch` that owns them. `buffers` sets the size of the shared ring (in 2MB
    /// slots) used for all worker memory contexts. Blocks until every worker has
    /// finished pre-faulting and is ready for work.
    pub fn spin_up(worker_count: usize, buffers: usize) -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        info!("Setting up io...");

        let cores: Vec<_> = cores.into_iter().take(worker_count).collect();
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
