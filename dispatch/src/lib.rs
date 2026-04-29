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
//! # let table = Arc::new(ParquetTable::from_directory(std::path::Path::new("/tmp")).unwrap());
//! // SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'
//! let results = table_input(&table, Projection::columns([13]), false)
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

use std::sync::mpsc::{Sender as StdSender, channel};
use std::sync::{Arc, Barrier, OnceLock};
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

use crate::worker::Worker;
pub use api::*;
pub use data_flow::WorkStatus;
pub use functions::*;
pub use io::IORequest;
pub use memory::ReadBuffer;
pub use operations::channels::{MpscSender, Sender};
pub use operations::nullary::Result as NullaryResult;
pub use operations::parquet::types::metadata::{
    ColumnChunkMeta, ColumnStatistics, RowGroupMetadata,
};
pub use operations::parquet::types::projection::Projection;
pub use operations::parquet::types::table::{Error as ParquetTableError, ParquetTable};
pub use operations::{
    IntKeyExtractor, Nullary, NullaryFactory, NullaryOperatorFactory, Operator, OrderBy,
    Result as OperatorResult, StringKeyExtractor,
};

#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8] = b"percpu_arena:percpu,oversize_threshold:0,\
muzzy_decay_ms:5000,dirty_decay_ms:10000,\
lg_extent_max_active_fit:8,background_thread:true\0";

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// Global singleton dispatcher. Initialized once via [`init`].
pub static DISPATCHER: OnceLock<Dispatcher> = OnceLock::new();

static NUM_WORKERS: OnceLock<usize> = OnceLock::new();

/// Returns the number of worker threads. Panics if [`init`] has not been called.
#[inline(always)]
pub fn num_workers() -> usize {
    *NUM_WORKERS.get().expect("num_workers called before init()")
}

/// Returns the global [`Dispatcher`]. Panics if [`init`] has not been called.
pub fn dispatcher() -> &'static Dispatcher {
    DISPATCHER
        .get()
        .expect("Dispatcher has not been initialized")
}

/// Initialize the global [`Dispatcher`] with `num_workers` worker threads.
///
/// Panics if `num_workers` exceeds the number of CPU cores.
/// Must be called before any queries. Safe to call multiple times (only the first
/// call has effect).
pub fn init(num_workers: usize) {
    let core_count = core_affinity::get_core_ids().unwrap().len();
    assert!(
        num_workers <= core_count,
        "num_workers ({num_workers}) exceeds core count ({core_count})"
    );
    NUM_WORKERS.get_or_init(|| num_workers);
    DISPATCHER.get_or_init(Dispatcher::new);
}

/// Global coordinator that owns the worker threads and distributes work to them.
///
/// Created once via [`init`] and accessed through [`dispatcher()`]. The `Dispatcher`
/// spawns one `Worker` per CPU core, each pinned to its core and connected via a
/// channel for receiving [`DataFlowBuilder`]s.
///
/// The dispatcher does not actively schedule — it simply provides [`push_data_flow`](Self::push_data_flow)
/// to send one `DataFlowBuilder` to each worker. The workers themselves drive execution
/// in their own event loops.
pub struct Dispatcher {
    worker_senders: Vec<StdSender<DataFlowBuilder>>,
    handles: Vec<JoinHandle<()>>,
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Dispatcher {
    pub fn new() -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        info!("Setting up io...");

        let cores: Vec<_> = cores.into_iter().take(num_workers()).collect();
        let mut threads = vec![];
        info!("Starting workers...");
        let barrier = Arc::new(Barrier::new(cores.len() + 1));
        let mut senders = vec![];
        for (i, core) in cores.into_iter().enumerate() {
            let (tx, rx) = channel();
            senders.push(tx);
            threads.push(Worker::create(i, core, rx, barrier.clone()));
        }
        barrier.wait();
        info!("All workers have begun...");

        Dispatcher {
            worker_senders: senders,
            handles: threads,
        }
    }

    /// Number of active worker threads (one per core).
    pub fn workers(&self) -> usize {
        self.handles.len()
    }

    /// Send each worker their pre-built DataFlow bundle.
    /// Each worker receives exactly one builder with their specific resources.
    pub fn push_data_flow(&self, builders: impl IntoIterator<Item = DataFlowBuilder>) {
        for (sender, builder) in self.worker_senders.iter().zip(builders) {
            sender.send(builder).unwrap();
        }
    }
}

pub type Identifier = usize;
