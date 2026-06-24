//! Support for the server's per-query `perf` profiling. Two pieces, both used
//! only with the `perf` feature:
//!
//! - A registry of worker-thread OS tids, so an out-of-crate profiler can scope
//!   `perf record -t` to the worker pool (and nothing else in the process).
//! - A global "profiled dataflows live" counter. While it is non-zero the worker
//!   event loop runs **only** profiled dataflows and pauses everything else
//!   (other queries, ingest encode) sharing the pool, so a capture is just the
//!   one dataflow under study rather than whatever else happened to be on-core.
//!
//! Pausing is the point: a profile is taken deliberately in staging and is
//! short, and it is the only way a single process-wide `perf record` can be sure
//! it sampled just the dataflow we asked for.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

/// Number of profiled dataflows currently live, maintained by
/// [`DataFlow`](crate::DataFlow) construction/drop. While `> 0` the worker runs
/// only profiled dataflows (see [`profiling_active`]).
pub(crate) static PROFILED_FLOWS: AtomicUsize = AtomicUsize::new(0);

/// True while any profiled dataflow is live: the worker should skip every
/// non-profiled dataflow so the profile isn't polluted by other work.
pub(crate) fn profiling_active() -> bool {
    PROFILED_FLOWS.load(Ordering::Relaxed) > 0
}

static WORKER_TIDS: LazyLock<Mutex<HashMap<usize, i32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record the calling worker thread's OS tid under its worker index. Called once
/// per worker at startup. `gettid` is Linux-only (where `perf` runs); elsewhere
/// the feature still compiles but the tid is a placeholder.
pub(crate) fn register_worker_tid(worker_idx: usize) {
    #[cfg(target_os = "linux")]
    let tid = nix::unistd::gettid().as_raw();
    #[cfg(not(target_os = "linux"))]
    let tid = 0i32;
    WORKER_TIDS.lock().unwrap().insert(worker_idx, tid);
}

/// OS tids of all started worker threads. An out-of-crate profiler scopes
/// `perf record -t <tids>` to the worker pool with these.
pub fn worker_tids() -> Vec<i32> {
    WORKER_TIDS.lock().unwrap().values().copied().collect()
}
