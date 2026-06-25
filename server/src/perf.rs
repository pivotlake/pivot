//! Per-query Linux `perf record` profiling, gated behind the `perf` feature.
//!
//! When a session runs `SET perf = 1` and the server was started with
//! `PIVOT_PERF_DIR` set, each query spawns its own `perf record` scoped to the
//! dispatch worker threads (`-t <tids>`). The query's dataflows are marked
//! profiled, which makes the workers run *only* that dataflow for its duration
//! (other queries and ingest encode on the pool pause; see
//! [`dispatch::DataFlowDispatcher::with_profiling`]). So the recording captures
//! just the dataflow under study: nothing else runs on the worker threads, and
//! non-worker threads (ingest receive/decode) aren't recorded at all. One fresh
//! report per query lands in the configured dir.
//!
//! Without the `perf` feature this module is not compiled, `libc` is not pulled
//! in, and `SET perf = 1` is an accepted no-op, so a normal build pays nothing.
//!
//! Config (read once from the environment):
//! - `PIVOT_PERF_DIR` (optional): directory the reports are written to; defaults
//!   to [`DEFAULT_PERF_DIR`] (`/tmp`).
//! - `PIVOT_PERF_ARGS` (optional): the `perf record` sampling arguments, default
//!   [`DEFAULT_PERF_ARGS`]. The server always adds `-t <worker tids>` and `-o <file>`.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime};

use std::os::unix::process::ExitStatusExt;
use tracing::{info, warn};

/// `perf record` sampling arguments used when `PIVOT_PERF_ARGS` is unset. Frame
/// pointer call graphs, since the server is built with `force-frame-pointers=yes`.
const DEFAULT_PERF_ARGS: &[&str] = &["--call-graph", "fp"];

/// Directory reports are written to when `PIVOT_PERF_DIR` is unset.
const DEFAULT_PERF_DIR: &str = "/tmp";

/// Upper bound on waiting for perf to attach (its output file to appear) before
/// running the query, so a short dataflow's start isn't missed.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(1);

/// Server-wide perf configuration, read once from the environment.
struct Config {
    dir: PathBuf,
    args: Vec<String>,
}

static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    let dir = std::env::var_os("PIVOT_PERF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PERF_DIR));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!(dir = %dir.display(), error = %e, "could not create perf report dir");
    }
    let args = match std::env::var("PIVOT_PERF_ARGS") {
        Ok(s) if !s.trim().is_empty() => s.split_whitespace().map(String::from).collect(),
        _ => DEFAULT_PERF_ARGS.iter().map(|s| s.to_string()).collect(),
    };
    Config { dir, args }
});

/// Per-report counter so reports launched within the same millisecond (or from
/// concurrent sessions) never collide on a filename.
static SEQ: AtomicUsize = AtomicUsize::new(0);

/// A running `perf record` scoped to the worker threads for one query. Dropping
/// it stops perf with `SIGINT` (so the report is flushed) and reaps the child.
/// Held across the query's execution and dropped when it ends, including on
/// cancellation or error.
pub struct PerfGuard {
    child: Child,
    report: PathBuf,
}

/// Start a `perf record -t <worker tids>` for the query about to run. Returns
/// `None` (a logged no-op) when profiling is not configured or perf could not be
/// started, so the query always still runs. The caller also marks the query's
/// dataflows profiled, which makes the workers run only that dataflow while this
/// guard is alive.
pub fn start(sql: &str) -> Option<PerfGuard> {
    let config = &*CONFIG;

    let tids = dispatch::worker_tids();
    if tids.is_empty() {
        warn!("no dispatch worker threads to profile; running query unprofiled");
        return None;
    }
    let tid_list = tids
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let report = config
        .dir
        .join(format!("pivotdb-{}-{seq}.perf.data", millis()));
    // Clear any leftover at this path (possible after a restart resets SEQ) so
    // `wait_for_attach`'s file-exists check can't false-positive on a stale file.
    let _ = std::fs::remove_file(&report);

    let child = Command::new("perf")
        .arg("record")
        .args(["-t", &tid_list])
        .args(&config.args)
        .args(["-o", &report.to_string_lossy()])
        .spawn();
    let child = match child {
        Ok(child) => child,
        Err(e) => {
            warn!(error = %e, "could not spawn `perf record`; running query unprofiled");
            return None;
        }
    };

    // Own the child in the guard immediately, before the fallible wait/log below,
    // so any panic still SIGINTs and reaps perf rather than leaking it.
    let guard = PerfGuard { child, report };
    wait_for_attach(&guard.report);
    info!(report = %guard.report.display(), %sql, "profiling query with perf");
    Some(guard)
}

/// Block (briefly) until `perf` has created its output file, our proxy for "it
/// has attached and started sampling". Bounded by [`ATTACH_TIMEOUT`].
fn wait_for_attach(report: &Path) {
    let started = Instant::now();
    while started.elapsed() < ATTACH_TIMEOUT {
        if report.exists() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn millis() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

impl Drop for PerfGuard {
    fn drop(&mut self) {
        // SIGINT, not `Child::kill` (SIGKILL): perf traps it to flush the report.
        // Safety: a standard signal to our own child's pid, not yet reaped.
        unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGINT) };
        match self.child.wait() {
            Ok(s) if s.success() || s.signal() == Some(libc::SIGINT) => {
                info!(report = %self.report.display(), "perf report written")
            }
            Ok(s) => warn!(report = %self.report.display(), %s, "perf record exited abnormally"),
            Err(e) => warn!(error = %e, "perf record could not be reaped"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CONFIG;
    use std::path::Path;

    #[test]
    fn report_dir_defaults_to_tmp() {
        // PIVOT_PERF_DIR is unset in the test environment, so it falls back.
        assert_eq!(CONFIG.dir, Path::new("/tmp"));
    }
}
