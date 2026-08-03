//! Process resource limits raised at startup.

use nix::sys::resource::{Resource, getrlimit, setrlimit};
use tracing::{info, warn};

/// Raise this process's soft open-file limit to its hard limit.
///
/// The server holds a descriptor per cached disk-cache object plus one io_uring
/// per worker, so a large `--disk-cache-max-objects` exhausts descriptors before
/// any memory or disk budget binds. Shells hand out a low soft default (1024 on
/// Ubuntu) while permitting far more, and raising your own soft limit needs no
/// privileges, so take the headroom at boot rather than expect `ulimit -n`.
///
/// Call before opening anything. Failure is logged, not fatal: the limit already
/// in force is often ample.
pub fn raise_open_file_limit() {
    let (soft, hard) = match getrlimit(Resource::RLIMIT_NOFILE) {
        Ok(limits) => limits,
        Err(errno) => {
            warn!(%errno, "cannot read the open-file limit; leaving it unchanged");
            return;
        }
    };

    if soft >= hard {
        info!(soft, hard, "open-file limit already at its hard ceiling");
        return;
    }

    match setrlimit(Resource::RLIMIT_NOFILE, hard, hard) {
        Ok(()) => info!(previous = soft, current = hard, "raised open-file limit"),
        Err(errno) => warn!(%errno, soft, hard, "cannot raise the open-file limit"),
    }
}
