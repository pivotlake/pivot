//! Driving the spec crate's async writers and a catalog client's calls from
//! blocking code.

use std::future::Future;
use std::sync::LazyLock;

/// Drive `future` to completion from a blocking thread: on the ambient
/// multi-thread runtime when there is one, else on a runtime of this crate's
/// own. A single-thread ambient runtime (a current-thread test) cannot be
/// blocked on from inside itself, so it is declined in favour of the crate's.
pub fn block_on<F: Future>(future: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        _ => OWN_RUNTIME.block_on(future),
    }
}

/// The runtime used when the process has no multi-thread runtime of its own.
/// A `static` so it is never dropped (dropping an owned runtime inside an
/// async context panics). Two workers: the futures it drives are few and
/// sequential.
static OWN_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the Iceberg runtime")
});
