//! Top-level server: the [`Server`] type, the public [`enum@Error`], and the
//! accept loop that ties them together.
//!
//! [`Server::serve`] does two things in one `tokio::select!`:
//!
//! 1. **Accept connections.** Each new TCP socket is handed to
//!    [`pgwire::tokio::process_socket`], which drives the full wire-protocol
//!    state machine (startup handshake, query loop, message framing) and
//!    calls back into the [`PivotHandlers`] bundle for each phase.
//! 2. **Watch the dispatch worker pool.** Every worker `JoinHandle` returned by
//!    [`Dispatch::into_parts`] is moved into a [`JoinSet`] of blocking tasks.
//!    Workers are designed to run forever, so a join ordinarily only completes
//!    once the shared exit flag (also returned from `into_parts`) flips. An
//!    earlier completion (panic, unexpected return) is treated as fatal and
//!    aborts the accept loop with an [`enum@Error`].
//!
//! On clean shutdown the accept loop sets the exit flag itself and drains the
//! worker watchers, so when [`Server::serve`] returns `Ok(())` every worker
//! thread has already terminated. Each datastore's own refresh and compaction
//! tasks watch that same exit flag and self-exit, so the server no longer
//! orchestrates them.

use crate::server::query_handler::PivotHandlers;
use catalog::PivotCatalog;
use catalog::metastore::Metastore;
#[cfg(test)]
use dispatch::DataFlowDispatcher;
use dispatch::{Dispatch, Shutdown};
use pgwire::tokio::{TlsAcceptor, process_socket};
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::task::{JoinError, JoinSet};
use tracing::{error, info, warn};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] io::Error),
    /// Boxed: the YAML parse error it carries is by far the largest thing in
    /// this enum, and every `Result<_, Error>` in the crate would pay for it on
    /// the success path too.
    #[error(transparent)]
    Config(#[from] Box<crate::server::config::Error>),
    #[error("invalid metastore configuration for `{path}`: {source}")]
    Metastore {
        path: PathBuf,
        /// Boxed for the same reason as [`Config`](Self::Config).
        #[source]
        source: Box<metastore_disk::Error>,
    },
    #[error("failed to open the datastores configured in `{path}`: {source}")]
    OpenDatastores {
        path: PathBuf,
        #[source]
        source: catalog::metastore::Error,
    },
    #[error("invalid metastore configuration: {0}")]
    InvalidCatalog(#[from] catalog::Error),
    #[error(transparent)]
    Tls(#[from] crate::server::tls::Error),
    #[error(transparent)]
    DiskCache(#[from] crate::server::config::DiskCacheError),
    #[error(transparent)]
    LogFilter(#[from] crate::logging::FilterError),
    #[error("worker watcher panic: {0}")]
    WorkerWatcherPanic(JoinError),
    #[error("dispatch worker failed: {0}")]
    DispatchWorkerFailed(String),
    #[error("dispatch worker died unexpectedly")]
    DispatchWorkerDied,
    #[error("failed to prepare the query planners: {0}")]
    PreparePlanners(crate::execution::Error),
    #[error(transparent)]
    MachineTooSmall(#[from] crate::memory::MachineTooSmall),
    #[error(
        "the buffer pool needs {} GiB but the machine only has {} GiB available right now; \
         every pool slot is faulted in at startup, so booting would be killed by the OOM \
         killer part way through. Free memory on the machine, or lower the budget with the \
         `memory` config key",
        .requested_bytes / GIB,
        .available_bytes / GIB,
    )]
    InsufficientMemory {
        requested_bytes: usize,
        available_bytes: usize,
    },
}

/// Bytes in a gibibyte, the unit memory budgets are reported in.
const GIB: usize = crate::memory::GIB as usize;

type Result<T, E = Error> = std::result::Result<T, E>;

/// Blocking-pool threads given a query planner before the server accepts
/// connections: enough for a few statements to plan at once. A thread beyond
/// these builds its own planner the first time it plans a statement.
const PREPARED_PLANNER_THREADS: usize = 4;

/// A pivotdb server instance.
///
/// Holds the TCP bind address, a [`JoinSet`] watching every dispatch worker
/// thread, and the configuration the pgwire handler bundle (query / startup /
/// cancel handlers) is built from when the accept loop starts. Construct one
/// with [`Server::new`] and drive it with [`Server::serve`].
pub struct Server {
    bind: SocketAddr,
    worker_watchers: JoinSet<std::thread::Result<()>>,
    shutdown: Shutdown,
    /// Statement executor every connection runs its queries through.
    query_executor: Arc<crate::execution::Executor>,
    /// Every datastore, presented as one composite catalog. Query binding
    /// starts here. [`serve`](Self::serve) starts each datastore's
    /// background maintenance when it begins serving and aborts it on shutdown,
    /// before the worker pool is torn down.
    catalog: Arc<PivotCatalog>,
    /// The metastore consulted for every connection's current user and
    /// authentication method.
    metastore: Arc<dyn Metastore>,
    /// Certificate to offer a connection that asks to encrypt itself. `None`
    /// (the default) turns those requests down and leaves every session
    /// plaintext; set it with [`with_tls`](Self::with_tls).
    tls: Option<TlsAcceptor>,
}

impl Server {
    /// Build a server from the bind address, an already-spun-up [`Dispatch`],
    /// the catalog used to resolve table names, and its metastore. The server
    /// keeps the metastore so every login can read current user credentials. It
    /// takes the `Dispatch` apart with [`Dispatch::into_parts`]: it clones the
    /// dispatcher for the query handler and adopts every worker `JoinHandle`
    /// for the shutdown / fault-detection path. Pass port `0` in `bind` to let
    /// the OS pick a free port (useful in tests).
    pub fn new(
        bind: SocketAddr,
        dispatch: Dispatch,
        catalog: Arc<PivotCatalog>,
        metastore: Arc<dyn Metastore>,
    ) -> Self {
        // Clone the dispatcher out *before* `into_parts` drops it; the query
        // handler needs it to compile every plan.
        let dispatcher = dispatch.dispatcher().clone();
        let (handles, shutdown) = dispatch.into_parts();
        let mut watchers = JoinSet::new();
        for handle in handles {
            watchers.spawn_blocking(move || handle.join());
        }
        let query_executor = Arc::new(crate::execution::Executor::new(
            catalog.clone(),
            dispatcher.clone(),
        ));
        Self {
            bind,
            shutdown,
            worker_watchers: watchers,
            query_executor,
            catalog,
            metastore,
            tls: None,
        }
    }

    /// Offer `acceptor`'s certificate to connections that ask to encrypt
    /// themselves, instead of turning them down. Build one with
    /// [`tls::build_acceptor`](crate::server::tls::build_acceptor). Off by default, and
    /// on or off it leaves a client that asks for plaintext with a plaintext
    /// session.
    pub fn with_tls(mut self, acceptor: TlsAcceptor) -> Self {
        self.tls = Some(acceptor);
        self
    }

    /// Run the accept loop until `shutdown` resolves or a worker exits.
    ///
    /// On `shutdown`, we flip the shared exit flag (the `Arc<AtomicBool>` we
    /// took from `Dispatch::into_parts`) and then await the worker-watcher
    /// tasks; they unblock once every worker observes the flag. The wait is
    /// short (a few worker-loop iterations) and bounded by the longest-running
    /// in-flight query.
    pub async fn serve(
        mut self,
        mut shutdown: impl std::future::Future<Output = ()> + Unpin,
    ) -> Result<()> {
        self.query_executor
            .prepare_planners(PREPARED_PLANNER_THREADS)
            .await
            .map_err(Error::PreparePlanners)?;
        let listener = TcpListener::bind(self.bind).await?;
        let tls = self.tls.clone();
        info!(addr = %self.bind, ssl = tls.is_some(), "listening for psql connections");

        let handlers = Arc::new(PivotHandlers::new(
            self.query_executor.clone(),
            self.metastore.clone(),
        ));

        // Start each datastore's background maintenance.
        self.catalog.start();

        loop {
            tokio::select! {
                // Prefer a clean shutdown over a worker exit if both fire on
                // the same poll: shutdown should look clean.
                biased;
                _ = &mut shutdown => {
                    info!("shutdown signalled, stopping workers");
                    // Stop each datastore's maintenance *before* tearing down the
                    // pool, so no refresh/compaction sweep races the workers'
                    // exit.
                    self.catalog.abort();
                    // Flip the shared exit flag: every dispatch worker watches it
                    // and exits.
                    self.shutdown.shutdown();
                    // Wait for every worker to observe the flag and exit. No
                    // need to inspect results; we initiated the shutdown.
                    while self.worker_watchers.join_next().await.is_some() {}
                    return Ok(());
                }
                Some(joined) = self.worker_watchers.join_next() => {
                    self.catalog.abort();
                    self.shutdown.shutdown();
                    return Err(match joined {
                        Ok(Ok(())) => {
                            error!("dispatch worker returned unexpectedly");
                            Error::DispatchWorkerDied
                        }
                        Ok(Err(payload)) => {
                            let msg = format_panic_payload(&payload);
                            error!(%msg, "dispatch worker panicked");
                            Error::DispatchWorkerFailed(msg)
                        }
                        Err(e) => {
                            return Err(Error::WorkerWatcherPanic(e));
                        }
                    })
                }
                accept = listener.accept() => {
                    match accept {
                        Ok((socket, peer)) => {
                            let handlers = handlers.clone();
                            // `process_socket` reads the client's `SSLRequest`
                            // and upgrades the socket itself; handing it the
                            // acceptor is the whole of enabling SSL.
                            let tls = tls.clone();
                            tokio::spawn(async move {
                                info!(?peer, "connection accepted");
                                if let Err(e) = process_socket(socket, tls, handlers).await {
                                    warn!(?e, ?peer, "connection error");
                                } else {
                                    info!(?peer, "connection closed");
                                }
                            });
                        }
                        Err(e) => error!(?e, "accept failed"),
                    }
                }
            }
        }
    }
}

/// Recover a printable message from a [`JoinHandle::join`](std::thread::JoinHandle::join) `Err` payload
/// (`Box<dyn Any + Send>`). Best-effort: non-string payloads degrade to a
/// placeholder.
fn format_panic_payload(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::{DEFAULT_DATASTORE_NAME, Datastore};
    use datastore_pivot::PivotDatastore;
    use std::collections::HashMap;
    use tokio::sync::oneshot;

    fn bind() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    /// An empty datastore on a test-owned directory: `Server::serve` only
    /// consults it on incoming queries, which these tests don't drive.
    fn catalog(dispatch: &Dispatch) -> (tempfile::TempDir, Arc<PivotCatalog>) {
        let directory = tempfile::tempdir().unwrap();
        let datastore: Arc<dyn Datastore> =
            PivotDatastore::open(&directory.path().to_string_lossy(), dispatch.dispatcher())
                .unwrap();
        let catalog = PivotCatalog::new(
            HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
            DEFAULT_DATASTORE_NAME.to_string(),
            metastore(),
        )
        .unwrap();
        (directory, Arc::new(catalog))
    }

    fn metastore() -> Arc<dyn Metastore> {
        #[derive(Debug)]
        struct TestMetastore;

        impl Metastore for TestMetastore {
            fn open_datastores(
                &self,
                _dispatcher: &DataFlowDispatcher,
            ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
                Ok(HashMap::new())
            }

            fn default_datastore_name(&self) -> &str {
                DEFAULT_DATASTORE_NAME
            }

            fn user_auth(&self, username: &str) -> Option<catalog::metastore::UserAuth> {
                (username == catalog::metastore::DEFAULT_USER_NAME)
                    .then_some(catalog::metastore::UserAuth::Trust)
            }
        }

        Arc::new(TestMetastore)
    }

    #[tokio::test]
    async fn shutdown_signal_returns_ok() {
        let (tx, rx) = oneshot::channel::<()>();
        let dispatch = Dispatch::spin_up(1, 32, None);
        let (_directory, catalog) = catalog(&dispatch);
        let server = Server::new(bind(), dispatch, catalog, metastore());

        let join = tokio::spawn(server.serve(Box::pin(async move {
            let _ = rx.await;
        })));
        tx.send(()).unwrap();
        let result = join.await.unwrap();

        assert!(matches!(result, Ok(())));
    }
}
