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

use crate::query_handler::{PivotHandlers, PlanCache};
use catalog::PivotCatalog;
use dispatch::{DataFlowDispatcher, Dispatch, Shutdown};
use metastore::Metastore;
use pgwire::tokio::process_socket;
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
    Config(#[from] Box<crate::config::Error>),
    #[error("invalid `metastore` section of `{path}`: {source}")]
    Metastore {
        path: PathBuf,
        /// Boxed for the same reason as [`Config`](Self::Config).
        #[source]
        source: Box<metastore_yaml::Error>,
    },
    #[error("failed to open the datastores configured in `{path}`: {source}")]
    OpenDatastores {
        path: PathBuf,
        #[source]
        source: metastore::Error,
    },
    #[error("invalid metastore configuration: {0}")]
    InvalidCatalog(#[from] catalog::Error),
    #[error("worker watcher panic: {0}")]
    WorkerWatcherPanic(JoinError),
    #[error("dispatch worker failed: {0}")]
    DispatchWorkerFailed(String),
    #[error("dispatch worker died unexpectedly")]
    DispatchWorkerDied,
}

type Result<T, E = Error> = std::result::Result<T, E>;

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
    /// Planned read queries shared by the PostgreSQL and in-process HTTP paths.
    plan_cache: Arc<PlanCache>,
    /// Cloned dispatcher, kept for the query handler (compiling plans) and the
    /// bundled web console's in-process queries.
    dispatcher: DataFlowDispatcher,
    /// Every datastore, presented as one composite catalog. Query binding and
    /// dashboard access start here. [`serve`](Self::serve) starts each datastore's
    /// background maintenance when it begins serving and aborts it on shutdown,
    /// before the worker pool is torn down.
    catalog: Arc<PivotCatalog>,
    /// Address for the optional bundled web dashboard (the `http` module).
    /// `None` (the default) leaves it off; set it with
    /// [`with_http_bind`](Self::with_http_bind).
    http_bind: Option<SocketAddr>,
    /// The metastore consulted for every connection's current user and
    /// authentication method.
    metastore: Arc<dyn Metastore>,
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
        // handler needs it to compile every plan, and the web console runs
        // queries on the worker pool.
        let dispatcher = dispatch.dispatcher().clone();
        let (handles, shutdown) = dispatch.into_parts();
        let mut watchers = JoinSet::new();
        for handle in handles {
            watchers.spawn_blocking(move || handle.join());
        }
        let plan_cache = Arc::new(PlanCache::default());
        Self {
            bind,
            shutdown,
            worker_watchers: watchers,
            plan_cache,
            dispatcher,
            catalog,
            http_bind: None,
            metastore,
        }
    }

    /// Also serve the bundled web dashboard (served by the `http` module) on `addr`
    /// while the server runs. Off by default.
    pub fn with_http_bind(mut self, addr: SocketAddr) -> Self {
        self.http_bind = Some(addr);
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
        let listener = TcpListener::bind(self.bind).await?;
        info!(addr = %self.bind, "listening for psql connections");

        let handlers = Arc::new(PivotHandlers::new(
            self.catalog.clone(),
            self.dispatcher.clone(),
            self.plan_cache.clone(),
            self.metastore.clone(),
        ));

        // Start each datastore's background maintenance.
        self.catalog.start();

        // Optionally serve the bundled web dashboard. It reads the engine's live
        // state directly - the catalog and the dispatcher (for the in-process
        // query console). Read-only except `/api/query`, so on shutdown we just
        // abort the task.
        let http_task = self.http_bind.map(|bind| {
            // The dashboard has no authentication of its own, and `/api/query`
            // runs arbitrary SQL. PostgreSQL users do not govern this endpoint.
            warn!(
                %bind,
                "web dashboard is unauthenticated and runs SQL; users govern only the PostgreSQL endpoint"
            );
            let state = crate::http::IntrospectState::new(
                self.catalog.clone(),
                self.dispatcher.clone(),
                self.plan_cache.clone(),
            );
            tokio::spawn(async move {
                if let Err(e) = crate::http::serve(bind, state, std::future::pending()).await {
                    error!(?e, "web dashboard server error");
                }
            })
        });

        loop {
            tokio::select! {
                // Prefer a clean shutdown over a worker exit if both fire on
                // the same poll: shutdown should look clean.
                biased;
                _ = &mut shutdown => {
                    info!("shutdown signalled, stopping workers");
                    if let Some(task) = &http_task {
                        task.abort();
                    }
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
                    if let Some(task) = &http_task {
                        task.abort();
                    }
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
                            tokio::spawn(async move {
                                info!(?peer, "connection accepted");
                                if let Err(e) = process_socket(socket, None, handlers).await {
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
    use datastore_delta::DeltaDatastore;
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
            DeltaDatastore::open(&directory.path().to_string_lossy(), dispatch.dispatcher())
                .unwrap();
        let catalog = PivotCatalog::new(
            HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .unwrap();
        (directory, Arc::new(catalog))
    }

    fn metastore() -> Arc<dyn Metastore> {
        struct TestMetastore;

        impl Metastore for TestMetastore {
            fn open_datastores(
                &self,
                _dispatcher: &DataFlowDispatcher,
            ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
                Ok(HashMap::new())
            }

            fn default_datastore_name(&self) -> &str {
                DEFAULT_DATASTORE_NAME
            }

            fn user_auth(&self, username: &str) -> Option<metastore::UserAuth> {
                (username == metastore::DEFAULT_USER_NAME).then_some(metastore::UserAuth::Trust)
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
