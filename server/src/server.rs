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
//! thread has already terminated.

use crate::query_handler::PivotHandlers;
use dispatch::{DataFlowDispatcher, Dispatch, Shutdown};
use ingest::{IngestConfig, Ingestor};
use pgwire::tokio::process_socket;
use planner::catalog::Catalog;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::task::{JoinError, JoinSet};
use tracing::{error, info, warn};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] io::Error),
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
/// thread, and the shared pgwire handlers bundle (query / startup / cancel
/// handlers) used by every connection. Construct one with [`Server::new`] and
/// drive it with [`Server::serve`].
pub struct Server {
    bind: SocketAddr,
    worker_watchers: JoinSet<std::thread::Result<()>>,
    shutdown: Shutdown,
    handlers: Arc<PivotHandlers>,
    /// Cloned dispatcher handed to the ingest sources so their Parquet encodes
    /// run on the same worker pool as queries.
    dispatcher: DataFlowDispatcher,
    /// Ingest sources to start when [`serve`](Self::serve) runs.
    ingests: Vec<IngestConfig>,
}

impl Server {
    /// Build a server from the bind address, an already-spun-up [`Dispatch`],
    /// and the catalog used to resolve table names. The server takes the
    /// `Dispatch` apart with [`Dispatch::into_parts`]: it clones the
    /// dispatcher for the query handler and adopts every worker `JoinHandle`
    /// for the shutdown / fault-detection path. Pass port `0` in `bind` to
    /// let the OS pick a free port (useful in tests).
    pub fn new(
        bind: SocketAddr,
        dispatch: Dispatch,
        catalog: Arc<dyn Catalog>,
        ingests: Vec<IngestConfig>,
    ) -> Self {
        // Clone the dispatcher out *before* `into_parts` drops it; the query
        // handler needs it to compile every plan, and the ingest sources need
        // it to encode Parquet on the worker pool.
        let dispatcher = dispatch.dispatcher().clone();
        let (handles, shutdown) = dispatch.into_parts();
        let mut watchers = JoinSet::new();
        for handle in handles {
            watchers.spawn_blocking(move || handle.join());
        }
        Self {
            bind,
            shutdown,
            worker_watchers: watchers,
            handlers: Arc::new(PivotHandlers::new(catalog, dispatcher.clone())),
            dispatcher,
            ingests,
        }
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

        // Start the configured ingest sources (OTLP receivers, etc.). They
        // encode Parquet on the dispatch workers, so they must be drained
        // before the workers stop. `Option` so the two terminal arms below can
        // each take ownership without the borrow checker tripping over the loop.
        let mut ingestor =
            Some(Ingestor::start(std::mem::take(&mut self.ingests), self.dispatcher.clone())?);

        loop {
            tokio::select! {
                // Prefer a clean shutdown over a worker exit if both fire on
                // the same poll: shutdown should look clean.
                biased;
                _ = &mut shutdown => {
                    info!("shutdown signalled, draining ingest then workers");
                    // Drain ingest first — the final flush encodes on the
                    // workers, which must still be alive.
                    if let Some(ingestor) = ingestor.take() {
                        ingestor.shutdown().await;
                    }
                    self.shutdown.shutdown();
                    // Wait for every worker to observe the flag and exit. No
                    // need to inspect results — we initiated the shutdown.
                    while self.worker_watchers.join_next().await.is_some() {}
                    return Ok(());
                }
                Some(joined) = self.worker_watchers.join_next() => {
                    // A worker died: flushing would hang on a dead worker, so
                    // stop the receivers without a final flush.
                    if let Some(ingestor) = ingestor.take() {
                        ingestor.abort();
                    }
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
                            let handlers = self.handlers.clone();
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
    use planner::catalog::{
        Catalog as PlannerCatalog, CreateTableRequest, Result as CatalogResult, Table,
    };
    use tokio::sync::oneshot;

    /// Minimal `Catalog` impl: `Server::serve` never touches the catalog
    /// (it's only consulted on incoming queries, which these tests don't drive),
    /// so every method is unreachable. Avoids pulling `ParquetCatalog` in.
    #[derive(Debug)]
    struct StubCatalog;

    impl PlannerCatalog for StubCatalog {
        fn table(&self, _name: &str) -> Option<Box<dyn Table>> {
            None
        }
        fn create_table(&self, _request: CreateTableRequest) -> CatalogResult<()> {
            Ok(())
        }
    }

    fn bind() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    fn catalog() -> Arc<dyn PlannerCatalog> {
        Arc::new(StubCatalog)
    }

    #[tokio::test]
    async fn shutdown_signal_returns_ok() {
        let (tx, rx) = oneshot::channel::<()>();
        let server = Server::new(bind(), Dispatch::spin_up(1, 32), catalog(), vec![]);

        let join = tokio::spawn(server.serve(Box::pin(async move {
            let _ = rx.await;
        })));
        tx.send(()).unwrap();
        let result = join.await.unwrap();

        assert!(matches!(result, Ok(())));
    }
}
