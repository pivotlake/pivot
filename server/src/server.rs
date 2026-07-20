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
use catalog::ParquetCatalog;
use compact::Compaction;
use dispatch::{DataFlowDispatcher, Dispatch, Shutdown};
use pgwire::tokio::process_socket;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::task::{JoinError, JoinSet};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

/// Default cadence of the background catalog refresh (the
/// `catalog_refresh_interval` argument of [`Server::new`]).
pub const DEFAULT_CATALOG_REFRESH: Duration = Duration::from_secs(30);

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
    /// Cloned dispatcher, kept for the query handler (compiling plans) and the
    /// bundled web console's in-process queries.
    dispatcher: DataFlowDispatcher,
    /// The concrete catalog, kept so the bundled compacter can read and swap the
    /// Parquet files a table holds. The query handlers hold it as a
    /// `dyn Catalog`.
    catalog: Arc<ParquetCatalog>,
    /// Target size for the bundled compacter (`0` = don't run one).
    compact_bytes: u64,
    /// The compacter's count trigger for sub-target (low-traffic) partitions.
    compact_min_files: usize,
    /// Address for the optional bundled web dashboard (the `http` module).
    /// `None` (the default) leaves it off; set it with
    /// [`with_http_bind`](Self::with_http_bind).
    http_bind: Option<SocketAddr>,
    /// How often the background sweep refreshes the in-memory catalog from the
    /// store (Delta versions + new footers). Queries snapshot the in-memory
    /// set, so this bounds staleness for externally committed data.
    catalog_refresh_interval: Duration,
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
        catalog: Arc<ParquetCatalog>,
        compact_bytes: u64,
        compact_min_files: usize,
        catalog_refresh_interval: Duration,
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
        Self {
            bind,
            shutdown,
            worker_watchers: watchers,
            handlers: Arc::new(PivotHandlers::new(catalog.clone(), dispatcher.clone())),
            dispatcher,
            catalog,
            compact_bytes,
            compact_min_files,
            http_bind: None,
            catalog_refresh_interval,
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

        // Start the bundled compacter. It re-encodes on the dispatch workers, so
        // it must be stopped before the workers stop. It reads and swaps a
        // table's files through the catalog.
        let compaction = Compaction::start(
            self.catalog.clone(),
            self.compact_bytes,
            self.compact_min_files,
        );

        // Optionally serve the bundled web dashboard. It reads the engine's live
        // state directly - the catalog, the compaction stats handle, and the
        // dispatcher (for the in-process query console). Read-only except
        // `/api/query`, so on shutdown we just abort the task.
        let http_task = self.http_bind.map(|bind| {
            let state = crate::http::IntrospectState::new(
                self.catalog.clone(),
                self.dispatcher.clone(),
                compaction.stats(),
            );
            tokio::spawn(async move {
                if let Err(e) = crate::http::serve(bind, state, std::future::pending()).await {
                    error!(?e, "web dashboard server error");
                }
            })
        });

        // Keep the in-memory catalog current: on an interval, reload every
        // table to its latest committed Delta version and fetch any new
        // files' footers. Queries bind against a snapshot of the in-memory set
        // and never read the store themselves, so this sweep is what makes
        // externally committed data (another process, a bucket writer)
        // visible. The catalog was fully loaded at open, so the
        // immediate first tick is skipped.
        let refresh_task = {
            let catalog = self.catalog.clone();
            let mut tick = tokio::time::interval(self.catalog_refresh_interval);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            tokio::spawn(async move {
                // A tokio interval fires its first tick immediately on
                // creation; consume it so the first refresh runs one full
                // interval from now (the catalog was just loaded at open).
                tick.tick().await;
                loop {
                    tick.tick().await;
                    let catalog = catalog.clone();
                    // The refresh drives footer-fetch dataflows and blocking
                    // store reads, so it runs off the reactor.
                    match tokio::task::spawn_blocking(move || catalog.refresh_catalog()).await {
                        Ok(Ok(_)) => {}
                        Ok(Err(e)) => warn!(error = %e, "catalog refresh failed"),
                        Err(e) => warn!(error = %e, "catalog refresh panicked"),
                    }
                }
            })
        };

        // `Option` so the two terminal arms below can each take ownership
        // without the borrow checker tripping over the loop.
        let mut compaction = Some(compaction);

        loop {
            tokio::select! {
                // Prefer a clean shutdown over a worker exit if both fire on
                // the same poll: shutdown should look clean.
                biased;
                _ = &mut shutdown => {
                    info!("shutdown signalled, stopping compaction then workers");
                    refresh_task.abort();
                    if let Some(task) = &http_task {
                        task.abort();
                    }
                    // Stop compaction first -- an in-flight merge encodes on the
                    // workers, which must still be alive.
                    if let Some(compaction) = compaction.take() {
                        compaction.shutdown().await;
                    }
                    self.shutdown.shutdown();
                    // Wait for every worker to observe the flag and exit. No
                    // need to inspect results — we initiated the shutdown.
                    while self.worker_watchers.join_next().await.is_some() {}
                    return Ok(());
                }
                Some(joined) = self.worker_watchers.join_next() => {
                    // A worker died: waiting on an in-flight merge would hang on
                    // the dead worker, so stop compaction without waiting.
                    refresh_task.abort();
                    if let Some(task) = &http_task {
                        task.abort();
                    }
                    if let Some(compaction) = compaction.take() {
                        compaction.abort();
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
    use tokio::sync::oneshot;

    fn bind() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    /// An empty in-memory catalog: `Server::serve` only consults it on
    /// incoming queries, which these tests don't drive.
    fn catalog(dispatch: &Dispatch) -> Arc<ParquetCatalog> {
        Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()))
    }

    #[tokio::test]
    async fn shutdown_signal_returns_ok() {
        let (tx, rx) = oneshot::channel::<()>();
        let dispatch = Dispatch::spin_up(1, 32, None);
        let catalog = catalog(&dispatch);
        let server = Server::new(
            bind(),
            dispatch,
            catalog,
            0,
            compact::DEFAULT_MIN_FILES_TO_MERGE,
            DEFAULT_CATALOG_REFRESH,
        );

        let join = tokio::spawn(server.serve(Box::pin(async move {
            let _ = rx.await;
        })));
        tx.send(()).unwrap();
        let result = join.await.unwrap();

        assert!(matches!(result, Ok(())));
    }
}
