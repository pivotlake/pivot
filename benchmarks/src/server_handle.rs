//! Boot the pivotdb server in-process for benchmarking.
//!
//! Picks a free local port, initialises a `dispatch` worker pool, and runs
//! `server::Server` on a dedicated background thread driving its own tokio
//! runtime — so the bench's blocking iteration loop on the main thread never
//! starves the accept loop. Returns a [`ServerHandle`] holding the bind port
//! plus a `oneshot::Sender` that triggers a clean shutdown when the handle is
//! dropped.

use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use catalog::ParquetCatalog;
use compact::{Compacter, DEFAULT_COMPACT_BYTES, DEFAULT_MIN_FILES_TO_MERGE};
use dispatch::{BUFFER_SIZE, Dispatch};
use server::Server;
use tokio::sync::oneshot;

fn total_memory_bytes() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}

/// Share of the machine's memory the ring takes, as a percentage — the same
/// `PIVOT_MEMORY_PCT` the real server reads, defaulting to the same 80, so a
/// benchmark run is sized like the server it stands in for.
///
/// Worth overriding on a machine that isn't the benchmark's alone: the ring is
/// prefaulted, so 80% of total is 80% whether or not something else is holding
/// memory, and the run is simply OOM-killed.
fn ring_buffers() -> usize {
    let pct: usize = dispatch::env::get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
    total_memory_bytes() * pct / 100 / BUFFER_SIZE
}

pub struct ServerHandle {
    port: u16,
    catalog: Arc<ParquetCatalog>,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Compact every table synchronously. A load of many small `INSERT`s leaves
    /// as many small files; this merges them into target-sized ones before the
    /// queries read them, the way the background compacter would over time.
    ///
    /// One sweep can leave a tail: a merge changes the file list, and a later
    /// candidate in the same sweep may be missed. So sweep until one merges
    /// nothing (`files_merged_in` is cumulative, so an unchanged total means the
    /// last sweep did no work).
    pub async fn compact(&self) {
        let compacter = Compacter::new(
            DEFAULT_COMPACT_BYTES,
            DEFAULT_MIN_FILES_TO_MERGE,
            // The poll interval is unused for a manual sweep.
            std::time::Duration::from_secs(1),
            self.catalog.clone(),
        );
        let mut merged_so_far = 0;
        loop {
            compacter.compact_all().await;
            let merged: u64 = compacter
                .snapshot()
                .per_table
                .values()
                .map(|table| table.files_merged_in)
                .sum();
            if merged == merged_so_far {
                break;
            }
            merged_so_far = merged;
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Pick a free local port by binding to `:0` and dropping the listener; the
/// kernel will not immediately reuse the port for the brief window before the
/// server thread re-binds.
fn pick_free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn wait_until_listening(addr: SocketAddr) -> std::io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut last_err = None;
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(_) => return Ok(()),
            Err(e) => last_err = Some(e),
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("timed out waiting for server")))
}

/// Start a pivotdb server with the given worker count, returning once the
/// listener is accepting connections. The catalog is empty; the runner sends
/// `CREATE TABLE` over the wire to populate it.
pub fn start(workers: usize) -> std::io::Result<ServerHandle> {
    let port = pick_free_port()?;
    let bind: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .expect("valid socket addr");

    let dispatch = Dispatch::spin_up(workers, ring_buffers(), None);
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    // Kept so the harness can run a compaction pass between load and queries;
    // the thread below takes its own clone.
    let handle_catalog = catalog.clone();

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let thread = thread::Builder::new()
        .name("pivot-bench-server".into())
        .spawn(move || {
            // The runtime gets few threads on purpose: its default (one per
            // core) would sit hundreds of mostly-idle threads next to the
            // pinned dispatch workers and preempt them on every wakeup.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(async move {
                let server = Server::new(
                    bind,
                    dispatch,
                    catalog,
                    0,
                    4,
                    server::DEFAULT_CATALOG_REFRESH,
                );
                let _ = server
                    .serve(Box::pin(async move {
                        let _ = shutdown_rx.await;
                    }))
                    .await;
            });
        })?;

    wait_until_listening(bind)?;

    Ok(ServerHandle {
        port,
        catalog: handle_catalog,
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    })
}
