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
use dispatch::{BUFFER_SIZE, Dispatch};
use server::Server;
use tokio::sync::oneshot;

fn total_memory_bytes() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}

pub struct ServerHandle {
    port: u16,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn port(&self) -> u16 {
        self.port
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

    let dispatch = Dispatch::spin_up(workers, total_memory_bytes() * 4 / 5 / BUFFER_SIZE, None);
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let thread = thread::Builder::new()
        .name("pivot-bench-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(async move {
                let server = Server::new(bind, dispatch, catalog, vec![], 0, 4);
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
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    })
}
