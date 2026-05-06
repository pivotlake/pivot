//! Shutdown integration test, isolated in its own test binary.
//!
//! `Server::serve` flips the process-global `dispatch::EXIT` flag on shutdown,
//! which permanently disables every worker for the rest of the process. Living
//! in a dedicated binary keeps that side-effect from cross-contaminating the
//! shared-server tests in `integration.rs`.

mod common;

use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use catalog::ParquetCatalog;
use common::{pick_free_port, wait_until_listening};
use server::Server;
use tokio::sync::oneshot;

/// `Server::serve` only returns `Ok(())` once every dispatch worker `JoinHandle`
/// has joined (drained from `worker_watchers`). So if the server thread
/// completes within the deadline with `Ok(())`, every worker thread terminated.
#[test]
fn shutdown_signal_drains_all_worker_threads() {
    let workers = 2;
    let handles = dispatch::init(workers);
    assert_eq!(
        handles.len(),
        workers,
        "dispatch should spawn one thread per worker"
    );
    let bind: SocketAddr = format!("127.0.0.1:{}", pick_free_port()).parse().unwrap();
    let catalog: Arc<dyn planner::catalog::Catalog> = Arc::new(ParquetCatalog::new());
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let server_thread = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let server = Server::new(bind, handles, catalog);
            server
                .serve(Box::pin(async move {
                    let _ = shutdown_rx.await;
                }))
                .await
        })
    });
    wait_until_listening(bind);
    shutdown_tx.send(()).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    while !server_thread.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        server_thread.is_finished(),
        "server did not shut down within 10s — worker threads likely didn't observe EXIT",
    );
    let result = server_thread.join().expect("server thread panicked");
    assert!(
        matches!(result, Ok(())),
        "expected Ok on clean shutdown, got {result:?}"
    );
}
