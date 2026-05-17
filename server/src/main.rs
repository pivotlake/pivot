//! Binary entry point for the pivotdb postgres-wire server.
//!
//! This internally plans & compiles queries using the `planner` crate (internally based on DuckDB's,
//! planner) with the `ParquetCatalog` and runs queries on `dispatch`

use std::net::SocketAddr;
use std::sync::Arc;

use catalog::ParquetCatalog;
use clap::Parser;
use server::{Error, Server};
use tracing::info;
use dispatch::{Dispatch, BUFFER_SIZE};

/// Postgres-wire-compatible server in front of pivotdb's dispatch engine.
#[derive(Parser, Debug)]
#[command(name = "pivot", about, version)]
struct Args {
    /// TCP socket to bind to.
    #[arg(long, default_value = "127.0.0.1:5432")]
    bind: SocketAddr,

    /// Number of dispatch worker threads. Defaults to the number of cores.
    #[arg(long)]
    workers: Option<usize>,
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}


/// Returns the total physical memory of the machine in bytes.
pub fn get_total_memory() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
        .total_memory() as usize
}

fn main() -> Result<(), Error> {
    init_tracing();
    let args = Args::parse();

    let workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    info!(workers, "initialising dispatch");
    let dispatch = Dispatch::spin_up(workers, get_total_memory() / 2 / BUFFER_SIZE);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let catalog = Arc::new(ParquetCatalog::new());
        let server = Server::new(args.bind, dispatch, catalog);
        let shutdown = Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        server.serve(shutdown).await
    })
}
