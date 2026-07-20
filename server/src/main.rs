//! Binary entry point for the pivotdb postgres-wire server.
//!
//! This internally plans & compiles queries using the `planner` crate (internally based on DuckDB's,
//! planner) with the `ParquetCatalog` and runs queries on `dispatch`

use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use catalog::ParquetCatalog;
use clap::Parser;
use dispatch::env::get_env_var_with_default;
use dispatch::{BUFFER_SIZE, Dispatch};
use server::{Error, Server};
use tracing::{error, info};

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

    /// Database directory. Tables created with `CREATE TABLE` (without their own
    /// `path`) live under here, and are reloaded on restart. Omit for an
    /// in-memory catalog — tables vanish on restart and must each name a `path`.
    #[arg(long)]
    path: Option<PathBuf>,

    /// Enable the on-disk cache for remote (S3/GCS) object reads, storing cached
    /// byte ranges under this directory (persists across restarts). Omit to
    /// disable. Only affects object-store reads; local files are unaffected.
    #[arg(long, value_name = "DIR")]
    disk_cache_dir: Option<PathBuf>,

    /// Disk-cache size budget in GiB (only with `--disk-cache-dir`).
    #[arg(long, default_value_t = 64.0, value_name = "GIB")]
    disk_cache_gb: f64,

    /// Disk-cache max object count - bounds open file descriptors, one per cached
    /// object (only with `--disk-cache-dir`). Keep below the process's fd limit.
    #[arg(long, default_value_t = 65536, value_name = "N")]
    disk_cache_max_objects: usize,

    /// Run the bundled compacter (OFF by default). The compacter merges and
    /// then deletes a table's small Parquet files, so it mutates the catalog;
    /// leave it off for a read-only server or an external reader, and only the
    /// process that owns the data (the writer) should enable it. The
    /// `--compact-*` tuning flags require this.
    #[arg(long)]
    compact: bool,

    /// Byte threshold the compacter merges a table's small Parquet files up to
    /// (CPU on the dispatch pool), once they amount to it. Requires `--compact`.
    #[arg(
        long,
        default_value_t = compact::DEFAULT_COMPACT_BYTES,
        requires = "compact",
        value_name = "BYTES"
    )]
    compact_bytes: u64,

    /// Compacter count trigger for low-traffic partitions: merge a sub-target
    /// partition's small files once this many accumulate (even below
    /// `--compact-bytes`). Lower means fewer tiny files per partition and more
    /// frequent sub-target merges. Requires `--compact`.
    #[arg(
        long,
        default_value_t = compact::DEFAULT_MIN_FILES_TO_MERGE,
        requires = "compact",
        value_name = "N"
    )]
    compact_min_files: usize,

    /// Also serve the bundled web dashboard (data-flow graph, live compaction
    /// stats, system metrics, SQL console) on this address. Omit to disable.
    #[arg(long, value_name = "ADDR")]
    http_bind: Option<SocketAddr>,

    /// How often (seconds) the background catalog refresh brings the in-memory
    /// table set up to date with the store: new Delta versions, new files'
    /// footers, and tables committed by other processes. Queries bind against
    /// a snapshot of that in-memory set, so this bounds how stale a query's
    /// view of *externally* committed data can be (this process's own INSERT
    /// and compaction publish their commits immediately).
    #[arg(long, default_value_t = 30, value_name = "SECS")]
    catalog_refresh_secs: u64,
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    // Only emit ANSI colour codes to an interactive terminal; when stdout (the
    // fmt subscriber's default writer) is redirected to a log file the escape
    // sequences are just noise that breaks grep/awk and bloats the file.
    let ansi = std::io::stdout().is_terminal();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .init();
}

/// Returns the total physical memory of the machine in bytes.
pub fn get_total_memory() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}

/// Build the remote-read disk cache from the `--disk-cache-*` flags, or `None`
/// when `--disk-cache-dir` is unset. A failure to open the directory is logged
/// and downgraded to `None`, so a cache problem never stops the server starting.
fn build_disk_cache(args: &Args) -> Option<Arc<dispatch::io::DiskCache>> {
    let dir = args.disk_cache_dir.clone()?;
    let byte_budget = (args.disk_cache_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    match dispatch::io::DiskCache::open(dir.clone(), byte_budget, args.disk_cache_max_objects) {
        Ok(cache) => {
            info!(
                dir = %dir.display(),
                gib = args.disk_cache_gb,
                max_objects = args.disk_cache_max_objects,
                "disk cache enabled"
            );
            Some(Arc::new(cache))
        }
        Err(e) => {
            error!(dir = %dir.display(), "failed to open disk cache, continuing without it: {e}");
            None
        }
    }
}

fn main() -> Result<(), Error> {
    init_tracing();
    let args = Args::parse();

    let workers = args.workers.unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    let disk_cache = build_disk_cache(&args);
    let memory_pct: usize = get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
    let pool_bytes = get_total_memory() * memory_pct / 100;
    let dispatch = Dispatch::spin_up(workers, pool_bytes / BUFFER_SIZE, disk_cache);

    // A handful of runtime threads handles the wire protocol comfortably; the
    // default (one per core) would put hundreds of mostly-idle threads next to
    // the pinned dispatch workers, and every wakeup of an unpinned thread on a
    // fully occupied box preempts a worker mid-stage.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;

    // Build the catalog on this (coordinator) thread, before the workers are
    // handed off: opening a persisted database reloads its tables, which reads
    // Parquet footers over the dispatch pool.
    let catalog = match args.path.as_deref() {
        Some(dir) => {
            let dir = dir.to_str().expect("database path must be valid UTF-8");
            Arc::new(
                ParquetCatalog::open(dir, dispatch.dispatcher()).unwrap_or_else(|e| {
                    error!("failed to open database `{dir}`: {e}");
                    std::process::exit(1);
                }),
            )
        }
        None => Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone())),
    };

    rt.block_on(async move {
        let compact_bytes = if args.compact { args.compact_bytes } else { 0 };
        let mut server = Server::new(
            args.bind,
            dispatch,
            catalog,
            compact_bytes,
            args.compact_min_files,
            Duration::from_secs(args.catalog_refresh_secs),
        );
        if let Some(addr) = args.http_bind {
            server = server.with_http_bind(addr);
        }
        let shutdown = Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        server.serve(shutdown).await
    })
}
