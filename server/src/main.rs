//! Binary entry point for the pivotdb postgres-wire server.
//!
//! This internally plans & compiles queries using the `planner` crate (internally based on DuckDB's,
//! planner) with the `DeltaDatastore` and runs queries on `dispatch`

use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use catalog::PivotCatalog;
use clap::Parser;
use dispatch::env::get_env_var_with_default;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore::Metastore;
use metastore_toml::TomlMetastore;
use server::{Error, Server};
use tracing::{error, info};

/// Postgres-wire-compatible server in front of pivotdb's dispatch engine.
#[derive(Parser, Debug)]
#[command(name = "pivot", about, version)]
struct Args {
    /// TCP socket to bind to.
    #[arg(long, default_value = "127.0.0.1:5432")]
    bind: SocketAddr,

    /// Memory budget for the buffer pool, as a human-readable size like `32g`
    /// or `512m` (suffixes k/m/g/t, base-1024, case-insensitive). Overrides the
    /// `PIVOT_MEMORY_PCT` percentage-of-total-RAM default when set.
    #[arg(long, value_name = "SIZE", value_parser = parse_memory_size)]
    memory: Option<usize>,

    /// Number of dispatch worker threads. Defaults to the number of cores.
    #[arg(long)]
    workers: Option<usize>,

    /// Metastore file (TOML) defining the datastores to serve, each a local
    /// directory or S3 root, attached as its own database
    /// (`SELECT * FROM <datastore>.main.<table>`). Exactly one datastore must set
    /// `default = true`; it is the current database (unqualified names resolve
    /// against it).
    #[arg(long, value_name = "FILE")]
    metastore: PathBuf,

    /// Enable the on-disk cache for remote (S3) object reads, storing cached
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

/// Parses a human-readable memory size like `32g`, `512m`, or `4096` into a
/// byte count. Accepts an optional base-1024 suffix (`k`, `m`, `g`, `t`, each
/// also accepting a trailing `b`), case-insensitive; a bare number is bytes.
fn parse_memory_size(input: &str) -> Result<usize, String> {
    let trimmed = input.trim();
    let digits_end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (number, suffix) = trimmed.split_at(digits_end);
    if number.is_empty() {
        return Err(format!("`{input}` has no leading number"));
    }
    let value: usize = number
        .parse()
        .map_err(|_| format!("`{number}` is not a valid number"))?;

    let multiplier: usize = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1024,
        "m" | "mb" => 1024 * 1024,
        "g" | "gb" => 1024 * 1024 * 1024,
        "t" | "tb" => 1024 * 1024 * 1024 * 1024,
        other => return Err(format!("`{other}` is not a known size suffix (k/m/g/t)")),
    };

    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("`{input}` overflows a byte count"))
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

/// Open every datastore from the required metastore file and assemble them into
/// one composite catalog. Each datastore self-manages its maintenance (the global
/// refresh cadence plus its own per-datastore compaction from the metastore file),
/// spawning its background tasks onto the ambient runtime, so this must run inside
/// `rt.block_on`. Exits the process on any open failure; there is nothing to serve
/// without it.
fn build_catalog(args: &Args, dispatcher: &DataFlowDispatcher) -> Arc<PivotCatalog> {
    let path = &args.metastore;
    let refresh_interval = Duration::from_secs(args.catalog_refresh_secs);
    let store = TomlMetastore::open(path, refresh_interval).unwrap_or_else(|e| {
        error!("failed to read metastore `{}`: {e}", path.display());
        std::process::exit(1);
    });
    let default_name = store.default_datastore_name().to_string();
    let datastores = store.open_datastores(dispatcher).unwrap_or_else(|e| {
        error!("failed to open datastores from `{}`: {e}", path.display());
        std::process::exit(1);
    });
    Arc::new(
        PivotCatalog::new(datastores, default_name).unwrap_or_else(|e| {
            error!("invalid metastore configuration: {e}");
            std::process::exit(1);
        }),
    )
}

fn main() -> Result<(), Error> {
    init_tracing();
    let args = Args::parse();

    let workers = args.workers.unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    let disk_cache = build_disk_cache(&args);
    let pool_bytes = match args.memory {
        Some(bytes) => bytes,
        None => {
            let memory_pct: usize = get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
            get_total_memory() * memory_pct / 100
        }
    };
    info!(pool_bytes, "buffer pool memory budget");
    let dispatch = Dispatch::spin_up(workers, pool_bytes / BUFFER_SIZE, disk_cache);

    // A handful of runtime threads handles the wire protocol comfortably; the
    // default (one per core) would put hundreds of mostly-idle threads next to
    // the pinned dispatch workers, and every wakeup of an unpinned thread on a
    // fully occupied box preempts a worker mid-stage.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let catalog = build_catalog(&args, dispatch.dispatcher());

        let mut server = Server::new(args.bind, dispatch, catalog);
        if let Some(addr) = args.http_bind {
            server = server.with_http_bind(addr);
        }
        let shutdown = Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        server.serve(shutdown).await
    })
}
