//! Binary entry point for the pivotdb postgres-wire server.
//!
//! This internally plans & compiles queries using the `planner` crate (internally based on DuckDB's,
//! planner) with the `DeltaDatastore` and runs queries on `dispatch`

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use catalog::PivotCatalog;
use clap::Parser;
use dispatch::env::get_env_var_with_default;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore::Metastore;
use metastore_disk::{DiskMetastore, MetastoreConfig};
use server::config::DiskCacheConfig;
use server::{Config, Error, Server, raise_open_file_limit};
use tracing::{error, info};

/// Postgres-wire-compatible server in front of pivotdb's dispatch engine.
#[derive(Parser, Debug)]
#[command(name = "pivot", about, version)]
struct Args {
    /// Config file (YAML) describing this instance. Its `server` section sets
    /// the endpoint, the memory and worker budgets, and the disk cache; every
    /// setting there has a default, so the section is optional. Its `metastore`
    /// section defines the datastores to serve, each a local directory or an S3
    /// or GCS root, attached as its own database
    /// (`SELECT * FROM <datastore>.main.<table>`), and the users that may
    /// connect. Exactly one datastore must set `default = true`; it is the
    /// current database (unqualified names resolve against it).
    #[arg(long, value_name = "FILE")]
    config: PathBuf,

    /// A second YAML file holding datastores and users, written the same way as
    /// the config file's `metastore` section but without the `metastore:` key
    /// above them. Its entries are merged with that section, so a datastore or
    /// a user may be defined in either file; a name defined in both stops
    /// startup, as does a file that cannot be read. Omit to serve exactly what
    /// the config file's `metastore` section defines.
    #[arg(long, value_name = "FILE")]
    metastore_file: Option<PathBuf>,
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

/// Build the remote-read disk cache from the config's `disk_cache` section, or
/// `None` when it has none. A failure to open the directory is logged and
/// downgraded to `None`, so a cache problem never stops the server starting.
fn build_disk_cache(config: Option<DiskCacheConfig>) -> Option<Arc<dispatch::io::DiskCache>> {
    let DiskCacheConfig {
        dir,
        size,
        max_objects,
    } = config?;
    match dispatch::io::DiskCache::open(dir.clone(), size.as_bytes(), max_objects) {
        Ok(cache) => {
            info!(
                dir = %dir.display(),
                bytes = size.as_bytes(),
                max_objects,
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

/// Build the process's one metastore from the config's `metastore` section and
/// the metastore file merged into it, if `--metastore-file` named one. The
/// returned object is shared by catalog construction and every later login, so
/// authentication always reaches the same live source rather than a startup copy
/// of its users.
fn build_metastore(
    config: MetastoreConfig,
    path: &Path,
    metastore_file: Option<&Path>,
    refresh_interval: Duration,
) -> Result<Arc<dyn Metastore>, Error> {
    let metastore =
        DiskMetastore::open(config, metastore_file, refresh_interval).map_err(|source| {
            Error::Metastore {
                path: path.to_path_buf(),
                source: Box::new(source),
            }
        })?;
    Ok(Arc::new(metastore))
}

/// Open every datastore from `metastore` and assemble them into one composite
/// catalog. Each datastore self-manages its maintenance (the global refresh
/// cadence plus its own per-datastore compaction), spawning its background tasks
/// onto the ambient runtime, so this must run inside `rt.block_on`.
fn build_catalog(
    path: &Path,
    metastore: &dyn Metastore,
    dispatcher: &DataFlowDispatcher,
) -> Result<Arc<PivotCatalog>, Error> {
    let default_name = metastore.default_datastore_name().to_string();
    let datastores =
        metastore
            .open_datastores(dispatcher)
            .map_err(|source| Error::OpenDatastores {
                path: path.to_path_buf(),
                source,
            })?;
    Ok(Arc::new(PivotCatalog::new(datastores, default_name)?))
}

fn run() -> Result<(), Error> {
    init_tracing();
    let args = Args::parse();
    let config = Config::open(&args.config).map_err(Box::new)?;
    let server_config = config.server;

    raise_open_file_limit();

    let workers = server_config
        .workers
        .unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    let disk_cache = build_disk_cache(server_config.disk_cache);
    let pool_bytes = match server_config.memory {
        Some(size) => size.as_bytes() as usize,
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
        let metastore = build_metastore(
            config.metastore,
            &args.config,
            args.metastore_file.as_deref(),
            server_config.refresh_interval.as_duration(),
        )?;
        let catalog = build_catalog(&args.config, metastore.as_ref(), dispatch.dispatcher())?;

        let mut server = Server::new(server_config.bind, dispatch, catalog, metastore);
        if let Some(addr) = server_config.http_bind {
            server = server.with_http_bind(addr);
        }
        let shutdown = Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        server.serve(shutdown).await
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
