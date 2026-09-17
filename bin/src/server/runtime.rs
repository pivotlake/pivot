//! Construction and foreground execution of a configured Pivot server.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use clap::Args;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore_disk::{DiskMetastore, MetastoreConfig};
use tracing::{error, info};

use crate::memory::{compute_default_pool_bytes, read_memory_pct};
use crate::server::config::DiskCacheConfig;
use crate::server::{Config, Error, Server, raise_open_file_limit};

/// Options accepted by `pivot server`.
#[derive(Args, Debug)]
pub struct ServerOptions {
    /// Config file (YAML) describing this instance. Its `server` section sets
    /// the endpoint, memory and worker budgets, and disk cache. Its `metastore`
    /// section defines the datastores and users served by this process.
    #[arg(long, value_name = "FILE")]
    config: PathBuf,

    /// A second YAML file holding datastores and users, without a surrounding
    /// `metastore` key. Its entries are merged with the main config.
    #[arg(long, value_name = "FILE")]
    metastore_file: Option<PathBuf>,
}

/// Targets whose `INFO` output is noise for an operator reading the server log.
const QUIET_TARGETS: &str = "delta_kernel=warn,delta_kernel_default_engine=warn";

fn init_tracing() {
    let requested = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let filter = tracing_subscriber::EnvFilter::try_new(format!("{QUIET_TARGETS},{requested}"))
        .unwrap_or_else(|error| {
            eprintln!("ignoring unparsable RUST_LOG ({requested:?}): {error}");
            tracing_subscriber::EnvFilter::new(format!("{QUIET_TARGETS},info"))
        });
    let ansi = std::io::stdout().is_terminal();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .init();
}

/// Wait for an interactive interrupt or the termination signal sent by a
/// service manager.
fn wait_for_shutdown_signal() -> impl std::future::Future<Output = ()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let terminate = signal(SignalKind::terminate());
        let interrupt = signal(SignalKind::interrupt());
        async move {
            let (mut terminate, mut interrupt) = match (terminate, interrupt) {
                (Ok(terminate), Ok(interrupt)) => (terminate, interrupt),
                (Err(error), _) | (_, Err(error)) => {
                    error!(%error, "failed to listen for shutdown signals");
                    return;
                }
            };
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = terminate.recv() => {}
            }
        }
    }

    #[cfg(not(unix))]
    async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            error!(%error, "failed to listen for Ctrl-C");
        }
    }
}

/// Return the machine's total physical memory in bytes.
fn get_total_memory() -> usize {
    read_memory().total_memory() as usize
}

/// Return how many bytes a new allocation on this machine can actually get hold
/// of: free pages plus what the kernel can reclaim without swapping.
fn get_available_memory() -> usize {
    read_memory().available_memory() as usize
}

/// One reading of the machine's memory, as the OS reports it now.
fn read_memory() -> sysinfo::System {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
}

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
        Err(error) => {
            error!(dir = %dir.display(), "failed to open disk cache, continuing without it: {error}");
            None
        }
    }
}

fn build_metastore(
    config: MetastoreConfig,
    path: &Path,
    metastore_file: Option<&Path>,
    refresh_interval: Duration,
) -> Result<Arc<DiskMetastore>, Error> {
    let metastore =
        DiskMetastore::open(config, metastore_file, refresh_interval).map_err(|source| {
            Error::Metastore {
                path: path.to_path_buf(),
                source: Box::new(source),
            }
        })?;
    Ok(Arc::new(metastore))
}

fn build_catalog(
    path: &Path,
    metastore: &Arc<DiskMetastore>,
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
    let catalog = PivotCatalog::new(datastores, default_name, metastore.clone())?
        .with_external_parquet_read_context(dispatcher, metastore.external_store_factory());
    Ok(Arc::new(catalog))
}

/// Run a configured server in the foreground until SIGINT or SIGTERM.
pub fn run(options: ServerOptions) -> Result<(), Error> {
    init_tracing();
    let config = Config::open(&options.config).map_err(Box::new)?;
    let server_config = config.server;

    let tls = server_config
        .tls
        .as_ref()
        .map(crate::server::tls::build_acceptor)
        .transpose()?;

    raise_open_file_limit();

    let workers = server_config
        .workers
        .unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    let disk_cache = build_disk_cache(server_config.disk_cache);
    let pool_bytes = match server_config.memory {
        Some(size) => size.as_bytes() as usize,
        None => compute_default_pool_bytes(get_total_memory() as u64, read_memory_pct())? as usize,
    };
    // The pool is a share of *total* memory, but every one of its slots is
    // faulted in while the workers start, so what it has to fit into is what the
    // machine has free. Asking for more than that is not a slower server: it is
    // an OOM kill part way through boot, which leaves nobody around to say why.
    let available_bytes = get_available_memory();
    info!(pool_bytes, available_bytes, "buffer pool memory budget");
    if pool_bytes > available_bytes {
        return Err(Error::InsufficientMemory {
            requested_bytes: pool_bytes,
            available_bytes,
        });
    }
    let dispatch = Dispatch::spin_up(workers, pool_bytes / BUFFER_SIZE, disk_cache);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let metastore = build_metastore(
            config.metastore,
            &options.config,
            options.metastore_file.as_deref(),
            server_config.refresh_interval.as_duration(),
        )?;
        let catalog = build_catalog(&options.config, &metastore, dispatch.dispatcher())?;
        let mut server = Server::new(server_config.bind, dispatch, catalog, metastore);
        if let Some(acceptor) = tls {
            server = server.with_tls(acceptor);
        }
        server.serve(Box::pin(wait_for_shutdown_signal())).await
    })
}
