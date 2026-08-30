//! Construction and foreground execution of a configured Pivot server.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use clap::Args;
use dispatch::env::get_env_var_with_default;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore_disk::{DiskMetastore, MetastoreConfig};
use tracing::{error, info};

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
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(terminate) => terminate,
                Err(error) => {
                    error!(%error, "failed to listen for the termination signal");
                    return;
                }
            };
        tokio::select! {
            interrupt = tokio::signal::ctrl_c() => {
                if let Err(error) = interrupt {
                    error!(%error, "failed to listen for Ctrl-C");
                }
            }
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    if let Err(error) = tokio::signal::ctrl_c().await {
        error!(%error, "failed to listen for Ctrl-C");
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
    let disk_cache = match server_config.disk_cache {
        Some(DiskCacheConfig {
            dir,
            size,
            max_objects,
        }) => Some(crate::resources::open_disk_cache(
            dir,
            size.as_bytes(),
            max_objects,
        )?),
        None => None,
    };
    let pool_bytes = match server_config.memory {
        Some(size) => size.as_bytes() as usize,
        None => {
            let memory_pct: usize = get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
            crate::resources::total_memory_bytes() * memory_pct / 100
        }
    };
    let available_bytes = crate::resources::check_pool_fits(pool_bytes)?;
    info!(pool_bytes, available_bytes, "buffer pool memory budget");
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
        if let Some(address) = server_config.http_bind {
            server = server.with_http_bind(address);
        }
        if let Some(acceptor) = tls {
            server = server.with_tls(acceptor);
        }
        server.serve(Box::pin(wait_for_shutdown_signal())).await
    })
}
