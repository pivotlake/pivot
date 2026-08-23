//! Construction and foreground execution of a Pivot server.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use clap::Args;
use datastore_delta::store::{ObjectStore, StoreScheme, open_store};
use datastore_delta::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
    DeltaDatastore, MaintenanceConfig, VacuumConfig, default_merge_target_bytes,
};
use dispatch::env::get_env_var_with_default;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore::{Metastore, UserAuth};
use metastore_disk::{DiskMetastore, DiskUserStore, MetastoreConfig};
use tracing::{error, info};

use crate::config::DiskCacheConfig;
use crate::{Config, Error, Server, raise_open_file_limit};

/// Options accepted by `pivot server`.
#[derive(Args, Debug)]
pub struct ServerOptions {
    /// Config file (YAML) describing this instance. Its `server` section sets
    /// the endpoint, memory and worker budgets, and disk cache. Its `metastore`
    /// section defines the datastores and users served by this process.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "datastore",
        conflicts_with_all = ["datastore", "default_datastore"]
    )]
    config: Option<PathBuf>,

    /// A second YAML file holding datastores and users, without a surrounding
    /// `metastore` key. Its entries are merged with the main config.
    #[arg(
        long,
        value_name = "FILE",
        requires = "config",
        conflicts_with_all = ["datastore", "default_datastore"]
    )]
    metastore_file: Option<PathBuf>,

    /// Datastore opened directly from a local path or object-store URI instead
    /// of `--config`. May be repeated as NAME=LOCATION; one unnamed LOCATION is
    /// served as `default`. S3 entries use AWS_ACCESS_KEY_ID and
    /// AWS_SECRET_ACCESS_KEY, with optional AWS_REGION (or AWS_DEFAULT_REGION)
    /// and AWS_ENDPOINT_URL. Users created through SQL are persisted in
    /// ./.pivot/metastore.yaml.
    #[arg(
        long,
        value_name = "[NAME=]LOCATION",
        required_unless_present = "config",
        conflicts_with_all = ["config", "metastore_file"]
    )]
    datastore: Vec<DirectDatastoreArg>,

    /// Default datastore for unqualified table names. Required when more than
    /// one `--datastore` is given; the sole datastore is inferred otherwise.
    #[arg(long, value_name = "NAME", requires = "datastore")]
    default_datastore: Option<String>,
}

/// One value supplied to the repeatable `--datastore [NAME=]LOCATION` option.
#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectDatastoreArg {
    name: Option<String>,
    location: String,
}

impl FromStr for DirectDatastoreArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let equals = value.find('=');
        let scheme = value.find("://");
        let (name, location) = match (equals, scheme) {
            (Some(equals), Some(scheme)) if scheme < equals => (None, value),
            (Some(equals), _) => (Some(&value[..equals]), &value[equals + 1..]),
            (None, _) => (None, value),
        };

        if let Some(name) = name {
            if name.is_empty() {
                return Err("the datastore name before `=` cannot be empty".to_string());
            }
            if name.trim() != name {
                return Err("the datastore name cannot begin or end with whitespace".to_string());
            }
            if name == datastore_system::DATASTORE_NAME {
                return Err(format!(
                    "`{name}` is reserved for Pivot's virtual system datastore"
                ));
            }
        }
        if location.is_empty() {
            return Err("the datastore location cannot be empty".to_string());
        }
        StoreScheme::of(location).map_err(|source| match name {
            Some(name) => format!("datastore `{name}` has an invalid location: {source}"),
            None => format!("datastore location `{location}` is invalid: {source}"),
        })?;
        Ok(Self {
            name: name.map(str::to_string),
            location: location.to_string(),
        })
    }
}

/// A direct datastore after an omitted name has been resolved to `default`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectDatastore {
    name: String,
    location: String,
}

/// The metastore half of the selected command-line form.
enum MetastoreInput {
    Config {
        path: PathBuf,
        metastore_file: Option<PathBuf>,
        config: MetastoreConfig,
    },
    Direct {
        datastores: Vec<DirectDatastore>,
        default_name: String,
    },
}

/// Fully resolved startup settings. A direct datastore uses the ordinary
/// server defaults; a config file supplies both halves.
struct ServerInput {
    server_config: crate::config::ServerConfig,
    metastore: MetastoreInput,
}

impl ServerInput {
    fn from_options(options: ServerOptions) -> Result<Self, Error> {
        let ServerOptions {
            config,
            metastore_file,
            datastore,
            default_datastore,
        } = options;
        match (
            config,
            metastore_file,
            datastore.is_empty(),
            default_datastore,
        ) {
            (Some(path), metastore_file, true, None) => {
                let config = Config::open(&path).map_err(Box::new)?;
                Ok(Self {
                    server_config: config.server,
                    metastore: MetastoreInput::Config {
                        path,
                        metastore_file,
                        config: config.metastore,
                    },
                })
            }
            (None, None, false, default_datastore) => {
                let (datastores, default_name) =
                    resolve_direct_datastores(datastore, default_datastore.as_deref())?;
                Ok(Self {
                    server_config: crate::config::ServerConfig::default(),
                    metastore: MetastoreInput::Direct {
                        datastores,
                        default_name,
                    },
                })
            }
            // Clap enforces the two valid forms. Keep the library entry point
            // defensive in case a future non-Clap caller constructs options in
            // this module.
            _ => Err(Error::InvalidDirectDatastore(
                "provide --config or one or more --datastore [NAME=]LOCATION options".to_string(),
            )),
        }
    }
}

fn resolve_direct_datastores(
    arguments: Vec<DirectDatastoreArg>,
    requested_default: Option<&str>,
) -> Result<(Vec<DirectDatastore>, String), Error> {
    if arguments.len() > 1 && arguments.iter().any(|argument| argument.name.is_none()) {
        return Err(Error::InvalidDirectDatastore(
            "every --datastore must have a NAME=LOCATION value when more than one is supplied"
                .to_string(),
        ));
    }

    let datastores: Vec<_> = arguments
        .into_iter()
        .map(|argument| DirectDatastore {
            name: argument
                .name
                .unwrap_or_else(|| DEFAULT_DATASTORE_NAME.to_string()),
            location: argument.location,
        })
        .collect();
    let mut names = std::collections::HashSet::new();
    for datastore in &datastores {
        if !names.insert(datastore.name.as_str()) {
            return Err(Error::InvalidDirectDatastore(format!(
                "datastore name `{}` was supplied more than once",
                datastore.name
            )));
        }
    }

    let default_name = match requested_default {
        Some(name) if names.contains(name) => name.to_string(),
        Some(name) => {
            return Err(Error::InvalidDirectDatastore(format!(
                "--default-datastore `{name}` does not name any --datastore entry"
            )));
        }
        None if datastores.len() == 1 => datastores[0].name.clone(),
        None => {
            return Err(Error::InvalidDirectDatastore(
                "multiple --datastore entries require --default-datastore <NAME>".to_string(),
            ));
        }
    };
    Ok((datastores, default_name))
}

/// Minimal metastore for directly supplied Delta datastores and the same
/// persistent user store a config-backed metastore supplies.
/// Opening through `open_store` is deliberate: its S3 path reads credentials
/// from `AWS_*`, while `DiskMetastore` continues requiring a configured secret.
#[derive(Debug)]
struct DirectMetastore {
    datastores: Vec<DirectDatastore>,
    default_name: String,
    refresh_interval: Duration,
    users: DiskUserStore,
}

impl DirectMetastore {
    fn open(
        datastores: Vec<DirectDatastore>,
        default_name: String,
        refresh_interval: Duration,
        metastore_path: &Path,
    ) -> Result<Self, metastore_disk::Error> {
        Ok(Self {
            datastores,
            default_name,
            refresh_interval,
            users: DiskUserStore::open_or_create(metastore_path)?,
        })
    }

    fn maintenance(&self) -> MaintenanceConfig {
        MaintenanceConfig {
            refresh_interval: self.refresh_interval,
            compaction: Some(CompactionConfig {
                target_bytes: DEFAULT_COMPACT_BYTES,
                merge_target_bytes: default_merge_target_bytes(DEFAULT_COMPACT_BYTES),
                min_files: DEFAULT_MIN_FILES_TO_MERGE,
                poll_interval: DEFAULT_COMPACT_POLL,
            }),
            vacuum: Some(VacuumConfig::default()),
        }
    }
}

impl Metastore for DirectMetastore {
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        self.datastores
            .iter()
            .map(|config| {
                let opened = (|| {
                    let store: Arc<dyn ObjectStore> = open_store(&config.location)?.into();
                    let datastore: Arc<dyn Datastore> =
                        DeltaDatastore::from_store(store, dispatcher, Some(self.maintenance()))?;
                    Ok(datastore)
                })();
                opened
                    .map(|datastore| (config.name.clone(), datastore))
                    .map_err(|source| {
                        Box::new(DirectDatastoreOpenError {
                            name: config.name.clone(),
                            location: config.location.clone(),
                            source,
                        }) as metastore::Error
                    })
            })
            .collect()
    }

    fn default_datastore_name(&self) -> &str {
        &self.default_name
    }

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        self.users.user_auth(username)
    }

    fn create_user(&self, username: &str, password: Option<&str>) -> metastore::Result<()> {
        self.users
            .create_user(username, password)
            .map_err(|error| Box::new(error) as metastore::Error)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("datastore `{name}` at `{location}`: {source}")]
struct DirectDatastoreOpenError {
    name: String,
    location: String,
    #[source]
    source: metastore::Error,
}

/// Targets whose `INFO` output is noise for an operator reading the server log.
const QUIET_TARGETS: &str = "delta_kernel=warn,delta_kernel_default_engine=warn";
const DIRECT_METASTORE_PATH: &str = ".pivot/metastore.yaml";

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

fn build_catalog(
    path: &Path,
    metastore: &Arc<dyn Metastore>,
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
    let catalog = PivotCatalog::new(datastores, default_name, metastore.clone())?;
    Ok(Arc::new(catalog))
}

/// Run a server in the foreground until SIGINT or SIGTERM.
pub fn run(options: ServerOptions) -> Result<(), Error> {
    init_tracing();
    let ServerInput {
        server_config,
        metastore: metastore_input,
    } = ServerInput::from_options(options)?;
    let crate::config::ServerConfig {
        bind,
        http_bind,
        memory,
        workers,
        refresh_interval,
        disk_cache,
        tls,
    } = server_config;

    let tls = tls.as_ref().map(crate::tls::build_acceptor).transpose()?;

    raise_open_file_limit();

    let workers = workers.unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    let disk_cache = build_disk_cache(disk_cache);
    let pool_bytes = match memory {
        Some(size) => size.as_bytes() as usize,
        None => {
            let memory_pct: usize = get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
            get_total_memory() * memory_pct / 100
        }
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
        let refresh_interval = refresh_interval.as_duration();
        let (metastore, catalog) = match metastore_input {
            MetastoreInput::Config {
                path,
                metastore_file,
                config,
            } => {
                let metastore =
                    build_metastore(config, &path, metastore_file.as_deref(), refresh_interval)?;
                let catalog = build_catalog(&path, &metastore, dispatch.dispatcher())?;
                (metastore, catalog)
            }
            MetastoreInput::Direct {
                datastores,
                default_name,
            } => {
                let metastore_path = PathBuf::from(DIRECT_METASTORE_PATH);
                let direct = DirectMetastore::open(
                    datastores,
                    default_name,
                    refresh_interval,
                    &metastore_path,
                )
                .map_err(|source| Error::Metastore {
                    path: metastore_path.clone(),
                    source: Box::new(source),
                })?;
                info!(path = %metastore_path.display(), "direct metastore ready");
                let metastore: Arc<dyn Metastore> = Arc::new(direct);
                let default_name = metastore.default_datastore_name().to_string();
                let datastores = metastore
                    .open_datastores(dispatch.dispatcher())
                    .map_err(Error::OpenDirectDatastores)?;
                let catalog = Arc::new(PivotCatalog::new(
                    datastores,
                    default_name,
                    metastore.clone(),
                )?);
                (metastore, catalog)
            }
        };

        let mut server = Server::new(bind, dispatch, catalog, metastore);
        if let Some(address) = http_bind {
            server = server.with_http_bind(address);
        }
        if let Some(acceptor) = tls {
            server = server.with_tls(acceptor);
        }
        server.serve(Box::pin(wait_for_shutdown_signal())).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unnamed_direct_datastore_keeps_its_complete_location() {
        for location in [
            "/var/lib/pivot",
            "file:///var/lib/pivot",
            "s3://bucket/db?key=a=b",
            "s3a://bucket/db",
            "gs://bucket/db",
        ] {
            let datastore: DirectDatastoreArg = location.parse().unwrap();
            assert_eq!(datastore.name, None);
            assert_eq!(datastore.location, location);
        }
    }

    #[test]
    fn a_named_direct_datastore_keeps_its_name_and_complete_location() {
        let datastore: DirectDatastoreArg = "warm=s3://bucket/db?key=a=b".parse().unwrap();

        assert_eq!(datastore.name.as_deref(), Some("warm"));
        assert_eq!(datastore.location, "s3://bucket/db?key=a=b");
    }

    #[test]
    fn direct_datastores_reject_malformed_or_reserved_values() {
        for value in [
            "",
            "=s3://bucket/db",
            "warm=",
            " warm=/tmp/warm",
            "system=/tmp/system",
            "https://example.com/db",
            "warm=https://example.com/db",
        ] {
            assert!(
                value.parse::<DirectDatastoreArg>().is_err(),
                "accepted invalid datastore {value:?}"
            );
        }
    }

    #[test]
    fn one_unnamed_datastore_is_named_default() {
        let arguments = vec!["/tmp/hot".parse().unwrap()];

        let (datastores, default_name) = resolve_direct_datastores(arguments, None).unwrap();

        assert_eq!(default_name, DEFAULT_DATASTORE_NAME);
        assert_eq!(datastores[0].name, DEFAULT_DATASTORE_NAME);
        assert_eq!(datastores[0].location, "/tmp/hot");
    }

    #[test]
    fn one_named_datastore_is_implicitly_the_default() {
        let arguments = vec!["hot=/tmp/hot".parse().unwrap()];

        assert_eq!(resolve_direct_datastores(arguments, None).unwrap().1, "hot");
    }

    #[test]
    fn multiple_named_datastores_require_an_existing_explicit_default() {
        let arguments = vec![
            "hot=/tmp/hot".parse().unwrap(),
            "warm=s3://bucket/warm".parse().unwrap(),
        ];

        let missing = resolve_direct_datastores(arguments.clone(), None).unwrap_err();
        assert!(
            missing.to_string().contains("require --default-datastore"),
            "{missing}"
        );
        let unknown = resolve_direct_datastores(arguments.clone(), Some("cold")).unwrap_err();
        assert!(
            unknown.to_string().contains("does not name any"),
            "{unknown}"
        );
        assert_eq!(
            resolve_direct_datastores(arguments, Some("warm"))
                .unwrap()
                .1,
            "warm"
        );
    }

    #[test]
    fn multiple_datastores_must_all_be_named() {
        for arguments in [
            vec!["/tmp/hot".parse().unwrap(), "/tmp/warm".parse().unwrap()],
            vec![
                "hot=/tmp/hot".parse().unwrap(),
                "s3://bucket/warm".parse().unwrap(),
            ],
        ] {
            let error = resolve_direct_datastores(arguments, Some("hot")).unwrap_err();
            assert!(error.to_string().contains("must have a NAME=LOCATION"));
        }
    }

    #[test]
    fn named_direct_datastore_names_must_be_unique() {
        let arguments = vec![
            "hot=/tmp/one".parse().unwrap(),
            "hot=/tmp/two".parse().unwrap(),
        ];

        let error = resolve_direct_datastores(arguments, Some("hot")).unwrap_err();
        assert!(error.to_string().contains("more than once"), "{error}");
    }

    #[test]
    fn direct_metastore_opens_every_named_datastore() {
        let hot = tempfile::tempdir().unwrap();
        let warm = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let metastore_path = state.path().join(".pivot/metastore.yaml");
        let metastore = DirectMetastore::open(
            vec![
                DirectDatastore {
                    name: "hot".to_string(),
                    location: hot.path().display().to_string(),
                },
                DirectDatastore {
                    name: "warm".to_string(),
                    location: warm.path().display().to_string(),
                },
            ],
            "hot".to_string(),
            datastore_delta::DEFAULT_REFRESH_INTERVAL,
            &metastore_path,
        )
        .unwrap();
        let dispatch = Dispatch::spin_up(1, 32, None);

        let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();

        assert_eq!(datastores.len(), 2);
        assert!(datastores.contains_key("hot"));
        assert!(datastores.contains_key("warm"));
        assert_eq!(metastore.default_datastore_name(), "hot");
        assert!(metastore_path.is_file());
        metastore.create_user("reader", None).unwrap();
        assert!(matches!(
            metastore.user_auth("reader"),
            Some(UserAuth::Trust)
        ));
        drop(datastores);
        dispatch.exit();
    }
}
