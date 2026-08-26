//! The config file: one YAML file describing a whole instance.
//!
//! The file has two sections, each owned by the code that acts on it. `server`
//! is [`ServerConfig`] below: the endpoint, the memory and worker budgets, the
//! disk cache. `metastore` is [`MetastoreConfig`]: the datastores to serve, the
//! secrets they are opened with, and the users that may connect. Both are read
//! in a single pass, so a mistake in either one is reported with its place in
//! the file and stops startup.
//!
//! ```yaml
//! server:
//!   bind: 0.0.0.0:5432
//!   memory: 32g
//!   workers: 16
//!   refresh_interval: 30s
//!   disk_cache:
//!     dir: /var/cache/pivot
//!     size: 64g
//!   tls:
//!     cert: /etc/pivot/server.crt
//!     key: /etc/pivot/server.key
//!
//! metastore:
//!   datastores:
//!     hot:
//!       kind: delta
//!       location: /var/lib/pivot
//!       default: true
//! ```
//!
//! Every field of `server` has a default, so the section may be omitted whole.
//! Unknown keys are rejected rather than ignored: a misspelled setting would
//! otherwise leave the server running on a default nobody asked for.
//!
//! The `metastore` section is not the only place datastores, secrets and users
//! may be written: the binary's `--metastore-file` names a second file of the
//! same shape, which the metastore merges into this section.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use datastore_delta::DEFAULT_REFRESH_INTERVAL;
use metastore_disk::{ByteSize, Interval, MetastoreConfig};
use serde::Deserialize;

/// The address the PostgreSQL endpoint binds to when `bind` is not set. Loopback
/// so an unconfigured server is not exposed to the network.
const DEFAULT_BIND: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5432);

/// The disk cache's byte budget when only `dir` is given.
const DEFAULT_DISK_CACHE_SIZE: ByteSize = ByteSize::from_bytes(64 * 1024 * 1024 * 1024);

/// The disk cache's object-count limit when only `dir` is given.
const DEFAULT_DISK_CACHE_MAX_OBJECTS: usize = 65536;

/// A parsed config file.
///
/// Unknown top-level keys are rejected: a section under a misspelled name would
/// otherwise leave the whole instance on defaults without a word about it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The instance itself. Every field has a default, so the section may be
    /// left out entirely.
    #[serde(default)]
    pub server: ServerConfig,
    /// The data to serve. Required: a server with no datastores has nothing to
    /// answer a query with.
    pub metastore: MetastoreConfig,
}

impl Config {
    /// Read and parse the config file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // The lossy form names the file in errors; the read itself keeps the
        // original bytes, which a path need not spell in UTF-8.
        let displayed = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: displayed.clone(),
            source,
        })?;
        Self::from_yaml(&text, &displayed)
    }

    /// Parse config from YAML text, using `path` in errors.
    fn from_yaml(text: &str, path: &str) -> Result<Self> {
        serde_yaml_ng::from_str(text).map_err(|source| Error::Parse {
            path: path.to_string(),
            source,
        })
    }
}

/// The `server` section: everything about the instance that is independent of
/// the data it serves.
///
/// The fields that stay `Option` here are the ones whose default is a property
/// of the machine (its core count, its total memory) rather than a constant, so
/// the binary resolves them at startup instead of baking them in here.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    /// TCP socket the PostgreSQL endpoint binds to.
    pub bind: SocketAddr,
    /// Also serve the bundled web dashboard (data-flow graph, live compaction
    /// stats, system metrics, SQL console) on this address. Omit to disable.
    pub http_bind: Option<SocketAddr>,
    /// Memory budget for the buffer pool, such as `32g`. Defaults to a
    /// percentage of the machine's total memory.
    pub memory: Option<ByteSize>,
    /// Number of dispatch worker threads. Defaults to the machine's core count.
    pub workers: Option<usize>,
    /// How often every datastore brings its in-memory table set up to date with
    /// the store. For shared remote stores, this bounds how stale a query's view
    /// of externally committed data can be; this process's own commits are
    /// visible immediately.
    pub refresh_interval: Interval,
    /// On-disk cache for remote (object store) reads. Omit to disable it; local
    /// files are never cached, they are read from the filesystem directly.
    pub disk_cache: Option<DiskCacheConfig>,
    /// Certificate the PostgreSQL endpoint presents to a client that asks to
    /// encrypt its connection. Omit to answer every such request with a refusal,
    /// leaving the endpoint plaintext-only.
    pub tls: Option<TlsConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND,
            http_bind: None,
            memory: None,
            workers: None,
            refresh_interval: Interval::from_duration(DEFAULT_REFRESH_INTERVAL),
            disk_cache: None,
            tls: None,
        }
    }
}

/// The `server.tls` section. Both fields are required: an endpoint that offers
/// encryption has to present a certificate, and a certificate is worth nothing
/// without the key that proves the server holds it.
///
/// Writing the section turns SSL on; it does not make it compulsory. A client
/// that asks to upgrade gets an encrypted session, and one that does not still
/// gets a plaintext one, which is how PostgreSQL's own `ssl = on` behaves.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    /// PEM file holding the certificate to present: the server's own first,
    /// followed by whatever intermediates a client needs to reach a root it
    /// trusts.
    pub cert: PathBuf,
    /// PEM file holding that certificate's private key, in any of PKCS#8,
    /// PKCS#1 or SEC1. Readable by the server's user and by nobody else.
    pub key: PathBuf,
}

/// The `server.disk_cache` section. `dir` is required: the section exists to
/// enable the cache, and a cache with no directory to live in cannot be one.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiskCacheConfig {
    /// Directory the cached byte ranges are stored in. The contents persist
    /// across restarts.
    pub dir: PathBuf,
    /// Size budget for the cached bytes.
    #[serde(default = "default_disk_cache_size")]
    pub size: ByteSize,
    /// Maximum number of cached objects. This bounds the open file descriptors
    /// the cache holds, one per object, so keep it below the process's limit.
    #[serde(default = "default_disk_cache_max_objects")]
    pub max_objects: usize,
}

fn default_disk_cache_size() -> ByteSize {
    DEFAULT_DISK_CACHE_SIZE
}

fn default_disk_cache_max_objects() -> usize {
    DEFAULT_DISK_CACHE_MAX_OBJECTS
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading config file `{path}`: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "parsing config file `{path}`: {source}\nthe file holds a `server` section (optional) and a `metastore` section (required)"
    )]
    Parse {
        path: String,
        source: serde_yaml_ng::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use catalog::metastore::Metastore;
    use metastore_disk::DiskMetastore;

    const METASTORE_SECTION: &str = "metastore:\n  datastores:\n    hot:\n      kind: delta\n      \
                                     location: /tmp/hot\n      default: true\n";

    /// A file whose `server` section holds `settings`, on top of a minimal
    /// `metastore` section.
    fn from_yaml_with(settings: &str) -> Result<Config> {
        Config::from_yaml(&format!("server:\n{settings}{METASTORE_SECTION}"), "test")
    }

    #[test]
    fn example_config_parses() {
        let yaml = include_str!("../../config.example.yaml");

        let config = Config::from_yaml(yaml, "config.example.yaml").unwrap();

        assert_eq!(config.server.bind, "127.0.0.1:5432".parse().unwrap());
    }

    #[test]
    fn one_file_configures_both_the_instance_and_its_data() {
        let config = from_yaml_with(
            "  bind: 0.0.0.0:5433\n  memory: 32g\n  workers: 8\n  refresh_interval: 5s\n",
        )
        .unwrap();
        let metastore = DiskMetastore::open(
            config.metastore,
            None,
            config.server.refresh_interval.as_duration(),
        )
        .unwrap();

        assert_eq!(config.server.bind, "0.0.0.0:5433".parse().unwrap());
        assert_eq!(
            config.server.memory.unwrap().as_bytes(),
            32 * 1024 * 1024 * 1024
        );
        assert_eq!(config.server.workers, Some(8));
        assert_eq!(
            config.server.refresh_interval.as_duration(),
            Duration::from_secs(5)
        );
        assert_eq!(metastore.default_datastore_name(), "hot");
    }

    #[test]
    fn an_omitted_server_section_leaves_the_instance_on_defaults() {
        let config = Config::from_yaml(METASTORE_SECTION, "test").unwrap();

        assert_eq!(config.server, ServerConfig::default());
        assert_eq!(config.server.bind, "127.0.0.1:5432".parse().unwrap());
        assert_eq!(
            config.server.refresh_interval.as_duration(),
            DEFAULT_REFRESH_INTERVAL
        );
    }

    #[test]
    fn a_file_without_a_metastore_section_is_rejected() {
        let error = Config::from_yaml("server:\n  workers: 8\n", "test")
            .err()
            .expect("a server with no data to serve should be rejected");

        assert!(
            error.to_string().contains("missing field `metastore`"),
            "{error}"
        );
    }

    #[test]
    fn a_misspelled_section_is_rejected_rather_than_dropped() {
        let yaml = format!("sever:\n  workers: 8\n{METASTORE_SECTION}");

        let error = Config::from_yaml(&yaml, "test").err().unwrap();

        assert!(
            error.to_string().contains("unknown field `sever`"),
            "{error}"
        );
    }

    #[test]
    fn a_misspelled_server_setting_is_rejected_rather_than_ignored() {
        let error = from_yaml_with("  bnid: 0.0.0.0:5433\n").err().unwrap();

        assert!(
            error.to_string().contains("unknown field `bnid`"),
            "{error}"
        );
    }

    #[test]
    fn a_disk_cache_takes_its_budgets_from_defaults() {
        let config = from_yaml_with("  disk_cache:\n    dir: /var/cache/pivot\n").unwrap();

        let cache = config.server.disk_cache.unwrap();
        assert_eq!(cache.dir, PathBuf::from("/var/cache/pivot"));
        assert_eq!(cache.size, DEFAULT_DISK_CACHE_SIZE);
        assert_eq!(cache.max_objects, DEFAULT_DISK_CACHE_MAX_OBJECTS);
    }

    #[test]
    fn a_tls_section_names_the_certificate_to_present() {
        let config = from_yaml_with(
            "  tls:\n    cert: /etc/pivot/server.crt\n    key: /etc/pivot/server.key\n",
        )
        .unwrap();

        let tls = config.server.tls.unwrap();
        assert_eq!(tls.cert, PathBuf::from("/etc/pivot/server.crt"));
        assert_eq!(tls.key, PathBuf::from("/etc/pivot/server.key"));
    }

    #[test]
    fn a_certificate_without_its_key_is_rejected() {
        let error = from_yaml_with("  tls:\n    cert: /etc/pivot/server.crt\n")
            .err()
            .expect("a certificate the server cannot prove it holds should be rejected");

        assert!(error.to_string().contains("missing field `key`"), "{error}");
    }

    #[test]
    fn a_disk_cache_without_a_directory_is_rejected() {
        let error = from_yaml_with("  disk_cache:\n    size: 8g\n")
            .err()
            .unwrap();

        assert!(error.to_string().contains("missing field `dir`"), "{error}");
    }

    #[test]
    fn a_missing_file_names_itself() {
        let error = Config::open("/nonexistent/pivot.yaml").err().unwrap();

        assert!(
            matches!(&error, Error::Read { path, .. } if path == "/nonexistent/pivot.yaml"),
            "{error}"
        );
    }
}
