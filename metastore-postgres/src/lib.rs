//! A PostgreSQL-backed [`metastore::Metastore`] provider.
//!
//! This crate owns the `metastore` section of PivotDB's config file when that
//! section selects `kind: postgres`. Instead of spelling datastores and users
//! inline the way the YAML provider does, the section points at a PostgreSQL
//! database that every instance of a cluster shares:
//!
//! ```yaml
//! metastore:
//!   kind: postgres
//!   url: postgres://pivot:secret@pg.internal:5432/pivot_metastore
//!   refresh_interval: 30s   # optional, as in the YAML provider
//! ```
//!
//! On startup the provider creates its schema, `pivot_metastore`, and its
//! tables when they do not exist yet, so the database only has to exist. Rows
//! are administered with plain SQL against that database; PivotDB's own
//! endpoint has no DDL for them:
//!
//! ```sql
//! INSERT INTO pivot_metastore.datastores (name, location, is_default)
//!     VALUES ('hot', 's3://my-bucket/pivot/', true);
//! INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier)
//!     VALUES ('analytics', 'scram-sha-256', 'pivot-scram-sha-256$4096:...$...');
//! ```
//!
//! Datastore rows carry the same fields as the YAML provider's `datastores`
//! entries: `kind` is the datastore format (only `delta` today, the column's
//! default), the storage backend is chosen from `location`'s scheme, the S3
//! credential columns apply only to an `s3://` location, and `compact` with its
//! tuning columns selects per-datastore compaction. Because the rows are shared
//! but compaction must run in at most one process per datastore, an instance
//! runs the rows' compaction settings only when its own config section sets
//! `compact: true`; start at most one instance with it. Exactly one row must
//! set `is_default = true`; a partial unique index enforces at most one across
//! every instance, and startup requires at least one. The rows are read once,
//! when the server opens its datastores at startup, so adding a datastore means
//! restarting instances.
//!
//! Users are read live: every login queries the database, so a user added or a
//! verifier rotated applies to the next login of every instance, no restarts.
//! An empty `users` table supplies the built-in trusted `pivot` user, exactly
//! like a YAML file without a `users` section; any inserted row replaces that
//! with the explicit allowlist. `scram_verifier` stores the
//! `pivot-scram-sha-256$...` form of [`metastore::format_scram_verifier`], not
//! a PostgreSQL `SCRAM-SHA-256` verifier copied from `pg_authid`.
//!
//! The `url` may carry a password and the datastore rows S3 credentials, so
//! the config file and the metastore database should be readable only by the
//! PivotDB processes and their operators. Connections are made without TLS, so
//! the database should be reached over a trusted network.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use catalog::Datastore;
use datastore_delta::store::{LocalStore, ObjectStore, S3Credentials, S3Store};
use datastore_delta::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
    DEFAULT_REFRESH_INTERVAL, DeltaDatastore, MaintenanceConfig,
};
use dispatch::DataFlowDispatcher;
use metastore::{DEFAULT_USER_NAME, Metastore, UserAuth, parse_scram_verifier};
use metastore_yaml::Interval;
use serde::Deserialize;

#[cfg(feature = "test-support")]
pub mod test_support;

/// Advisory-lock key serialising [`migrate`] across instances. Arbitrary (the
/// bytes spell `pivot_ms`), but it must stay the same in every PivotDB version
/// that shares a database.
const SCHEMA_LOCK_KEY: i64 = 0x7069766f745f6d73;

/// Everything [`migrate`] creates, idempotently, in the fixed `pivot_metastore`
/// schema (fixed rather than configured: every instance of a cluster must
/// resolve the same tables, and a second cluster gets its own database).
///
/// The partial unique index makes "at most one default datastore" a property of
/// the shared database rather than of any one instance's validation. The users
/// checks keep a row's method and verifier consistent, so a login lookup can
/// trust the pair.
const SCHEMA: &str = "
CREATE SCHEMA IF NOT EXISTS pivot_metastore;
CREATE TABLE IF NOT EXISTS pivot_metastore.datastores (
    name text PRIMARY KEY,
    kind text NOT NULL DEFAULT 'delta',
    location text NOT NULL,
    is_default boolean NOT NULL DEFAULT false,
    compact boolean NOT NULL DEFAULT false,
    compact_bytes bigint CHECK (compact_bytes > 0),
    compact_min_files bigint CHECK (compact_min_files > 0),
    region text,
    access_key_id text,
    secret_access_key text,
    endpoint text
);
CREATE UNIQUE INDEX IF NOT EXISTS datastores_one_default
    ON pivot_metastore.datastores ((true)) WHERE is_default;
CREATE TABLE IF NOT EXISTS pivot_metastore.users (
    name text PRIMARY KEY,
    auth_method text NOT NULL CHECK (auth_method IN ('trust', 'scram-sha-256')),
    scram_verifier text,
    CHECK ((auth_method = 'scram-sha-256') = (scram_verifier IS NOT NULL))
);
";

/// The config file's `metastore` section when it selects `kind: postgres`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresMetastoreConfig {
    /// Connection string of the shared metastore database, in any form libpq
    /// accepts: a `postgres://user:password@host:port/database` URL or a
    /// `host=... user=...` keyword string.
    url: String,
    /// How often every datastore refreshes its table set from the store, as in
    /// the YAML provider.
    #[serde(default = "default_refresh_interval")]
    refresh_interval: Interval,
    /// Honour the datastore rows' `compact` settings in this instance. Off by
    /// default: the rows are shared by every instance, but compaction must run
    /// in at most one process per datastore, so it is the instance started
    /// with this flag (at most one) that compacts, not the whole cluster.
    #[serde(default)]
    compact: bool,
}

impl PostgresMetastoreConfig {
    /// A section holding only the database `url`, every other setting on its
    /// default: what parsing `metastore: {kind: postgres, url: ...}` yields.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            refresh_interval: default_refresh_interval(),
            compact: false,
        }
    }
}

fn default_refresh_interval() -> Interval {
    Interval::from_duration(DEFAULT_REFRESH_INTERVAL)
}

/// A metastore backed by a shared PostgreSQL database.
///
/// [`connect`](Self::connect) prepares the schema and validates that a default
/// datastore is configured. Datastores are read when
/// [`Metastore::open_datastores`] is called at startup; users are read on every
/// [`Metastore::user_auth`] lookup, so administration applies live.
pub struct PostgresMetastore {
    /// Reconnection source: the parsed `url`, kept so a dropped connection can
    /// be replaced without restarting the server.
    connect_config: postgres::Config,
    /// The one live connection, replaced when it breaks. Logins are rare and
    /// datastores are read once, so a single serialised connection is plenty.
    client: Mutex<Option<postgres::Client>>,
    default_name: String,
    refresh_interval: Duration,
    /// Whether this instance runs the rows' per-datastore compaction settings.
    compact: bool,
}

/// Redacted: the connection configuration may carry the database password.
impl std::fmt::Debug for PostgresMetastore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PostgresMetastore")
    }
}

impl PostgresMetastore {
    /// Connect to the metastore database, create the `pivot_metastore` schema
    /// and tables when absent, and read which datastore is the default. Fails
    /// when the database is unreachable or no row sets `is_default = true`, so
    /// a misconfigured cluster stops at startup rather than at the first query.
    pub fn connect(config: PostgresMetastoreConfig) -> Result<Self> {
        let connect_config: postgres::Config = config.url.parse().map_err(Error::Url)?;
        let (client, default_name) = run_off_runtime(|| {
            let mut client = open_client(&connect_config)?;
            migrate(&mut client).map_err(Error::Migrate)?;
            let default_name = read_default_datastore_name(&mut client)?;
            Ok::<_, Error>((client, default_name))
        })?;
        Ok(Self {
            connect_config,
            client: Mutex::new(Some(client)),
            default_name,
            refresh_interval: config.refresh_interval.as_duration(),
            compact: config.compact,
        })
    }

    /// Run `operation` on the shared connection, opening one when none is live.
    /// When the connection turns out to have died underneath us (database
    /// restart, dropped network), open a fresh one and retry once, so a single
    /// broken connection does not fail a login.
    fn with_client<T: Send>(
        &self,
        operation: impl Fn(&mut postgres::Client) -> Result<T, postgres::Error> + Sync,
    ) -> Result<T> {
        run_off_runtime(|| {
            let mut slot = self.client.lock().unwrap();
            let client = match slot.as_mut() {
                Some(client) => client,
                None => slot.insert(open_client(&self.connect_config)?),
            };
            match operation(client) {
                Ok(value) => Ok(value),
                Err(_) if client.is_closed() => {
                    let client = slot.insert(open_client(&self.connect_config)?);
                    operation(client).map_err(Error::Query)
                }
                Err(error) => Err(Error::Query(error)),
            }
        })
    }

    fn build_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>> {
        let rows = self.with_client(|client| {
            client.query(
                "SELECT name, kind, location, compact, compact_bytes, compact_min_files, \
                 region, access_key_id, secret_access_key, endpoint \
                 FROM pivot_metastore.datastores",
                &[],
            )
        })?;
        rows.iter()
            .map(|row| {
                let name: String = row.get("name");
                let kind: String = row.get("kind");
                let store = open_store(&name, row)?;
                let maintenance = MaintenanceConfig {
                    refresh_interval: self.refresh_interval,
                    // Gated on this instance's own `compact` flag: the rows are
                    // shared, and only one process per datastore may compact.
                    compaction: if self.compact {
                        compaction(&name, row)?
                    } else {
                        None
                    },
                };
                let datastore: Arc<dyn Datastore> = match kind.as_str() {
                    "delta" => DeltaDatastore::from_store(store, dispatcher, Some(maintenance))?,
                    other => {
                        return Err(Error::Datastore {
                            name,
                            message: format!("unsupported kind `{other}` (expected `delta`)"),
                        });
                    }
                };
                Ok((name, datastore))
            })
            .collect()
    }

    /// One login's lookup. A missing row on an entirely empty table yields the
    /// built-in trusted `pivot` user, mirroring the YAML provider's behaviour
    /// for a file without a `users` section.
    fn look_up_user(&self, username: &str) -> Result<Option<UserAuth>> {
        let row = self.with_client(|client| {
            client.query_opt(
                "SELECT auth_method, scram_verifier FROM pivot_metastore.users WHERE name = $1",
                &[&username],
            )
        })?;
        let Some(row) = row else {
            if username != DEFAULT_USER_NAME {
                return Ok(None);
            }
            let any_users = self.with_client(|client| {
                client.query_one("SELECT EXISTS (SELECT 1 FROM pivot_metastore.users)", &[])
            })?;
            return Ok((!any_users.get::<_, bool>(0)).then_some(UserAuth::Trust));
        };
        let method: String = row.get("auth_method");
        let verifier: Option<String> = row.get("scram_verifier");
        match (method.as_str(), verifier) {
            ("trust", None) => Ok(Some(UserAuth::Trust)),
            ("scram-sha-256", Some(text)) => parse_scram_verifier(&text)
                .map(|verifier| Some(UserAuth::ScramSha256(verifier)))
                .map_err(|message| Error::User {
                    name: username.to_string(),
                    message,
                }),
            // The schema's checks forbid these combinations, so reaching here
            // means the table was altered; refuse the login rather than guess.
            (method, verifier) => Err(Error::User {
                name: username.to_string(),
                message: format!(
                    "auth method `{method}` disagrees with its verifier being {}",
                    if verifier.is_some() { "set" } else { "absent" }
                ),
            }),
        }
    }
}

impl Metastore for PostgresMetastore {
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        self.build_datastores(dispatcher)
            .map_err(|error| Box::new(error) as metastore::Error)
    }

    fn default_datastore_name(&self) -> &str {
        &self.default_name
    }

    fn user_auth(&self, username: &str) -> metastore::Result<Option<UserAuth>> {
        self.look_up_user(username)
            .map_err(|error| Box::new(error) as metastore::Error)
    }
}

/// Run `operation` off any ambient tokio runtime. The synchronous `postgres`
/// client drives its own single-thread runtime with `block_on`, which panics
/// when entered from a thread already executing async code (the server's login
/// path); a scoped thread has no such context. Outside a runtime the call is
/// direct.
fn run_off_runtime<T: Send>(operation: impl FnOnce() -> T + Send) -> T {
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::scope(|scope| {
            scope
                .spawn(operation)
                .join()
                .expect("metastore database operation panicked")
        })
    } else {
        operation()
    }
}

fn open_client(config: &postgres::Config) -> Result<postgres::Client> {
    config.connect(postgres::NoTls).map_err(Error::Connect)
}

/// Create the schema, serialised across instances with an advisory lock: two
/// servers starting at once would otherwise race the `IF NOT EXISTS` creations,
/// which PostgreSQL reports as duplicate-key errors on its own catalogs.
fn migrate(client: &mut postgres::Client) -> Result<(), postgres::Error> {
    let mut transaction = client.transaction()?;
    transaction.execute("SELECT pg_advisory_xact_lock($1)", &[&SCHEMA_LOCK_KEY])?;
    transaction.batch_execute(SCHEMA)?;
    transaction.commit()
}

/// The name of the one datastore flagged `is_default = true`. The unique index
/// keeps a second flag from ever being stored, but the multiple case is still
/// reported (the index could have been dropped by hand) rather than picking one.
fn read_default_datastore_name(client: &mut postgres::Client) -> Result<String> {
    let rows = client
        .query(
            "SELECT name FROM pivot_metastore.datastores WHERE is_default ORDER BY name",
            &[],
        )
        .map_err(Error::Query)?;
    let mut names: Vec<String> = rows.iter().map(|row| row.get(0)).collect();
    if names.len() > 1 {
        return Err(Error::MultipleDefaults(names));
    }
    names.pop().ok_or(Error::MissingDefault)
}

/// Build one row's object store, mirroring the YAML provider: an `s3://` (or
/// `s3a://`) location opens an S3 store with the row's inline credentials
/// (`region`, `access_key_id`, `secret_access_key` are required); anything else
/// is a local path (an optional `file://` scheme is stripped).
fn open_store(name: &str, row: &postgres::Row) -> Result<Arc<dyn ObjectStore>> {
    let location: String = row.get("location");
    if location.starts_with("s3://") || location.starts_with("s3a://") {
        let credentials = S3Credentials {
            region: require_column(name, row, "region")?,
            access_key: require_column(name, row, "access_key_id")?,
            secret_key: require_column(name, row, "secret_access_key")?,
            endpoint: row.get("endpoint"),
        };
        Ok(Arc::new(S3Store::with_credentials(&location, credentials)?))
    } else {
        let path = location.strip_prefix("file://").unwrap_or(&location);
        Ok(Arc::new(LocalStore::new(path)))
    }
}

fn require_column(name: &str, row: &postgres::Row, column: &str) -> Result<String> {
    row.get::<_, Option<String>>(column)
        .ok_or_else(|| Error::Datastore {
            name: name.to_string(),
            message: format!("`{column}` is required for an s3:// location"),
        })
}

/// A row's compaction settings, or `None` when `compact` is off. Fills the
/// tuning columns' defaults; the schema's checks keep stored values positive.
fn compaction(name: &str, row: &postgres::Row) -> Result<Option<CompactionConfig>> {
    if !row.get::<_, bool>("compact") {
        return Ok(None);
    }
    let target_bytes = match row.get::<_, Option<i64>>("compact_bytes") {
        Some(bytes) => u64::try_from(bytes).map_err(|_| Error::Datastore {
            name: name.to_string(),
            message: format!("`compact_bytes` must be positive, got {bytes}"),
        })?,
        None => DEFAULT_COMPACT_BYTES,
    };
    let min_files = match row.get::<_, Option<i64>>("compact_min_files") {
        Some(files) => usize::try_from(files).map_err(|_| Error::Datastore {
            name: name.to_string(),
            message: format!("`compact_min_files` must be positive, got {files}"),
        })?,
        None => DEFAULT_MIN_FILES_TO_MERGE,
    };
    Ok(Some(CompactionConfig {
        target_bytes,
        min_files,
        poll_interval: DEFAULT_COMPACT_POLL,
    }))
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid metastore database `url`: {0}")]
    Url(#[source] postgres::Error),
    #[error("connecting to the metastore database: {0}")]
    Connect(#[source] postgres::Error),
    #[error("preparing the metastore schema: {0}")]
    Migrate(#[source] postgres::Error),
    #[error("querying the metastore database: {0}")]
    Query(#[source] postgres::Error),
    #[error(
        "no default datastore is configured; mark exactly one row of \
         `pivot_metastore.datastores` with `is_default = true`"
    )]
    MissingDefault,
    #[error(
        "multiple datastores are marked `is_default = true` ({}); exactly one may be", .0.join(", ")
    )]
    MultipleDefaults(Vec<String>),
    #[error("datastore `{name}`: {message}")]
    Datastore { name: String, message: String },
    #[error("user `{name}`: {message}")]
    User { name: String, message: String },
    #[error(transparent)]
    Store(#[from] datastore_delta::store::StoreError),
    #[error(transparent)]
    Delta(#[from] datastore_delta::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_without_a_refresh_interval_uses_the_default() {
        let config: PostgresMetastoreConfig =
            serde_yaml_ng::from_str("url: postgres://localhost/pivot").unwrap();

        assert_eq!(
            config.refresh_interval.as_duration(),
            DEFAULT_REFRESH_INTERVAL
        );
    }

    #[test]
    fn a_section_without_a_url_is_rejected() {
        let result: std::result::Result<PostgresMetastoreConfig, _> =
            serde_yaml_ng::from_str("refresh_interval: 30s");

        assert!(result.is_err());
    }

    #[test]
    fn a_misspelled_setting_is_rejected_rather_than_ignored() {
        let result: std::result::Result<PostgresMetastoreConfig, _> =
            serde_yaml_ng::from_str("url: postgres://localhost/pivot\nrefresh_intervall: 30s");

        assert!(result.is_err());
    }

    #[test]
    fn a_malformed_url_is_rejected_before_any_connection() {
        let error =
            PostgresMetastore::connect(PostgresMetastoreConfig::new("not a url")).unwrap_err();

        assert!(matches!(error, Error::Url(_)), "{error}");
    }
}
