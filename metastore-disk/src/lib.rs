//! A disk-backed [`metastore::Metastore`] provider.
//!
//! This crate owns the `metastore` section of PivotDB's config file: the
//! datastores to serve and the users that may connect. The server reads the
//! file, keeps its own `server` section, and hands this section over as a
//! [`MetastoreConfig`]. The scalars both sections are written with,
//! [`ByteSize`] and [`Interval`], are defined here too.
//!
//! A section of the same shape may also live in a file of its own, which the
//! server names with `--metastore-file` and this crate reads and merges into
//! the config file's section ([`DiskMetastore::open`]). The two are peers: a
//! datastore or a user may be written in either, and the rules below apply to
//! the merged whole. A name written in both files stops startup rather than one
//! silently winning, since the two entries are free to disagree about location
//! or credentials.
//!
//! Every datastore and user keeps the [`Origin`] it was merged from, because the
//! two files are not interchangeable once the server is running: the config file
//! is the operator's, and the metastore file is the server's own to rewrite.
//!
//! `kind` is the datastore format (only `delta` today). The storage backend is
//! chosen from `location`: a plain path (or `file://`) opens a local store, an
//! `s3://` URI opens an S3 store, a `gs://` URI opens a Google Cloud Storage
//! store. The S3 credential fields apply only to an `s3://` location, and
//! `credentials_file` only to a `gs://` one.
//!
//! Exactly one datastore must set `default = true`; it becomes the current
//! database, so unqualified table names and DDL resolve against it. Its name is
//! free (it need not be called `default`).
//!
//! ```yaml
//! metastore:
//!   datastores:
//!     hot:
//!       kind: delta
//!       location: /var/lib/pivot     # local path -> local store
//!       default: true                # the current database
//!     warm:
//!       kind: delta
//!       location: s3://my-bucket/pivot/
//!       region: us-east-1            # required for s3://
//!       access_key_id: AKIA...       # required for s3://
//!       secret_access_key: "..."
//!       # endpoint: http://localhost:9000
//!       # compact: true              # optional
//!     cold:
//!       kind: delta
//!       location: gs://my-bucket/pivot/
//!       # credentials_file: /etc/pivot/gcs-key.json   # else ambient ADC
//! ```
//!
//! Each entry under `users` names a user that may authenticate to the
//! PostgreSQL endpoint. Its nested `auth` selects one authentication method. A
//! user may explicitly be trusted without a password:
//!
//! ```yaml
//! metastore:
//!   users:
//!     reader:
//!       auth:
//!         method: trust
//! ```
//!
//! Or it may authenticate with a SCRAM-SHA-256 verifier derived from its
//! password:
//!
//! ```yaml
//! metastore:
//!   users:
//!     analytics:
//!       auth:
//!         method: scram-sha-256
//!         verifier: "pivot-scram-sha-256$4096:cGVwcGVy...$Zm9vYmFy..."
//! ```
//!
//! A user named `pivot` is always served, so a server is reachable whatever else
//! its users are. It is trusted unless a file defines `pivot` itself, which
//! takes over its authentication method entirely: give it a
//! `scram-sha-256` verifier to require a password of it.
//!
//! An S3 datastore's credentials are inline, so the file holds secrets and should
//! be readable only by the PivotDB process (a GCS datastore's key stays in the
//! file `credentials_file` points at). A verifier is not a password (the
//! password cannot be recovered from it), but it is still worth the same care.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use catalog::Datastore;
use datastore_delta::store::{GcsStore, LocalStore, ObjectStore, S3Credentials, S3Store};
use datastore_delta::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
    DEFAULT_VACUUM_POLL, DeltaDatastore, MaintenanceConfig, VacuumConfig,
};
use dispatch::DataFlowDispatcher;
use metastore::{DEFAULT_USER_NAME, Metastore, UserAuth, parse_scram_verifier};
use serde::Deserialize;

mod units;

pub use units::{ByteSize, Interval};

/// A metastore backed by the config file's `metastore` section, and by the
/// standalone file merged into it.
///
/// The configuration is structurally validated by [`open`](Self::open). Object
/// stores and their Delta datastores are opened when
/// [`Metastore::open_datastores`] is called.
pub struct DiskMetastore {
    datastore_configs: HashMap<String, ConfigDefinition<DatastoreConfig>>,
    default_name: String,
    /// Every user's authentication method, keyed by user name. The built-in
    /// `pivot` user is always among them. Verifiers are
    /// decoded once, by [`open`](Self::open), so a malformed one fails startup
    /// rather than a login.
    user_auth: HashMap<String, ConfigDefinition<UserAuth>>,
    /// How often every datastore this metastore opens refreshes its table set
    /// from the store. Global (all datastores share the cadence, which the
    /// server takes from its own section); compaction, in contrast, is
    /// configured per datastore.
    refresh_interval: Duration,
}

impl DiskMetastore {
    /// Build the metastore from the config file's `metastore` section, merged
    /// with the datastores and users in `metastore_file` when the server was
    /// given one.
    ///
    /// A `metastore_file` that cannot be read stops startup: a metastore file
    /// that was asked for and is not there means datastores or users are
    /// missing, which would otherwise show up as tables that do not exist and
    /// logins that are refused.
    pub fn open(
        config: MetastoreConfig,
        metastore_file: Option<&Path>,
        refresh_interval: Duration,
    ) -> Result<Self> {
        let mut datastores = into_config_definitions(config.datastores, Origin::ConfigFile);
        let mut users = into_config_definitions(config.users, Origin::ConfigFile);
        if let Some(path) = metastore_file {
            let disk = parse_config(&std::fs::read_to_string(path)?)?;
            merge_definitions(
                &mut datastores,
                into_config_definitions(disk.datastores, Origin::MetastoreFile),
            )
            .map_err(|names| Error::ConflictingDatastores { names })?;
            merge_definitions(
                &mut users,
                into_config_definitions(disk.users, Origin::MetastoreFile),
            )
            .map_err(|names| Error::ConflictingUsers { names })?;
        }
        Self::from_definitions(datastores, users, refresh_interval)
    }

    /// Validate the merged definitions: exactly one default datastore, and users
    /// whose verifiers decode.
    fn from_definitions(
        datastores: HashMap<String, ConfigDefinition<DatastoreConfig>>,
        users: HashMap<String, ConfigDefinition<UserConfig>>,
        refresh_interval: Duration,
    ) -> Result<Self> {
        // The default datastore is the one flagged `default = true`, not one with
        // a reserved name. Exactly one is required: it is the current database.
        let mut defaults: Vec<String> = datastores
            .iter()
            .filter(|(_, definition)| definition.value.is_default)
            .map(|(name, _)| name.clone())
            .collect();
        if defaults.len() > 1 {
            defaults.sort();
            return Err(Error::MultipleDefaults(defaults));
        }
        let default_name = defaults.pop().ok_or(Error::MissingDefault)?;
        let mut user_auth = users
            .into_iter()
            .map(|(name, definition)| {
                let auth = definition
                    .value
                    .into_auth()
                    .map_err(|message| Error::User {
                        name: name.clone(),
                        message,
                    })?;
                Ok((
                    name,
                    ConfigDefinition {
                        origin: definition.origin,
                        value: auth,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        // The built-in user is always present.
        user_auth
            .entry(DEFAULT_USER_NAME.to_string())
            .or_insert_with(|| ConfigDefinition {
                origin: Origin::BuiltIn,
                value: UserAuth::Trust,
            });
        Ok(Self {
            datastore_configs: datastores,
            default_name,
            user_auth,
            refresh_interval,
        })
    }

    fn build_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>> {
        self.datastore_configs
            .iter()
            .map(|(name, definition)| {
                let config = &definition.value;
                let store = config.open_store(name)?;
                let maintenance = MaintenanceConfig {
                    refresh_interval: self.refresh_interval,
                    compaction: config.compaction(),
                    vacuum: config.vacuum(),
                };
                let datastore: Arc<dyn Datastore> = match config.kind {
                    DatastoreKind::Delta => {
                        DeltaDatastore::from_store(store, dispatcher, Some(maintenance))?
                    }
                };
                Ok((name.clone(), datastore))
            })
            .collect()
    }
}

/// Where a config definition came from.
///
/// The config file is the operator's, so the server never rewrites it and
/// runtime changes to what it defines are refused; the metastore file is the
/// server's own to rewrite. Recording the fact rather than the consequence
/// leaves room for the several rules that read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The config file's `metastore` section.
    ConfigFile,
    /// The file named by `--metastore-file`.
    MetastoreFile,
    /// Neither file: the built-in trusted [`DEFAULT_USER_NAME`] user, served
    /// whenever no file defined one by that name.
    BuiltIn,
}

/// One entry as a file defines it, and which file that was.
struct ConfigDefinition<T> {
    origin: Origin,
    value: T,
}

/// Note every entry of one file's map as defined by that file.
fn into_config_definitions<T>(
    entries: HashMap<String, T>,
    origin: Origin,
) -> HashMap<String, ConfigDefinition<T>> {
    entries
        .into_iter()
        .map(|(name, value)| (name, ConfigDefinition { origin, value }))
        .collect()
}

/// Fold `incoming` into `existing`. A name defined on both sides comes back
/// instead of being merged: the two entries may disagree about where a datastore
/// lives or how a user authenticates, and whichever one lost would do so
/// invisibly.
fn merge_definitions<T>(
    existing: &mut HashMap<String, ConfigDefinition<T>>,
    incoming: HashMap<String, ConfigDefinition<T>>,
) -> std::result::Result<(), Vec<String>> {
    let conflicting = find_conflicting_names(existing, &incoming);
    if !conflicting.is_empty() {
        return Err(conflicting);
    }
    existing.extend(incoming);
    Ok(())
}

impl Metastore for DiskMetastore {
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

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        self.user_auth
            .get(username)
            .map(|definition| definition.value.clone())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading the metastore file: {source}")]
    Read {
        #[from]
        source: std::io::Error,
    },
    #[error("parsing the metastore configuration: {source}")]
    Parse {
        #[from]
        source: serde_yaml_ng::Error,
    },
    #[error(
        "datastores defined both in the config file's `metastore` section and in the metastore file: {}",
        .names.join(", ")
    )]
    ConflictingDatastores { names: Vec<String> },
    #[error(
        "users defined both in the config file's `metastore` section and in the metastore file: {}",
        .names.join(", ")
    )]
    ConflictingUsers { names: Vec<String> },
    #[error("datastore `{name}`: {message}")]
    Datastore { name: String, message: String },
    #[error("user `{name}`: {message}")]
    User { name: String, message: String },
    #[error(
        "no default datastore is configured; mark exactly one entry under `datastores` with `default: true`"
    )]
    MissingDefault,
    #[error(
        "multiple datastores are marked `default = true` ({}); exactly one may be", .0.join(", ")
    )]
    MultipleDefaults(Vec<String>),
    #[error(transparent)]
    Store(#[from] datastore_delta::store::StoreError),
    #[error(transparent)]
    Delta(#[from] datastore_delta::Error),
}

/// The datastores and users of a metastore, as written: the config file's
/// `metastore` section, and the standalone metastore file, share this shape.
///
/// Unknown keys are rejected: a misspelled `users` section would otherwise be
/// dropped in silence and unexpectedly select the built-in trusted `pivot` user.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetastoreConfig {
    #[serde(default)]
    datastores: HashMap<String, DatastoreConfig>,
    #[serde(default)]
    users: HashMap<String, UserConfig>,
}

/// The names `incoming` would overwrite in `existing`, sorted so the error lists
/// them in the same order every run.
fn find_conflicting_names<T, U>(
    existing: &HashMap<String, T>,
    incoming: &HashMap<String, U>,
) -> Vec<String> {
    let mut names: Vec<String> = incoming
        .keys()
        .filter(|name| existing.contains_key(*name))
        .cloned()
        .collect();
    names.sort();
    names
}

/// Parse one metastore configuration from YAML text, using `path` in errors.
fn parse_config(text: &str) -> Result<MetastoreConfig> {
    Ok(serde_yaml_ng::from_str(text)?)
}

/// One user and the authentication method nested inside it. Keeping the user as
/// a struct leaves room for later user-level fields such as roles.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UserConfig {
    auth: UserAuthConfig,
}

impl UserConfig {
    fn into_auth(self) -> std::result::Result<UserAuth, String> {
        match self.auth {
            UserAuthConfig::Trust {} => Ok(UserAuth::Trust),
            UserAuthConfig::ScramSha256 { verifier } => {
                parse_scram_verifier(&verifier).map(UserAuth::ScramSha256)
            }
        }
    }
}

/// YAML representation of a user's one authentication method. Variant-specific
/// fields live inside the variant, so `trust` cannot carry a verifier and SCRAM
/// cannot omit one.
#[derive(Deserialize)]
#[serde(tag = "method", deny_unknown_fields)]
enum UserAuthConfig {
    #[serde(rename = "trust")]
    Trust {},
    #[serde(rename = "scram-sha-256")]
    ScramSha256 { verifier: String },
}

/// One datastore's configuration. `kind` is the datastore format; the storage
/// backend (local filesystem vs S3) is inferred from `location`'s scheme, and
/// the S3 credential fields apply only when `location` is an `s3://` URI.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatastoreConfig {
    kind: DatastoreKind,
    location: String,
    /// Marks this datastore as the default: the current database, the target of
    /// unqualified table names and DDL. Exactly one datastore must set it.
    #[serde(rename = "default", default)]
    is_default: bool,
    /// Run this datastore's own background compaction. On by default; the
    /// `compact_*` tuning fields apply only when it is on. Compaction rewrites a
    /// table's small Parquet files, so run it in only one process per datastore
    /// (set `compact: false` on the others).
    #[serde(default = "default_true")]
    compact: bool,
    /// Per-table byte threshold compaction merges small files up to (a size such
    /// as `128m` or `1g`). Defaults to [`DEFAULT_COMPACT_BYTES`].
    compact_bytes: Option<ByteSize>,
    /// Count trigger for a low-traffic partition's small files (merge once this
    /// many accumulate, even below `compact_bytes`). Defaults to
    /// [`DEFAULT_MIN_FILES_TO_MERGE`].
    compact_min_files: Option<usize>,
    /// Run this datastore's own background vacuum. On by default: the vacuumer
    /// deletes unreferenced data files and superseded commit JSONs past their
    /// retention. Set `vacuum: false` on a read-only server, or where another
    /// process owns physical cleanup.
    #[serde(default = "default_true")]
    vacuum: bool,
    region: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    endpoint: Option<String>,
    /// A Google service-account (or authorized-user) JSON key file, for a
    /// `gs://` location. Omit to resolve credentials from the ambient
    /// Application Default Credentials chain instead.
    credentials_file: Option<String>,
}

/// Serde default for the `compact` and `vacuum` toggles: both maintenance loops
/// run unless a datastore explicitly turns them off.
fn default_true() -> bool {
    true
}

impl DatastoreConfig {
    /// This datastore's compaction settings, or `None` when `compact` is off.
    /// Fills the tuning fields' defaults.
    fn compaction(&self) -> Option<CompactionConfig> {
        if !self.compact {
            return None;
        }
        Some(CompactionConfig {
            target_bytes: self
                .compact_bytes
                .map(ByteSize::as_bytes)
                .unwrap_or(DEFAULT_COMPACT_BYTES),
            min_files: self.compact_min_files.unwrap_or(DEFAULT_MIN_FILES_TO_MERGE),
            poll_interval: DEFAULT_COMPACT_POLL,
        })
    }

    /// This datastore's vacuum settings, or `None` when `vacuum` is off. On by
    /// default; each table's `delta.deletedFileRetentionDuration` governs the
    /// deletion window.
    fn vacuum(&self) -> Option<VacuumConfig> {
        if !self.vacuum {
            return None;
        }
        Some(VacuumConfig {
            poll_interval: DEFAULT_VACUUM_POLL,
        })
    }
}

/// The datastore format. Only [`Delta`](Self::Delta) is supported today; adding
/// another (Iceberg, ...) is a new variant plus its arm in
/// [`build_datastores`](DiskMetastore::build_datastores).
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DatastoreKind {
    Delta,
}

impl DatastoreConfig {
    /// Build this datastore's object store. The backend is chosen from
    /// `location`: an `s3://` (or `s3a://`) URI opens an S3 store with the
    /// datastore's inline credentials (`region`, `access_key_id`,
    /// `secret_access_key` are required); anything else is a local path (an
    /// optional `file://` scheme is stripped).
    fn open_store(&self, name: &str) -> Result<Arc<dyn ObjectStore>> {
        if is_s3_location(&self.location) {
            let credentials = S3Credentials {
                region: require(name, &self.region, "region")?,
                access_key: require(name, &self.access_key_id, "access_key_id")?,
                secret_key: require(name, &self.secret_access_key, "secret_access_key")?,
                endpoint: self.endpoint.clone(),
            };
            Ok(Arc::new(S3Store::with_credentials(
                &self.location,
                credentials,
            )?))
        } else if is_gcs_location(&self.location) {
            Ok(Arc::new(match &self.credentials_file {
                Some(path) => GcsStore::with_credentials_file(&self.location, path)?,
                None => GcsStore::from_uri(&self.location)?,
            }))
        } else {
            let path = self
                .location
                .strip_prefix("file://")
                .unwrap_or(&self.location);
            Ok(Arc::new(LocalStore::new(path)))
        }
    }
}

/// Whether a location is an S3 URI (`s3://` / `s3a://`). The storage backend is
/// inferred from the scheme, not configured.
fn is_s3_location(location: &str) -> bool {
    location.starts_with("s3://") || location.starts_with("s3a://")
}

/// Whether a location is a Google Cloud Storage URI (`gs://`); a location that
/// is neither this nor [`is_s3_location`] is a local path.
fn is_gcs_location(location: &str) -> bool {
    location.starts_with("gs://")
}

/// Return a required credential, pointing at the environment alternative when
/// an inline value is absent.
fn require(name: &str, value: &Option<String>, field: &str) -> Result<String> {
    value.clone().ok_or_else(|| Error::Datastore {
        name: name.to_string(),
        message: format!("`{field}` is required"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::DEFAULT_DATASTORE_NAME;
    use datastore_delta::DEFAULT_REFRESH_INTERVAL;
    use metastore::ScramVerifier;
    use std::io::Write;
    use std::time::Duration;

    /// Open a `metastore` section on its own, at whatever refresh cadence the
    /// tests that do not exercise it would rather not spell out.
    fn from_yaml(text: &str) -> Result<DiskMetastore> {
        DiskMetastore::open(parse_config(text)?, None, DEFAULT_REFRESH_INTERVAL)
    }

    /// Write `text` to a metastore file of its own, kept alive by the returned
    /// handle.
    fn metastore_file(text: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        file
    }

    #[test]
    fn compaction_is_per_datastore() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
    compact_bytes: 128m
  warm:
    kind: delta
    location: /tmp/warm
    compact: false
"#;

        let store = from_yaml(yaml).unwrap();

        let hot = store.datastore_configs["hot"].value.compaction();
        let warm = store.datastore_configs["warm"].value.compaction();
        // Compaction is on by default (hot omits `compact`), and off only where a
        // datastore turns it off explicitly (warm).
        assert_eq!(hot.unwrap().target_bytes, 128 * 1024 * 1024);
        assert!(warm.is_none());
    }

    #[test]
    fn every_datastore_shares_the_refresh_interval_the_server_configured() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
"#;

        let store = DiskMetastore::open(
            parse_config(yaml).unwrap(),
            None,
            Duration::from_millis(500),
        )
        .unwrap();

        assert_eq!(store.refresh_interval, Duration::from_millis(500));
    }

    #[test]
    fn a_setting_of_the_server_section_is_not_accepted_here() {
        let yaml = r#"
refresh_interval: 500ms
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
"#;

        let error = from_yaml(yaml).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn a_misspelled_datastore_setting_is_rejected_rather_than_ignored() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
    compact_byte: 128m
"#;

        let error = from_yaml(yaml).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn missing_default_datastore_is_rejected() {
        let yaml = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
"#;

        let error = from_yaml(yaml)
            .err()
            .expect("a metastore without a default should be rejected");

        assert!(matches!(&error, Error::MissingDefault));
        assert_eq!(
            error.to_string(),
            "no default datastore is configured; mark exactly one entry under `datastores` with `default: true`"
        );
    }

    #[test]
    fn unknown_datastore_kind_is_rejected() {
        let yaml = r#"
datastores:
  default:
    kind: iceberg
    location: /tmp/default
"#;

        let result = from_yaml(yaml);

        assert!(matches!(result, Err(Error::Parse { .. })));
    }

    #[test]
    fn multiple_defaults_are_rejected() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  warm:
    kind: delta
    location: /tmp/warm
    default: true
"#;

        let result = from_yaml(yaml);

        assert!(matches!(result, Err(Error::MultipleDefaults(names)) if names == ["hot", "warm"]));
    }

    #[test]
    fn default_is_the_flagged_datastore_whatever_its_name() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  warm:
    kind: delta
    location: /tmp/warm
"#;

        let store = from_yaml(yaml).unwrap();

        assert_eq!(store.default_datastore_name(), "hot");
    }

    #[test]
    fn s3_location_without_keys_is_rejected() {
        let yaml = r#"
datastores:
  default:
    kind: delta
    location: /tmp/default
    default: true
  warm:
    kind: delta
    location: s3://bucket/prefix
"#;

        let store = from_yaml(yaml).unwrap();
        let err = store.datastore_configs["warm"]
            .value
            .open_store("warm")
            .unwrap_err();

        assert!(matches!(err, Error::Datastore { .. }));
    }

    #[test]
    fn local_s3_and_gcs_locations_open_stores() {
        let yaml = r#"
datastores:
  default:
    kind: delta
    location: /tmp/default
    default: true
  warm:
    kind: delta
    location: s3://bucket/prefix
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
  cold:
    kind: delta
    location: gs://bucket/prefix
"#;

        let store = from_yaml(yaml).unwrap();

        assert!(
            store.datastore_configs[DEFAULT_DATASTORE_NAME]
                .value
                .open_store(DEFAULT_DATASTORE_NAME)
                .is_ok()
        );
        assert!(
            store.datastore_configs["warm"]
                .value
                .open_store("warm")
                .is_ok()
        );
        // A GCS store resolves its credentials lazily (at the first request),
        // so opening one needs no ambient Google credentials.
        let cold = store.datastore_configs["cold"]
            .value
            .open_store("cold")
            .unwrap();
        assert_eq!(cold.location_uri(), "gs://bucket/prefix");
    }

    /// A `metastore` section, and a metastore file, opened as one metastore.
    fn open_merged(section: &str, disk: &str) -> Result<DiskMetastore> {
        let file = metastore_file(disk);
        DiskMetastore::open(
            parse_config(section).unwrap(),
            Some(file.path()),
            DEFAULT_REFRESH_INTERVAL,
        )
    }

    #[test]
    fn datastores_and_users_of_both_files_are_served_together() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
users:
  reader:
    auth:
      method: trust
"#;
        let disk = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
users:
  writer:
    auth:
      method: trust
"#;

        let store = open_merged(section, disk).unwrap();

        assert_eq!(store.datastore_configs.len(), 2);
        assert_eq!(store.datastore_configs["warm"].value.location, "/tmp/warm");
        assert_eq!(store.default_datastore_name(), "hot");
        assert!(matches!(store.user_auth("reader"), Some(UserAuth::Trust)));
        assert!(matches!(store.user_auth("writer"), Some(UserAuth::Trust)));
    }

    #[test]
    fn every_datastore_and_user_knows_which_file_defined_it() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
users:
  reader:
    auth:
      method: trust
"#;
        let disk = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
users:
  writer:
    auth:
      method: trust
"#;

        let store = open_merged(section, disk).unwrap();

        assert_eq!(store.datastore_configs["hot"].origin, Origin::ConfigFile);
        assert_eq!(
            store.datastore_configs["warm"].origin,
            Origin::MetastoreFile
        );
        assert_eq!(store.user_auth["reader"].origin, Origin::ConfigFile);
        assert_eq!(store.user_auth["writer"].origin, Origin::MetastoreFile);
    }

    #[test]
    fn the_user_no_file_defined_is_marked_as_built_in() {
        let store = from_yaml_with("").unwrap();

        assert_eq!(store.user_auth[DEFAULT_USER_NAME].origin, Origin::BuiltIn);
    }

    #[test]
    fn the_default_datastore_may_be_the_one_on_disk() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
"#;
        let disk = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
    default: true
"#;

        let store = open_merged(section, disk).unwrap();

        assert_eq!(store.default_datastore_name(), "warm");
    }

    #[test]
    fn a_default_in_each_file_is_rejected() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
"#;
        let disk = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
    default: true
"#;

        let error = open_merged(section, disk).err().unwrap();

        assert!(
            matches!(&error, Error::MultipleDefaults(names) if *names == ["hot", "warm"]),
            "{error}"
        );
    }

    #[test]
    fn a_datastore_defined_in_both_files_is_rejected_rather_than_shadowed() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  warm:
    kind: delta
    location: /tmp/warm
"#;
        let disk = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/elsewhere
"#;

        let error = open_merged(section, disk).err().unwrap();

        assert!(
            matches!(&error, Error::ConflictingDatastores { names, .. } if *names == ["warm"]),
            "{error}"
        );
    }

    #[test]
    fn a_user_defined_in_both_files_is_rejected_rather_than_shadowed() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
users:
  analytics:
    auth:
      method: trust
"#;
        let disk = r#"
users:
  analytics:
    auth:
      method: trust
"#;

        let error = open_merged(section, disk).err().unwrap();

        assert!(
            matches!(&error, Error::ConflictingUsers { names, .. } if *names == ["analytics"]),
            "{error}"
        );
    }

    #[test]
    fn a_metastore_file_that_is_not_there_stops_startup() {
        let section = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
"#;

        let error = DiskMetastore::open(
            parse_config(section).unwrap(),
            Some(Path::new("/nonexistent/metastore.yaml")),
            DEFAULT_REFRESH_INTERVAL,
        )
        .err()
        .unwrap();

        assert!(matches!(&error, Error::Read { .. }), "{error}");
    }

    /// A metastore with a default datastore and whatever `users` adds.
    fn from_yaml_with(users: &str) -> Result<DiskMetastore> {
        let yaml = format!(
            "datastores:\n  default:\n    kind: delta\n    location: /tmp/default\n    \
             default: true\n{users}"
        );
        from_yaml(&yaml)
    }

    fn verifier_for(password: &str) -> String {
        metastore::format_scram_verifier(&ScramVerifier {
            salt: vec![1; 16],
            salted_password: password
                .as_bytes()
                .iter()
                .cycle()
                .take(32)
                .copied()
                .collect(),
        })
    }

    #[test]
    fn a_file_without_users_provides_the_builtin_pivot_user() {
        for users in ["", "users: {}\n"] {
            let store = from_yaml_with(users).unwrap();

            assert!(matches!(
                store.user_auth(DEFAULT_USER_NAME),
                Some(UserAuth::Trust)
            ));
            assert!(store.user_auth("analytics").is_none());
        }
    }

    #[test]
    fn the_builtin_user_is_served_alongside_the_users_a_file_defines() {
        let users = r#"
users:
  reader:
    auth:
      method: trust
  writer:
    auth:
      method: trust
"#;

        let store = from_yaml_with(users).unwrap();

        assert!(matches!(
            store.user_auth(DEFAULT_USER_NAME),
            Some(UserAuth::Trust)
        ));
        assert_eq!(store.user_auth[DEFAULT_USER_NAME].origin, Origin::BuiltIn);
        assert!(store.user_auth("reader").is_some());
        assert!(store.user_auth("writer").is_some());
    }

    #[test]
    fn a_file_that_defines_the_builtin_user_decides_how_it_authenticates() {
        let users = format!(
            "users:\n  {DEFAULT_USER_NAME}:\n    auth:\n      method: scram-sha-256\n      \
             verifier: \"{}\"\n",
            verifier_for("a")
        );

        let store = from_yaml_with(&users).unwrap();

        let Some(UserAuth::ScramSha256(verifier)) = store.user_auth(DEFAULT_USER_NAME) else {
            panic!("a configured `{DEFAULT_USER_NAME}` should keep its own method");
        };
        assert_eq!(verifier.salted_password, [b'a'; 32]);
        assert_eq!(
            store.user_auth[DEFAULT_USER_NAME].origin,
            Origin::ConfigFile
        );
    }

    #[test]
    fn each_scram_user_keeps_its_own_verifier() {
        let users = format!(
            "users:\n  analytics:\n    auth:\n      method: scram-sha-256\n      \
             verifier: \"{}\"\n  ingest:\n    auth:\n      method: scram-sha-256\n      \
             verifier: \"{}\"\n",
            verifier_for("a"),
            verifier_for("b")
        );

        let store = from_yaml_with(&users).unwrap();

        let Some(UserAuth::ScramSha256(analytics)) = store.user_auth("analytics") else {
            panic!("analytics should use SCRAM-SHA-256");
        };
        assert_eq!(analytics.salted_password, [b'a'; 32]);
        let Some(UserAuth::ScramSha256(ingest)) = store.user_auth("ingest") else {
            panic!("ingest should use SCRAM-SHA-256");
        };
        assert_eq!(ingest.salted_password, [b'b'; 32]);
        assert!(store.user_auth("nobody").is_none());
    }

    #[test]
    fn a_user_can_explicitly_be_trusted_without_a_password() {
        let users = r#"
users:
  reader:
    auth:
      method: trust
"#;

        let store = from_yaml_with(users).unwrap();

        assert!(matches!(store.user_auth("reader"), Some(UserAuth::Trust)));
    }

    #[test]
    fn trust_rejects_a_verifier_instead_of_ignoring_it() {
        let users = r#"
users:
  reader:
    auth:
      method: trust
      verifier: irrelevant
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn scram_requires_a_verifier() {
        let users = r#"
users:
  analytics:
    auth:
      method: scram-sha-256
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn an_empty_user_is_not_silently_trusted() {
        let users = r#"
users:
  analytics: {}
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn a_misspelled_user_section_is_rejected_rather_than_dropped() {
        let users = r#"
user:
  analytics:
    auth:
      method: trust
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn an_unknown_authentication_method_is_rejected() {
        let users = r#"
users:
  analytics:
    auth:
      method: kerberos
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(matches!(error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn a_malformed_verifier_is_rejected_at_parse_time() {
        let users = r#"
users:
  analytics:
    auth:
      method: scram-sha-256
      verifier: hunter2
"#;

        let error = from_yaml_with(users).err().unwrap();

        assert!(
            matches!(&error, Error::User { name, .. } if name == "analytics"),
            "{error}"
        );
    }
}
