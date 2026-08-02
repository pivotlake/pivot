//! A YAML-backed [`metastore::Metastore`] provider.
//!
//! `kind` is the datastore format (only `delta` today). The storage backend is
//! chosen from `location`: a plain path (or `file://`) opens a local store, an
//! `s3://` URI opens an S3 store. The S3 credential fields apply only to an
//! `s3://` location.
//!
//! Exactly one datastore must set `default = true`; it becomes the current
//! database, so unqualified table names and DDL resolve against it. Its name is
//! free (it need not be called `default`).
//!
//! ```yaml
//! datastores:
//!   hot:
//!     kind: delta
//!     location: /var/lib/pivot       # local path -> local store
//!     default: true                  # the current database
//!   warm:
//!     kind: delta
//!     location: s3://my-bucket/pivot/
//!     region: us-east-1              # required for s3://
//!     access_key_id: AKIA...         # required for s3://
//!     secret_access_key: "..."
//!     # endpoint: http://localhost:9000
//!     # compact: true                # optional
//! ```
//!
//! Each entry under `users` names a user that may authenticate to the
//! PostgreSQL endpoint. Its nested `auth` selects one authentication method. A
//! user may explicitly be trusted without a password:
//!
//! ```yaml
//! users:
//!   reader:
//!     auth:
//!       method: trust
//! ```
//!
//! Or it may authenticate with a SCRAM-SHA-256 verifier derived from its
//! password:
//!
//! ```yaml
//! users:
//!   analytics:
//!     auth:
//!       method: scram-sha-256
//!       verifier: "pivot-scram-sha-256$4096:cGVwcGVy...$Zm9vYmFy..."
//! ```
//!
//! When `users` is omitted or empty, the provider supplies one built-in trusted
//! user named `pivot`. Defining any users replaces that default with the
//! configured allowlist.
//!
//! An S3 datastore's credentials are inline, so the file holds secrets and should
//! be readable only by the PivotDB process. A verifier is not a password (the
//! password cannot be recovered from it), but it is still worth the same care.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use catalog::Datastore;
use datastore_delta::store::{LocalStore, ObjectStore, S3Credentials, S3Store};
use datastore_delta::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
    DeltaDatastore, MaintenanceConfig,
};
use dispatch::DataFlowDispatcher;
use metastore::{DEFAULT_USER_NAME, Metastore, UserAuth, parse_scram_verifier};
use serde::Deserialize;

/// A metastore backed by a parsed YAML file.
///
/// Configuration is parsed and structurally validated by [`open`](Self::open).
/// Object stores and their Delta datastores are opened when
/// [`Metastore::open_datastores`] is called.
pub struct YamlMetastore {
    datastore_configs: HashMap<String, DatastoreConfig>,
    default_name: String,
    /// Every user's authentication method, keyed by user name. A file with no
    /// users receives the built-in trusted `pivot` user. Verifiers are decoded
    /// once, by [`from_yaml`](Self::from_yaml), so a malformed one fails startup
    /// rather than a login.
    user_auth: HashMap<String, UserAuth>,
    /// How often every datastore this metastore opens refreshes its table set
    /// from the store. Global (all datastores share the cadence); compaction, in
    /// contrast, is configured per datastore.
    refresh_interval: Duration,
}

impl YamlMetastore {
    /// Read and parse the metastore file at `path`, applying `refresh_interval`
    /// to every datastore it opens.
    pub fn open(path: impl AsRef<Path>, refresh_interval: Duration) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&text, &path.display().to_string(), refresh_interval)
    }

    /// Parse metastore configuration from YAML text, using `path` in errors.
    pub fn from_yaml(text: &str, path: &str, refresh_interval: Duration) -> Result<Self> {
        let file: MetastoreFile = serde_yaml_ng::from_str(text).map_err(|source| Error::Parse {
            path: path.to_string(),
            source,
        })?;
        // The default datastore is the one flagged `default = true`, not one with
        // a reserved name. Exactly one is required: it is the current database.
        let mut defaults: Vec<String> = file
            .datastores
            .iter()
            .filter(|(_, config)| config.is_default)
            .map(|(name, _)| name.clone())
            .collect();
        if defaults.len() > 1 {
            defaults.sort();
            return Err(Error::MultipleDefaults(defaults));
        }
        let default_name = defaults.pop().ok_or(Error::MissingDefault)?;
        let mut user_auth = file
            .users
            .into_iter()
            .map(|(name, config)| {
                let auth = config.into_auth().map_err(|message| Error::User {
                    name: name.clone(),
                    message,
                })?;
                Ok((name, auth))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        if user_auth.is_empty() {
            user_auth.insert(DEFAULT_USER_NAME.to_string(), UserAuth::Trust);
        }
        Ok(Self {
            datastore_configs: file.datastores,
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
            .map(|(name, config)| {
                let store = config.open_store(name)?;
                let maintenance = MaintenanceConfig {
                    refresh_interval: self.refresh_interval,
                    compaction: config.compaction(name)?,
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

impl Metastore for YamlMetastore {
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
        self.user_auth.get(username).cloned()
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading metastore file `{path}`: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing metastore file `{path}`: {source}")]
    Parse {
        path: String,
        source: serde_yaml_ng::Error,
    },
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

/// Unknown top-level keys are rejected: a misspelled `users` section would
/// otherwise be dropped in silence and unexpectedly select the built-in
/// trusted `pivot` user.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetastoreFile {
    #[serde(default)]
    datastores: HashMap<String, DatastoreConfig>,
    #[serde(default)]
    users: HashMap<String, UserConfig>,
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
struct DatastoreConfig {
    kind: DatastoreKind,
    location: String,
    /// Marks this datastore as the default: the current database, the target of
    /// unqualified table names and DDL. Exactly one datastore must set it.
    #[serde(rename = "default", default)]
    is_default: bool,
    /// Run this datastore's own background compaction. Off by default; the
    /// `compact_*` tuning fields apply only when it is on. Compaction rewrites a
    /// table's small Parquet files, so enable it in only one process per
    /// datastore.
    #[serde(default)]
    compact: bool,
    /// Per-table byte threshold compaction merges small files up to (a size such
    /// as `128m` or `1g`). Defaults to [`DEFAULT_COMPACT_BYTES`].
    compact_bytes: Option<String>,
    /// Count trigger for a low-traffic partition's small files (merge once this
    /// many accumulate, even below `compact_bytes`). Defaults to
    /// [`DEFAULT_MIN_FILES_TO_MERGE`].
    compact_min_files: Option<usize>,
    region: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    endpoint: Option<String>,
}

impl DatastoreConfig {
    /// This datastore's compaction settings, or `None` when `compact` is off.
    /// Parses `compact_bytes` and fills the tuning fields' defaults.
    fn compaction(&self, name: &str) -> Result<Option<CompactionConfig>> {
        if !self.compact {
            return Ok(None);
        }
        let target_bytes = match &self.compact_bytes {
            Some(size) => parse_byte_size(size).map_err(|message| Error::Datastore {
                name: name.to_string(),
                message,
            })?,
            None => DEFAULT_COMPACT_BYTES,
        };
        Ok(Some(CompactionConfig {
            target_bytes,
            min_files: self.compact_min_files.unwrap_or(DEFAULT_MIN_FILES_TO_MERGE),
            poll_interval: DEFAULT_COMPACT_POLL,
        }))
    }
}

/// Parse a human-readable byte size such as `128m`, `1g`, or `4096` into a byte
/// count. Accepts an optional base-1024 suffix (`k`/`m`/`g`/`t`, each also with a
/// trailing `b`), case-insensitive; a bare number is bytes.
fn parse_byte_size(input: &str) -> std::result::Result<u64, String> {
    let trimmed = input.trim();
    let digits_end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (number, suffix) = trimmed.split_at(digits_end);
    let value: u64 = number
        .parse()
        .map_err(|_| format!("`{input}` is not a valid size"))?;
    let multiplier: u64 = match suffix.trim().to_ascii_lowercase().as_str() {
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

/// The datastore format. Only [`Delta`](Self::Delta) is supported today; adding
/// another (Iceberg, ...) is a new variant plus its arm in
/// [`build_datastores`](YamlMetastore::build_datastores).
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
        } else {
            let path = self
                .location
                .strip_prefix("file://")
                .unwrap_or(&self.location);
            Ok(Arc::new(LocalStore::new(path)))
        }
    }
}

/// Whether a location is an S3 URI (`s3://` / `s3a://`); otherwise it is a local
/// path. The storage backend is inferred from the scheme, not configured.
fn is_s3_location(location: &str) -> bool {
    location.starts_with("s3://") || location.starts_with("s3a://")
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
    use metastore::ScramVerifier;
    use std::time::Duration;

    #[test]
    fn compaction_is_per_datastore() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
    compact: true
    compact_bytes: 128m
  warm:
    kind: delta
    location: /tmp/warm
"#;

        let store = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30)).unwrap();

        let hot = store.datastore_configs["hot"].compaction("hot").unwrap();
        let warm = store.datastore_configs["warm"].compaction("warm").unwrap();
        assert_eq!(hot.unwrap().target_bytes, 128 * 1024 * 1024);
        assert!(warm.is_none());
    }

    #[test]
    fn missing_default_datastore_is_rejected() {
        let yaml = r#"
datastores:
  warm:
    kind: delta
    location: /tmp/warm
"#;

        let error = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30))
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

        let result = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30));

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

        let result = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30));

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

        let store = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30)).unwrap();

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

        let store = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30)).unwrap();
        let err = store.datastore_configs["warm"]
            .open_store("warm")
            .unwrap_err();

        assert!(matches!(err, Error::Datastore { .. }));
    }

    #[test]
    fn local_and_s3_locations_open_stores() {
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
"#;

        let store = YamlMetastore::from_yaml(yaml, "test", Duration::from_secs(30)).unwrap();

        assert!(
            store.datastore_configs[DEFAULT_DATASTORE_NAME]
                .open_store(DEFAULT_DATASTORE_NAME)
                .is_ok()
        );
        assert!(store.datastore_configs["warm"].open_store("warm").is_ok());
    }

    /// A metastore with a default datastore and whatever `users` adds.
    fn from_yaml_with(users: &str) -> Result<YamlMetastore> {
        let yaml = format!(
            "datastores:\n  default:\n    kind: delta\n    location: /tmp/default\n    \
             default: true\n{users}"
        );
        YamlMetastore::from_yaml(&yaml, "test", Duration::from_secs(30))
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
