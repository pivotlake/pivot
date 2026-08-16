//! A disk-backed [`catalog::metastore::Metastore`] provider.
//!
//! This crate owns the `metastore` section of PivotDB's config file: the
//! datastores to serve, the secrets they are opened with, and the users that
//! may connect. The server reads the file, keeps its own `server` section, and
//! hands this section over as a [`MetastoreConfig`]. The scalars both sections
//! are written with, [`ByteSize`] and [`Interval`], are defined here too.
//!
//! A section of the same shape may also live in a file of its own, which the
//! server names with `--metastore-file` and this crate reads and merges into
//! the config file's section ([`DiskMetastore::open`]). The two are peers: a
//! datastore, a secret or a user may be written in either, and the rules below
//! apply to the merged whole. A name written in both files stops startup rather
//! than one silently winning, since the two entries are free to disagree about
//! location or credentials.
//!
//! Each file's definitions are held as a section of their own, because the two
//! files are not interchangeable once the server is running: the config file
//! is the operator's, and the metastore file is the server's own to rewrite
//! (which is exactly serialising its section back out).
//!
//! `kind` is the datastore format (only `delta` today). The storage backend is
//! chosen from `location`: a plain path (or `file://`) opens a local store, an
//! `s3://` URI opens an S3 store, a `gs://` URI opens a Google Cloud Storage
//! store. A datastore carries no credentials of its own; they come from the
//! `secrets` section below.
//!
//! Exactly one datastore must set `default = true`; it becomes the current
//! database, so unqualified table names and DDL resolve against it. Its name is
//! free (it need not be called `default`).
//!
//! Each entry under `secrets` holds the credentials for the paths its `scope`
//! covers, so a bucket's keys are written once however many datastores sit in
//! it. `type` names the backend (`s3` or `gcs`) and carries that backend's
//! fields. The most specific scope covering a location authenticates it; two
//! secrets may not claim the same scope, so which one that is never depends on
//! the order they were written in. A secret with no `scope` covers every
//! location of its type.
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
//!       # compact: true              # optional
//!     cold:
//!       kind: delta
//!       location: gs://my-bucket/pivot/
//!   secrets:
//!     my-bucket:
//!       type: s3
//!       scope: s3://my-bucket/       # omit to cover every s3:// location
//!       region: us-east-1
//!       access_key_id: AKIA...
//!       secret_access_key: "..."
//!       # endpoint: http://localhost:9000   # MinIO / S3-compatible
//!     google:
//!       type: gcs
//!       scope: gs://my-bucket/
//!       credentials_file: /etc/pivot/gcs-key.json
//! ```
//!
//! An `s3://` datastore needs a secret covering it: without one there is
//! nothing to sign its requests with, and startup stops. A `gs://` datastore
//! without one falls back to the ambient Application Default Credentials chain
//! (`GOOGLE_APPLICATION_CREDENTIALS`, the file
//! `gcloud auth application-default login` writes, or the workload identity of
//! the Google compute instance).
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
//! An S3 secret's keys are inline, so a file holding one should be readable only
//! by the PivotDB process (a GCS secret's key stays in the file
//! `credentials_file` points at). A verifier is not a password (the password
//! cannot be recovered from it), but it is still worth the same care.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use catalog::Datastore;
use catalog::metastore::{
    DEFAULT_USER_NAME, Metastore, SCRAM_ITERATIONS, SCRAM_SALT_LEN, ScramVerifier, UserAuth,
    format_scram_verifier, parse_scram_verifier,
};
use datastore_delta::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
    DEFAULT_VACUUM_POLL, DeltaDatastore, MaintenanceConfig, VacuumConfig,
    default_merge_target_bytes,
};
use dispatch::DataFlowDispatcher;
use object_storage::{
    ExternalStoreFactory, GcsStore, LocalStore, ObjectStore, S3Store, StoreError, StoreScheme,
    local_path,
};
use pgwire::api::auth::sasl::scram::gen_salted_password;
use serde::{Deserialize, Serialize};

mod secrets;
mod units;

use secrets::{SecretConfig, Secrets};
pub use units::{ByteSize, Interval};

/// A metastore backed by the config file's `metastore` section, and by the
/// standalone file merged into it.
///
/// The configuration is structurally validated by [`open`](Self::open). Object
/// stores and their Delta datastores are opened when
/// [`Metastore::open_datastores`] is called.
#[derive(Debug)]
pub struct DiskMetastore {
    /// The server config file's `metastore` section: the operator's, never
    /// rewritten, so it needs no lock.
    server_config: MetastoreConfig,
    /// What the metastore file holds: the server's own to rewrite.
    metastore_config: RwLock<MetastoreConfig>,
    /// The file named by `--metastore-file`.
    metastore_file: Option<PathBuf>,
    /// The secrets of both configs (config file + metastore file) as one set.
    secrets: Arc<Secrets>,
    default_name: String,
    /// How often every datastore this metastore opens refreshes its table set
    /// from the store. Global (all datastores share the cadence, which the
    /// server takes from its own section); compaction, in contrast, is
    /// configured per datastore.
    refresh_interval: Duration,
}

/// Opens external read-only stores with the same scoped-secret policy as a
/// configured datastore. The non-wildcard store root selects credentials and
/// is also the root addressed by the returned store.
#[derive(Debug)]
struct SecretExternalStoreFactory {
    secrets: Arc<Secrets>,
}

impl ExternalStoreFactory for SecretExternalStoreFactory {
    fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
        match StoreScheme::of(root_uri)? {
            StoreScheme::S3 => {
                let credentials = self.secrets.resolve_s3(root_uri).ok_or_else(|| {
                    StoreError::Config(format!(
                        "no S3 secret covers external store root `{root_uri}`"
                    ))
                })?;
                Ok(Arc::new(S3Store::with_credentials(root_uri, credentials)?))
            }
            StoreScheme::Gcs => Ok(Arc::new(match self.secrets.resolve_gcs(root_uri) {
                Some(credentials_file) => {
                    GcsStore::with_credentials_file(root_uri, credentials_file)?
                }
                None => GcsStore::with_default_credentials(root_uri)?,
            })),
            StoreScheme::Local => Ok(Arc::new(LocalStore::new(local_path(root_uri))?)),
        }
    }
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
        server_config: MetastoreConfig,
        metastore_file: Option<&Path>,
        refresh_interval: Duration,
    ) -> Result<Self> {
        let metastore_config = match metastore_file {
            Some(path) => read_config(path)?,
            None => MetastoreConfig::default(),
        };
        check_no_conflicts(&server_config, &metastore_config)?;
        let secrets = Arc::new(Secrets::build([
            &server_config.secrets,
            &metastore_config.secrets,
        ])?);
        let default_name = find_default_datastore(&server_config, &metastore_config)?;
        Ok(Self {
            server_config,
            metastore_config: RwLock::new(metastore_config),
            metastore_file: metastore_file.map(Path::to_path_buf),
            secrets,
            default_name,
            refresh_interval,
        })
    }

    /// Open external files with the same scoped-secret policy used by this
    /// metastore's configured datastores.
    pub fn external_store_factory(&self) -> Arc<dyn ExternalStoreFactory> {
        Arc::new(SecretExternalStoreFactory {
            secrets: self.secrets.clone(),
        })
    }

    /// Add a user to the metastore config, durably: refuse a name either
    /// config (or the built-in) already defines, rewrite the metastore file
    /// with the user added, then serve the result from memory. A password
    /// becomes a stored SCRAM verifier over a fresh random salt; no password
    /// means [`UserAuth::Trust`]. Requires a metastore file: the server
    /// config file is the operator's, and the server never rewrites it.
    ///
    /// The metastore file is this server's own, and nothing else writes it,
    /// so the config in memory is authoritative and the rewrite simply
    /// serialises it; the write lock serialises writers.
    pub fn create_user(&self, username: &str, password: Option<&str>) -> Result<()> {
        let Some(path) = &self.metastore_file else {
            return Err(Error::NoMetastoreFile);
        };
        if username == DEFAULT_USER_NAME || self.server_config.users.contains_key(username) {
            return Err(Error::UserExists {
                name: username.to_string(),
            });
        }
        let mut metastore_config = self.metastore_config.write().unwrap();
        if metastore_config.users.contains_key(username) {
            return Err(Error::UserExists {
                name: username.to_string(),
            });
        }
        let auth = derive_user_auth(password);
        metastore_config
            .users
            .insert(username.to_string(), UserConfig { auth });
        if let Err(error) = write_config(path, &metastore_config) {
            metastore_config.users.remove(username);
            return Err(error);
        }
        Ok(())
    }

    /// Remove a user from the metastore config, durably: rewrite the metastore
    /// file without the user, then serve the result from memory. Only a user
    /// the metastore file defines can be removed; the built-in `pivot` and any
    /// user in the server config file are refused (that file is the
    /// operator's, and the server never rewrites it). Mirrors
    /// [`create_user`](Self::create_user), including the write-lock protocol
    /// and the re-insert when the rewrite fails.
    pub fn drop_user(&self, username: &str) -> Result<()> {
        if username == DEFAULT_USER_NAME {
            return Err(Error::DropBuiltinUser);
        }
        if self.server_config.users.contains_key(username) {
            return Err(Error::DropConfigFileUser {
                name: username.to_string(),
            });
        }
        let mut metastore_config = self.metastore_config.write().unwrap();
        let Some(removed) = metastore_config.users.remove(username) else {
            return Err(Error::NoSuchUser {
                name: username.to_string(),
            });
        };
        // A user can only have entered the metastore config through the file,
        // so the file exists whenever the removal above found one.
        let path = self
            .metastore_file
            .as_ref()
            .expect("a metastore user was loaded from the metastore file");
        if let Err(error) = write_config(path, &metastore_config) {
            metastore_config.users.insert(username.to_string(), removed);
            return Err(error);
        }
        Ok(())
    }

    fn build_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>> {
        let metastore_config = self.metastore_config.read().unwrap();
        self.server_config
            .datastores
            .iter()
            .chain(metastore_config.datastores.iter())
            .map(|(name, config)| {
                let store = config.open_store(name, &self.secrets)?;
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

/// Reject a name defined in both configs. The two entries may disagree about
/// where a datastore lives or how a user authenticates, and whichever one
/// lost would do so invisibly.
fn check_no_conflicts(
    server_config: &MetastoreConfig,
    metastore_config: &MetastoreConfig,
) -> Result<()> {
    let names = find_conflicting_names(&server_config.datastores, &metastore_config.datastores);
    if !names.is_empty() {
        return Err(Error::ConflictingDatastores { names });
    }
    let names = find_conflicting_names(&server_config.users, &metastore_config.users);
    if !names.is_empty() {
        return Err(Error::ConflictingUsers { names });
    }
    let names = find_conflicting_names(&server_config.secrets, &metastore_config.secrets);
    if !names.is_empty() {
        return Err(Error::ConflictingSecrets { names });
    }
    Ok(())
}

/// The name of the one datastore across the two configs flagged
/// `default = true` (not one with a reserved name). Exactly one is required:
/// it is the current database.
fn find_default_datastore(
    server_config: &MetastoreConfig,
    metastore_config: &MetastoreConfig,
) -> Result<String> {
    let mut defaults: Vec<String> = server_config
        .datastores
        .iter()
        .chain(metastore_config.datastores.iter())
        .filter(|(_, datastore)| datastore.is_default)
        .map(|(name, _)| name.clone())
        .collect();
    if defaults.len() > 1 {
        defaults.sort();
        return Err(Error::MultipleDefaults(defaults));
    }
    defaults.pop().ok_or(Error::MissingDefault)
}

impl Metastore for DiskMetastore {
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        self.build_datastores(dispatcher)
            .map_err(|error| Box::new(error) as catalog::metastore::Error)
    }

    fn default_datastore_name(&self) -> &str {
        &self.default_name
    }

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        if let Some(user) = self.server_config.users.get(username) {
            return Some(user.auth.clone());
        }
        if let Some(user) = self.metastore_config.read().unwrap().users.get(username) {
            return Some(user.auth.clone());
        }
        // The built-in trusted user, served whenever no file defined one by
        // that name, so a server is reachable whatever else its users are.
        (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
    }

    fn create_user(
        &self,
        username: &str,
        password: Option<&str>,
    ) -> catalog::metastore::Result<()> {
        DiskMetastore::create_user(self, username, password)
            .map_err(|error| Box::new(error) as catalog::metastore::Error)
    }

    fn drop_user(&self, username: &str) -> catalog::metastore::Result<()> {
        DiskMetastore::drop_user(self, username)
            .map_err(|error| Box::new(error) as catalog::metastore::Error)
    }
}

/// The [`UserAuth`] a password grants: a SCRAM verifier over a fresh random
/// salt, or trust when no password was given. The verifier is what gets
/// stored; the password itself is dropped here.
///
/// The salted password is pgwire's own derivation (SASLprep normalisation,
/// then PBKDF2-HMAC-SHA256), so what this stores is exactly what pgwire
/// verifies a login's proof against.
fn derive_user_auth(password: Option<&str>) -> UserAuth {
    match password {
        None => UserAuth::Trust,
        Some(password) => {
            let salt = rand::random::<[u8; SCRAM_SALT_LEN]>().to_vec();
            UserAuth::ScramSha256(ScramVerifier {
                salted_password: gen_salted_password(password, &salt, SCRAM_ITERATIONS),
                salt,
            })
        }
    }
}

/// Read and validate the metastore file's section.
fn read_config(path: &Path) -> Result<MetastoreConfig> {
    parse_config(&std::fs::read_to_string(path)?)
}

/// Write `config` as the metastore file's new content, via a sibling
/// temporary file renamed into place so a crash never leaves the file half
/// written.
fn write_config(path: &Path, config: &MetastoreConfig) -> Result<()> {
    let temporary = path.with_extension("rewrite");
    std::fs::write(&temporary, serde_yaml_ng::to_string(config)?)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
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
    #[error(
        "secrets defined both in the config file's `metastore` section and in the metastore file: {}",
        .names.join(", ")
    )]
    ConflictingSecrets { names: Vec<String> },
    #[error(
        "secrets `{first}` and `{second}` are both scoped to `{scope}`; a scope may name only one secret"
    )]
    DuplicateScope {
        first: String,
        second: String,
        scope: String,
    },
    #[error("secret `{name}`: scope `{scope}` must be under `{expected}`")]
    SecretScope {
        name: String,
        scope: String,
        expected: &'static str,
    },
    #[error("datastore `{name}`: no secret is scoped to `{location}`; define one under `secrets`")]
    NoSecret { name: String, location: String },
    #[error("user `{name}` already exists")]
    UserExists { name: String },
    #[error("user `{name}` does not exist")]
    NoSuchUser { name: String },
    #[error(
        "cannot drop user `{}`: it is the built-in default user",
        DEFAULT_USER_NAME
    )]
    DropBuiltinUser,
    #[error(
        "cannot drop user `{name}`: it is defined in the server config file, which the server does not rewrite"
    )]
    DropConfigFileUser { name: String },
    #[error("creating a user requires a metastore file; start the server with --metastore-file")]
    NoMetastoreFile,
    #[error(
        "no default datastore is configured; mark exactly one entry under `datastores` with `default: true`"
    )]
    MissingDefault,
    #[error(
        "multiple datastores are marked `default = true` ({}); exactly one may be", .0.join(", ")
    )]
    MultipleDefaults(Vec<String>),
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    #[error(transparent)]
    Delta(#[from] datastore_delta::Error),
}

/// The datastores and users of a metastore, as written: the config file's
/// `metastore` section, and the standalone metastore file, share this shape.
///
/// Unknown keys are rejected: a misspelled `users` section would otherwise be
/// dropped in silence and unexpectedly select the built-in trusted `pivot` user.
///
/// Serialised sorted by name, so rewriting the metastore file is
/// deterministic rather than in hash order.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetastoreConfig {
    #[serde(default, serialize_with = "sorted_by_name")]
    datastores: HashMap<String, DatastoreConfig>,
    #[serde(default, serialize_with = "sorted_by_name")]
    users: HashMap<String, UserConfig>,
    #[serde(default, serialize_with = "sorted_by_name")]
    secrets: HashMap<String, SecretConfig>,
}

/// Serialize a map's entries in name order.
fn sorted_by_name<S: serde::Serializer, T: Serialize>(
    map: &HashMap<String, T>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    let sorted: std::collections::BTreeMap<&String, &T> = map.iter().collect();
    sorted.serialize(serializer)
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
///
/// The verifier is decoded by deserialisation itself (via [`RawUserConfig`]),
/// so a parsed config holds ready [`UserAuth`]s and a malformed verifier fails
/// the file's parse rather than a login; serialisation re-encodes it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(try_from = "RawUserConfig", into = "RawUserConfig")]
struct UserConfig {
    auth: UserAuth,
}

/// [`UserConfig`] as the file spells it, the verifier still encoded.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RawUserConfig {
    auth: UserAuthConfig,
}

impl TryFrom<RawUserConfig> for UserConfig {
    type Error = String;

    fn try_from(raw: RawUserConfig) -> std::result::Result<Self, String> {
        let auth = match raw.auth {
            UserAuthConfig::Trust {} => UserAuth::Trust,
            UserAuthConfig::ScramSha256 { verifier } => {
                UserAuth::ScramSha256(parse_scram_verifier(&verifier)?)
            }
        };
        Ok(Self { auth })
    }
}

impl From<UserConfig> for RawUserConfig {
    fn from(user: UserConfig) -> Self {
        Self {
            auth: match user.auth {
                UserAuth::Trust => UserAuthConfig::Trust {},
                UserAuth::ScramSha256(verifier) => UserAuthConfig::ScramSha256 {
                    verifier: format_scram_verifier(&verifier),
                },
            },
        }
    }
}

/// YAML representation of a user's one authentication method. Variant-specific
/// fields live inside the variant, so `trust` cannot carry a verifier and SCRAM
/// cannot omit one.
#[derive(Deserialize, Serialize)]
#[serde(tag = "method", deny_unknown_fields)]
enum UserAuthConfig {
    #[serde(rename = "trust")]
    Trust {},
    #[serde(rename = "scram-sha-256")]
    ScramSha256 { verifier: String },
}

/// One datastore's configuration. `kind` is the datastore format; the storage
/// backend (local filesystem, S3, GCS) is inferred from `location`'s scheme,
/// and the credentials a remote one is opened with come from the secret scoped
/// to that location rather than from the datastore itself.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DatastoreConfig {
    kind: DatastoreKind,
    location: String,
    /// Marks this datastore as the default: the current database, the target of
    /// unqualified table names and DDL. Exactly one datastore must set it.
    #[serde(rename = "default", default)]
    is_default: bool,
    /// Run this datastore's own background compaction. On by default;
    /// `compact_bytes` applies only when it is on. Compaction rewrites small
    /// Parquet files and overlapping large ones, so run it in only one process
    /// per datastore (set `compact: false` on the others).
    #[serde(default = "default_true")]
    compact: bool,
    /// Layout-compaction output target (such as `128m` or `1g`). Files strictly
    /// below half this size are small-file candidates; files at least half full
    /// are layout candidates. A single row group may exceed the target. Defaults
    /// to [`DEFAULT_COMPACT_BYTES`].
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_bytes: Option<ByteSize>,
    /// Accumulated small-file bytes that immediately trigger a merge. Defaults
    /// to 1.3 times `compact_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_merge_bytes: Option<ByteSize>,
    /// Count at which sub-target small files may use the balance fallback.
    /// Defaults to [`DEFAULT_MIN_FILES_TO_MERGE`].
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_min_files: Option<usize>,
    /// Run this datastore's own background vacuum. On by default: the vacuumer
    /// deletes unreferenced data files and superseded commit JSONs past their
    /// retention. Set `vacuum: false` on a read-only server, or where another
    /// process owns physical cleanup.
    #[serde(default = "default_true")]
    vacuum: bool,
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
        let target_bytes = self
            .compact_bytes
            .map(ByteSize::as_bytes)
            .unwrap_or(DEFAULT_COMPACT_BYTES);
        Some(CompactionConfig {
            target_bytes,
            merge_target_bytes: self
                .compact_merge_bytes
                .map(ByteSize::as_bytes)
                .unwrap_or_else(|| default_merge_target_bytes(target_bytes)),
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
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum DatastoreKind {
    Delta,
}

impl DatastoreConfig {
    /// Build this datastore's object store. The backend is chosen from
    /// `location`: an `s3://` (or `s3a://`) URI opens an S3 store, a `gs://`
    /// URI a Google Cloud Storage store, and a path with no scheme (or a
    /// `file://` one, which is stripped) a local store. A location whose scheme
    /// names none of these is refused, so the server fails to start rather than
    /// serving an empty datastore out of a directory named after the URI.
    ///
    /// A remote store is opened with the secret scoped to its location. An S3
    /// location needs one: without it there is nothing to sign a request with.
    /// A GCS location without one falls back to the ambient Application
    /// Default Credentials chain, which on Google compute is the whole
    /// configuration a store needs.
    fn open_store(&self, name: &str, secrets: &Secrets) -> Result<Arc<dyn ObjectStore>> {
        match StoreScheme::of(&self.location)? {
            StoreScheme::S3 => {
                let credentials =
                    secrets
                        .resolve_s3(&self.location)
                        .ok_or_else(|| Error::NoSecret {
                            name: name.to_string(),
                            location: self.location.clone(),
                        })?;
                Ok(Arc::new(S3Store::with_credentials(
                    &self.location,
                    credentials,
                )?))
            }
            StoreScheme::Gcs => Ok(Arc::new(match secrets.resolve_gcs(&self.location) {
                Some(credentials_file) => {
                    GcsStore::with_credentials_file(&self.location, credentials_file)?
                }
                None => GcsStore::with_default_credentials(&self.location)?,
            })),
            StoreScheme::Local => Ok(Arc::new(LocalStore::new(local_path(&self.location))?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::DEFAULT_DATASTORE_NAME;
    use datastore_delta::DEFAULT_REFRESH_INTERVAL;
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

    /// The config serving `name`, or "builtin" for the user neither defines.
    fn user_source(store: &DiskMetastore, name: &str) -> &'static str {
        if store.server_config.users.contains_key(name) {
            "server config"
        } else if store
            .metastore_config
            .read()
            .unwrap()
            .users
            .contains_key(name)
        {
            "metastore config"
        } else if store.user_auth(name).is_some() {
            "builtin"
        } else {
            panic!("no user `{name}`")
        }
    }

    /// Open the object store the named datastore addresses, with the secrets
    /// of the metastore serving it.
    fn open_store(store: &DiskMetastore, name: &str) -> Result<Arc<dyn ObjectStore>> {
        datastore(store, name).open_store(name, &store.secrets)
    }

    /// A datastore's definition, whichever config holds it.
    fn datastore(store: &DiskMetastore, name: &str) -> DatastoreConfig {
        if let Some(datastore) = store.server_config.datastores.get(name) {
            return datastore.clone();
        }
        store.metastore_config.read().unwrap().datastores[name].clone()
    }

    /// Open a config `section` merged with a metastore file holding `disk`,
    /// keeping the file alive so a test can rewrite and reopen it.
    fn open_with_file(section: &str, disk: &str) -> (DiskMetastore, tempfile::NamedTempFile) {
        let file = metastore_file(disk);
        (reopen(section, &file), file)
    }

    /// Open `section` merged with the metastore file as a fresh store, as a
    /// restarted server would.
    fn reopen(section: &str, file: &tempfile::NamedTempFile) -> DiskMetastore {
        DiskMetastore::open(
            parse_config(section).unwrap(),
            Some(file.path()),
            DEFAULT_REFRESH_INTERVAL,
        )
        .unwrap()
    }

    /// A minimal config-file `metastore` section: one default datastore.
    const HOT_SECTION: &str =
        "datastores:\n  hot:\n    kind: delta\n    location: /tmp/hot\n    default: true\n";

    /// A section whose `warm` datastore sits in a bucket, for the secret tests
    /// to authenticate. Its `secrets` block is whatever each test appends.
    const WARM_SECTION: &str = "\
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  warm:
    kind: delta
    location: s3://analytics/warm/data
";

    #[test]
    fn compaction_is_per_datastore() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
    compact_bytes: 128m
    compact_merge_bytes: 160m
    compact_min_files: 42
  warm:
    kind: delta
    location: /tmp/warm
    compact: false
"#;

        let store = from_yaml(yaml).unwrap();

        let hot = datastore(&store, "hot").compaction().unwrap();
        let warm = datastore(&store, "warm").compaction();

        // Compaction is on by default (hot omits `compact`), and off only where a
        // datastore turns it off explicitly (warm).
        assert_eq!(hot.target_bytes, 128 * 1024 * 1024);
        assert_eq!(hot.merge_target_bytes, 160 * 1024 * 1024);
        assert_eq!(hot.min_files, 42);
        assert!(warm.is_none());
    }

    #[test]
    fn compaction_knobs_have_defaults() {
        let store = from_yaml(HOT_SECTION).unwrap();

        let config = datastore(&store, "hot").compaction().unwrap();

        assert_eq!(config.target_bytes, DEFAULT_COMPACT_BYTES);
        assert_eq!(
            config.merge_target_bytes,
            default_merge_target_bytes(DEFAULT_COMPACT_BYTES)
        );
        assert_eq!(config.min_files, DEFAULT_MIN_FILES_TO_MERGE);
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

        let error = from_yaml(yaml).expect_err("a metastore without a default should be rejected");

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
    fn an_s3_datastore_no_secret_covers_is_rejected() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  elsewhere:
    type: s3
    scope: s3://other-bucket/
    region: us-east-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let store = from_yaml(&yaml).unwrap();
        let error = open_store(&store, "warm").unwrap_err();

        assert!(matches!(&error, Error::NoSecret { .. }), "{error}");
    }

    #[test]
    fn a_datastore_located_under_an_unknown_scheme_is_rejected() {
        let yaml = "\
datastores:
  hot:
    kind: delta
    location: blob://analytics/hot
    default: true
";

        let store = from_yaml(yaml).unwrap();
        let error = open_store(&store, "hot").unwrap_err();

        assert!(matches!(&error, Error::Store(_)), "{error}");
    }

    #[test]
    fn a_datastore_is_opened_with_the_secret_scoped_to_its_location() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  elsewhere:
    type: s3
    scope: s3://other-bucket/
    region: us-east-1
    access_key_id: AKIA
    secret_access_key: secret
  analytics:
    type: s3
    scope: s3://analytics/
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let store = from_yaml(&yaml).unwrap();

        // A virtual-hosted S3 origin carries the region, so it names the secret
        // the store was opened with.
        assert!(
            open_store(&store, "warm")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
    }

    #[test]
    fn the_most_specific_scope_covering_a_location_authenticates_it() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  bucket:
    type: s3
    scope: s3://analytics/
    region: us-east-1
    access_key_id: AKIA
    secret_access_key: secret
  warm:
    type: s3
    scope: s3://analytics/warm
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let store = from_yaml(&yaml).unwrap();

        assert!(
            open_store(&store, "warm")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
    }

    #[test]
    fn a_secret_without_a_scope_covers_every_bucket() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  aws:
    type: s3
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let store = from_yaml(&yaml).unwrap();

        assert!(
            open_store(&store, "warm")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
    }

    #[test]
    fn a_scope_covers_a_deeper_path_but_not_a_longer_bucket_name() {
        // One scope, two datastores: `covered` sits under the bucket it names,
        // `adjacent` in a bucket whose name merely starts with it.
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  covered:
    kind: delta
    location: s3://analytics/warm/data
  adjacent:
    kind: delta
    location: s3://analyticsarchive/warm
secrets:
  analytics:
    type: s3
    scope: s3://analytics
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#;

        let store = from_yaml(yaml).unwrap();

        assert!(
            open_store(&store, "covered")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
        let error = open_store(&store, "adjacent").unwrap_err();
        assert!(matches!(&error, Error::NoSecret { .. }), "{error}");
    }

    #[test]
    fn a_scope_covers_whole_path_segments_only() {
        // `warm` lives under `s3://analytics/warm/`, which this scope shares a
        // name prefix with and nothing more.
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  analytics:
    type: s3
    scope: s3://analytics/war
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let store = from_yaml(&yaml).unwrap();
        let error = open_store(&store, "warm").unwrap_err();

        assert!(matches!(&error, Error::NoSecret { .. }), "{error}");
    }

    #[test]
    fn two_secrets_on_the_same_scope_are_rejected() {
        // The same bucket in two spellings: `s3a://` addresses what `s3://`
        // does, and a trailing slash pins down nothing more.
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  first:
    type: s3
    scope: s3://analytics/
    region: us-east-1
    access_key_id: AKIA
    secret_access_key: secret
  second:
    type: s3
    scope: s3a://analytics
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let error = from_yaml(&yaml).err().unwrap();

        assert!(
            matches!(&error, Error::DuplicateScope { first, second, scope }
                if first == "first" && second == "second" && scope == "s3://analytics"),
            "{error}"
        );
    }

    #[test]
    fn two_secrets_covering_a_whole_backend_are_rejected() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  first:
    type: s3
    region: us-east-1
    access_key_id: AKIA
    secret_access_key: secret
  second:
    type: s3
    scope: s3://
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let error = from_yaml(&yaml).err().unwrap();

        assert!(
            matches!(&error, Error::DuplicateScope { scope, .. } if scope == "s3://"),
            "{error}"
        );
    }

    #[test]
    fn a_secret_scoped_to_another_backend_is_rejected() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  analytics:
    type: s3
    scope: gs://analytics/
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#
        );

        let error = from_yaml(&yaml).err().unwrap();

        assert!(
            matches!(&error, Error::SecretScope { name, expected, .. }
                if name == "analytics" && *expected == "s3://"),
            "{error}"
        );
    }

    #[test]
    fn a_gcs_datastore_no_secret_covers_resolves_its_credentials_ambiently() {
        let yaml = r#"
datastores:
  hot:
    kind: delta
    location: /tmp/hot
    default: true
  cold:
    kind: delta
    location: gs://analytics/cold
"#;

        let store = from_yaml(yaml).unwrap();

        assert_eq!(
            open_store(&store, "cold").unwrap().location_uri(),
            "gs://analytics/cold"
        );
    }

    #[test]
    fn a_secret_in_the_metastore_file_serves_a_datastore_in_the_config_file() {
        let disk = r#"
secrets:
  analytics:
    type: s3
    scope: s3://analytics/
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#;

        let store = open_merged(WARM_SECTION, disk).unwrap();

        assert!(
            open_store(&store, "warm")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
    }

    #[test]
    fn a_secret_defined_in_both_files_is_rejected_rather_than_shadowed() {
        let secret = r#"
secrets:
  analytics:
    type: s3
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#;

        let error = open_merged(&format!("{WARM_SECTION}{secret}"), secret)
            .err()
            .unwrap();

        assert!(
            matches!(&error, Error::ConflictingSecrets { names } if *names == ["analytics"]),
            "{error}"
        );
    }

    #[test]
    fn a_secret_in_the_metastore_file_survives_a_rewrite() {
        let disk = "secrets:\n  analytics:\n    type: s3\n    scope: s3://analytics/\n    \
                    region: eu-west-1\n    access_key_id: AKIA\n    secret_access_key: secret\n";
        let (store, file) = open_with_file(WARM_SECTION, disk);

        store.create_user("walt", None).unwrap();

        let reopened = reopen(WARM_SECTION, &file);
        assert!(
            open_store(&reopened, "warm")
                .unwrap()
                .describe()
                .contains("eu-west-1")
        );
    }

    #[test]
    fn a_secrets_keys_stay_out_of_its_debug_output() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  analytics:
    type: s3
    region: eu-west-1
    access_key_id: AKIAEXPOSED
    secret_access_key: hunter2
"#
        );

        let store = from_yaml(&yaml).unwrap();

        let printed = format!("{:?}", store.server_config);
        assert!(!printed.contains("AKIAEXPOSED"), "{printed}");
        assert!(!printed.contains("hunter2"), "{printed}");
    }

    #[test]
    fn a_secret_carrying_another_types_field_is_rejected() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  analytics:
    type: s3
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
    credentials_file: /etc/pivot/gcs-key.json
"#
        );

        let error = from_yaml(&yaml).err().unwrap();

        assert!(matches!(&error, Error::Parse { .. }), "{error}");
    }

    #[test]
    fn an_s3_secret_without_its_keys_is_rejected() {
        let yaml = format!(
            "{WARM_SECTION}{}",
            r#"
secrets:
  analytics:
    type: s3
    region: eu-west-1
"#
        );

        let error = from_yaml(&yaml).err().unwrap();

        assert!(matches!(&error, Error::Parse { .. }), "{error}");
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
  cold:
    kind: delta
    location: gs://bucket/prefix
secrets:
  aws:
    type: s3
    region: eu-west-1
    access_key_id: AKIA
    secret_access_key: secret
"#;

        let store = from_yaml(yaml).unwrap();

        assert!(open_store(&store, DEFAULT_DATASTORE_NAME).is_ok());
        assert!(open_store(&store, "warm").is_ok());
        // A GCS store no secret covers resolves its credentials lazily from the
        // ambient chain (at the first request), so opening one needs no
        // Google credentials here.
        let cold = open_store(&store, "cold").unwrap();
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

        assert_eq!(datastore(&store, "warm").location, "/tmp/warm");
        assert_eq!(datastore(&store, "hot").location, "/tmp/hot");
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

        assert!(store.server_config.datastores.contains_key("hot"));
        assert!(
            store
                .metastore_config
                .read()
                .unwrap()
                .datastores
                .contains_key("warm")
        );
        assert_eq!(user_source(&store, "reader"), "server config");
        assert_eq!(user_source(&store, "writer"), "metastore config");
    }

    #[test]
    fn the_user_no_file_defined_is_served_as_built_in() {
        let store = from_yaml_with("").unwrap();

        assert_eq!(user_source(&store, DEFAULT_USER_NAME), "builtin");
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

    /// The stored verifier line for a password, in the form a YAML file holds
    /// it: `pivot-scram-sha-256$<iterations>:<base64 salt>$<base64 salted password>`.
    fn verifier_for(password: &str) -> String {
        use base64::Engine;
        let base64 = base64::engine::general_purpose::STANDARD;
        let salted_password: Vec<u8> = password
            .as_bytes()
            .iter()
            .cycle()
            .take(32)
            .copied()
            .collect();
        format!(
            "pivot-scram-sha-256$4096:{}${}",
            base64.encode([1u8; 16]),
            base64.encode(&salted_password)
        )
    }

    #[test]
    fn a_created_user_is_served_and_survives_reopening() {
        let disk = "datastores:\n  warm:\n    kind: delta\n    location: /tmp/warm\n    \
                    compact: false\n    compact_bytes: 128m\n    compact_merge_bytes: 160m\n    \
                    compact_min_files: 42\n";
        let (store, file) = open_with_file(HOT_SECTION, disk);

        store.create_user("walt", Some("w")).unwrap();

        assert!(matches!(
            store.user_auth("walt"),
            Some(UserAuth::ScramSha256(_))
        ));
        assert_eq!(user_source(&store, "walt"), "metastore config");
        // The rewrite carried the file's datastore through unchanged,
        // tuning fields included.
        let reopened = reopen(HOT_SECTION, &file);
        assert!(matches!(
            reopened.user_auth("walt"),
            Some(UserAuth::ScramSha256(_))
        ));
        let warm = datastore(&reopened, "warm");
        assert_eq!(warm.location, "/tmp/warm");
        assert!(!warm.compact);
        assert_eq!(
            warm.compact_bytes.unwrap().as_bytes(),
            128 * 1024 * 1024,
            "the size survives re-encoding through its written form"
        );
        assert_eq!(
            warm.compact_merge_bytes.unwrap().as_bytes(),
            160 * 1024 * 1024
        );
        assert_eq!(warm.compact_min_files, Some(42));
    }

    #[test]
    fn a_created_trust_user_round_trips() {
        let (store, file) = open_with_file(HOT_SECTION, "users: {}\n");

        store.create_user("walt", None).unwrap();

        assert!(matches!(
            reopen(HOT_SECTION, &file).user_auth("walt"),
            Some(UserAuth::Trust)
        ));
    }

    #[test]
    fn creating_a_second_user_keeps_the_first_and_the_datastore_in_the_file() {
        let disk = "datastores:\n  warm:\n    kind: delta\n    location: /tmp/warm\n";
        let (store, file) = open_with_file(HOT_SECTION, disk);

        store.create_user("walt", None).unwrap();
        store.create_user("jesse", None).unwrap();

        let reopened = reopen(HOT_SECTION, &file);
        assert!(reopened.user_auth("walt").is_some());
        assert!(reopened.user_auth("jesse").is_some());
        assert_eq!(datastore(&reopened, "warm").location, "/tmp/warm");
    }

    #[test]
    fn create_user_rejects_every_existing_user() {
        let section = format!("{HOT_SECTION}users:\n  reader:\n    auth:\n      method: trust\n");
        let disk = "users:\n  writer:\n    auth:\n      method: trust\n";
        let (store, _file) = open_with_file(&section, disk);

        for existing in ["reader", "writer", DEFAULT_USER_NAME] {
            let error = store.create_user(existing, None).unwrap_err();

            assert!(matches!(error, Error::UserExists { .. }), "{error}");
        }
    }

    #[test]
    fn create_user_without_a_metastore_file_is_refused() {
        let store = from_yaml_with("").unwrap();

        let error = store.create_user("walt", None).unwrap_err();

        assert!(matches!(error, Error::NoMetastoreFile), "{error}");
    }

    #[test]
    fn a_dropped_user_is_no_longer_served_and_stays_gone_after_reopening() {
        let (store, file) = open_with_file(HOT_SECTION, "users: {}\n");
        store.create_user("walt", Some("w")).unwrap();

        store.drop_user("walt").unwrap();

        assert!(store.user_auth("walt").is_none());
        assert!(reopen(HOT_SECTION, &file).user_auth("walt").is_none());
    }

    #[test]
    fn dropping_a_missing_user_fails() {
        let (store, _file) = open_with_file(HOT_SECTION, "users: {}\n");

        let error = store.drop_user("walt").unwrap_err();

        assert!(matches!(error, Error::NoSuchUser { .. }), "{error}");
    }

    #[test]
    fn dropping_the_builtin_user_is_refused() {
        let (store, _file) = open_with_file(HOT_SECTION, "users: {}\n");

        let error = store.drop_user(DEFAULT_USER_NAME).unwrap_err();

        assert!(matches!(error, Error::DropBuiltinUser), "{error}");
    }

    #[test]
    fn dropping_a_server_config_user_is_refused() {
        let section = format!("{HOT_SECTION}users:\n  reader:\n    auth:\n      method: trust\n");
        let (store, _file) = open_with_file(&section, "users: {}\n");

        let error = store.drop_user("reader").unwrap_err();

        assert!(matches!(error, Error::DropConfigFileUser { .. }), "{error}");
        assert!(store.user_auth("reader").is_some());
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
        assert_eq!(user_source(&store, DEFAULT_USER_NAME), "builtin");
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
        assert_eq!(user_source(&store, DEFAULT_USER_NAME), "server config");
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

        assert!(matches!(&error, Error::Parse { .. }), "{error}");
        assert!(error.to_string().contains("verifier"), "{error}");
    }
}
