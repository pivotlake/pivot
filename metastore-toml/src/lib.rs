//! A TOML-backed [`metastore::Metastore`] provider.
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
//! ```toml
//! [datastore.hot]
//! kind = "delta"
//! location = "/var/lib/pivot"        # local path -> local store
//! default = true                     # the current database
//!
//! [datastore.warm]
//! kind = "delta"
//! location = "s3://my-bucket/pivot/" # s3:// -> S3 store
//! region = "us-east-1"
//! access_key_id = "AKIA..."
//! secret_access_key = "..."
//! # session_token = "..."    # optional
//! # endpoint = "http://localhost:9000" # optional (MinIO / S3-compatible)
//! # source = "env"           # optional: use AWS_* environment variables
//! ```
//!
//! Inline credentials mean the file holds secrets, so it should be readable
//! only by the PivotDB process. `source = "env"` avoids storing credentials in
//! the file by deferring to the ambient environment.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use catalog::Datastore;
use datastore_delta::store::{LocalStore, ObjectStore, S3Credentials, S3Store};
use datastore_delta::{DeltaDatastore, MaintenanceConfig};
use dispatch::DataFlowDispatcher;
use metastore::Metastore;
use serde::Deserialize;

/// AWS region assumed for an S3 datastore that does not name one.
const DEFAULT_S3_REGION: &str = "us-east-1";

/// A metastore backed by a parsed TOML file.
///
/// Configuration is parsed and structurally validated by [`open`](Self::open).
/// Object stores and their Delta datastores are opened when
/// [`Metastore::open_datastores`] is called.
pub struct TomlMetastore {
    datastore_configs: HashMap<String, DatastoreConfig>,
    default_name: String,
    /// Background maintenance every datastore this metastore opens self-manages
    /// (refresh cadence and optional compaction). Passed straight through to
    /// [`DeltaDatastore::from_store`].
    maintenance: MaintenanceConfig,
}

impl TomlMetastore {
    /// Read and parse the metastore file at `path`, applying `maintenance` to
    /// every datastore it opens.
    pub fn open(path: impl AsRef<Path>, maintenance: MaintenanceConfig) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text, &path.display().to_string(), maintenance)
    }

    /// Parse metastore configuration from TOML text, using `path` in errors.
    pub fn from_toml(text: &str, path: &str, maintenance: MaintenanceConfig) -> Result<Self> {
        let file: MetastoreFile = toml::from_str(text).map_err(|source| Error::Parse {
            path: path.to_string(),
            source,
        })?;
        // The default datastore is the one flagged `default = true`, not one with
        // a reserved name. Exactly one is required: it is the current database.
        let mut defaults: Vec<String> = file
            .datastore
            .iter()
            .filter(|(_, config)| config.is_default)
            .map(|(name, _)| name.clone())
            .collect();
        if defaults.len() > 1 {
            defaults.sort();
            return Err(Error::MultipleDefaults(defaults));
        }
        let default_name = defaults.pop().ok_or(Error::MissingDefault)?;
        Ok(Self {
            datastore_configs: file.datastore,
            default_name,
            maintenance,
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
                let datastore: Arc<dyn Datastore> = match config.kind {
                    DatastoreKind::Delta => DeltaDatastore::from_store(
                        name.clone(),
                        store,
                        dispatcher,
                        Some(self.maintenance.clone()),
                    )?,
                };
                Ok((name.clone(), datastore))
            })
            .collect()
    }
}

impl Metastore for TomlMetastore {
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
        source: toml::de::Error,
    },
    #[error("datastore `{name}`: {message}")]
    Datastore { name: String, message: String },
    #[error(
        "no datastore is marked `default = true`; exactly one is required (it is the current database)"
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

#[derive(Deserialize)]
struct MetastoreFile {
    #[serde(default)]
    datastore: HashMap<String, DatastoreConfig>,
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
    region: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    endpoint: Option<String>,
    #[serde(default)]
    source: CredentialSource,
}

/// The datastore format. Only [`Delta`](Self::Delta) is supported today; adding
/// another (Iceberg, ...) is a new variant plus its arm in
/// [`build_datastores`](TomlMetastore::build_datastores).
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DatastoreKind {
    Delta,
}

/// Where a remote datastore's credentials come from.
#[derive(Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CredentialSource {
    /// Given inline in this datastore's configuration (the default).
    #[default]
    Inline,
    /// Taken from the ambient environment (`AWS_*` for S3).
    Env,
}

impl DatastoreConfig {
    /// Build this datastore's object store. The backend is chosen from
    /// `location`: an `s3://` (or `s3a://`) URI opens an S3 store with the
    /// configured credentials; anything else is a local path (an optional
    /// `file://` scheme is stripped).
    fn open_store(&self, name: &str) -> Result<Arc<dyn ObjectStore>> {
        if is_s3_location(&self.location) {
            let store = match self.source {
                CredentialSource::Env => S3Store::from_uri(&self.location)?,
                CredentialSource::Inline => {
                    let credentials = S3Credentials {
                        region: self
                            .region
                            .clone()
                            .unwrap_or_else(|| DEFAULT_S3_REGION.to_string()),
                        access_key: require(name, &self.access_key_id, "access_key_id")?,
                        secret_key: require(name, &self.secret_access_key, "secret_access_key")?,
                        session_token: self.session_token.clone(),
                        endpoint: self.endpoint.clone(),
                    };
                    S3Store::with_credentials(&self.location, credentials)?
                }
            };
            Ok(Arc::new(store))
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
        message: format!("`{field}` is required (or set `source = \"env\"`)"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A no-compaction maintenance config for the parse tests, which never open
    /// datastores and so never act on it.
    fn test_maintenance() -> MaintenanceConfig {
        MaintenanceConfig {
            refresh_interval: Duration::from_secs(30),
            compaction: None,
        }
    }

    #[test]
    fn missing_default_datastore_is_rejected() {
        let toml = r#"
            [datastore.warm]
            kind = "delta"
            location = "/tmp/warm"
        "#;

        let result = TomlMetastore::from_toml(toml, "test", test_maintenance());

        assert!(matches!(result, Err(Error::MissingDefault)));
    }

    #[test]
    fn unknown_datastore_kind_is_rejected() {
        let toml = r#"
            [datastore.default]
            kind = "iceberg"
            location = "/tmp/default"
        "#;

        let result = TomlMetastore::from_toml(toml, "test", test_maintenance());

        assert!(matches!(result, Err(Error::Parse { .. })));
    }

    #[test]
    fn multiple_defaults_are_rejected() {
        let toml = r#"
            [datastore.hot]
            kind = "delta"
            location = "/tmp/hot"
            default = true

            [datastore.warm]
            kind = "delta"
            location = "/tmp/warm"
            default = true
        "#;

        let result = TomlMetastore::from_toml(toml, "test", test_maintenance());

        assert!(matches!(result, Err(Error::MultipleDefaults(names)) if names == ["hot", "warm"]));
    }

    #[test]
    fn default_is_the_flagged_datastore_whatever_its_name() {
        let toml = r#"
            [datastore.hot]
            kind = "delta"
            location = "/tmp/hot"
            default = true

            [datastore.warm]
            kind = "delta"
            location = "/tmp/warm"
        "#;

        let store = TomlMetastore::from_toml(toml, "test", test_maintenance()).unwrap();

        assert_eq!(store.default_datastore_name(), "hot");
    }

    #[test]
    fn s3_location_without_keys_is_rejected() {
        let toml = r#"
            [datastore.default]
            kind = "delta"
            location = "/tmp/default"
            default = true

            [datastore.warm]
            kind = "delta"
            location = "s3://bucket/prefix"
        "#;

        let store = TomlMetastore::from_toml(toml, "test", test_maintenance()).unwrap();
        let err = store.datastore_configs["warm"]
            .open_store("warm")
            .unwrap_err();

        assert!(matches!(err, Error::Datastore { .. }));
    }

    #[test]
    fn local_and_s3_locations_open_stores() {
        let toml = r#"
            [datastore.default]
            kind = "delta"
            location = "/tmp/default"
            default = true

            [datastore.warm]
            kind = "delta"
            location = "s3://bucket/prefix"
            region = "eu-west-1"
            access_key_id = "AKIA"
            secret_access_key = "secret"
        "#;

        let store = TomlMetastore::from_toml(toml, "test", test_maintenance()).unwrap();

        assert!(
            store.datastore_configs["default"]
                .open_store("default")
                .is_ok()
        );
        assert!(store.datastore_configs["warm"].open_store("warm").is_ok());
    }
}
