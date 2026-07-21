//! [`TomlMetastore`]: datastores defined in a TOML file on disk.
//!
//! ```toml
//! [datastore.default]         # required — DuckDB's current database
//! kind = "local"
//! location = "/var/lib/pivot"
//!
//! [datastore.warm]
//! kind = "s3"
//! location = "s3://my-bucket/pivot/"
//! region = "us-east-1"
//! access_key_id = "AKIA..."
//! secret_access_key = "..."
//! # session_token = "..."     # optional
//! # endpoint = "http://localhost:9000"   # optional (MinIO / S3-compatible)
//! # source = "env"            # optional: ignore the inline keys, use AWS_* env
//! ```
//!
//! Inline credentials mean the file holds secrets — keep it `0600`. `source =
//! "env"` avoids inlining by deferring to the ambient environment instead.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use catalog::Datastore;
use datastore_delta::ParquetCatalog;
use datastore_delta::store::{LocalStore, ObjectStore, S3Credentials, S3Store};
use dispatch::DataFlowDispatcher;
use serde::Deserialize;

use crate::{DEFAULT_DATASTORE_NAME, Error, Metastore, Result};

/// AWS region assumed for an S3 datastore that doesn't name one.
const DEFAULT_S3_REGION: &str = "us-east-1";

/// A metastore backed by a parsed TOML file. The datastores are validated at
/// [`open`](Self::open) time; the object stores and catalogs are built lazily in
/// [`datastores`](Metastore::datastores).
pub struct TomlMetastore {
    datastores: HashMap<String, DatastoreConfig>,
}

impl TomlMetastore {
    /// Read and parse the metastore file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text, &path.display().to_string())
    }

    /// Parse metastore config from TOML text, `path` naming it in errors.
    pub fn from_toml(text: &str, path: &str) -> Result<Self> {
        let file: MetastoreFile = toml::from_str(text).map_err(|source| Error::Parse {
            path: path.to_string(),
            source,
        })?;
        if !file.datastore.contains_key(DEFAULT_DATASTORE_NAME) {
            return Err(Error::MissingDefault);
        }
        Ok(Self {
            datastores: file.datastore,
        })
    }
}

impl Metastore for TomlMetastore {
    fn datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>> {
        self.datastores
            .iter()
            .map(|(name, config)| {
                let store = config.open_store(name)?;
                let catalog = ParquetCatalog::from_store(name.clone(), store, dispatcher)?;
                Ok((name.clone(), Arc::new(catalog) as Arc<dyn Datastore>))
            })
            .collect()
    }
}

#[derive(Deserialize)]
struct MetastoreFile {
    #[serde(default)]
    datastore: HashMap<String, DatastoreConfig>,
}

/// One datastore's config, tagged by `kind` (`local` / `s3`).
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DatastoreConfig {
    Local {
        location: String,
    },
    S3 {
        location: String,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        endpoint: Option<String>,
        #[serde(default)]
        source: CredentialSource,
    },
}

/// Where a remote datastore's credentials come from.
#[derive(Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CredentialSource {
    /// Given inline in this datastore's config (the default).
    #[default]
    Inline,
    /// Taken from the ambient environment: `AWS_*` for S3.
    Env,
}

impl DatastoreConfig {
    /// Build this datastore's object store, resolving credentials as configured.
    fn open_store(&self, name: &str) -> Result<Arc<dyn ObjectStore>> {
        let store: Arc<dyn ObjectStore> = match self {
            DatastoreConfig::Local { location } => Arc::new(LocalStore::new(location)),

            DatastoreConfig::S3 {
                location,
                source: CredentialSource::Env,
                ..
            } => Arc::new(S3Store::from_uri(location)?),
            DatastoreConfig::S3 {
                location,
                region,
                access_key_id,
                secret_access_key,
                session_token,
                endpoint,
                source: CredentialSource::Inline,
            } => {
                let credentials = S3Credentials {
                    region: region
                        .clone()
                        .unwrap_or_else(|| DEFAULT_S3_REGION.to_string()),
                    access_key: require(name, access_key_id, "access_key_id")?,
                    secret_key: require(name, secret_access_key, "secret_access_key")?,
                    session_token: session_token.clone(),
                    endpoint: endpoint.clone(),
                };
                Arc::new(S3Store::with_credentials(location, credentials)?)
            }
        };
        Ok(store)
    }
}

/// A required credential field, erroring with a message that points at the
/// `source = "env"` alternative when it's missing.
fn require(name: &str, value: &Option<String>, field: &str) -> Result<String> {
    value.clone().ok_or_else(|| Error::Datastore {
        name: name.to_string(),
        message: format!("`{field}` is required (or set `source = \"env\"`)"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_default_datastore_is_rejected() {
        let toml = r#"
            [datastore.warm]
            kind = "local"
            location = "/tmp/warm"
        "#;

        let result = TomlMetastore::from_toml(toml, "test");

        assert!(matches!(result, Err(Error::MissingDefault)));
    }

    #[test]
    fn s3_inline_without_keys_is_rejected() {
        let toml = r#"
            [datastore.default]
            kind = "local"
            location = "/tmp/default"

            [datastore.warm]
            kind = "s3"
            location = "s3://bucket/prefix"
        "#;

        let store = TomlMetastore::from_toml(toml, "test").unwrap();
        let err = store.datastores["warm"].open_store("warm").unwrap_err();

        assert!(matches!(err, Error::Datastore { .. }));
    }

    #[test]
    fn local_and_env_s3_open_stores() {
        let toml = r#"
            [datastore.default]
            kind = "local"
            location = "/tmp/default"

            [datastore.warm]
            kind = "s3"
            location = "s3://bucket/prefix"
            region = "eu-west-1"
            access_key_id = "AKIA"
            secret_access_key = "secret"
        "#;

        let store = TomlMetastore::from_toml(toml, "test").unwrap();

        assert!(store.datastores["default"].open_store("default").is_ok());
        assert!(store.datastores["warm"].open_store("warm").is_ok());
    }
}
