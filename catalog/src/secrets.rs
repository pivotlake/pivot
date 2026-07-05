//! Secrets: named credential bundles (`CREATE SECRET`), part of the catalog.
//!
//! A secret pairs a set of key-value options (`key_id`, `secret`, `region`,
//! ...) with the path prefixes it applies to (its scope). Consumers - the S3
//! object store when signing a request - look a secret up by the path they are
//! about to touch; the secret whose scope is the longest matching prefix wins,
//! as in DuckDB's secrets manager.
//!
//! Persistent secrets (the default) live at [`SECRETS_KEY`] in the database's
//! object store, so a database rooted on S3 carries its secrets with it.
//! Temporary secrets (`CREATE TEMPORARY SECRET`) live only in this registry
//! and vanish on restart.
//!
//! # Security model
//!
//! Following DuckDB's stored secrets, values are persisted **unencrypted**;
//! what protects them is access control on the storage:
//!
//! - on a local database, the secrets file is written owner-only (mode 600),
//!   and a file readable by group/other is refused on load;
//! - on a remote database, the bucket's access policy is the boundary - anyone
//!   who can read the bucket can read the secrets, exactly like the data.
//!
//! Sensitive option values (see
//! [`secret_option_is_redacted`]) render as `redacted` in every
//! human-readable output: `pivot_secrets()`, plan displays, and logs.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, RwLock};

use planner::catalog::{CreateSecretRequest, DropSecretRequest, secret_option_is_redacted};
use thiserror::Error as ThisError;

use crate::store::{ObjectPath, ObjectStore};

/// The store key the persistent secrets are kept under, next to the database
/// manifest. Written whole on every change (plain put, last-writer-wins, same
/// single-control-plane-writer model as the database manifest).
pub const SECRETS_KEY: &str = "_pivot_secrets.json";

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("secret `{0}` already exists (use CREATE OR REPLACE SECRET to overwrite it)")]
    SecretExists(String),
    #[error("secret `{0}` does not exist")]
    SecretNotFound(String),
    #[error(
        "the persisted secrets object `{SECRETS_KEY}` is readable by other users; refusing to load it (chmod 600 it to proceed)"
    )]
    SecretsFileNotPrivate,
    #[error("parsing persisted secrets: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// One secret: its identity, the path prefixes it applies to, and its
/// key-value options. This is also the persisted (JSON) form.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Secret {
    pub name: String,
    /// The secret's TYPE (e.g. `s3`), lowercased.
    pub secret_type: String,
    /// How the values were produced (`config` for explicit values).
    pub provider: String,
    /// Path prefixes the secret applies to; longest match wins on lookup.
    pub scope: Vec<String>,
    /// The key-value options, keys lowercased. Ordered so listings are
    /// deterministic.
    pub options: BTreeMap<String, String>,
}

impl Secret {
    /// The option's value, or `None` when the secret doesn't carry the key.
    pub fn option(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str)
    }

    /// The length of the longest scope prefix that `path` starts with, or
    /// `None` when no scope matches.
    fn match_score(&self, path: &str) -> Option<usize> {
        self.scope
            .iter()
            .filter(|prefix| path.starts_with(prefix.as_str()))
            .map(|prefix| prefix.len())
            .max()
    }

    /// The options as a single `key=value;...` string with sensitive values
    /// replaced by `redacted` - the `pivot_secrets()` rendering, mirroring
    /// DuckDB's `duckdb_secrets()`.
    pub fn redacted_options(&self) -> String {
        self.options
            .iter()
            .map(|(key, value)| {
                if secret_option_is_redacted(key) {
                    format!("{key}=redacted")
                } else {
                    format!("{key}={value}")
                }
            })
            .collect::<Vec<_>>()
            .join(";")
    }
}

/// A [`Secret`] as the registry holds it: the secret plus whether it is
/// temporary (in-memory only, never persisted).
#[derive(Debug, Clone)]
pub struct SecretEntry {
    pub secret: Secret,
    pub temporary: bool,
}

/// The persisted form: every persistent secret, whole. JSON, like the
/// database manifest.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct SecretsFile {
    secrets: Vec<Secret>,
}

/// The catalog's secrets, keyed by name (one namespace for temporary and
/// persistent secrets). Shared between the catalog (which creates/drops) and
/// the object store (which looks credentials up per request), so a secret
/// created mid-session applies to the very next request.
#[derive(Debug, Default)]
pub struct SecretsRegistry {
    secrets: RwLock<HashMap<String, SecretEntry>>,
    /// Serializes create/drop (mutate + persist) against each other. The map
    /// lock alone cannot be held across the persist: the S3 store's put calls
    /// back into [`lookup`](Self::lookup) to resolve its signing credentials,
    /// which would deadlock on the map lock.
    writer: Mutex<()>,
}

impl SecretsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the persistent secrets from `store`, refusing a secrets object
    /// that other users can read (see the module doc's security model). A
    /// database with no secrets object loads empty.
    pub fn load(&self, store: &dyn ObjectStore) -> Result<()> {
        let key = ObjectPath::new(SECRETS_KEY);
        let Some(bytes) = store.get(&key)? else {
            return Ok(());
        };
        if !store.is_private(&key)? {
            return Err(Error::SecretsFileNotPrivate);
        }
        let file: SecretsFile = serde_json::from_slice(&bytes)?;
        let mut secrets = self.secrets.write().unwrap();
        for secret in file.secrets {
            secrets.insert(
                secret.name.clone(),
                SecretEntry {
                    secret,
                    temporary: false,
                },
            );
        }
        Ok(())
    }

    /// Register the secret described by `request`, persisting the new
    /// persistent set to `store` unless the secret is temporary.
    pub fn create_secret(
        &self,
        request: CreateSecretRequest,
        store: &dyn ObjectStore,
    ) -> Result<()> {
        let temporary = request.temporary;
        let entry = SecretEntry {
            secret: Secret {
                name: request.name.clone(),
                secret_type: request.secret_type,
                provider: request.provider,
                scope: request.scope,
                options: request.options,
            },
            temporary,
        };

        let _writer = self.writer.lock().unwrap();
        // The map lock is taken per step, never across the persist (see
        // `writer`); the writer mutex keeps the steps of concurrent
        // creates/drops from interleaving.
        let replaced = {
            let mut secrets = self.secrets.write().unwrap();
            if secrets.contains_key(&request.name) {
                if request.if_not_exists {
                    return Ok(());
                }
                if !request.or_replace {
                    return Err(Error::SecretExists(request.name));
                }
            }
            secrets.insert(request.name.clone(), entry)
        };
        // A replaced persistent secret is rewritten (or dropped from the
        // store, when the replacement is temporary) by writing the resulting
        // persistent set whole.
        let replaced_persistent = replaced.as_ref().is_some_and(|old| !old.temporary);
        if temporary && !replaced_persistent {
            return Ok(());
        }
        self.persist_or_restore(store, request.name, replaced)
    }

    /// Drop the named secret. `request.temporary` narrows which kind may be
    /// dropped: dropping a persistent secret with `DROP TEMPORARY SECRET`
    /// (or vice versa) is "not found".
    pub fn drop_secret(&self, request: DropSecretRequest, store: &dyn ObjectStore) -> Result<()> {
        let _writer = self.writer.lock().unwrap();
        let removed = {
            let mut secrets = self.secrets.write().unwrap();
            let matches = secrets
                .get(&request.name)
                .is_some_and(|entry| request.temporary.is_none_or(|t| entry.temporary == t));
            if !matches {
                if request.if_exists {
                    return Ok(());
                }
                return Err(Error::SecretNotFound(request.name));
            }
            secrets.remove(&request.name).unwrap()
        };
        if removed.temporary {
            return Ok(());
        }
        self.persist_or_restore(store, request.name, Some(removed))
    }

    /// Persist the current persistent set; if the store write fails, restore
    /// `previous` under `name` (`None` removes the entry), undoing the map
    /// mutation the caller just made so the registry never claims a
    /// durability the store didn't deliver. Callers hold the `writer` mutex.
    fn persist_or_restore(
        &self,
        store: &dyn ObjectStore,
        name: String,
        previous: Option<SecretEntry>,
    ) -> Result<()> {
        let result = self.persist(store);
        if result.is_err() {
            let mut secrets = self.secrets.write().unwrap();
            match previous {
                Some(old) => secrets.insert(name, old),
                None => secrets.remove(&name),
            };
        }
        result
    }

    /// Write the persistent secrets whole to the store, owner-only on local
    /// backends (see [`ObjectStore::put_private`]). Reads the map under a
    /// short-lived lock and writes with none held: the S3 store's put resolves
    /// its own signing credentials through [`lookup`](Self::lookup). Callers
    /// hold the `writer` mutex.
    fn persist(&self, store: &dyn ObjectStore) -> Result<()> {
        let mut file = SecretsFile {
            secrets: self
                .secrets
                .read()
                .unwrap()
                .values()
                .filter(|entry| !entry.temporary)
                .map(|entry| entry.secret.clone())
                .collect(),
        };
        file.secrets.sort_by(|a, b| a.name.cmp(&b.name));
        let bytes = serde_json::to_vec_pretty(&file)?;
        store.put_private(&ObjectPath::new(SECRETS_KEY), &bytes)?;
        Ok(())
    }

    /// The secret that applies to `path` for `secret_type`: the longest
    /// matching scope prefix wins; on a tie a temporary secret beats a
    /// persistent one, then the lexicographically smaller name (DuckDB's
    /// tie-break order).
    pub fn lookup(&self, path: &str, secret_type: &str) -> Option<Secret> {
        // The tie-break rank, defined once for both sides of the comparison.
        fn rank<'a>(score: &usize, entry: &'a SecretEntry) -> (usize, bool, Reverse<&'a str>) {
            (*score, entry.temporary, Reverse(entry.secret.name.as_str()))
        }
        let secrets = self.secrets.read().unwrap();
        secrets
            .values()
            .filter(|entry| entry.secret.secret_type.eq_ignore_ascii_case(secret_type))
            .filter_map(|entry| entry.secret.match_score(path).map(|score| (score, entry)))
            .max_by(|(score_a, a), (score_b, b)| rank(score_a, a).cmp(&rank(score_b, b)))
            .map(|(_, entry)| entry.secret.clone())
    }

    /// Every secret, sorted by name - the `pivot_secrets()` listing.
    pub fn list(&self) -> Vec<SecretEntry> {
        let mut entries: Vec<SecretEntry> =
            self.secrets.read().unwrap().values().cloned().collect();
        entries.sort_by(|a, b| a.secret.name.cmp(&b.secret.name));
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::LocalStore;

    fn request(name: &str, scope: &[&str]) -> CreateSecretRequest {
        CreateSecretRequest {
            name: name.to_string(),
            secret_type: "s3".to_string(),
            provider: "config".to_string(),
            scope: scope.iter().map(|s| s.to_string()).collect(),
            options: BTreeMap::from([
                ("key_id".to_string(), format!("key-of-{name}")),
                ("secret".to_string(), "hunter2".to_string()),
            ]),
            temporary: false,
            or_replace: false,
            if_not_exists: false,
        }
    }

    #[test]
    fn lookup_prefers_longest_scope_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        let registry = SecretsRegistry::new();

        registry
            .create_secret(request("wide", &["s3://bucket"]), &store)
            .unwrap();
        registry
            .create_secret(request("narrow", &["s3://bucket/inner"]), &store)
            .unwrap();

        let hit = registry
            .lookup("s3://bucket/inner/file.parquet", "s3")
            .unwrap();
        assert_eq!(hit.name, "narrow");
        let hit = registry.lookup("s3://bucket/other.parquet", "s3").unwrap();
        assert_eq!(hit.name, "wide");
        assert!(
            registry
                .lookup("s3://elsewhere/file.parquet", "s3")
                .is_none()
        );
        assert!(
            registry
                .lookup("s3://bucket/inner/file.parquet", "gcs")
                .is_none()
        );
    }

    #[test]
    fn create_conflicts_error_without_or_replace() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        let registry = SecretsRegistry::new();

        registry
            .create_secret(request("mine", &["s3://"]), &store)
            .unwrap();

        assert_eq!(
            registry.lookup("s3://any/where", "s3").unwrap().name,
            "mine"
        );
        let err = registry
            .create_secret(request("mine", &["s3://"]), &store)
            .unwrap_err();
        assert!(matches!(err, Error::SecretExists(_)));

        let mut replace = request("mine", &["s3://only-here"]);
        replace.or_replace = true;
        registry.create_secret(replace, &store).unwrap();
        assert!(registry.lookup("s3://any/where", "s3").is_none());
    }

    #[test]
    fn persistent_secrets_survive_reload_and_temporary_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());

        let registry = SecretsRegistry::new();
        registry
            .create_secret(request("durable", &["s3://"]), &store)
            .unwrap();
        let mut temp = request("fleeting", &["s3://"]);
        temp.temporary = true;
        registry.create_secret(temp, &store).unwrap();

        let reloaded = SecretsRegistry::new();
        reloaded.load(&store).unwrap();
        let names: Vec<String> = reloaded.list().into_iter().map(|e| e.secret.name).collect();
        assert_eq!(names, vec!["durable"]);
    }

    #[test]
    fn drop_removes_from_store_and_respects_qualifier() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        let registry = SecretsRegistry::new();
        registry
            .create_secret(request("mine", &["s3://"]), &store)
            .unwrap();

        // A temporary-only drop must not touch a persistent secret.
        let err = registry
            .drop_secret(
                DropSecretRequest {
                    name: "mine".to_string(),
                    if_exists: false,
                    temporary: Some(true),
                },
                &store,
            )
            .unwrap_err();
        assert!(matches!(err, Error::SecretNotFound(_)));

        registry
            .drop_secret(
                DropSecretRequest {
                    name: "mine".to_string(),
                    if_exists: false,
                    temporary: None,
                },
                &store,
            )
            .unwrap();
        let reloaded = SecretsRegistry::new();
        reloaded.load(&store).unwrap();
        assert!(reloaded.list().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn world_readable_secrets_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        let registry = SecretsRegistry::new();
        registry
            .create_secret(request("mine", &["s3://"]), &store)
            .unwrap();

        let path = dir.path().join(SECRETS_KEY);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = SecretsRegistry::new().load(&store).unwrap_err();
        assert!(matches!(err, Error::SecretsFileNotPrivate));
    }
}
