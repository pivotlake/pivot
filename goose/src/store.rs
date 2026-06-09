//! Object storage for the catalog: the **control plane** — listing the
//! `_goose_log/` directory, fetching snapshot bytes, and the atomic
//! create-if-absent that the CAS commit loop is built on.
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3/GCS go over [`ureq`] (blocking HTTP + rustls).
//! S3 requests are signed with `aws_sigv4::http_request::sign` — a pure function
//! we call inline (the tokio it transitively links is never driven). Credentials
//! come from the environment. This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use crate::metadata::{CatalogSnapshot, MetadataError};
use std::fmt::Debug;
use std::path::PathBuf;

mod gcs;
mod s3;
pub use gcs::GcsStore;
pub use s3::S3Store;

/// Directory (key prefix) under the catalog root holding the versioned snapshot
/// files: `_goose_log/<zero-padded-version>.json`.
const LOG_DIR: &str = "_goose_log";
/// Zero-pad versions to a fixed width so lexical key order matches numeric
/// version order.
const VERSION_WIDTH: usize = 20;
/// Optimistic-retry ceiling for a contended commit.
const MAX_COMMIT_RETRIES: u32 = 10_000;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io error on `{key}`: {source}")]
    Io {
        key: String,
        #[source]
        source: std::io::Error,
    },
    #[error("http error talking to object store: {0}")]
    Http(String),
    #[error("unsupported catalog uri `{0}` (expected a local path, file://, s3://, or gs://)")]
    UnsupportedUri(String),
    #[error("missing credential/config: {0}")]
    Config(String),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("catalog commit lost too many races (>{0} retries)")]
    TooMuchContention(u32),
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// The outcome of an atomic create-if-absent — the CAS primitive underlying a
/// catalog commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutOutcome {
    /// The object did not exist and now holds our bytes.
    Created,
    /// Another writer won the race; our bytes were not written.
    AlreadyExists,
}

/// A flat key→bytes object store rooted at one catalog (which, in goose, is one
/// table). Keys are relative to that root, e.g. `_goose_log/0000…1.json`.
pub trait ObjectStore: Debug + Send + Sync {
    /// Fetch an object in full, or `None` if it does not exist.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Atomically create `key` only if it does not already exist. This single
    /// operation is the catalog's compare-and-swap: a lost race returns
    /// [`PutOutcome::AlreadyExists`] without overwriting.
    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<PutOutcome>;

    /// List object keys directly under `prefix` (one level, not recursive),
    /// returned as full keys relative to the root.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Produce a time-limited URL that GETs `key` with no auth headers — so the
    /// io_uring HTTP reader can range-read it directly. Remote backends sign a
    /// URL; the local backend has no URL and returns an error (callers resolve
    /// local files to filesystem paths instead).
    fn presign_get(&self, key: &str) -> Result<url::Url> {
        Err(StoreError::Config(format!(
            "{} cannot presign `{key}`: only object-store backends produce URLs",
            self.describe()
        )))
    }

    /// A human-readable description of where this store points, for diagnostics.
    fn describe(&self) -> String;
}

/// Presign a GET URL for an absolute object-store location (`s3://…`/`gs://…`):
/// open the right backend for its bucket and sign the object's key. Used to turn
/// a snapshot's remote data-file location into something the ring can fetch.
pub fn presign_get(location: &str) -> Result<url::Url> {
    let (root, key) = split_bucket(location)?;
    open_store(&root)?.presign_get(&key)
}

/// Split an absolute object URI into `(bucket-root-uri, object-key)`, e.g.
/// `s3://b/p/x.parquet` → (`s3://b`, `p/x.parquet`).
fn split_bucket(location: &str) -> Result<(String, String)> {
    for scheme in ["s3://", "s3a://", "gs://"] {
        if let Some(rest) = location.strip_prefix(scheme) {
            let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
            return Ok((format!("{scheme}{bucket}"), key.to_string()));
        }
    }
    Err(StoreError::UnsupportedUri(location.to_string()))
}

/// Open the object store for a catalog root URI: `s3://bucket/prefix`,
/// `gs://bucket/prefix`, or a local path (optionally `file://`).
pub fn open_store(uri: &str) -> Result<Box<dyn ObjectStore>> {
    if uri.starts_with("s3://") || uri.starts_with("s3a://") {
        Ok(Box::new(S3Store::from_uri(uri)?))
    } else if uri.starts_with("gs://") {
        Ok(Box::new(GcsStore::from_uri(uri)?))
    } else {
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        Ok(Box::new(LocalStore::new(path)))
    }
}

/// Load the latest committed snapshot for a catalog: LIST `_goose_log/`, take
/// the highest version, GET and parse it. This is the once-per-query "latest"
/// read, called when resolving a remote table. Returns the empty snapshot if the
/// log has no commits yet.
pub fn latest_snapshot(store: &dyn ObjectStore) -> Result<CatalogSnapshot> {
    let keys = store.list(LOG_DIR)?;
    let latest = keys
        .iter()
        .filter_map(|k| parse_version(k).map(|v| (v, k)))
        .max_by_key(|(v, _)| *v)
        .map(|(_, k)| k.clone());

    let Some(key) = latest else {
        return Ok(CatalogSnapshot::empty());
    };
    let bytes = store
        .get(&key)?
        .ok_or_else(|| StoreError::Http(format!("snapshot {key} vanished between list and get")))?;
    Ok(CatalogSnapshot::from_slice(&bytes)?)
}

/// Compare-and-swap commit: read the latest snapshot, apply `mutate` to a copy,
/// then atomically create `_goose_log/<version+1>.json`. On a lost race
/// (another writer committed that version first) it reloads and retries. This
/// single create-if-absent is the only synchronization the catalog needs.
///
/// `mutate` may return an error to abort the commit (no retry); a storage error
/// on the create also aborts. Returns the committed snapshot.
pub fn commit<F>(store: &dyn ObjectStore, mut mutate: F) -> Result<CatalogSnapshot>
where
    F: FnMut(&mut CatalogSnapshot) -> Result<()>,
{
    for _ in 0..MAX_COMMIT_RETRIES {
        let mut snap = latest_snapshot(store)?;
        let base = snap.version;
        mutate(&mut snap)?;
        snap.version = base + 1;
        snap.format_version = crate::metadata::FORMAT_VERSION;
        let key = format!(
            "{LOG_DIR}/{:0width$}.json",
            snap.version,
            width = VERSION_WIDTH
        );
        match store.put_if_absent(&key, &snap.to_vec())? {
            PutOutcome::Created => return Ok(snap),
            PutOutcome::AlreadyExists => continue,
        }
    }
    Err(StoreError::TooMuchContention(MAX_COMMIT_RETRIES))
}

/// Parse a snapshot version from a `_goose_log/<version>.json` key.
fn parse_version(key: &str) -> Option<i64> {
    key.rsplit('/')
        .next()?
        .strip_suffix(".json")?
        .parse::<i64>()
        .ok()
}

/// The local-filesystem backend: keys are paths under `root`.
#[derive(Debug)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl ObjectStore for LocalStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path_for(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<PutOutcome> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                key: key.to_string(),
                source,
            })?;
        }
        // `create_new` is an atomic O_EXCL create — the local CAS primitive.
        use std::io::Write;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                f.write_all(data).map_err(|source| StoreError::Io {
                    key: key.to_string(),
                    source,
                })?;
                Ok(PutOutcome::Created)
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok(PutOutcome::AlreadyExists)
            }
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let dir = self.path_for(prefix);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // A not-yet-created log directory lists as empty, not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(StoreError::Io {
                    key: prefix.to_string(),
                    source,
                });
            }
        };
        let mut keys = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            if let Some(name) = entry.file_name().to_str() {
                keys.push(format!("{}/{}", prefix.trim_end_matches('/'), name));
            }
        }
        Ok(keys)
    }

    fn describe(&self) -> String {
        format!("local:{}", self.root.display())
    }
}

/// Join a relative catalog key onto an in-bucket prefix, preserving the prefix's
/// (possibly empty) value. Shared by the remote backends.
fn join_prefix(prefix: &str, key: &str) -> String {
    let prefix = prefix.trim_matches('/');
    if prefix.is_empty() {
        key.trim_start_matches('/').to_string()
    } else {
        format!("{prefix}/{}", key.trim_start_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Column, DataFile, FORMAT_VERSION, Schema, Table};

    fn snapshot(version: i64, table: &str) -> CatalogSnapshot {
        CatalogSnapshot {
            format_version: FORMAT_VERSION,
            version,
            schemas: vec![Schema {
                name: "main".into(),
                tables: vec![Table {
                    name: table.into(),
                    columns: vec![Column {
                        name: "id".into(),
                        type_sql: "INTEGER".into(),
                    }],
                    files: vec![DataFile {
                        location: "_goose_data/a.parquet".into(),
                        size: 123,
                        row_count: 1,
                    }],
                }],
            }],
        }
    }

    fn commit(store: &dyn ObjectStore, version: i64, snap: &CatalogSnapshot) {
        let key = format!("{LOG_DIR}/{version:020}.json");
        assert_eq!(
            store.put_if_absent(&key, &snap.to_vec()).unwrap(),
            PutOutcome::Created
        );
    }

    #[test]
    fn local_put_if_absent_is_a_cas() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert_eq!(
            store.put_if_absent("k", b"first").unwrap(),
            PutOutcome::Created
        );
        // Second writer loses the race; original bytes are untouched.
        assert_eq!(
            store.put_if_absent("k", b"second").unwrap(),
            PutOutcome::AlreadyExists
        );
        assert_eq!(store.get("k").unwrap().unwrap(), b"first");
    }

    #[test]
    fn local_get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.get("nope").unwrap().is_none());
        assert!(store.list(LOG_DIR).unwrap().is_empty());
    }

    #[test]
    fn latest_snapshot_picks_highest_version() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        // Commit out of order to prove sorting is by parsed version.
        commit(&store, 2, &snapshot(2, "events"));
        commit(&store, 10, &snapshot(10, "latest"));
        commit(&store, 1, &snapshot(1, "first"));

        let snap = latest_snapshot(&store).unwrap();
        assert_eq!(snap.version, 10);
        assert!(snap.table("main", "latest").is_some());
        assert!(snap.table("main", "events").is_none());
    }

    #[test]
    fn latest_snapshot_of_empty_log_is_version_zero() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert_eq!(latest_snapshot(&store).unwrap().version, 0);
    }

    #[test]
    fn open_store_routes_local_and_file_uri() {
        let dir = tempfile::tempdir().unwrap();
        let store = open_store(dir.path().to_str().unwrap()).unwrap();
        commit(store.as_ref(), 1, &snapshot(1, "t"));
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let reopened = open_store(&uri).unwrap();
        assert_eq!(latest_snapshot(reopened.as_ref()).unwrap().version, 1);
    }
}
