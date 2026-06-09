//! A generic key→bytes object store — local filesystem, S3, or GCS — and nothing
//! catalog-specific. It knows how to `get`/`list`/atomically-create objects and
//! presign GET URLs; the catalog control plane (`_goose_log/` snapshots, the CAS
//! commit loop) lives one layer up in [`crate::table_store`].
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3/GCS go over [`ureq`] (blocking HTTP + rustls).
//! S3 requests are signed with `aws_sigv4::http_request::sign` — a pure function
//! we call inline (the tokio it transitively links is never driven). Credentials
//! come from the environment. This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use std::fmt::Debug;

mod gcs;
mod local;
mod s3;
pub use gcs::GcsStore;
pub use local::LocalStore;
pub use s3::S3Store;

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

    /// Atomically replace `key` with `data` (overwriting any existing object).
    /// Backs a small, rarely-written mutable control file like the table
    /// manifest; reads see either the old or the new object whole, never a torn
    /// write. (Concurrent writers are last-writer-wins — fine for the manifest,
    /// which a single server writes on the occasional `CREATE TABLE`.)
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

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

    #[test]
    fn open_store_routes_local_and_file_uri() {
        let dir = tempfile::tempdir().unwrap();
        let store = open_store(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(store.put_if_absent("k", b"v").unwrap(), PutOutcome::Created);
        // Reopening through a `file://` URI lands on the same root.
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let reopened = open_store(&uri).unwrap();
        assert_eq!(reopened.get("k").unwrap().unwrap(), b"v");
    }
}
