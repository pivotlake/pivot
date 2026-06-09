//! A generic key→bytes object store — local filesystem, S3, or GCS — and nothing
//! catalog-specific. It knows how to `get`/`put`/`list` objects and turn a key
//! into a ring-readable [`DataFileLocation`]; the table manifest and catalog
//! build on it one layer up.
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3/GCS go over [`ureq`] (blocking HTTP + rustls).
//! S3 requests are signed with `aws_sigv4::http_request::sign` — a pure function
//! we call inline (the tokio it transitively links is never driven). Credentials
//! come from the environment. This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use std::fmt::Debug;
use std::path::{Path, PathBuf};

mod gcs;
mod local;
mod memory;
mod s3;
pub use gcs::GcsStore;
pub use local::LocalStore;
pub use memory::MemoryStore;
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

/// One object returned by [`ObjectStore::list`]: its key (relative to the store
/// root) and size in bytes. The size lets a remote reader locate a Parquet
/// footer without a separate HEAD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    pub key: String,
    pub size: u64,
}

/// How the io_uring data plane should read one of a store's objects — a local
/// backend by filesystem path, a remote one by a presigned GET URL. The size
/// travels with it, so a reader can locate (e.g.) a Parquet footer without a
/// separate `stat`/HEAD. Produced by [`ObjectStore::data_file`]; which variant
/// you get is entirely the store's business, not the caller's.
#[derive(Clone, Debug)]
pub enum DataFileLocation {
    Local { path: PathBuf, size: u64 },
    Remote { url: url::Url, size: u64 },
}

/// A flat key→bytes object store rooted at one database. Keys are relative to
/// that root, e.g. `_pivot_manifest.json` or `events/a.parquet`.
pub trait ObjectStore: Debug + Send + Sync {
    /// Fetch an object in full, or `None` if it does not exist.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Atomically replace `key` with `data` (overwriting any existing object).
    /// Backs a small, rarely-written mutable control file like the table
    /// manifest; reads see either the old or the new object whole, never a torn
    /// write. (Concurrent writers are last-writer-wins — fine for the manifest,
    /// which a single server writes on the occasional `CREATE TABLE`.)
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// List objects directly under `prefix` (one level, not recursive), as keys
    /// (relative to the root) paired with their sizes.
    fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>>;

    /// How the io_uring reader should fetch object `key` (`size` bytes): a local
    /// backend yields a filesystem path, a remote one a presigned GET URL. The
    /// default store serves no readable data files (e.g. a pure in-memory store
    /// holds only the manifest) — local/remote backends override it.
    fn data_file(&self, _key: &str, _size: u64) -> Result<DataFileLocation> {
        Err(StoreError::Config(
            "this store has no readable data files".to_string(),
        ))
    }
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

/// The `*.parquet` files directly under a local directory, each as a
/// [`DataFileLocation::Local`] carrying its size. A missing directory yields
/// none. This is how a local table's data files are enumerated for the
/// metadata-fetch dataflow.
pub fn local_parquet_files(dir: &Path) -> std::io::Result<Vec<DataFileLocation>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "parquet") {
            let meta = entry.metadata()?;
            if meta.is_file() {
                files.push(DataFileLocation::Local {
                    path,
                    size: meta.len(),
                });
            }
        }
    }
    Ok(files)
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
        store.put("k", b"v").unwrap();
        // Reopening through a `file://` URI lands on the same root.
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let reopened = open_store(&uri).unwrap();
        assert_eq!(reopened.get("k").unwrap().unwrap(), b"v");
    }
}
