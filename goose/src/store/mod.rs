//! A generic key→bytes object store — local filesystem, S3, or GCS — and nothing
//! catalog-specific. It knows how to `get`/`put`/`list`/`delete` objects, do a
//! conditional create ([`ObjectStore::put_if_absent`], the table log's CAS),
//! and turn a key into a ring-readable [`DataFile`]; the table manifest,
//! table log, and catalog build on it one layer up.
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3/GCS go over [`ureq`] (blocking HTTP + rustls).
//! S3 requests are signed with `aws_sigv4::http_request::sign` — a pure function
//! we call inline (the tokio it transitively links is never driven). Credentials
//! come from the environment. This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use std::fmt::Debug;
use std::path::PathBuf;

mod gcs;
mod local;
mod object_path;
mod s3;
pub use gcs::GcsStore;
pub use local::LocalStore;
pub use object_path::ObjectPath;
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

/// A table's data file: its [`ObjectPath`] in the store and its size in bytes.
/// The single durable file identity — the [table manifest] records a
/// `Vec<FileRef>`, [`ObjectStore::list`] returns these, and the catalog and
/// compacter speak them. The path reads the file directly (no re-joining a
/// location); the size lets a reader locate a Parquet footer without a separate
/// HEAD/`stat`.
///
/// [table manifest]: crate::manifest::TableManifest
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileRef {
    pub path: ObjectPath,
    pub size: u64,
}

/// A [`FileRef`] located for reading: its identity (`file`, whose size locates
/// the footer without a `stat`/HEAD) plus where its bytes live (`source`).
/// Produced transiently by [`ObjectStore::data_file`] and consumed straight by
/// the metadata fetcher, which stamps the `file` onto the `TableFile` it emits —
/// so the file's identity travels with its bytes through the load.
#[derive(Clone, Debug)]
pub struct DataFile {
    pub file: FileRef,
    pub source: DataFileSource,
}

/// Where a data file's bytes live: a local filesystem path (read via the
/// io_uring file path) or a concrete — already presigned — URL (read via HTTP
/// range requests on the same ring). Which variant a store yields is entirely
/// its business, not the caller's.
#[derive(Clone, Debug)]
pub enum DataFileSource {
    Local(PathBuf),
    Remote(url::Url),
}

impl DataFile {
    /// A data file on the local filesystem; its [`FileRef`] path is the
    /// filesystem path (these constructors feed the whole-directory readers,
    /// where the path is the file's identity directly).
    pub fn local(path: PathBuf, size: u64) -> Self {
        Self {
            file: FileRef {
                path: ObjectPath::new(path.to_string_lossy().into_owned()),
                size,
            },
            source: DataFileSource::Local(path),
        }
    }

    /// A data file at a concrete (already-presigned) remote URL; its [`FileRef`]
    /// path is the URL's path component.
    pub fn remote(url: url::Url, size: u64) -> Self {
        Self {
            file: FileRef {
                path: ObjectPath::new(url.path()),
                size,
            },
            source: DataFileSource::Remote(url),
        }
    }
}

/// A flat key→bytes object store rooted at one database. [`ObjectPath`] keys are
/// relative to that root, e.g. `_pivot_manifest.json` or `events/a.parquet`.
pub trait ObjectStore: Debug + Send + Sync {
    /// Fetch an object in full, or `None` if it does not exist.
    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>>;

    /// Atomically replace `key` with `data` (overwriting any existing object).
    /// Backs a small, rarely-written mutable control file like the database
    /// manifest; reads see either the old or the new object whole, never a torn
    /// write. (Concurrent writers are last-writer-wins — fine for the database
    /// index, which a single server writes on the occasional `CREATE TABLE`.)
    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()>;

    /// Create `key` with `data` only if it does not already exist — the
    /// compare-and-swap the versioned [table manifest](crate::manifest) builds
    /// its commits on. `Ok(true)` means this writer created the object; `Ok(false)`
    /// means the key was already there (the caller lost the race: re-read the
    /// latest state and retry at the next version).
    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<bool>;

    /// Delete `key`. Deleting an object that does not exist is not an error —
    /// the caller's goal (key absent) is already met.
    fn delete(&self, key: &ObjectPath) -> Result<()>;

    /// List objects directly under `prefix` (one level, not recursive), as
    /// [`FileRef`]s — each a full [`ObjectPath`] (`prefix` joined with the
    /// object's name) paired with its size.
    fn list(&self, prefix: &ObjectPath) -> Result<Vec<FileRef>>;

    /// How the io_uring reader should fetch object `key` (`size` bytes): a local
    /// backend yields a filesystem path, a remote one a presigned GET URL.
    fn data_file(&self, key: &ObjectPath, size: u64) -> Result<DataFile>;
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

/// The final path segment of a raw object-store key string (a backend's listing
/// response), as the manifest records the object's name.
pub(crate) fn key_name(key: &str) -> String {
    key.rsplit('/').next().unwrap_or(key).to_string()
}

/// The in-bucket object key a remote backend should address for a store key.
/// An [absolute](ObjectPath::is_absolute) key is taken from the bucket root,
/// ignoring `prefix` (the database's own prefix within the bucket); any other
/// key lives under `prefix`.
pub(crate) fn object_key(prefix: &str, key: &ObjectPath) -> String {
    if key.is_absolute() {
        key.as_str().trim_start_matches('/').to_string()
    } else {
        let prefix = prefix.trim_matches('/');
        if prefix.is_empty() {
            key.as_str().to_string()
        } else {
            format!("{prefix}/{}", key.as_str())
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_relative_lives_under_prefix_absolute_at_bucket_root() {
        // Relative keys hang under the database's own prefix in the bucket.
        assert_eq!(object_key("mydb", &ObjectPath::new("events/a.parquet")), "mydb/events/a.parquet");
        // An absolute key escapes the database prefix to the bucket root.
        assert_eq!(object_key("mydb", &ObjectPath::new("/shared/a.parquet")), "shared/a.parquet");
        // Bucket root with no database prefix configured behaves the same.
        assert_eq!(object_key("", &ObjectPath::new("events/a.parquet")), "events/a.parquet");
        assert_eq!(object_key("", &ObjectPath::new("/shared/a.parquet")), "shared/a.parquet");
    }

    #[test]
    fn open_store_routes_local_and_file_uri() {
        let dir = tempfile::tempdir().unwrap();
        let store = open_store(dir.path().to_str().unwrap()).unwrap();
        let key = ObjectPath::new("k");
        store.put(&key, b"v").unwrap();
        // Reopening through a `file://` URI lands on the same root.
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let reopened = open_store(&uri).unwrap();
        assert_eq!(reopened.get(&key).unwrap().unwrap(), b"v");
    }
}
