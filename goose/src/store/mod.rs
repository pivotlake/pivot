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

/// A table's data file: its name within the table's data location and its size
/// in bytes. The single durable file identity — the [table log] records a
/// `Vec<FileRef>`, [`ObjectStore::list`] returns these, and the catalog and
/// compacter speak them. The size lets a reader locate a Parquet footer without
/// a separate HEAD/`stat`.
///
/// [table log]: crate::table_log
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileRef {
    pub name: String,
    pub size: u64,
}

/// A [`FileRef`] located for reading: its size (locates the footer without a
/// `stat`/HEAD) and where its bytes live. Produced transiently by
/// [`ObjectStore::data_file`] / [`local_parquet_files`] and consumed straight by
/// the metadata fetcher — never stored. The file's *name* isn't needed to read
/// it; it stays in the [`FileRef`] the caller already holds.
#[derive(Clone, Debug)]
pub struct DataFile {
    pub size: u64,
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
    /// A data file on the local filesystem.
    pub fn local(path: PathBuf, size: u64) -> Self {
        Self {
            size,
            source: DataFileSource::Local(path),
        }
    }
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

    /// Create `key` with `data` only if it does not already exist — the
    /// compare-and-swap the versioned [table log](crate::table_log) builds its
    /// commits on. `Ok(true)` means this writer created the object; `Ok(false)`
    /// means the key was already there (the caller lost the race: re-read the
    /// latest state and retry at the next version).
    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<bool>;

    /// Delete `key`. Deleting an object that does not exist is not an error —
    /// the caller's goal (key absent) is already met.
    fn delete(&self, key: &str) -> Result<()>;

    /// List objects directly under `prefix` (one level, not recursive), as
    /// [`FileRef`]s (name within `prefix`, paired with size).
    fn list(&self, prefix: &str) -> Result<Vec<FileRef>>;

    /// How the io_uring reader should fetch object `key` (`size` bytes): a local
    /// backend yields a filesystem path, a remote one a presigned GET URL. The
    /// default store serves no readable data files (e.g. a pure in-memory store
    /// holds only the manifest) — local/remote backends override it.
    fn data_file(&self, _key: &str, _size: u64) -> Result<DataFile> {
        Err(StoreError::Config(
            "this store has no readable data files".to_string(),
        ))
    }

    /// Whether this store is a remote object store (S3/GCS) rather than the
    /// server's local filesystem. The catalog uses it to decide what an
    /// *absolute* table path means: on a local store an absolute path is a
    /// directory on the server's disk; on a remote store it is a key from the
    /// bucket root (see [`object_key`]). Local/in-memory stores keep the
    /// default.
    fn is_remote(&self) -> bool {
        false
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
/// [`FileRef`] (name + size). A missing directory yields none. This is how a
/// local table's data files are enumerated; the catalog locates each for the
/// metadata-fetch dataflow.
pub fn local_parquet_files(dir: &Path) -> std::io::Result<Vec<FileRef>> {
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
                files.push(FileRef {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    size: meta.len(),
                });
            }
        }
    }
    Ok(files)
}

/// The final path segment of a store key — the object's name within its
/// directory/prefix, as the table log records it.
pub(crate) fn key_name(key: &str) -> String {
    key.rsplit('/').next().unwrap_or(key).to_string()
}

/// Join a relative key onto a (possibly empty) directory/prefix. Shared by the
/// remote backends and the catalog's location handling.
pub(crate) fn join_prefix(prefix: &str, key: &str) -> String {
    let prefix = prefix.trim_matches('/');
    if prefix.is_empty() {
        key.trim_start_matches('/').to_string()
    } else {
        format!("{prefix}/{}", key.trim_start_matches('/'))
    }
}

/// The in-bucket object key a remote backend should address for a store key.
/// A leading `/` marks an **absolute** key — taken from the bucket root,
/// ignoring `prefix` (the database's own prefix within the bucket); any other
/// key lives under `prefix`. The "absolute = the store's root" companion to
/// [`join_prefix`].
pub(crate) fn object_key(prefix: &str, key: &str) -> String {
    if key.starts_with('/') {
        key.trim_start_matches('/').to_string()
    } else {
        join_prefix(prefix, key)
    }
}

/// Join `name` onto a table `location` for a store key, **preserving** whether
/// the location is absolute (a leading `/`, meaning the bucket root). The
/// absolute-aware companion to [`join_prefix`], used where the resulting key is
/// later handed to a store that interprets the leading `/` itself
/// ([`object_key`]).
pub(crate) fn location_key(location: &str, name: &str) -> String {
    let joined = join_prefix(location, name);
    if location.starts_with('/') {
        format!("/{joined}")
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_relative_lives_under_prefix_absolute_at_bucket_root() {
        // Relative keys hang under the database's own prefix in the bucket.
        assert_eq!(object_key("mydb", "events/a.parquet"), "mydb/events/a.parquet");
        // An absolute key escapes the database prefix to the bucket root.
        assert_eq!(object_key("mydb", "/shared/a.parquet"), "shared/a.parquet");
        // Bucket root with no database prefix configured behaves the same.
        assert_eq!(object_key("", "events/a.parquet"), "events/a.parquet");
        assert_eq!(object_key("", "/shared/a.parquet"), "shared/a.parquet");
    }

    #[test]
    fn location_key_preserves_absoluteness_for_the_store_to_interpret() {
        // A relative table location yields a relative key (store prepends its prefix).
        assert_eq!(location_key("events", "a.parquet"), "events/a.parquet");
        // An absolute table location keeps its leading slash so the store reads
        // it from the bucket root.
        assert_eq!(location_key("/shared/events", "a.parquet"), "/shared/events/a.parquet");
    }

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
