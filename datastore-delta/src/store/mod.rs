//! A generic key→bytes object store — local filesystem, S3, or GCS — and nothing
//! catalog-specific. It knows how to `get`/`put`/`list`/`delete` objects, do a
//! conditional create ([`ObjectStore::put_if_absent`], the table log's CAS),
//! and turn a key into a ring-readable [`DataFile`]; the table manifest,
//! table log, and catalog build on it one layer up.
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3 and GCS go over [`ureq`] (blocking HTTP +
//! rustls). S3 requests are signed with `aws_sigv4::http_request::sign` — a pure
//! function we call inline (the tokio it transitively links is never driven);
//! GCS requests carry an OAuth2 bearer token. Credentials
//! come from whoever opened the store: a metastore hands them in per datastore,
//! and [`open_store`] falls back to the environment. A backend also builds the
//! async client Delta Kernel reads the same location's `_delta_log` with
//! ([`ObjectStore::build_delta_object_store`]), so the two address it alike.
//! This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use delta_kernel::object_store::DynObjectStore;
use dispatch::io::{AuthHeader, OpenFile, RemoteFile, open_direct_read};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

mod gcs;
mod local;
mod object_path;
mod s3;
pub use gcs::GcsStore;
pub use local::LocalStore;
pub use object_path::ObjectPath;
pub use s3::{S3Credentials, S3Store};

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
    #[error("cannot open `{uri}` for Delta: {source}")]
    DeltaObjectStore {
        uri: String,
        #[source]
        source: delta_kernel::object_store::Error,
    },
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// A table's data file: its [`ObjectPath`] in the store and its size in bytes.
/// The single durable file identity — a Delta log entry records one,
/// [`ObjectStore::list`] returns these, and the catalog and compacter speak
/// them. The path reads the file directly (no re-joining a location); the size
/// lets a reader locate a Parquet footer without a separate HEAD/`stat`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileRef {
    pub path: ObjectPath,
    pub size: u64,
}

/// A listed object paired with its storage modification time (Unix
/// milliseconds): the backend's own last-modified stamp — a filesystem mtime, or
/// S3's `LastModified`. Kept out of [`FileRef`] because that timestamp is a
/// backend clock reading, not part of a file's durable identity; only vacuum's
/// orphan detection needs it, where an unreferenced file's mtime is the sole
/// "how new is this" signal for deciding it is safe to delete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedObject {
    pub file: FileRef,
    pub modified_unix_ms: u64,
}

impl FileRef {
    /// Turn this ref into a [`DataFile`] to read it for a table at `location`:
    /// its path is relative to the table's location (an absolute one escapes to
    /// the store root), so resolve it against `location` for the read source —
    /// keeping the ref itself as the identity that flows onto the `TableFile` and
    /// matches the manifest.
    pub(crate) fn into_data_file(
        self,
        store: &dyn ObjectStore,
        location: &ObjectPath,
    ) -> Result<DataFile> {
        let source = store.source(&location.resolve(&self.path))?;
        Ok(DataFile { file: self, source })
    }
}

/// A [`FileRef`] located for reading: its identity (`file`, whose size locates
/// the footer without a `stat`/HEAD) plus where its bytes live (`source`).
/// Produced transiently by [`FileRef::into_data_file`] and consumed straight by
/// the metadata fetcher, which stamps the `file` onto the `TableFile` it emits —
/// so the file's identity travels with its bytes through the load.
#[derive(Clone, Debug)]
pub struct DataFile {
    pub file: FileRef,
    pub source: DataFileLocation,
}

/// Where a data file's bytes live, for reading or writing: a local filesystem
/// path (via the io_uring file path) or a remote URL (HTTP on the same ring). A
/// remote URL either carries its own auth (an S3 presigned URL, `auth: None`) or
/// pairs a stable URL with an [`AuthHeader`] that mints a fresh bearer token per
/// request. Which variant a store yields is its business, not the caller's.
///
/// The same key maps to a different URL per direction — [`source`](ObjectStore::source)
/// builds the read URL, [`sink`](ObjectStore::sink) the write URL (e.g. S3 presigns
/// a GET vs a PUT) — so both return this one type. Uploads are always a `PUT`.
#[derive(Clone)]
pub enum DataFileLocation {
    Local(PathBuf),
    Remote {
        url: url::Url,
        auth: Option<AuthHeader>,
    },
}

impl DataFileLocation {
    /// Open this read location as a ring-readable [`OpenFile`]: a local path
    /// becomes an `O_DIRECT` fd, a remote URL a [`RemoteFile`] (`size` locates
    /// its footer tail with no HEAD probe). Only valid once the file exists, so
    /// callers open it after the listing that found it or the upload that wrote
    /// it. This is the read (`GET`) counterpart; a `sink()` location is a `PUT`.
    pub(crate) fn open_read(self, size: u64) -> std::io::Result<OpenFile> {
        Ok(match self {
            DataFileLocation::Local(path) => OpenFile::Local(Arc::new(open_direct_read(&path)?)),
            DataFileLocation::Remote { url, auth } => {
                OpenFile::Remote(Arc::new(RemoteFile::open(url, auth, size)?))
            }
        })
    }
}

impl Debug for DataFileLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(path) => f.debug_tuple("Local").field(path).finish(),
            Self::Remote { url, auth } => f
                .debug_struct("Remote")
                .field("url", url)
                .field("auth", &auth.is_some())
                .finish(),
        }
    }
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
            source: DataFileLocation::Local(path),
        }
    }

    /// A data file at a self-authenticating remote URL (no per-request auth); its
    /// [`FileRef`] path is the URL's path component.
    pub fn remote(url: url::Url, size: u64) -> Self {
        Self {
            file: FileRef {
                path: ObjectPath::new(url.path()),
                size,
            },
            source: DataFileLocation::Remote { url, auth: None },
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

    /// List objects directly under `prefix` (one level, not recursive), each as
    /// a [`ListedObject`] — a **relative** [`ObjectPath`] (the object's name
    /// within `prefix`) with its size and its storage modification time (Unix
    /// ms).
    fn list(&self, prefix: &ObjectPath) -> Result<Vec<ListedObject>>;

    /// How the io_uring reader should fetch object `key`: a local backend yields
    /// a filesystem path, a remote one a GET URL — either presigned or paired
    /// with a per-request bearer token. (Identity — the [`FileRef`] — is the
    /// caller's; this is only how to read the bytes.)
    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation>;

    /// How a worker should asynchronously create a data object at `key`.
    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation>;

    /// Prime any credentials workers will need for writes. Called on the
    /// coordinator before launching an INSERT dataflow.
    fn prepare_write(&self) -> Result<()> {
        Ok(())
    }

    /// Ensure the directory named by `prefix` exists so a writer can create
    /// objects under it. A local filesystem backend must create it, because a
    /// writer that addresses the prefix as a real path — Delta Kernel canonicalizes
    /// a table root before writing its log — fails if the directory is missing.
    fn create_dir(&self, prefix: &ObjectPath) -> Result<()>;

    /// A human-readable description of where this store is rooted - e.g.
    /// `file:///var/lib/pivot`, `s3://bucket/prefix`, or `gs://bucket/prefix`.
    /// Purely for diagnostics and introspection (a dashboard showing whether a
    /// table lives on local disk or object storage); never an addressable key.
    /// The default falls back to the backend's `Debug` form.
    fn describe(&self) -> String {
        format!("{self:?}")
    }

    /// The machine-addressable URI this store is rooted at — `file:///path` for
    /// a local store, the `s3://bucket/prefix` or `gs://bucket/prefix` it was
    /// opened with for a remote one. This
    /// is what a durable reference to the store's contents (e.g. a table's
    /// Delta log location) is derived from, so it must be a parseable URL whose
    /// path includes the store's key prefix; contrast [`describe`](Self::describe),
    /// which is free-form text for humans. Required (no default): a backend
    /// cannot silently fall back to something unaddressable.
    fn location_uri(&self) -> String;

    /// Build the separate asynchronous object-store client Delta Kernel uses to
    /// read this store's `_delta_log`. This returns Delta Kernel's
    /// [`DynObjectStore`] trait object, not this blocking [`ObjectStore`], but
    /// configures it from the same location and credentials as this backend.
    /// Required (no default): a backend cannot silently leave Delta Kernel to
    /// resolve its own credentials, which for an unconfigured S3 client can mean
    /// a slow failed probe of the instance metadata service.
    fn build_delta_object_store(&self) -> Result<Arc<DynObjectStore>>;
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

/// Percent-encode one object-key component for a URL path segment or query
/// value per RFC 3986: unreserved characters pass through, everything else
/// (including `/`) is escaped.
pub(crate) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &byte in s.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Parse an object store's listing timestamp (RFC 3339, always UTC — S3's
/// `LastModified`, GCS's `updated`) into Unix milliseconds. Returns `None` on
/// any malformed field, which the caller turns into a listing error rather than
/// a silently-wrong (too-old) timestamp.
pub(crate) fn parse_iso8601_millis(s: &str) -> Option<u64> {
    let nanos = arrow_cast::parse::string_to_timestamp_nanos(s).ok()?;
    u64::try_from(nanos / 1_000_000).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_relative_lives_under_prefix_absolute_at_bucket_root() {
        // Relative keys hang under the database's own prefix in the bucket.
        assert_eq!(
            object_key("mydb", &ObjectPath::new("events/a.parquet")),
            "mydb/events/a.parquet"
        );
        // An absolute key escapes the database prefix to the bucket root.
        assert_eq!(
            object_key("mydb", &ObjectPath::new("/shared/a.parquet")),
            "shared/a.parquet"
        );
        // Bucket root with no database prefix configured behaves the same.
        assert_eq!(
            object_key("", &ObjectPath::new("events/a.parquet")),
            "events/a.parquet"
        );
        assert_eq!(
            object_key("", &ObjectPath::new("/shared/a.parquet")),
            "shared/a.parquet"
        );
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
