//! Object storage for the catalog: listing the `_goose_log/` directory and
//! fetching snapshot bytes.
//!
//! This is the catalog **control plane** — a thin wrapper over the
//! [`object_store`] crate (local fs, S3, GCS). Per query the planner asks for
//! the latest snapshot ([`GooseStore::latest_snapshot`]); that's one LIST plus
//! one GET of a few KB of JSON. It runs off the io_uring ring because
//! `object_store` already owns the cloud credentials and a LIST is not a
//! range-GET the ring can serve — but it's rare and tiny, so it costs nothing
//! next to the hot column-chunk reads (which stay on the ring). Data-file
//! *bytes* never flow through here.
//!
//! `object_store`'s API is async; we drive it from the synchronous planner via a
//! per-store current-thread Tokio runtime. The calls are infrequent and off the
//! hot path, so `block_on` is fine. (Planning is synchronous — it is not invoked
//! from inside another runtime, so there is no nested-runtime hazard.)

use crate::metadata::{CatalogSnapshot, MetadataError};
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::ObjectStore;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Directory (key prefix) under the catalog root holding the versioned snapshot
/// files: `_goose_log/<zero-padded-version>.json`.
const LOG_DIR: &str = "_goose_log";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("unsupported catalog uri `{0}` (expected a local path, file://, s3://, or gs://)")]
    UnsupportedUri(String),
    #[error("building object store for `{uri}`: {source}")]
    Build {
        uri: String,
        #[source]
        source: object_store::Error,
    },
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("could not start async runtime for object store: {0}")]
    Runtime(#[source] std::io::Error),
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// A catalog root on some object store: the backing client, the base prefix
/// within it (everything — `_goose_log/` and data files — lives under this), and
/// a runtime to drive the async API synchronously.
pub struct GooseStore {
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    rt: Runtime,
    uri: String,
}

impl std::fmt::Debug for GooseStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GooseStore").field("uri", &self.uri).finish()
    }
}

impl GooseStore {
    /// Open the catalog root at `uri`: `s3://bucket/prefix`, `gs://bucket/prefix`,
    /// or a local path (optionally `file://`). Cloud credentials come from the
    /// environment (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, workload identity),
    /// resolved by `object_store`.
    pub fn open(uri: &str) -> Result<Self> {
        let (store, root) = build_object_store(uri)?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(StoreError::Runtime)?;
        Ok(Self {
            store,
            root,
            rt,
            uri: uri.to_string(),
        })
    }

    /// The once-per-query "latest snapshot" load: LIST `_goose_log/`, pick the
    /// highest version, GET and parse it. Returns the empty snapshot if the log
    /// is empty (a catalog root that exists but has no commits yet).
    pub fn latest_snapshot(&self) -> Result<CatalogSnapshot> {
        let log_prefix = self.root.child(LOG_DIR);

        let latest_key = self.rt.block_on(async {
            // The log is a flat directory of `<version>.json` files, so a single
            // delimited list (not a recursive stream) enumerates every commit.
            let listing = self.store.list_with_delimiter(Some(&log_prefix)).await?;
            let best = listing
                .objects
                .into_iter()
                .filter_map(|meta| parse_version(&meta.location).map(|v| (v, meta.location)))
                .max_by_key(|(v, _)| *v)
                .map(|(_, key)| key);
            Ok::<_, object_store::Error>(best)
        })?;

        let Some(key) = latest_key else {
            return Ok(CatalogSnapshot::empty());
        };

        let bytes = self
            .rt
            .block_on(async { self.store.get(&key).await?.bytes().await })?;
        Ok(CatalogSnapshot::from_slice(&bytes)?)
    }

    /// The catalog root URI, for diagnostics.
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

/// Parse a snapshot version from a `_goose_log/<version>.json` path's filename.
/// Returns `None` for anything that isn't a `<digits>.json` file.
fn parse_version(path: &StorePath) -> Option<i64> {
    path.filename()?.strip_suffix(".json")?.parse::<i64>().ok()
}

/// Build an [`ObjectStore`] and base prefix from a catalog-root URI. Mirrors
/// ingest's `build_object_store` so the workspace resolves one client per scheme.
fn build_object_store(uri: &str) -> Result<(Arc<dyn ObjectStore>, StorePath)> {
    let build_err = |source| StoreError::Build {
        uri: uri.to_string(),
        source,
    };

    if let Some(rest) = uri.strip_prefix("gs://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(build_err)?;
        Ok((Arc::new(store), StorePath::from(prefix)))
    } else if let Some(rest) = uri
        .strip_prefix("s3://")
        .or_else(|| uri.strip_prefix("s3a://"))
    {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(build_err)?;
        Ok((Arc::new(store), StorePath::from(prefix)))
    } else {
        // Local filesystem. The store is rooted at the directory, so the base
        // prefix within it is empty.
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        let store = LocalFileSystem::new_with_prefix(path).map_err(build_err)?;
        Ok((Arc::new(store), StorePath::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Column, DataFile, Schema, Table, FORMAT_VERSION};
    use std::fs;

    fn snapshot(version: i64, table: &str) -> CatalogSnapshot {
        CatalogSnapshot {
            format_version: FORMAT_VERSION,
            version,
            schemas: vec![Schema {
                name: "main".into(),
                tables: vec![Table {
                    name: table.into(),
                    columns: vec![Column { name: "id".into(), type_sql: "INTEGER".into() }],
                    files: vec![DataFile {
                        location: "_goose_data/main/t/a.parquet".into(),
                        size: Some(123),
                        row_count: 1,
                    }],
                }],
            }],
        }
    }

    fn write_log(root: &std::path::Path, version: i64, snap: &CatalogSnapshot) {
        let log = root.join(LOG_DIR);
        fs::create_dir_all(&log).unwrap();
        fs::write(log.join(format!("{version:020}.json")), snap.to_vec()).unwrap();
    }

    #[test]
    fn empty_log_yields_empty_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(LOG_DIR)).unwrap();
        let store = GooseStore::open(dir.path().to_str().unwrap()).unwrap();
        let snap = store.latest_snapshot().unwrap();
        assert_eq!(snap.version, 0);
    }

    #[test]
    fn picks_highest_version_across_commits() {
        let dir = tempfile::tempdir().unwrap();
        // Write out of order to prove it sorts by parsed version, not listing order.
        write_log(dir.path(), 2, &snapshot(2, "events"));
        write_log(dir.path(), 10, &snapshot(10, "latest"));
        write_log(dir.path(), 1, &snapshot(1, "first"));

        let store = GooseStore::open(dir.path().to_str().unwrap()).unwrap();
        let snap = store.latest_snapshot().unwrap();
        assert_eq!(snap.version, 10);
        assert!(snap.table("main", "latest").is_some());
        assert!(snap.table("main", "events").is_none());
    }

    #[test]
    fn file_uri_prefix_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        write_log(dir.path(), 1, &snapshot(1, "t"));
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let store = GooseStore::open(&uri).unwrap();
        assert_eq!(store.latest_snapshot().unwrap().version, 1);
    }
}
