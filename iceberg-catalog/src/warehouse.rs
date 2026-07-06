//! Resolving the absolute URIs Iceberg metadata records (`s3://bucket/...`,
//! `gs://bucket/...`, `file:///...`) onto `catalog::store` backends: small
//! control-plane fetches (manifest lists, manifests) and ring-readable
//! [`DataFile`]s for the Parquet data itself. Stores are opened per bucket and
//! cached, since every URI in a warehouse repeats the same few roots.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use catalog::store::{DataFile, FileRef, ObjectPath, ObjectStore, open_store};

use crate::{Error, Result};

/// An opened warehouse: a cache of object stores keyed by store root
/// (`s3://bucket`, `gs://bucket`, or `/` for local paths).
#[derive(Debug, Default)]
pub(crate) struct WarehouseReader {
    stores: Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
}

impl WarehouseReader {
    /// Fetch the object at `uri` in full (`Ok(None)` when it does not exist) -
    /// manifest lists, manifests, and metadata JSON, the small reads iceberg's
    /// planning does through [`storage`](crate::storage).
    pub(crate) fn try_fetch(&self, uri: &str) -> Result<Option<Vec<u8>>> {
        let (store, key) = self.resolve_store(uri)?;
        Ok(store.get(&key)?)
    }

    /// Locate the Parquet data file at `uri` for the engine: its identity keeps
    /// the full URI, its size comes from the manifest (so no HEAD request), and
    /// its read source is whatever the store yields (a local path, a presigned
    /// URL, or a URL plus bearer auth).
    pub(crate) fn locate_data_file(&self, uri: &str, size: u64) -> Result<DataFile> {
        let (store, key) = self.resolve_store(uri)?;
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(uri),
                size,
            },
            source: store.source(&key)?,
        })
    }

    /// The store serving `uri` (opened once per root and cached) and the key of
    /// `uri` within it.
    fn resolve_store(&self, uri: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath)> {
        let (root, key) = split_uri(uri)?;
        let mut stores = self.stores.lock().unwrap();
        let store = match stores.get(&root) {
            Some(store) => store.clone(),
            None => {
                let store: Arc<dyn ObjectStore> = open_store(&root)?.into();
                stores.insert(root, store.clone());
                store
            }
        };
        Ok((store, ObjectPath::new(key)))
    }
}

/// Split an absolute warehouse URI into the store root to open and the key
/// within that store. Object-store URIs split at the bucket; `file://` URIs and
/// bare absolute paths use a store rooted at `/` with the absolute path as the
/// key (which `LocalStore` reads as-is).
fn split_uri(uri: &str) -> Result<(String, String)> {
    for scheme in ["s3://", "s3a://", "gs://"] {
        if let Some(rest) = uri.strip_prefix(scheme) {
            let (bucket, key) = rest
                .split_once('/')
                .filter(|(bucket, key)| !bucket.is_empty() && !key.is_empty())
                .ok_or_else(|| Error::UnsupportedUri(uri.to_string()))?;
            return Ok((format!("{scheme}{bucket}"), key.to_string()));
        }
    }
    // Both file-URI shapes appear in the wild: `file:///abs` (empty authority)
    // and Java writers' `file:/abs` (no authority).
    let path = uri
        .strip_prefix("file://")
        .or_else(|| uri.strip_prefix("file:"))
        .unwrap_or(uri);
    if !path.starts_with('/') {
        return Err(Error::UnsupportedUri(uri.to_string()));
    }
    Ok(("/".to_string(), path.to_string()))
}

#[cfg(test)]
mod tests {
    use super::split_uri;

    #[test]
    fn splits_object_store_uris_at_the_bucket() {
        assert_eq!(
            split_uri("s3://lake/warehouse/db/t/data/x.parquet").unwrap(),
            (
                "s3://lake".to_string(),
                "warehouse/db/t/data/x.parquet".to_string()
            )
        );
        assert_eq!(
            split_uri("gs://lake/w/m.avro").unwrap(),
            ("gs://lake".to_string(), "w/m.avro".to_string())
        );
    }

    #[test]
    fn file_uris_and_bare_paths_resolve_to_the_local_root() {
        assert_eq!(
            split_uri("file:///tmp/w/m.avro").unwrap(),
            ("/".to_string(), "/tmp/w/m.avro".to_string())
        );
        // Java writers render local URIs with a single slash (no authority).
        assert_eq!(
            split_uri("file:/tmp/w/m.avro").unwrap(),
            ("/".to_string(), "/tmp/w/m.avro".to_string())
        );
        assert_eq!(
            split_uri("/tmp/w/m.avro").unwrap(),
            ("/".to_string(), "/tmp/w/m.avro".to_string())
        );
    }

    #[test]
    fn rejects_relative_paths_and_unknown_schemes() {
        assert!(split_uri("warehouse/m.avro").is_err());
        assert!(split_uri("abfs://container/x").is_err());
        assert!(split_uri("s3://bucket-only").is_err());
    }
}
