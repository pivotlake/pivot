//! iceberg's [`FileIO`](iceberg::io::FileIO) backed by `catalog::store`: the
//! metadata objects `plan_files` reads (table metadata JSON, manifest lists,
//! manifests) come through the same object-store backends and environment
//! credentials as everything else in pivot - no second storage stack in the
//! binary. Read-only: this catalog never writes, so every write entry point
//! errors.
//!
//! The sync store calls run under [`spawn_blocking`](tokio::task::spawn_blocking)
//! so `plan_files`' internal manifest-fetch concurrency actually overlaps the
//! GETs instead of serializing them on the runtime thread.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use iceberg::io::{FileMetadata, FileRead, FileWrite, InputFile, OutputFile};
use iceberg::io::{Storage, StorageConfig, StorageFactory};

use crate::warehouse::WarehouseReader;

/// Builds [`PivotStorage`] for every scheme; the reader routes each URI to the
/// right `catalog::store` backend itself.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PivotStorageFactory;

#[typetag::serde(name = "pivot")]
impl StorageFactory for PivotStorageFactory {
    fn build(&self, _config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(PivotStorage::default()))
    }
}

/// The read-only [`Storage`] over `catalog::store`. Objects are fetched whole:
/// everything read through here is small catalog metadata (KBs of JSON/Avro),
/// and `catalog::store` deliberately has no ranged GET (ranged reads belong to
/// the ring).
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct PivotStorage {
    #[serde(skip)]
    reader: Arc<WarehouseReader>,
}

/// Fetch `path` in full off the blocking pool (`None` when the object does not
/// exist), so concurrent metadata reads overlap instead of serializing the
/// runtime thread.
async fn try_fetch_blocking(
    reader: &Arc<WarehouseReader>,
    path: &str,
) -> iceberg::Result<Option<Vec<u8>>> {
    let reader = reader.clone();
    let path = path.to_string();
    tokio::task::spawn_blocking(move || reader.try_fetch(&path))
        .await
        .map_err(|e| {
            iceberg::Error::new(
                iceberg::ErrorKind::Unexpected,
                format!("storage read task failed: {e}"),
            )
        })?
        .map_err(to_iceberg_error)
}

/// [`try_fetch_blocking`], with a missing object as an error naming the path.
async fn fetch_blocking(reader: &Arc<WarehouseReader>, path: &str) -> iceberg::Result<Vec<u8>> {
    try_fetch_blocking(reader, path).await?.ok_or_else(|| {
        iceberg::Error::new(
            iceberg::ErrorKind::Unexpected,
            format!("`{path}` does not exist in the object store"),
        )
    })
}

fn to_iceberg_error(e: crate::Error) -> iceberg::Error {
    iceberg::Error::new(iceberg::ErrorKind::Unexpected, e.to_string())
}

fn read_only_error(operation: &str) -> iceberg::Error {
    iceberg::Error::new(
        iceberg::ErrorKind::FeatureUnsupported,
        format!("pivot's iceberg storage is read-only ({operation})"),
    )
}

#[async_trait]
#[typetag::serde(name = "pivot")]
impl Storage for PivotStorage {
    async fn exists(&self, path: &str) -> iceberg::Result<bool> {
        // `catalog::store` has no stat; a full read of a small metadata object
        // is an acceptable existence probe on this cold path.
        Ok(try_fetch_blocking(&self.reader, path).await?.is_some())
    }

    async fn metadata(&self, path: &str) -> iceberg::Result<FileMetadata> {
        let bytes = fetch_blocking(&self.reader, path).await?;
        Ok(FileMetadata {
            size: bytes.len() as u64,
        })
    }

    async fn read(&self, path: &str) -> iceberg::Result<Bytes> {
        Ok(Bytes::from(fetch_blocking(&self.reader, path).await?))
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        Ok(Box::new(PivotFileRead {
            reader: self.reader.clone(),
            path: path.to_string(),
        }))
    }

    async fn write(&self, _path: &str, _bs: Bytes) -> iceberg::Result<()> {
        Err(read_only_error("write"))
    }

    async fn writer(&self, _path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        Err(read_only_error("writer"))
    }

    async fn delete(&self, _path: &str) -> iceberg::Result<()> {
        Err(read_only_error("delete"))
    }

    async fn delete_prefix(&self, _path: &str) -> iceberg::Result<()> {
        Err(read_only_error("delete_prefix"))
    }

    fn new_input(&self, path: &str) -> iceberg::Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, _path: &str) -> iceberg::Result<OutputFile> {
        Err(read_only_error("new_output"))
    }
}

/// A [`FileRead`] over one object: each ranged read fetches the object whole
/// and slices it. Fine here - the planning path reads manifests in full, so
/// this is a rarely-taken compatibility surface, not a data path.
#[derive(Debug)]
struct PivotFileRead {
    reader: Arc<WarehouseReader>,
    path: String,
}

#[async_trait]
impl FileRead for PivotFileRead {
    async fn read(&self, range: std::ops::Range<u64>) -> iceberg::Result<Bytes> {
        let bytes = fetch_blocking(&self.reader, &self.path).await?;
        let (start, end) = (range.start as usize, range.end as usize);
        if end > bytes.len() || start > end {
            return Err(iceberg::Error::new(
                iceberg::ErrorKind::DataInvalid,
                format!(
                    "range {start}..{end} out of bounds for `{}` ({} bytes)",
                    self.path,
                    bytes.len()
                ),
            ));
        }
        Ok(Bytes::from(bytes).slice(start..end))
    }
}
