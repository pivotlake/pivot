//! The iceberg crate's file IO over Pivot's object stores.
//!
//! A query never uses it: every file a scan needs is read through Pivot's
//! ring. A commit does: the crate's transaction writes the manifests and the
//! manifest list of the snapshot it adds, reads the current ones, and deletes
//! the ones it wrote that nothing names (all of them when the commit was not
//! applied, the manifests it merged into another when it was), through the
//! file IO of the table it commits. An object is read whole: the storage
//! serves no ranged reads, which only a Parquet or Puffin reader would ask
//! for, and Pivot reads those files with its own engine.
//!
//! Every read, write and delete goes straight to the store, on one of the
//! runtime's blocking threads. Nothing read here passes through the ring or
//! the caches the queries' reads of the same objects fill, so a commit asks
//! the store for an object every time its transaction reads it. No call
//! drives a dataflow, so the file IO asks nothing of the worker pool: it is
//! driven on a runtime from whatever thread may wait on the store.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileIO, FileIOBuilder, FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind};
use object_storage::{ObjectPath, ObjectStore};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::metadata::layout::TableLocation;

/// Why a location could not be served.
pub type LocateError = Box<dyn std::error::Error + Send + Sync>;

/// Which store the object at a location is in, and its key there.
pub trait ObjectLocator: Send + Sync + 'static {
    fn locate(&self, uri: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath), LocateError>;
}

/// A table's files all live under its root; a location under any
/// other is refused.
impl ObjectLocator for TableLocation {
    fn locate(&self, uri: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath), LocateError> {
        Ok((self.store().clone(), self.find_key_for_uri(uri)?))
    }
}

impl TableLocation {
    /// The file IO of the table, its files read and written through its
    /// store.
    pub fn build_file_io(&self) -> FileIO {
        let storage = PivotStorage::new(Arc::new(self.clone()));
        FileIOBuilder::new(Arc::new(FixedStorageFactory { storage })).build()
    }
}

/// The crate's storage over Pivot's stores. Clones share the locator: the
/// crate clones a storage into every input and output file it hands out.
#[derive(Clone)]
pub struct PivotStorage {
    locator: Arc<dyn ObjectLocator>,
}

impl PivotStorage {
    pub fn new(locator: Arc<dyn ObjectLocator>) -> Self {
        Self { locator }
    }

    /// Run `call` against the store of the object at `uri` on a blocking
    /// thread: the store's calls block on the network, and the runtime's
    /// workers must not.
    async fn run_store_call<T: Send + 'static>(
        &self,
        uri: &str,
        call: impl FnOnce(&dyn ObjectStore, &ObjectPath) -> object_storage::Result<T> + Send + 'static,
    ) -> iceberg::Result<T> {
        let locator = self.locator.clone();
        let uri = uri.to_string();
        let store_result = tokio::task::spawn_blocking(move || -> Result<T, LocateError> {
            let (store, key) = locator.locate(&uri)?;
            Ok(call(store.as_ref(), &key)?)
        })
        .await
        .map_err(|panicked| {
            IcebergError::new(ErrorKind::Unexpected, "a store call panicked").with_source(panicked)
        })?;
        store_result.map_err(|error| {
            IcebergError::new(ErrorKind::Unexpected, format!("the table's store: {error}"))
        })
    }

    /// The size of the object at `uri`, or `None` when there is none.
    async fn size_of(&self, uri: &str) -> iceberg::Result<Option<u64>> {
        self.run_store_call(uri, |store, key| store.object_size(key))
            .await
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for PivotStorage {
    async fn exists(&self, path: &str) -> iceberg::Result<bool> {
        Ok(self.size_of(path).await?.is_some())
    }

    async fn metadata(&self, path: &str) -> iceberg::Result<FileMetadata> {
        let size = self.size_of(path).await?.ok_or_else(|| not_found(path))?;
        Ok(FileMetadata { size })
    }

    async fn read(&self, path: &str) -> iceberg::Result<Bytes> {
        let bytes = self
            .run_store_call(path, |store, key| store.get(key))
            .await?
            .ok_or_else(|| not_found(path))?;
        Ok(Bytes::from(bytes))
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        Err(IcebergError::new(
            ErrorKind::FeatureUnsupported,
            format!("`{path}` cannot be read by range: this storage reads metadata objects whole"),
        ))
    }

    async fn write(&self, path: &str, bytes: Bytes) -> iceberg::Result<()> {
        self.run_store_call(path, move |store, key| store.put(key, &bytes))
            .await
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        Ok(Box::new(BufferedWrite {
            storage: self.clone(),
            uri: path.to_string(),
            buffer: Vec::new(),
        }))
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        self.run_store_call(path, |store, key| store.delete(key))
            .await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        self.run_store_call(path, delete_under).await
    }

    async fn delete_stream(&self, mut paths: BoxStream<'static, String>) -> iceberg::Result<()> {
        while let Some(path) = paths.next().await {
            self.delete(&path).await?;
        }
        Ok(())
    }

    fn new_input(&self, path: &str) -> iceberg::Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> iceberg::Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

impl fmt::Debug for PivotStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PivotStorage").finish_non_exhaustive()
    }
}

/// Delete every object under `prefix`, level by level.
fn delete_under(store: &dyn ObjectStore, prefix: &ObjectPath) -> object_storage::Result<()> {
    let listing = store.list(prefix)?;
    for object in listing.objects {
        store.delete(&prefix.join(object.file.path.as_str()))?;
    }
    for child in listing.prefixes {
        delete_under(store, &prefix.join(child.as_str()))?;
    }
    Ok(())
}

/// A file written whole when closed: the store takes an object in one put.
struct BufferedWrite {
    storage: PivotStorage,
    uri: String,
    buffer: Vec<u8>,
}

#[async_trait]
impl FileWrite for BufferedWrite {
    async fn write(&mut self, bytes: Bytes) -> iceberg::Result<()> {
        self.buffer.extend_from_slice(&bytes);
        Ok(())
    }

    async fn close(&mut self) -> iceberg::Result<()> {
        let bytes = Bytes::from(std::mem::take(&mut self.buffer));
        self.storage.write(&self.uri, bytes).await
    }
}

fn not_found(uri: &str) -> IcebergError {
    IcebergError::new(
        ErrorKind::DataInvalid,
        format!("`{uri}` is not in the table's store"),
    )
}

/// Hands every file IO the one storage it was built around.
struct FixedStorageFactory {
    storage: PivotStorage,
}

#[typetag::serde]
impl StorageFactory for FixedStorageFactory {
    fn build(&self, _config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(self.storage.clone()))
    }
}

impl fmt::Debug for FixedStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedStorageFactory")
            .finish_non_exhaustive()
    }
}

/// The crate's storage traits are object-safe through typetag, which asks
/// every implementation to serialize. A storage bound to this process's
/// stores serializes to its name alone and is never deserialized: a file IO
/// is rebuilt by loading the table, not from bytes.
impl Serialize for PivotStorage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("PivotStorage")
    }
}

impl<'de> Deserialize<'de> for PivotStorage {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a storage is bound to a process's store and is not deserialized",
        ))
    }
}

impl Serialize for FixedStorageFactory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("FixedStorageFactory")
    }
}

impl<'de> Deserialize<'de> for FixedStorageFactory {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a storage factory is bound to a process's store and is not deserialized",
        ))
    }
}
