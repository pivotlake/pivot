//! The iceberg crate's file IO over Pivot's object stores.
//!
//! A query never uses it: every file a scan needs is read through Pivot's
//! ring. A commit does: the crate's transaction writes the manifests and the
//! manifest list of the snapshot it adds, and reads the current ones, through
//! the file IO of the table it commits. Those objects are small, read once
//! and written once, so they go straight to the store, not over the pool, and
//! whole: the storage serves no ranged reads, which only a Parquet or Puffin
//! reader would ask for, and Pivot reads those files with its own engine.

use std::fmt;
use std::sync::{Arc, Mutex};

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

use crate::store::TableStore;

/// Why a location could not be served.
pub type LocateError = Box<dyn std::error::Error + Send + Sync>;

/// Which store the object at a location is in, and its key there.
pub trait ObjectLocator: Send + Sync + 'static {
    fn locate(&self, location: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath), LocateError>;
}

/// A table's files all live under its store's root; a location under any
/// other is refused.
impl ObjectLocator for TableStore {
    fn locate(&self, location: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath), LocateError> {
        Ok((self.store().clone(), self.key_of(location)?))
    }
}

impl TableStore {
    /// The file IO of a table whose files are read and written through this
    /// store, with a record of every object written through it.
    pub fn build_recording_file_io(&self) -> (FileIO, WrittenObjects) {
        let written = WrittenObjects::default();
        let storage = PivotStorage {
            locator: Arc::new(self.clone()),
            written: Some(written.clone()),
        };
        let file_io = FileIOBuilder::new(Arc::new(OneStorageFactory { storage })).build();
        (file_io, written)
    }
}

/// The objects a commit wrote, each by its location with its bytes: a commit
/// that does not succeed deletes them, and one that does reads its own
/// manifest list back from here rather than from the store.
#[derive(Clone, Default)]
pub struct WrittenObjects(Arc<Mutex<Vec<(String, Bytes)>>>);

impl WrittenObjects {
    /// The bytes written at `location`, if the commit wrote there.
    pub fn find(&self, location: &str) -> Option<Bytes> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|(written, _)| written == location)
            .map(|(_, bytes)| bytes.clone())
    }

    /// Delete every object written, through `store`. Best effort: an object
    /// left behind is an orphan a vacuum reclaims.
    pub fn withdraw(&self, store: &TableStore) {
        for (location, _) in self.0.lock().unwrap().drain(..) {
            let deleted = store
                .key_of(&location)
                .and_then(|key| Ok(store.store().delete(&key)?));
            if let Err(error) = deleted {
                tracing::warn!(location = %location, error = %error, "deleting a withdrawn commit's object failed");
            }
        }
    }
}

/// The crate's storage over Pivot's stores. Clones share the locator: the
/// crate clones a storage into every input and output file it hands out.
#[derive(Clone)]
pub struct PivotStorage {
    locator: Arc<dyn ObjectLocator>,
    /// Where the locations written are recorded, when the file IO was built
    /// to record them.
    written: Option<WrittenObjects>,
}

impl PivotStorage {
    pub fn new(locator: Arc<dyn ObjectLocator>) -> Self {
        Self {
            locator,
            written: None,
        }
    }

    /// Run `call` against the store of `location` on a blocking thread: the
    /// store's calls block on the network, and the runtime's workers must
    /// not.
    async fn blocking<T: Send + 'static>(
        &self,
        location: &str,
        call: impl FnOnce(&dyn ObjectStore, &ObjectPath) -> object_storage::Result<T> + Send + 'static,
    ) -> iceberg::Result<T> {
        let locator = self.locator.clone();
        let location = location.to_string();
        tokio::task::spawn_blocking(move || -> Result<T, LocateError> {
            let (store, key) = locator.locate(&location)?;
            Ok(call(store.as_ref(), &key)?)
        })
        .await
        .map_err(|panicked| {
            IcebergError::new(ErrorKind::Unexpected, "a store call panicked").with_source(panicked)
        })?
        .map_err(|error| {
            IcebergError::new(ErrorKind::Unexpected, format!("the table's store: {error}"))
        })
    }

    /// The size of the object at `path`, or `None` when there is none.
    async fn size_of(&self, path: &str) -> iceberg::Result<Option<u64>> {
        self.blocking(path, |store, key| store.object_size(key))
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
            .blocking(path, |store, key| store.get(key))
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
        let stored = bytes.clone();
        self.blocking(path, move |store, key| store.put(key, &stored))
            .await?;
        if let Some(written) = &self.written {
            written.0.lock().unwrap().push((path.to_string(), bytes));
        }
        Ok(())
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        Ok(Box::new(BufferedWrite {
            storage: self.clone(),
            path: path.to_string(),
            buffer: Vec::new(),
        }))
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        self.blocking(path, |store, key| store.delete(key)).await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        self.blocking(path, delete_under).await
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
    path: String,
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
        self.storage.write(&self.path, bytes).await
    }
}

fn not_found(path: &str) -> IcebergError {
    IcebergError::new(
        ErrorKind::DataInvalid,
        format!("`{path}` is not in the table's store"),
    )
}

/// Hands every file IO the one storage it was built around.
struct OneStorageFactory {
    storage: PivotStorage,
}

#[typetag::serde]
impl StorageFactory for OneStorageFactory {
    fn build(&self, _config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(self.storage.clone()))
    }
}

impl fmt::Debug for OneStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneStorageFactory").finish_non_exhaustive()
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

impl Serialize for OneStorageFactory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("OneStorageFactory")
    }
}

impl<'de> Deserialize<'de> for OneStorageFactory {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a storage factory is bound to a process's store and is not deserialized",
        ))
    }
}
