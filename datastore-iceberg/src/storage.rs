//! The iceberg crate's file IO over Pivot's object stores.
//!
//! The catalog client builds a file IO for every table it loads, from the
//! factory here and the storage properties the load-table response carried,
//! among them the credentials the catalog vended for the table. A query never
//! uses it: every file a scan needs is read through Pivot's ring. A commit
//! does: the crate's transaction writes the manifest and manifest list of the
//! snapshot it adds, and reads the current manifest list, through the file IO
//! of the table it commits. Those objects are small, read once and written
//! once, so they go straight to the store, not over the pool, and whole: the
//! storage serves no ranged reads, which only a Parquet or Puffin reader
//! would ask for, and Pivot reads those files with its own engine.
//!
//! A file IO belongs to one table, whose files all live under one root (the
//! bucket, or the local filesystem). The storage opens that root's store on
//! the first location it is asked for and serves every later one under it; a
//! location under another root is refused rather than served through a store
//! it was not opened for.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind};
use object_storage::{ExternalStoreFactory, ObjectPath, ObjectStore};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::store::{is_same_root, open_store_at_root, read_vended_s3_credentials, split_location};

/// Builds the file IO of every table the catalog client loads: a storage that
/// opens the table's root through the process's store factory, or with the
/// credentials the catalog vended for the table when it vended any.
pub(crate) struct PivotStorageFactory {
    store_factory: Arc<dyn ExternalStoreFactory>,
}

impl PivotStorageFactory {
    pub(crate) fn new(store_factory: Arc<dyn ExternalStoreFactory>) -> Self {
        Self { store_factory }
    }
}

#[typetag::serde]
impl StorageFactory for PivotStorageFactory {
    /// Nothing is opened here: the root is known only from the first
    /// location the storage serves, and most file IOs, those of the tables a
    /// refresh loads, serve none.
    fn build(&self, config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(PivotStorage {
            inner: Arc::new(StorageInner {
                store_factory: self.store_factory.clone(),
                properties: config.props().clone(),
                store: OnceLock::new(),
            }),
        }))
    }
}

impl fmt::Debug for PivotStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PivotStorageFactory")
            .finish_non_exhaustive()
    }
}

/// The crate's storage over one table's root. Clones share the opened store:
/// the crate clones a storage into every input and output file it hands out.
#[derive(Clone)]
pub(crate) struct PivotStorage {
    inner: Arc<StorageInner>,
}

struct StorageInner {
    store_factory: Arc<dyn ExternalStoreFactory>,
    /// The table's storage properties, as the catalog client merged them: the
    /// vended credentials are read from here when the root is opened.
    properties: HashMap<String, String>,
    /// The store opened on the first location served, with the root it
    /// serves, set once: every later location must be under that root.
    store: OnceLock<OpenedStore>,
}

struct OpenedStore {
    root: String,
    object_store: Arc<dyn ObjectStore>,
}

impl StorageInner {
    /// The store `location` is served through and its key under the root,
    /// opening the root on the first location served and refusing a location
    /// under any other.
    fn locate(&self, location: &str) -> crate::Result<(Arc<dyn ObjectStore>, ObjectPath)> {
        let (root, key) = split_location(location)?;
        let opened = match self.store.get() {
            Some(opened) => opened,
            None => {
                let vended = read_vended_s3_credentials(location, &self.properties)?;
                let object_store = open_store_at_root(&root, self.store_factory.as_ref(), vended)?;
                // Two first locations racing here open two stores; the one
                // set serves both, and the other is checked below as any
                // later location is.
                let _ = self.store.set(OpenedStore {
                    root: root.clone(),
                    object_store,
                });
                self.store.get().expect("set just above")
            }
        };
        if !is_same_root(&opened.root, &root) {
            return Err(crate::Error::LocationOutsideRoot {
                location: location.to_string(),
                root: opened.root.clone(),
            });
        }
        Ok((opened.object_store.clone(), key))
    }
}

impl PivotStorage {
    /// Run `call` against the store on a blocking thread: the store's calls
    /// block on the network, and the runtime's workers must not.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&StorageInner) -> crate::Result<T> + Send + 'static,
    ) -> iceberg::Result<T> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || call(&inner))
            .await
            .map_err(|panicked| {
                IcebergError::new(ErrorKind::Unexpected, "a store call panicked")
                    .with_source(panicked)
            })?
            .map_err(into_iceberg_error)
    }

    /// The size of the object at `path`, or `None` when there is none.
    async fn size_of(&self, path: &str) -> iceberg::Result<Option<u64>> {
        let location = path.to_string();
        self.blocking(move |inner| {
            let (store, key) = inner.locate(&location)?;
            Ok(store.object_size(&key)?)
        })
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
        let location = path.to_string();
        let bytes = self
            .blocking(move |inner| {
                let (store, key) = inner.locate(&location)?;
                Ok(store.get(&key)?)
            })
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
        let location = path.to_string();
        self.blocking(move |inner| {
            let (store, key) = inner.locate(&location)?;
            Ok(store.put(&key, &bytes)?)
        })
        .await
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        Ok(Box::new(BufferedWrite {
            storage: self.clone(),
            path: path.to_string(),
            buffer: Vec::new(),
        }))
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |inner| {
            let (store, key) = inner.locate(&location)?;
            Ok(store.delete(&key)?)
        })
        .await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |inner| {
            let (store, prefix) = inner.locate(&location)?;
            delete_under(store.as_ref(), &prefix)
        })
        .await
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
        f.debug_struct("PivotStorage")
            .field("root", &self.inner.store.get().map(|opened| &opened.root))
            .finish_non_exhaustive()
    }
}

/// Delete every object under `prefix`, level by level.
fn delete_under(store: &dyn ObjectStore, prefix: &ObjectPath) -> crate::Result<()> {
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

fn into_iceberg_error(error: crate::Error) -> IcebergError {
    IcebergError::new(ErrorKind::Unexpected, "the table's store").with_source(error)
}

/// The crate's storage traits are object-safe through typetag, which asks
/// every implementation to serialize. A storage bound to this process's
/// stores serializes to its name alone and is never deserialized: a file IO
/// is rebuilt by loading the table, not from bytes.
impl Serialize for PivotStorageFactory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("PivotStorageFactory")
    }
}

impl<'de> Deserialize<'de> for PivotStorageFactory {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a storage factory is bound to a process's stores and is not deserialized",
        ))
    }
}

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

#[cfg(test)]
mod tests {
    use object_storage::{LocalStore, StoreError};

    use super::*;

    /// Opens the local filesystem for a `file:///` root.
    #[derive(Debug)]
    struct LocalFactory;

    impl ExternalStoreFactory for LocalFactory {
        fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
            if root_uri != "file:///" {
                return Err(StoreError::Config(format!(
                    "not a local root: `{root_uri}`"
                )));
            }
            Ok(Arc::new(LocalStore::new("/")?))
        }
    }

    /// A storage as the catalog client would build one for a table that was
    /// vended no credentials.
    fn storage() -> Arc<dyn Storage> {
        PivotStorageFactory::new(Arc::new(LocalFactory))
            .build(&StorageConfig::new())
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_object_written_through_a_writer_is_read_back_by_its_location() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage();
        let location = format!("file://{}/m0.avro", directory.path().display());

        let mut writer = storage.writer(&location).await.unwrap();
        writer.write(Bytes::from_static(b"mani")).await.unwrap();
        writer.write(Bytes::from_static(b"fest")).await.unwrap();
        writer.close().await.unwrap();

        assert!(storage.exists(&location).await.unwrap());
        assert_eq!(storage.metadata(&location).await.unwrap().size, 8);
        assert_eq!(storage.read(&location).await.unwrap().as_ref(), b"manifest");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_ranged_reader_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage();
        let location = format!("file://{}/m0.avro", directory.path().display());
        storage
            .write(&location, Bytes::from_static(b"manifest"))
            .await
            .unwrap();

        let error = storage.reader(&location).await.err().expect("refused");

        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(error.to_string().contains("m0.avro"), "{error}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_object_that_is_not_there_is_reported_as_missing() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage();
        let location = format!("file://{}/none.avro", directory.path().display());

        assert!(!storage.exists(&location).await.unwrap());
        let error = storage.read(&location).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_location_outside_the_root_first_served_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage();
        let location = format!("file://{}/m0.avro", directory.path().display());
        storage
            .write(&location, Bytes::from_static(b"x"))
            .await
            .unwrap();

        let error = storage
            .write("s3://other/metadata/m0.avro", Bytes::from_static(b"x"))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("outside"), "{error}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_input_and_output_files_of_a_storage_share_its_store() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage();
        let location = format!("file://{}/m0.avro", directory.path().display());

        let mut writer = storage
            .new_output(&location)
            .unwrap()
            .writer()
            .await
            .unwrap();
        writer.write(Bytes::from_static(b"manifest")).await.unwrap();
        writer.close().await.unwrap();
        let read_back = storage.new_input(&location).unwrap().read().await.unwrap();
        let elsewhere = storage
            .new_input("s3://other/m0.avro")
            .unwrap()
            .read()
            .await
            .unwrap_err();

        assert_eq!(read_back.as_ref(), b"manifest");
        assert!(elsewhere.to_string().contains("outside"), "{elsewhere}");
    }
}
