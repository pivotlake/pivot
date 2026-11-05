//! The iceberg crate's file IO over a table's store.
//!
//! The crate reads a table's manifest list and manifests through the file IO
//! of the table, and a commit writes its own through it. The file IO built
//! here reads and writes whole objects through the table's store, signed with
//! the keys the catalog vended for the table, so what the crate reads to plan
//! a scan comes through the same store every read of the table does. A
//! store's calls block, so each runs off the runtime's workers.

use std::fmt;
use std::ops::Range;
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
use object_storage::ObjectPath;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::store::TableStore;

/// The file IO of a table whose files live in `store`.
pub(crate) fn build_file_io(store: &TableStore) -> FileIO {
    FileIOBuilder::new(Arc::new(TableStorageFactory {
        store: store.clone(),
    }))
    .build()
}

/// Builds the storage of one table's file IO: the crate asks for it once,
/// when the file IO is first used.
struct TableStorageFactory {
    store: TableStore,
}

#[typetag::serde]
impl StorageFactory for TableStorageFactory {
    fn build(&self, _: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(TableStorage {
            store: self.store.clone(),
        }))
    }
}

impl fmt::Debug for TableStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStorageFactory")
            .field("table", &self.store.table_name())
            .finish_non_exhaustive()
    }
}

/// The crate's storage over one table's store: a file is located by its full
/// location, which must be under the table's root, and read or written whole.
#[derive(Clone)]
struct TableStorage {
    store: TableStore,
}

impl TableStorage {
    /// Run `call` against the store on a blocking thread: the store's calls
    /// block on the network, and the runtime's workers must not.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&TableStore) -> crate::Result<T> + Send + 'static,
    ) -> iceberg::Result<T> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || call(&store))
            .await
            .map_err(|panicked| {
                IcebergError::new(ErrorKind::Unexpected, "a store call panicked")
                    .with_source(panicked)
            })?
            .map_err(into_iceberg_error)
    }

    /// The size of the object at `path`, or `None` when there is none.
    async fn size_of(&self, path: &str) -> iceberg::Result<Option<u64>> {
        let path = path.to_string();
        self.blocking(move |store| store.object_size(&path)).await
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for TableStorage {
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
            .blocking(move |store| store.get_object(&location))
            .await?
            .ok_or_else(|| not_found(path))?;
        Ok(Bytes::from(bytes))
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        Ok(Box::new(ObjectRead {
            bytes: self.read(path).await?,
        }))
    }

    async fn write(&self, path: &str, bytes: Bytes) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |store| store.put_object(&location, &bytes))
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
        self.blocking(move |store| store.delete_object(&location))
            .await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |store| delete_under(store, &store.key_of(&location)?))
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

impl fmt::Debug for TableStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStorage")
            .field("table", &self.store.table_name())
            .finish_non_exhaustive()
    }
}

/// Delete every object under `prefix`, level by level.
fn delete_under(store: &TableStore, prefix: &ObjectPath) -> crate::Result<()> {
    let listing = store.store().list(prefix)?;
    for object in listing.objects {
        store
            .store()
            .delete(&prefix.join(object.file.path.as_str()))?;
    }
    for child in listing.prefixes {
        delete_under(store, &prefix.join(child.as_str()))?;
    }
    Ok(())
}

/// A file read whole when opened: what the crate reads this way, a manifest
/// list or a manifest, is small and read once.
struct ObjectRead {
    bytes: Bytes,
}

#[async_trait]
impl FileRead for ObjectRead {
    async fn read(&self, range: Range<u64>) -> iceberg::Result<Bytes> {
        let out_of_bounds = || {
            IcebergError::new(
                ErrorKind::DataInvalid,
                format!(
                    "range {}..{} is outside the object's {} bytes",
                    range.start,
                    range.end,
                    self.bytes.len()
                ),
            )
        };
        let start = usize::try_from(range.start).map_err(|_| out_of_bounds())?;
        let end = usize::try_from(range.end).map_err(|_| out_of_bounds())?;
        if start > end || end > self.bytes.len() {
            return Err(out_of_bounds());
        }
        Ok(self.bytes.slice(start..end))
    }
}

/// A file written whole when closed: the store takes an object in one put.
struct BufferedWrite {
    storage: TableStorage,
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
impl Serialize for TableStorageFactory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("TableStorageFactory")
    }
}

impl<'de> Deserialize<'de> for TableStorageFactory {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a table storage factory is bound to a process's stores and is not deserialized",
        ))
    }
}

impl Serialize for TableStorage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("TableStorage")
    }
}

impl<'de> Deserialize<'de> for TableStorage {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a table storage is bound to a process's stores and is not deserialized",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use iceberg::io::FileIO;
    use object_storage::{ExternalStoreFactory, LocalStore, ObjectStore, local_path};

    use super::*;
    use crate::rest_catalog::block_on;

    /// Opens the local filesystem for any root under it.
    #[derive(Debug)]
    struct LocalFactory;

    impl ExternalStoreFactory for LocalFactory {
        fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
            Ok(Arc::new(LocalStore::new(local_path(root_uri))?))
        }
    }

    /// The file IO of a table whose metadata lives under `directory`.
    fn local_file_io(directory: &std::path::Path) -> FileIO {
        let metadata_location = format!("file://{}/metadata/v1.json", directory.display());
        build_file_io(&TableStore::open("db.t", &metadata_location, &LocalFactory, None).unwrap())
    }

    #[test]
    fn an_object_written_through_the_file_io_is_read_back_through_it() {
        let directory = tempfile::tempdir().unwrap();
        let location = format!("file://{}/m0.avro", directory.path().display());
        let file_io = local_file_io(directory.path());

        let read_back = block_on(async {
            file_io
                .new_output(&location)?
                .write(Bytes::from_static(b"manifest"))
                .await?;
            file_io.new_input(&location)?.read().await
        })
        .unwrap();

        assert_eq!(&read_back[..], b"manifest");
    }

    #[test]
    fn a_missing_object_is_reported_as_invalid_data() {
        let directory = tempfile::tempdir().unwrap();
        let location = format!("file://{}/missing.avro", directory.path().display());
        let file_io = local_file_io(directory.path());

        let error = block_on(async { file_io.new_input(&location)?.read().await }).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn a_location_outside_the_tables_root_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let file_io = local_file_io(directory.path());

        let error = block_on(async { file_io.new_input("s3://elsewhere/m0.avro")?.read().await })
            .unwrap_err();

        assert!(error.to_string().contains("the table's store"), "{error}");
    }
}
