//! The iceberg crate's file IO for the tables a REST catalog serves.
//!
//! The catalog client builds a file IO for every table it loads, from the
//! factory here and the storage properties the load-table response carried,
//! among them the credentials the catalog vended for the table. What the file
//! IO is used for, and how it serves an object, is the shared storage's to
//! say: see [`iceberg_tables::io::storage`].
//!
//! A file IO belongs to one table, whose files all live under one root (the
//! bucket, or the local filesystem). The storage opens that root's store on
//! the first location it is asked for and serves every later one under it; a
//! location under another root is refused rather than served through a store
//! it was not opened for.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

use iceberg::io::{Storage, StorageConfig, StorageFactory};
use iceberg_tables::io::storage::{LocateError, ObjectLocator, PivotStorage};
use iceberg_tables::io::store::{open_store_at_root, read_vended_s3_credentials};
use iceberg_tables::metadata::layout::{is_same_root, split_location};
use object_storage::{ExternalStoreFactory, ObjectPath, ObjectStore};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
        Ok(Arc::new(PivotStorage::new(Arc::new(LazilyOpenedRoot {
            store_factory: self.store_factory.clone(),
            properties: config.props().clone(),
            store: OnceLock::new(),
        }))))
    }
}

impl fmt::Debug for PivotStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PivotStorageFactory")
            .finish_non_exhaustive()
    }
}

/// One table's root, opened on the first location served.
struct LazilyOpenedRoot {
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

impl LazilyOpenedRoot {
    /// The store `location` is served through and its key under the root,
    /// opening the root on the first location served and refusing a location
    /// under any other.
    fn locate_under_root(
        &self,
        location: &str,
    ) -> crate::Result<(Arc<dyn ObjectStore>, ObjectPath)> {
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

impl ObjectLocator for LazilyOpenedRoot {
    fn locate(&self, location: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath), LocateError> {
        Ok(self.locate_under_root(location)?)
    }
}

/// The crate's storage traits are object-safe through typetag, which asks
/// every implementation to serialize. A factory bound to this process's
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

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use iceberg::ErrorKind;
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
