use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use object_storage::{
    AmbientExternalStoreFactory, DataFileLocation, DirectoryListing, ExternalStoreFactory,
    ObjectPath, ObjectStore, Result, StoreConnection, StoreError,
};

/// Records objects admitted to Pivot's reader, including reads served by caches.
#[derive(Clone, Debug, Default)]
pub struct ReadLog {
    paths: Arc<Mutex<Vec<String>>>,
    fail_next_footer: Arc<AtomicBool>,
}

impl ReadLog {
    pub fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.paths.lock().unwrap())
    }

    pub fn fail_next_footer(&self) {
        self.fail_next_footer.store(true, Ordering::SeqCst);
    }
}

impl ExternalStoreFactory for ReadLog {
    fn open(&self, root_uri: &str) -> Result<Arc<dyn ObjectStore>> {
        Ok(Arc::new(RecordingStore {
            inner: AmbientExternalStoreFactory.open(root_uri)?,
            reads: self.clone(),
        }))
    }
}

#[derive(Debug)]
struct RecordingStore {
    inner: Arc<dyn ObjectStore>,
    reads: ReadLog,
}

impl ObjectStore for RecordingStore {
    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.reads.paths.lock().unwrap().push(key.to_string());
        if key.as_str().ends_with(".parquet")
            && self.reads.fail_next_footer.swap(false, Ordering::SeqCst)
        {
            return Err(StoreError::Io {
                key: key.to_string(),
                source: std::io::Error::other("injected footer read failure"),
            });
        }
        self.inner.source(key)
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
        self.inner.put(key, data)
    }

    fn update(
        &self,
        key: &ObjectPath,
        apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> Result<()> {
        self.inner.update(key, apply)
    }

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        self.inner.delete(key)
    }

    fn list_with_name_prefix(
        &self,
        prefix: &ObjectPath,
        name_prefix: &str,
    ) -> Result<DirectoryListing> {
        self.inner.list_with_name_prefix(prefix, name_prefix)
    }

    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.inner.sink(key)
    }

    fn create_dir(&self, prefix: &ObjectPath) -> Result<()> {
        self.inner.create_dir(prefix)
    }

    fn absolute_key(&self, key: &ObjectPath) -> Result<ObjectPath> {
        self.inner.absolute_key(key)
    }

    fn location_uri(&self) -> String {
        self.inner.location_uri()
    }

    fn connection(&self) -> StoreConnection {
        self.inner.connection()
    }
}
