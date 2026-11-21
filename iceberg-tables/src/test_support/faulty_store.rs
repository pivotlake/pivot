//! An object store that counts the requests made of it and fails the ones a
//! test names.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use object_storage::{
    DataFileLocation, DirectoryListing, ObjectPath, ObjectStore, StoreConnection, StoreError,
};

/// What a [`FaultyStore`] does to the calls made of it, and how many of
/// them it saw.
#[derive(Debug, Default)]
pub struct Faults {
    /// How many whole-object reads were asked for.
    pub object_reads: AtomicUsize,
    /// How many manifests were located to be read off their source, which
    /// is how a table reads a manifest it parses: not a whole-object read.
    pub manifests_sourced: AtomicUsize,
    /// How many listings were asked for.
    pub listings: AtomicUsize,
    /// How many writes of a metadata file are applied and then answered
    /// with an error, as when the store's answer is lost on the way.
    pub lost_put_answers: AtomicUsize,
    /// Whether every write of a metadata file fails without being applied.
    pub fail_metadata_puts: AtomicBool,
    /// Whether every whole-object read fails.
    pub fail_gets: AtomicBool,
}

/// The requests a [`FaultyStore`] counted.
pub struct Requests {
    pub object_reads: usize,
    pub manifests_sourced: usize,
    pub listings: usize,
}

impl Faults {
    /// The requests counted since the last take.
    pub fn take_requests(&self) -> Requests {
        Requests {
            object_reads: self.object_reads.swap(0, Ordering::SeqCst),
            manifests_sourced: self.manifests_sourced.swap(0, Ordering::SeqCst),
            listings: self.listings.swap(0, Ordering::SeqCst),
        }
    }
}

/// A store that fails the calls its [`Faults`] name and passes the rest on.
#[derive(Debug)]
pub struct FaultyStore {
    pub inner: Arc<dyn ObjectStore>,
    pub faults: Arc<Faults>,
}

fn injected_failure(key: &ObjectPath) -> StoreError {
    StoreError::Io {
        key: key.to_string(),
        source: std::io::Error::other("injected failure"),
    }
}

impl ObjectStore for FaultyStore {
    fn source(&self, key: &ObjectPath) -> object_storage::Result<DataFileLocation> {
        if key.as_str().ends_with(".avro") {
            self.faults.manifests_sourced.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.source(key)
    }

    fn get(&self, key: &ObjectPath) -> object_storage::Result<Option<Vec<u8>>> {
        self.faults.object_reads.fetch_add(1, Ordering::SeqCst);
        if self.faults.fail_gets.load(Ordering::SeqCst) {
            return Err(injected_failure(key));
        }
        self.inner.get(key)
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> object_storage::Result<()> {
        self.inner.put(key, data)
    }

    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> object_storage::Result<()> {
        if !key.as_str().ends_with(".metadata.json") {
            return self.inner.put_if_absent(key, data);
        }
        if self.faults.fail_metadata_puts.load(Ordering::SeqCst) {
            return Err(injected_failure(key));
        }
        let is_answer_lost = self
            .faults
            .lost_put_answers
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok();
        self.inner.put_if_absent(key, data)?;
        if is_answer_lost {
            return Err(injected_failure(key));
        }
        Ok(())
    }

    fn update(
        &self,
        key: &ObjectPath,
        apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> object_storage::Result<()> {
        self.inner.update(key, apply)
    }

    fn delete(&self, key: &ObjectPath) -> object_storage::Result<()> {
        self.inner.delete(key)
    }

    fn list_filtered(
        &self,
        directory: &ObjectPath,
        name_prefix: &str,
        after: Option<&str>,
    ) -> object_storage::Result<DirectoryListing> {
        self.faults.listings.fetch_add(1, Ordering::SeqCst);
        self.inner.list_filtered(directory, name_prefix, after)
    }

    fn sink(&self, key: &ObjectPath) -> object_storage::Result<DataFileLocation> {
        self.inner.sink(key)
    }

    fn create_dir(&self, prefix: &ObjectPath) -> object_storage::Result<()> {
        self.inner.create_dir(prefix)
    }

    fn absolute_key(&self, key: &ObjectPath) -> object_storage::Result<ObjectPath> {
        self.inner.absolute_key(key)
    }

    fn location_uri(&self) -> String {
        self.inner.location_uri()
    }

    fn connection(&self) -> StoreConnection {
        self.inner.connection()
    }
}
