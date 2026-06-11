//! The default (ephemeral) database's store. Catalog metadata (the manifest and
//! table logs — relative keys) lives in an in-memory `HashMap` and vanishes on
//! restart. A table's *data*, though, is real files on disk, named by an
//! **absolute** key (a `WITH (path = …)` directory); those keys are delegated to
//! a [`LocalStore`] on the local filesystem.
//!
//! Splitting on absolute-vs-relative is a deliberate stopgap — the two halves
//! duplicate `LocalStore`'s filesystem handling rather than share an
//! abstraction. Good enough until the store layer is unified.

use super::{DataFile, FileRef, LocalStore, ObjectStore, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug)]
pub struct MemoryStore {
    objects: Mutex<HashMap<String, Vec<u8>>>,
    /// Where absolute (filesystem) keys are served from. Its root is unused —
    /// absolute keys ignore it — but it anchors the in-memory database's data at
    /// the current directory.
    disk: LocalStore,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            objects: Mutex::new(HashMap::new()),
            disk: LocalStore::new(cwd),
        }
    }
}

/// Whether `key` is a filesystem path (table data) rather than an in-memory
/// metadata key.
fn on_disk(key: &str) -> bool {
    Path::new(key).is_absolute()
}

impl ObjectStore for MemoryStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if on_disk(key) {
            return self.disk.get(key);
        }
        Ok(self.objects.lock().unwrap().get(key).cloned())
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        if on_disk(key) {
            return self.disk.put(key, data);
        }
        self.objects
            .lock()
            .unwrap()
            .insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<bool> {
        if on_disk(key) {
            return self.disk.put_if_absent(key, data);
        }
        match self.objects.lock().unwrap().entry(key.to_string()) {
            std::collections::hash_map::Entry::Occupied(_) => Ok(false),
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(data.to_vec());
                Ok(true)
            }
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        if on_disk(key) {
            return self.disk.delete(key);
        }
        self.objects.lock().unwrap().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<FileRef>> {
        if on_disk(prefix) {
            return self.disk.list(prefix);
        }
        let prefix = format!("{}/", prefix.trim_end_matches('/'));
        Ok(self
            .objects
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(key, bytes)| FileRef {
                name: super::key_name(key),
                size: bytes.len() as u64,
            })
            .collect())
    }

    fn data_file(&self, key: &str, size: u64) -> Result<DataFile> {
        self.disk.data_file(key, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_lists_under_prefix() {
        let store = MemoryStore::new();
        assert!(store.get("k").unwrap().is_none());
        store.put("a/x.parquet", b"12345").unwrap();
        store.put("a/y.parquet", b"6").unwrap();
        store.put("b/z.parquet", b"7").unwrap();

        assert_eq!(store.get("a/x.parquet").unwrap().unwrap(), b"12345");
        let mut under_a: Vec<_> = store
            .list("a")
            .unwrap()
            .into_iter()
            .map(|o| (o.name, o.size))
            .collect();
        under_a.sort();
        assert_eq!(
            under_a,
            vec![
                ("x.parquet".to_string(), 5),
                ("y.parquet".to_string(), 1),
            ]
        );
    }

    #[test]
    fn put_if_absent_creates_once_and_delete_is_idempotent() {
        let store = MemoryStore::new();
        assert!(store.put_if_absent("k", b"first").unwrap());
        assert!(!store.put_if_absent("k", b"second").unwrap());
        assert_eq!(store.get("k").unwrap().unwrap(), b"first");
        store.delete("k").unwrap();
        assert!(store.get("k").unwrap().is_none());
        store.delete("k").unwrap();
    }

    #[test]
    fn put_overwrites() {
        let store = MemoryStore::new();
        store.put("k", b"first").unwrap();
        store.put("k", b"second").unwrap();
        assert_eq!(store.get("k").unwrap().unwrap(), b"second");
    }
}
