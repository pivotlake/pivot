//! An in-memory [`ObjectStore`] — a `HashMap` of key → bytes. Backs the default
//! (ephemeral) database: it holds the table manifest in memory, so tables vanish
//! on restart. It serves no readable data files of its own (those come from the
//! `WITH (path = …)` directories on the local filesystem), so it inherits the
//! trait's default [`data_file`](ObjectStore::data_file).

use super::{ObjectMeta, ObjectStore, Result};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Default)]
pub struct MemoryStore {
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ObjectStore for MemoryStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.objects.lock().unwrap().get(key).cloned())
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.objects
            .lock()
            .unwrap()
            .insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<bool> {
        match self.objects.lock().unwrap().entry(key.to_string()) {
            std::collections::hash_map::Entry::Occupied(_) => Ok(false),
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(data.to_vec());
                Ok(true)
            }
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.objects.lock().unwrap().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let prefix = format!("{}/", prefix.trim_end_matches('/'));
        Ok(self
            .objects
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(key, bytes)| ObjectMeta {
                key: key.clone(),
                size: bytes.len() as u64,
            })
            .collect())
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
            .map(|o| (o.key, o.size))
            .collect();
        under_a.sort();
        assert_eq!(
            under_a,
            vec![
                ("a/x.parquet".to_string(), 5),
                ("a/y.parquet".to_string(), 1),
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
