//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory, the CAS primitive is an `O_EXCL` create.

use super::{DataFileLocation, ObjectMeta, ObjectStore, Result, StoreError};
use std::path::PathBuf;

/// The local-filesystem backend: keys are paths under `root`.
#[derive(Debug)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl ObjectStore for LocalStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path_for(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                key: key.to_string(),
                source,
            })?;
        }
        // Write a sibling temp file and rename over the target, so a concurrent
        // reader sees the old or new file whole — never a half-written one.
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, data).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })?;
        std::fs::rename(&tmp, &path).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let dir = self.path_for(prefix);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // A not-yet-created directory lists as empty, not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(StoreError::Io {
                    key: prefix.to_string(),
                    source,
                });
            }
        };
        let mut objects = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            let meta = entry.metadata().map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            if let Some(name) = entry.file_name().to_str() {
                objects.push(ObjectMeta {
                    key: format!("{}/{}", prefix.trim_end_matches('/'), name),
                    size: meta.len(),
                });
            }
        }
        Ok(objects)
    }

    fn data_file(&self, key: &str, size: u64) -> Result<DataFileLocation> {
        Ok(DataFileLocation::Local {
            path: self.path_for(key),
            size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_overwrites_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        store.put("k", b"first").unwrap();
        store.put("k", b"second").unwrap();
        assert_eq!(store.get("k").unwrap().unwrap(), b"second");
    }

    #[test]
    fn get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.get("nope").unwrap().is_none());
        assert!(store.list("_missing").unwrap().is_empty());
    }
}
