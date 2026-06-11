//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory, the CAS primitive is an `O_EXCL` create.

use super::{DataFile, DataFileSource, FileRef, ObjectStore, Result, StoreError};
use std::path::{Path, PathBuf};

/// The local-filesystem backend: keys are paths under `root`.
#[derive(Debug)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The filesystem path for `key`. A relative key lives under the store
    /// root; an **absolute** key is taken as-is — it names a location outside
    /// the root (an external data directory). Made explicit rather than relying
    /// on `Path::join`'s absolute-component behavior.
    fn path_for(&self, key: &str) -> PathBuf {
        let key = Path::new(key);
        if key.is_absolute() {
            key.to_path_buf()
        } else {
            self.root.join(key)
        }
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

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<bool> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                key: key.to_string(),
                source,
            })?;
        }
        // Write the full content to a writer-unique temp file, then `link` it
        // to the target name: `link` fails with `AlreadyExists` if the target
        // exists, so creation is atomic *and* a concurrent reader can never see
        // a partially-written object.
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, data).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })?;
        let created = match std::fs::hard_link(&tmp, &path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        };
        let _ = std::fs::remove_file(&tmp);
        created
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path_for(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<FileRef>> {
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
            // One level only: skip subdirectories (a table's data files are flat).
            if !meta.is_file() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                objects.push(FileRef {
                    name: name.to_string(),
                    size: meta.len(),
                });
            }
        }
        Ok(objects)
    }

    fn data_file(&self, key: &str, size: u64) -> Result<DataFile> {
        Ok(DataFile {
            size,
            source: DataFileSource::Local(self.path_for(key)),
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
    fn put_if_absent_creates_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.put_if_absent("v/1.json", b"first").unwrap());
        assert!(!store.put_if_absent("v/1.json", b"second").unwrap());
        // The loser's bytes never land.
        assert_eq!(store.get("v/1.json").unwrap().unwrap(), b"first");
    }

    #[test]
    fn delete_removes_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        store.put("k", b"v").unwrap();
        store.delete("k").unwrap();
        assert!(store.get("k").unwrap().is_none());
        store.delete("k").unwrap();
    }

    #[test]
    fn get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.get("nope").unwrap().is_none());
        assert!(store.list("_missing").unwrap().is_empty());
    }
}
