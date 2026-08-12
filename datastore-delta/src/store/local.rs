//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory. A Delta datastore opened over it takes an exclusive root lock.

use super::{DataFileLocation, FileRef, ListedObject, ObjectPath, ObjectStore, Result, StoreError};
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::local::LocalFileSystem;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    fn path_for(&self, key: &ObjectPath) -> PathBuf {
        if key.is_absolute() {
            PathBuf::from(key.as_str())
        } else {
            self.root.join(key.as_str())
        }
    }
}

impl ObjectStore for LocalStore {
    fn local_root(&self) -> Option<&Path> {
        Some(&self.root)
    }

    fn describe(&self) -> String {
        format!("file://{}", self.root.display())
    }

    fn location_uri(&self) -> String {
        format!("file://{}", self.root.display())
    }

    fn build_delta_object_store(&self) -> Result<Arc<DynObjectStore>> {
        // Keys reach this client as absolute paths from the table URI, so it is
        // rooted at the filesystem rather than at this store's root.
        Ok(Arc::new(LocalFileSystem::new()))
    }

    fn create_dir(&self, prefix: &ObjectPath) -> Result<()> {
        std::fs::create_dir_all(self.path_for(prefix)).map_err(|source| StoreError::Io {
            key: prefix.to_string(),
            source,
        })
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path_for(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
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

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        match std::fs::remove_file(self.path_for(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn list(&self, prefix: &ObjectPath) -> Result<Vec<ListedObject>> {
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
            let modified = meta.modified().map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            let modified_unix_ms = modified
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            if let Some(name) = entry.file_name().to_str() {
                objects.push(ListedObject {
                    file: FileRef {
                        path: ObjectPath::new(name),
                        size: meta.len(),
                    },
                    modified_unix_ms,
                });
            }
        }
        Ok(objects)
    }

    fn absolute_key(&self, key: &ObjectPath) -> Result<ObjectPath> {
        // A store rooted at a relative directory would otherwise yield a key that
        // only resolves from the process's working directory.
        let path = std::path::absolute(self.path_for(key)).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })?;
        Ok(ObjectPath::new(path.to_string_lossy().into_owned()))
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        Ok(DataFileLocation::Local(self.path_for(key)))
    }

    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        Ok(DataFileLocation::Local(self.path_for(key)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(key: &str) -> ObjectPath {
        ObjectPath::new(key)
    }

    #[test]
    fn put_overwrites_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        store.put(&p("k"), b"first").unwrap();
        store.put(&p("k"), b"second").unwrap();
        assert_eq!(store.get(&p("k")).unwrap().unwrap(), b"second");
    }

    #[test]
    fn delete_removes_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        store.put(&p("k"), b"v").unwrap();
        store.delete(&p("k")).unwrap();
        assert!(store.get(&p("k")).unwrap().is_none());
        store.delete(&p("k")).unwrap();
    }

    #[test]
    fn get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.get(&p("nope")).unwrap().is_none());
        assert!(store.list(&p("_missing")).unwrap().is_empty());
    }
}
