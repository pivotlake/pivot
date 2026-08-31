//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory.

use super::{
    DataFileLocation, DirectoryListing, FileRef, ListedObject, ObjectPath, ObjectStore, Result,
    StoreConnection, StoreError,
};
use std::path::{Path, PathBuf};

/// The local-filesystem backend: keys are paths under `root`.
#[derive(Debug)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    /// Open a store rooted at `root`, resolving a relative path against the
    /// process's current directory once so later I/O cannot change meaning if
    /// the process changes its working directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let requested = root.into();
        let root = std::path::absolute(&requested).map_err(|source| StoreError::Io {
            key: requested.display().to_string(),
            source,
        })?;
        Ok(Self { root })
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
        url::Url::from_directory_path(&self.root)
            .expect("LocalStore roots are made absolute by the constructor")
            .into()
    }

    fn connection(&self) -> StoreConnection {
        StoreConnection::Local
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

    /// Read-modify-write under an exclusive advisory lock, so two writers —
    /// in-process or across processes — never interleave their cycles. The lock
    /// lives on a sibling `<name>.lock` file, because the object itself is
    /// replaced by rename and a lock on it would not survive the swap; it is
    /// held from the read until the replacing rename lands.
    fn update(
        &self,
        key: &ObjectPath,
        apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> Result<()> {
        let io_error = |source| StoreError::Io {
            key: key.to_string(),
            source,
        };
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(io_error)?;
        lock.lock().map_err(io_error)?;
        let current = match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(source) => return Err(io_error(source)),
        };
        let Some(replacement) = apply(current) else {
            return Ok(());
        };
        self.put(key, &replacement)
        // Dropping `lock` releases the advisory lock.
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

    fn list_with_name_prefix(
        &self,
        prefix: &ObjectPath,
        name_prefix: &str,
    ) -> Result<DirectoryListing> {
        let dir = self.path_for(prefix);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // A not-yet-created directory lists as empty, not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(DirectoryListing::default());
            }
            Err(source) => {
                return Err(StoreError::Io {
                    key: prefix.to_string(),
                    source,
                });
            }
        };
        let mut objects = Vec::new();
        let mut prefixes = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !name.starts_with(name_prefix) {
                continue;
            }
            let file_type = entry.file_type().map_err(|source| StoreError::Io {
                key: prefix.join(&name).to_string(),
                source,
            })?;
            if file_type.is_dir() {
                prefixes.push(ObjectPath::new(name));
                continue;
            }
            let meta = entry.metadata().map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
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
            objects.push(ListedObject {
                file: FileRef {
                    path: ObjectPath::new(name),
                    size: meta.len(),
                },
                modified_unix_ms,
            });
        }
        Ok(DirectoryListing { objects, prefixes })
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
        let store = LocalStore::new(dir.path()).unwrap();
        store.put(&p("k"), b"first").unwrap();
        store.put(&p("k"), b"second").unwrap();
        assert_eq!(store.get(&p("k")).unwrap().unwrap(), b"second");
    }

    #[test]
    fn delete_removes_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();
        store.put(&p("k"), b"v").unwrap();
        store.delete(&p("k")).unwrap();
        assert!(store.get(&p("k")).unwrap().is_none());
        store.delete(&p("k")).unwrap();
    }

    #[test]
    fn update_creates_a_missing_object_and_applies_over_the_current_one() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();

        store
            .update(&p("k"), &mut |current| {
                assert!(current.is_none());
                Some(b"1".to_vec())
            })
            .unwrap();
        store
            .update(&p("k"), &mut |current| {
                assert_eq!(current.unwrap(), b"1");
                Some(b"2".to_vec())
            })
            .unwrap();

        assert_eq!(store.get(&p("k")).unwrap().unwrap(), b"2");
    }

    #[test]
    fn update_returning_none_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();
        store.put(&p("k"), b"kept").unwrap();

        store.update(&p("k"), &mut |_| None).unwrap();

        assert_eq!(store.get(&p("k")).unwrap().unwrap(), b"kept");
    }

    #[test]
    fn get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();
        assert!(store.get(&p("nope")).unwrap().is_none());
        assert_eq!(
            store.list(&p("_missing")).unwrap(),
            DirectoryListing::default()
        );
    }
}
