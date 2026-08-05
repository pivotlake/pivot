//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory, the CAS primitive is an `O_EXCL` create.

use super::{DataFileLocation, FileRef, ObjectPath, ObjectStore, Result, StoreError};
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::local::LocalFileSystem;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the staging files of concurrent writers within this process.
/// Paired with the pid, which distinguishes processes sharing one store.
static NEXT_STAGING_FILE: AtomicU64 = AtomicU64::new(0);

/// A staging path beside `path` that no other writer will pick.
///
/// The name has to be unique per *writer*, not per process. Two threads writing
/// one key would otherwise derive the same staging path, and then one truncates
/// and rewrites the file the other is about to publish, or removes it from under
/// them. The publish still succeeds, so a writer reports success having stored
/// another's bytes. The counter separates writers within a process; the pid
/// separates processes sharing the store.
///
/// `with_extension` replaces the key's own extension, so a staging file for
/// `000...006.json` is not itself named `.json` and never looks like a log
/// version to a listing.
fn staging_path(path: &std::path::Path) -> PathBuf {
    path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        NEXT_STAGING_FILE.fetch_add(1, Ordering::Relaxed)
    ))
}

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
        // reader sees the old or new file whole — never a half-written one. The
        // name is per-writer for the same reason as in `put_if_absent`: a shared
        // one lets two writers interleave into a single staging file.
        let tmp = staging_path(&path);
        std::fs::write(&tmp, data).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })?;
        std::fs::rename(&tmp, &path).map_err(|source| StoreError::Io {
            key: key.to_string(),
            source,
        })
    }

    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<bool> {
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
        //
        // The staging name is per-writer (see `staging_path`); a shared one lets
        // two committers of the same version interleave into one file, so the
        // link publishes bytes the winner never wrote.
        let tmp = staging_path(&path);
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

    fn list(&self, prefix: &ObjectPath) -> Result<Vec<FileRef>> {
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
                    path: ObjectPath::new(name),
                    size: meta.len(),
                });
            }
        }
        Ok(objects)
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
    fn put_if_absent_creates_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.put_if_absent(&p("v/1.json"), b"first").unwrap());
        assert!(!store.put_if_absent(&p("v/1.json"), b"second").unwrap());
        // The loser's bytes never land.
        assert_eq!(store.get(&p("v/1.json")).unwrap().unwrap(), b"first");
    }

    /// Racing writers must not corrupt each other's staging. Exactly one wins
    /// the key, and what lands has to be the winner's own bytes: a shared
    /// staging file lets a loser's content be published under the winner's
    /// success, which is how a Delta log ends up naming a version whose actions
    /// were never written.
    #[test]
    fn concurrent_put_if_absent_stores_the_winners_own_bytes() {
        // A race needs repeating to be caught reliably; each round is a fresh
        // key so the rounds do not interfere.
        for round in 0..64 {
            let dir = tempfile::tempdir().unwrap();
            let store = LocalStore::new(dir.path());
            let key = p(&format!("_delta_log/{round:020}.json"));
            let writers: Vec<String> = (0..8).map(|w| format!("writer-{w}")).collect();

            let winners: Vec<&String> = std::thread::scope(|scope| {
                let handles: Vec<_> = writers
                    .iter()
                    .map(|content| {
                        let store = &store;
                        let key = &key;
                        scope.spawn(move || {
                            store
                                .put_if_absent(key, content.as_bytes())
                                .unwrap()
                                .then_some(content)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().unwrap())
                    .collect()
            });

            assert_eq!(winners.len(), 1, "exactly one writer creates the key");
            let stored = store.get(&key).unwrap().unwrap();
            assert_eq!(
                String::from_utf8(stored).unwrap(),
                *winners[0],
                "the key holds bytes a different writer wrote"
            );
        }
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
