//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory.

use super::{
    DataFileLocation, DirectoryListing, FileRef, ListedObject, ObjectPath, ObjectStore, Result,
    StoreConnection, StoreError,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The local-filesystem backend: keys are paths under `root`.
#[derive(Debug)]
pub struct LocalStore {
    root: PathBuf,
}

/// Distinguishes the temp files one process has in flight, so two writes of
/// one key never share a staging file.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A staging path beside `target` that no other writer, in this process or
/// another, is using: dot-prefixed so a listing by name prefix never sees it,
/// and unique by process id and an in-process counter.
fn unique_temp_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!(".{name}.{}-{counter}.tmp", std::process::id()))
}

/// Write `data` to a fresh staging file beside `target`, creating the parent
/// directory, and hand back the staging path to publish by rename or link.
fn stage_bytes(target: &Path, key: &ObjectPath, data: &[u8]) -> Result<PathBuf> {
    let io_error = |source| StoreError::Io {
        key: key.to_string(),
        source,
    };
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(io_error)?;
    }
    let staged = unique_temp_path(target);
    std::fs::write(&staged, data).map_err(io_error)?;
    Ok(staged)
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
        // Write a staging file and rename over the target, so a concurrent
        // reader sees the old or new file whole — never a half-written one.
        let staged = stage_bytes(&path, key, data)?;
        std::fs::rename(&staged, &path).map_err(|source| {
            let _ = std::fs::remove_file(&staged);
            StoreError::Io {
                key: key.to_string(),
                source,
            }
        })
    }

    /// Publish by hard link instead of rename: linking a new name to the
    /// staged inode fails with `AlreadyExists` when the name is taken, which is
    /// the atomic "create only if absent" the filesystem offers, and the name
    /// appears with the whole object behind it. The staging file is removed
    /// whether the link landed or not.
    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
        let path = self.path_for(key);
        let staged = stage_bytes(&path, key, data)?;
        let linked = std::fs::hard_link(&staged, &path);
        let _ = std::fs::remove_file(&staged);
        match linked {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(StoreError::VersionConflict {
                    key: key.to_string(),
                })
            }
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
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

    fn list_from(
        &self,
        prefix: &ObjectPath,
        name_prefix: &str,
        after_name: &str,
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
            if !name.starts_with(name_prefix) || name.as_str() <= after_name {
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
    fn put_if_absent_creates_once_then_conflicts_and_keeps_the_first_object() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();

        store.put_if_absent(&p("t/v1"), b"first").unwrap();
        let second = store.put_if_absent(&p("t/v1"), b"second");

        assert!(matches!(second, Err(StoreError::VersionConflict { .. })));
        assert_eq!(store.get(&p("t/v1")).unwrap().unwrap(), b"first");
    }

    #[test]
    fn put_if_absent_leaves_no_staging_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();
        store.put_if_absent(&p("k"), b"1").unwrap();

        let _ = store.put_if_absent(&p("k"), b"2");

        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["k"]);
    }

    #[test]
    fn racing_creates_of_one_key_admit_exactly_one_writer() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(LocalStore::new(dir.path()).unwrap());

        let outcomes: Vec<_> = (0..16u8)
            .map(|writer| {
                let store = store.clone();
                std::thread::spawn(move || store.put_if_absent(&p("v"), &[writer]))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        let winners = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
        assert_eq!(winners, 1);
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome, Ok(()) | Err(StoreError::VersionConflict { .. })))
        );
    }

    #[test]
    fn concurrent_puts_of_one_key_all_succeed_and_leave_one_whole_object() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(LocalStore::new(dir.path()).unwrap());

        let handles: Vec<_> = (0..8u8)
            .map(|writer| {
                let store = store.clone();
                std::thread::spawn(move || store.put(&p("k"), &[writer; 64]))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        let bytes = store.get(&p("k")).unwrap().unwrap();
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|byte| *byte == bytes[0]));
    }

    #[test]
    fn list_from_returns_only_names_after_the_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path()).unwrap();
        for name in ["v01", "v02", "v03", "w01"] {
            store.put(&p(&format!("d/{name}")), b"x").unwrap();
        }

        let listed = store.list_from(&p("d"), "v", "v02").unwrap();

        let mut names: Vec<_> = listed
            .objects
            .iter()
            .map(|object| object.file.path.to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["v03"]);
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
