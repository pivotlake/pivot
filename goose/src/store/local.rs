//! The local-filesystem [`ObjectStore`] backend: keys are paths under a root
//! directory, the CAS primitive is an `O_EXCL` create.

use super::{ObjectStore, PutOutcome, Result, StoreError};
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

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<PutOutcome> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                key: key.to_string(),
                source,
            })?;
        }
        // `create_new` is an atomic O_EXCL create — the local CAS primitive.
        use std::io::Write;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                f.write_all(data).map_err(|source| StoreError::Io {
                    key: key.to_string(),
                    source,
                })?;
                Ok(PutOutcome::Created)
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok(PutOutcome::AlreadyExists)
            }
            Err(source) => Err(StoreError::Io {
                key: key.to_string(),
                source,
            }),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let dir = self.path_for(prefix);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // A not-yet-created log directory lists as empty, not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(StoreError::Io {
                    key: prefix.to_string(),
                    source,
                });
            }
        };
        let mut keys = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| StoreError::Io {
                key: prefix.to_string(),
                source,
            })?;
            if let Some(name) = entry.file_name().to_str() {
                keys.push(format!("{}/{}", prefix.trim_end_matches('/'), name));
            }
        }
        Ok(keys)
    }

    fn describe(&self) -> String {
        format!("local:{}", self.root.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_if_absent_is_a_cas() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert_eq!(
            store.put_if_absent("k", b"first").unwrap(),
            PutOutcome::Created
        );
        // Second writer loses the race; original bytes are untouched.
        assert_eq!(
            store.put_if_absent("k", b"second").unwrap(),
            PutOutcome::AlreadyExists
        );
        assert_eq!(store.get("k").unwrap().unwrap(), b"first");
    }

    #[test]
    fn get_missing_is_none_and_list_of_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.get("nope").unwrap().is_none());
        assert!(store.list("_missing").unwrap().is_empty());
    }
}
