//! Exclusive process ownership for a local pivotlake datastore root.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{Error, Result};

const LOCK_FILE_NAME: &str = ".pivot.lock";

/// The open file whose advisory lock owns one local datastore for this process.
#[derive(Debug)]
pub(super) struct LocalDatastoreLock {
    _file: File,
}

impl LocalDatastoreLock {
    pub(super) fn acquire(requested_root: &Path) -> Result<Self> {
        std::fs::create_dir_all(requested_root).map_err(|source| Error::DatastoreLock {
            path: requested_root.to_path_buf(),
            source,
        })?;
        let root = requested_root
            .canonicalize()
            .map_err(|source| Error::DatastoreLock {
                path: requested_root.to_path_buf(),
                source,
            })?;
        let mut lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(LOCK_FILE_NAME))
            .map_err(|source| Error::DatastoreLock {
                path: root.clone(),
                source,
            })?;

        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let owner = read_owner(&mut lock)
                    .map(|pid| format!(" (PID {pid})"))
                    .unwrap_or_default();
                return Err(Error::DatastoreInUse { path: root, owner });
            }
            Err(TryLockError::Error(source)) => {
                return Err(Error::DatastoreLock { path: root, source });
            }
        }

        write_owner(&mut lock).map_err(|source| Error::DatastoreLock { path: root, source })?;
        Ok(Self { _file: lock })
    }
}

fn read_owner(file: &mut File) -> Option<u32> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut owner = String::new();
    file.read_to_string(&mut owner).ok()?;
    owner.trim().parse().ok()
}

fn write_owner(file: &mut File) -> std::io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    writeln!(file, "{}", std::process::id())?;
    file.sync_data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_datastore_lock_is_exclusive() {
        let dir = tempfile::tempdir().unwrap();
        let first = LocalDatastoreLock::acquire(dir.path()).unwrap();

        let error = LocalDatastoreLock::acquire(dir.path()).unwrap_err();

        assert!(matches!(error, Error::DatastoreInUse { .. }));
        assert!(error.to_string().contains(&std::process::id().to_string()));
        assert!(dir.path().join(LOCK_FILE_NAME).is_file());

        drop(first);
        LocalDatastoreLock::acquire(dir.path()).unwrap();
    }
}
