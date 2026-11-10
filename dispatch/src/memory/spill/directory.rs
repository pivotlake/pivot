//! Where spill files live. The user names one directory for them, a
//! [`SpillDirectory`], and a pool that was given none spills nothing. Under
//! it the files are laid out so that cleaning up is removing a directory:
//!
//! ```text
//! <spill directory>/dataflow-<id>/worker-<idx>/operator-<idx>/partition-<id>
//! ```
//!
//! Operator indices and partition ids repeat on every worker running a
//! dataflow, so each worker keeps its own subtree. A worker removes an
//! operator's directory when the operator finishes, its own directory when its
//! dataflow drops, and the dataflow's directory once no worker's subtree is
//! left in it.
//!
//! Dataflow ids start over in every process, so two processes spilling into
//! one directory would write each other's files. A lock file in the directory
//! keeps a second one out for as long as the first has the directory open.
//! Whoever holds the lock therefore knows that the dataflow directories it
//! found on opening belong to a process that died before removing them, and
//! opening removes them.

use crate::Identifier;
use crate::io::LocalFile;
use crate::memory::spill::SPILL_BLOCK_SIZE;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

const LOCK_FILE_NAME: &str = "lock";
const PROBE_FILE_NAME: &str = "probe";
const DATAFLOW_DIRECTORY_PREFIX: &str = "dataflow-";

/// A directory that has been checked to hold spill files, locked for this
/// process. Only [`open`](Self::open) makes one, so whatever spills into a
/// `SpillDirectory` writes where a spill file is known to work and where no
/// other process writes. Every worker of a pool, and every dataflow that
/// spilled, holds a clone of the pool's one; the lock is let go when the last
/// clone drops.
#[derive(Clone, Debug)]
pub struct SpillDirectory {
    held: Arc<HeldDirectory>,
}

#[derive(Debug)]
struct HeldDirectory {
    root: PathBuf,
    /// The directory's lock file, locked. Closing it lets the lock go, and so
    /// does the death of the process.
    _lock: File,
}

/// Why a directory cannot hold this process's spill files.
#[derive(Debug, Error)]
pub enum SpillDirectoryError {
    /// Something else has the directory open and may be spilling into it.
    #[error(
        "spill directory `{}` is already in use: each running instance needs a spill \
         directory of its own",
        .directory.display()
    )]
    InUse { directory: PathBuf },
    /// What was being done in the directory, and the error that came back.
    #[error("spill directory `{}` cannot be used: {action}: {source}", .directory.display())]
    Unusable {
        directory: PathBuf,
        action: &'static str,
        #[source]
        source: std::io::Error,
    },
}

/// One spill block of memory, aligned the way direct IO wants the memory it
/// reads and writes.
#[repr(C, align(4096))]
struct AlignedBlock([u8; SPILL_BLOCK_SIZE]);

const _: () = assert!(std::mem::align_of::<AlignedBlock>() == SPILL_BLOCK_SIZE);

impl SpillDirectory {
    /// Take `root` as the directory spill files go under, creating it if it
    /// is not there. The directory is locked for this process, the spill
    /// files a dead process left in it are removed, and a spill file is seen
    /// to work in it: a file created there the way spill files are takes a
    /// block-aligned write and gives the block back. Spill files are opened
    /// for direct IO, which not every filesystem takes (tmpfs before Linux
    /// 6.6 does not, nor do some overlay and network filesystems). One that
    /// does not fails here, when the directory is named, instead of failing
    /// the first query that spills.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, SpillDirectoryError> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .map_err(describe_failure(&root, "creating the directory"))?;
        let lock = lock_directory(&root)?;
        remove_leftover_spill_files(&root)?;
        check_a_spill_file_works(&root)?;
        Ok(Self {
            held: Arc::new(HeldDirectory { root, _lock: lock }),
        })
    }

    /// The directory the user named.
    pub fn path(&self) -> &Path {
        &self.held.root
    }

    fn dataflow_directory(&self, dataflow: Identifier) -> PathBuf {
        self.held
            .root
            .join(format!("{DATAFLOW_DIRECTORY_PREFIX}{dataflow}"))
    }

    fn worker_directory(&self, dataflow: Identifier, worker: usize) -> PathBuf {
        self.dataflow_directory(dataflow)
            .join(format!("worker-{worker}"))
    }

    pub(crate) fn operator_directory(
        &self,
        dataflow: Identifier,
        worker: usize,
        operator: Identifier,
    ) -> PathBuf {
        self.worker_directory(dataflow, worker)
            .join(format!("operator-{operator}"))
    }

    pub(crate) fn partition_file_path(
        &self,
        dataflow: Identifier,
        worker: usize,
        operator: Identifier,
        spill_partition_id: usize,
    ) -> PathBuf {
        self.operator_directory(dataflow, worker, operator)
            .join(format!("partition-{spill_partition_id}"))
    }

    /// Remove everything `operator` spilled on `worker`. Nothing to do when the
    /// operator never spilled.
    pub(crate) fn remove_operator_directory(
        &self,
        dataflow: Identifier,
        worker: usize,
        operator: Identifier,
    ) -> std::io::Result<()> {
        remove_tree_if_present(self.operator_directory(dataflow, worker, operator))
    }

    /// Remove everything `worker` spilled for `dataflow`, and the dataflow's
    /// directory too once it is the last worker to leave it.
    pub(crate) fn remove_worker_directory(
        &self,
        dataflow: Identifier,
        worker: usize,
    ) -> std::io::Result<()> {
        remove_tree_if_present(self.worker_directory(dataflow, worker))?;
        match std::fs::remove_dir(self.dataflow_directory(dataflow)) {
            Ok(()) => Ok(()),
            // Another worker's subtree is still in it, or no worker ever spilled.
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::DirectoryNotEmpty | ErrorKind::NotFound
                ) =>
            {
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

fn remove_tree_if_present(path: PathBuf) -> std::io::Result<()> {
    match std::fs::remove_dir_all(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Lock `root` for this process, by locking the lock file in it. Fails with
/// [`SpillDirectoryError::InUse`] when the lock is already taken.
fn lock_directory(root: &Path) -> Result<File, SpillDirectoryError> {
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(LOCK_FILE_NAME))
        .map_err(describe_failure(root, "creating its lock file"))?;
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(SpillDirectoryError::InUse {
            directory: root.to_path_buf(),
        }),
        Err(TryLockError::Error(source)) => Err(describe_failure(root, "locking it")(source)),
    }
}

/// Remove what a process that died left in `root`: its dataflows' directories
/// and its probe file. The caller holds the directory's lock, which is what
/// makes them a dead process's and not a running one's. Anything else in the
/// directory is not a spill file and stays.
fn remove_leftover_spill_files(root: &Path) -> Result<(), SpillDirectoryError> {
    let describe = || describe_failure(root, "removing the spill files left in it");
    for entry in std::fs::read_dir(root).map_err(describe())? {
        let entry = entry.map_err(describe())?;
        let is_dataflow_directory = entry
            .file_name()
            .as_encoded_bytes()
            .starts_with(DATAFLOW_DIRECTORY_PREFIX.as_bytes());
        if is_dataflow_directory {
            std::fs::remove_dir_all(entry.path()).map_err(describe())?;
        }
    }
    remove_file_if_present(&root.join(PROBE_FILE_NAME)).map_err(describe())
}

/// Create a probe file in `root` the way spill files are created, write a
/// block to it and read the block back, and remove the file whether or not
/// that worked.
fn check_a_spill_file_works(root: &Path) -> Result<(), SpillDirectoryError> {
    let probe_path = root.join(PROBE_FILE_NAME);
    let probe = crate::io::create_direct_read_write(&probe_path).map_err(describe_failure(
        root,
        "creating a file opened for direct IO, which spill files need and not every \
         filesystem takes",
    ))?;

    let round_trip = write_and_read_back_a_block(root, &probe);
    drop(probe);
    let removal = std::fs::remove_file(&probe_path)
        .map_err(describe_failure(root, "removing the probe file"));
    round_trip.and(removal)
}

fn write_and_read_back_a_block(root: &Path, probe: &LocalFile) -> Result<(), SpillDirectoryError> {
    let written = Box::new(AlignedBlock([0xA5; SPILL_BLOCK_SIZE]));
    let mut read_back = Box::new(AlignedBlock([0; SPILL_BLOCK_SIZE]));
    probe.write_all_at(&written.0, 0).map_err(describe_failure(
        root,
        "writing a block to a file opened for direct IO",
    ))?;
    probe
        .read_exact_at(&mut read_back.0, 0)
        .and_then(|()| {
            if read_back.0 == written.0 {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "the block read back is not the block written",
                ))
            }
        })
        .map_err(describe_failure(
            root,
            "reading a block back from a file opened for direct IO",
        ))
}

/// Turns the error that `action` failed with into `root`'s refusal.
fn describe_failure<'a>(
    root: &'a Path,
    action: &'static str,
) -> impl FnOnce(std::io::Error) -> SpillDirectoryError + 'a {
    move |source| SpillDirectoryError::Unusable {
        directory: root.to_path_buf(),
        action,
        source,
    }
}

fn remove_file_if_present(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// A spill directory of a test's own, in a temporary directory that is
/// removed with it.
#[cfg(test)]
pub(crate) struct TestSpillDirectory {
    spill_directory: SpillDirectory,
    _root: tempfile::TempDir,
}

#[cfg(test)]
impl TestSpillDirectory {
    pub(crate) fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        Self {
            spill_directory: SpillDirectory::open(root.path()).unwrap(),
            _root: root,
        }
    }
}

#[cfg(test)]
impl std::ops::Deref for TestSpillDirectory {
    type Target = SpillDirectory;

    fn deref(&self) -> &SpillDirectory {
        &self.spill_directory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_that_takes_a_spill_file_is_accepted_and_holds_only_its_lock_file() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("not").join("there").join("yet");

        let spill_directory = SpillDirectory::open(&root).unwrap();

        let left_in_it: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(spill_directory.path(), root);
        assert_eq!(left_in_it, [LOCK_FILE_NAME]);
    }

    #[test]
    fn a_directory_that_is_already_open_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let _held = SpillDirectory::open(root.path()).unwrap();

        let refusal = SpillDirectory::open(root.path()).unwrap_err().to_string();

        assert!(refusal.contains("is already in use"), "{refusal}");
    }

    #[test]
    fn a_directory_can_be_opened_again_once_every_clone_has_dropped() {
        let root = tempfile::tempdir().unwrap();
        let opened = SpillDirectory::open(root.path()).unwrap();
        let clone = opened.clone();
        drop(opened);
        let while_a_clone_lives = SpillDirectory::open(root.path()).is_ok();
        drop(clone);

        let after_the_last_clone = SpillDirectory::open(root.path()).is_ok();

        assert!(!while_a_clone_lives);
        assert!(after_the_last_clone);
    }

    #[test]
    fn opening_a_directory_removes_the_spill_files_left_in_it_and_nothing_else() {
        let root = tempfile::tempdir().unwrap();
        let leftover_operator = root.path().join("dataflow-3/worker-0/operator-1");
        std::fs::create_dir_all(&leftover_operator).unwrap();
        std::fs::write(leftover_operator.join("partition-7"), "spilled rows").unwrap();
        let unrelated = root.path().join("notes.txt");
        std::fs::write(&unrelated, "not a spill file").unwrap();

        let _spill_directory = SpillDirectory::open(root.path()).unwrap();

        assert!(!root.path().join("dataflow-3").exists());
        assert!(unrelated.exists());
    }

    #[test]
    fn a_directory_that_cannot_be_created_is_refused_by_name() {
        let parent = tempfile::tempdir().unwrap();
        let occupied = parent.path().join("occupied");
        std::fs::write(&occupied, "a file, where a directory is wanted").unwrap();
        let root = occupied.join("spill");

        let refusal = SpillDirectory::open(&root).unwrap_err().to_string();

        let expected = format!(
            "spill directory `{}` cannot be used: creating the directory: ",
            root.display()
        );
        assert!(refusal.starts_with(&expected), "{refusal}");
    }

    #[test]
    fn removing_a_worker_leaves_the_dataflow_directory_to_the_other_workers() {
        let spill_directory = TestSpillDirectory::new();
        let dataflow = 1;
        std::fs::create_dir_all(spill_directory.operator_directory(dataflow, 0, 2)).unwrap();
        std::fs::create_dir_all(spill_directory.operator_directory(dataflow, 1, 2)).unwrap();

        spill_directory
            .remove_worker_directory(dataflow, 0)
            .unwrap();
        let after_first = spill_directory.dataflow_directory(dataflow).exists();
        spill_directory
            .remove_worker_directory(dataflow, 1)
            .unwrap();
        let after_last = spill_directory.dataflow_directory(dataflow).exists();

        assert!(after_first);
        assert!(!after_last);
    }

    #[test]
    fn removing_what_was_never_spilled_is_not_an_error() {
        let spill_directory = TestSpillDirectory::new();
        let dataflow = 2;

        spill_directory
            .remove_operator_directory(dataflow, 0, 0)
            .unwrap();
        spill_directory
            .remove_worker_directory(dataflow, 0)
            .unwrap();
    }
}
