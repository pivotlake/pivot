//! Where spill files live, laid out so that cleaning up is removing a
//! directory:
//!
//! ```text
//! <spill root>/dataflow-<id>/worker-<idx>/operator-<idx>/partition-<id>
//! ```
//!
//! Operator indices and partition ids repeat on every worker running a
//! dataflow, so each worker keeps its own subtree. A worker removes an
//! operator's directory when the operator finishes, its own directory when its
//! dataflow drops, and the dataflow's directory once no worker's subtree is
//! left in it.

use crate::Identifier;
use std::io::ErrorKind;
use std::path::PathBuf;

fn spill_root() -> PathBuf {
    std::env::temp_dir().join("pivot-spill")
}

fn dataflow_directory(dataflow: Identifier) -> PathBuf {
    spill_root().join(format!("dataflow-{dataflow}"))
}

fn worker_directory(dataflow: Identifier, worker: usize) -> PathBuf {
    dataflow_directory(dataflow).join(format!("worker-{worker}"))
}

pub(crate) fn operator_directory(
    dataflow: Identifier,
    worker: usize,
    operator: Identifier,
) -> PathBuf {
    worker_directory(dataflow, worker).join(format!("operator-{operator}"))
}

pub(crate) fn partition_file_path(
    dataflow: Identifier,
    worker: usize,
    operator: Identifier,
    spill_partition_id: usize,
) -> PathBuf {
    operator_directory(dataflow, worker, operator).join(format!("partition-{spill_partition_id}"))
}

/// Remove everything `operator` spilled on `worker`. Nothing to do when the
/// operator never spilled.
pub(crate) fn remove_operator_directory(
    dataflow: Identifier,
    worker: usize,
    operator: Identifier,
) -> std::io::Result<()> {
    remove_tree_if_present(operator_directory(dataflow, worker, operator))
}

/// Remove everything `worker` spilled for `dataflow`, and the dataflow's
/// directory too once it is the last worker to leave it.
pub(crate) fn remove_worker_directory(dataflow: Identifier, worker: usize) -> std::io::Result<()> {
    remove_tree_if_present(worker_directory(dataflow, worker))?;
    match std::fs::remove_dir(dataflow_directory(dataflow)) {
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

fn remove_tree_if_present(path: PathBuf) -> std::io::Result<()> {
    match std::fs::remove_dir_all(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_worker_leaves_the_dataflow_directory_to_the_other_workers() {
        let dataflow = 9_000_001;
        std::fs::create_dir_all(operator_directory(dataflow, 0, 2)).unwrap();
        std::fs::create_dir_all(operator_directory(dataflow, 1, 2)).unwrap();

        remove_worker_directory(dataflow, 0).unwrap();
        let after_first = dataflow_directory(dataflow).exists();
        remove_worker_directory(dataflow, 1).unwrap();
        let after_last = dataflow_directory(dataflow).exists();

        assert!(after_first);
        assert!(!after_last);
    }

    #[test]
    fn removing_what_was_never_spilled_is_not_an_error() {
        let dataflow = 9_000_002;

        remove_operator_directory(dataflow, 0, 0).unwrap();
        remove_worker_directory(dataflow, 0).unwrap();
    }
}
