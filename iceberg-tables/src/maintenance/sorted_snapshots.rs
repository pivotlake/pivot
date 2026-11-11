//! A table's snapshots from the oldest to the current one, each the parent
//! of the next. A table whose history is not linear is refused: one with a
//! branch or a tag besides `main`, or with a snapshot that is neither the
//! current one nor an ancestor of it.
//!
//! Expiring snapshots and deleting what they retired both take a table's
//! history to be linear: the snapshots that expire are the oldest of it
//! ([`retention`](super::retention)), and what they retired is found from
//! them and the oldest snapshot kept, with no other snapshot read
//! ([`expired_objects`](super::expired_objects)). A second branch, a tag or
//! a snapshot that is neither the current one nor an ancestor of it may
//! still read a file that the current snapshot or one of its ancestors
//! retired, and nothing else checks for one, so both ask here first
//! and a table whose history is not linear is refused whole, on every round
//! and whether or not anything of it is past its window: none of its
//! snapshots expires and nothing of it is deleted. Such a table keeps its
//! previous metadata files and every entry of its metadata log too, for as
//! long as it is refused. Reading it and committing to it are not affected.
//! Branches and tags are tracked in
//! <https://github.com/pivotlake/pivot/issues/101>.

use std::collections::HashSet;

use iceberg::spec::{MAIN_BRANCH, SnapshotRef, TableMetadata};

use crate::{Error, Result};

/// The snapshots of the table `metadata` describes, which errors name
/// `table`, oldest first: every snapshot from the oldest to the current
/// one, each the parent of the next, and none for a table without a
/// snapshot. They are told from the metadata alone, with nothing read. The
/// table is refused ([`Error::NonLinearHistory`]) when it has a branch or a
/// tag besides `main`, or a snapshot that is neither the current one nor an
/// ancestor of it: one that following parents from the current snapshot
/// does not reach, which is one whose parent the metadata no longer holds
/// as much as one written beside the current snapshot's ancestors.
pub(crate) fn list_sorted_snapshots<'m>(
    table: &str,
    metadata: &'m TableMetadata,
) -> Result<Vec<&'m SnapshotRef>> {
    let refuse = |reason: String| Error::NonLinearHistory {
        table: table.to_string(),
        reason,
    };
    let mut other_references: Vec<(&str, bool)> = metadata
        .refs()
        .iter()
        .filter(|(name, _)| name.as_str() != MAIN_BRANCH)
        .map(|(name, reference)| (name.as_str(), reference.is_branch()))
        .collect();
    other_references.sort_unstable();
    if let Some((name, is_branch)) = other_references.first() {
        let kind = if *is_branch { "branch" } else { "tag" };
        return Err(refuse(format!(
            "it has the {kind} `{name}` besides `{MAIN_BRANCH}`"
        )));
    }

    let snapshot_count = metadata.snapshots().len();
    let mut snapshots: Vec<&SnapshotRef> = Vec::with_capacity(snapshot_count);
    let mut next = metadata.current_snapshot();
    while let Some(snapshot) = next {
        if snapshots.len() == snapshot_count {
            return Err(refuse(
                "the parents of its snapshots form a cycle".to_string(),
            ));
        }
        snapshots.push(snapshot);
        next = snapshot
            .parent_snapshot_id()
            .and_then(|parent_id| metadata.snapshot_by_id(parent_id));
    }
    if snapshots.len() < snapshot_count {
        let reached_snapshot_ids: HashSet<i64> = snapshots
            .iter()
            .map(|snapshot| snapshot.snapshot_id())
            .collect();
        let unreached_snapshot_id = metadata
            .snapshots()
            .map(|snapshot| snapshot.snapshot_id())
            .filter(|snapshot_id| !reached_snapshot_ids.contains(snapshot_id))
            .min()
            .expect("following parents reached fewer snapshots than the table has");
        return Err(refuse(format!(
            "its snapshot {unreached_snapshot_id} is neither the current snapshot nor an ancestor of it"
        )));
    }
    snapshots.reverse();
    Ok(snapshots)
}
