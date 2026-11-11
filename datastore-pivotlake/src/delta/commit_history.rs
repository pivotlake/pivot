//! What the commits this process made to a Delta table changed, kept so a
//! partition's index built at one version is brought to another without
//! the log being replayed.
//!
//! A table's copies in a process stand at different versions: the one a
//! writer commits through moves ahead of a compaction round's and of a
//! running query's. The cache holds one index per partition, the newest
//! version's that asked, so a copy at another version finds an index that
//! is not its own. When every commit between the two versions was made
//! here, the index is brought across them by the files those commits
//! removed and added in the partition, forward or backward; a commit made
//! elsewhere breaks the chain, and the partition is then read from the log.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use table_file_index::IndexedRecord;

use super::file_record::{DeltaFile, DeltaIndex};

/// How much of the removed and added files' records the history holds, by
/// their estimated memory; past it the oldest commits are forgotten.
const COMMIT_HISTORY_HELD_BYTES: u64 = 16 * 1024 * 1024;

/// The files one commit removed from and added to one partition.
pub(super) struct PartitionChange {
    pub partition: Arc<[u8]>,
    pub removed: Vec<Arc<DeltaFile>>,
    pub added: Vec<Arc<DeltaFile>>,
}

/// What the commit that wrote one version changed.
struct CommittedChanges {
    /// The version the commit wrote, one above the one it was made on.
    version: u64,
    changes: Vec<PartitionChange>,
    /// The estimated memory of the records the changes hold.
    heap_bytes: u64,
}

impl CommittedChanges {
    /// What the commit changed in `partition`.
    fn list_changes_in<'c>(
        &'c self,
        partition: &'c [u8],
    ) -> impl Iterator<Item = &'c PartitionChange> {
        self.changes
            .iter()
            .filter(move |change| change.partition.as_ref() == partition)
    }
}

/// The changes of a run of consecutive versions this process committed,
/// oldest first.
#[derive(Default)]
pub(super) struct CommitHistory {
    commits: VecDeque<CommittedChanges>,
    held_bytes: u64,
}

impl CommitHistory {
    /// Remember that the commit that wrote `version` made `changes`. A
    /// version that does not follow the latest one remembered starts the
    /// history over: what lies between was committed elsewhere.
    pub(super) fn record(&mut self, version: u64, changes: Vec<PartitionChange>) {
        let follows_latest = self
            .commits
            .back()
            .is_none_or(|latest| latest.version + 1 == version);
        if !follows_latest {
            self.forget();
        }
        let heap_bytes = changes
            .iter()
            .flat_map(|change| change.removed.iter().chain(&change.added))
            .map(|file| file.estimate_heap_bytes())
            .sum();
        self.held_bytes += heap_bytes;
        self.commits.push_back(CommittedChanges {
            version,
            changes,
            heap_bytes,
        });
        while self.held_bytes > COMMIT_HISTORY_HELD_BYTES && self.commits.len() > 1 {
            let oldest = self
                .commits
                .pop_front()
                .expect("more than one commit is remembered");
            self.held_bytes -= oldest.heap_bytes;
        }
    }

    /// Forget every commit: a version was committed elsewhere.
    pub(super) fn forget(&mut self) {
        self.commits.clear();
        self.held_bytes = 0;
    }

    /// `index`, the files of `partition` at the version `built_at`, as the
    /// partition's files at the version `wanted`; `None` when a commit
    /// between the two versions is not remembered.
    pub(super) fn bring_index(
        &self,
        index: &DeltaIndex,
        partition: &[u8],
        built_at: u64,
        wanted: u64,
    ) -> Option<DeltaIndex> {
        let (earlier, later) = (built_at.min(wanted), built_at.max(wanted));
        let crossed = self.list_commits_between(earlier, later)?;
        let mut brought = index.clone();
        if built_at < wanted {
            let later_changes = crossed
                .iter()
                .flat_map(|commit| commit.list_changes_in(partition));
            for change in later_changes {
                apply_change(&mut brought, &change.removed, &change.added);
            }
        } else {
            let earlier_changes = crossed
                .iter()
                .rev()
                .flat_map(|commit| commit.list_changes_in(partition));
            for change in earlier_changes {
                apply_change(&mut brought, &change.added, &change.removed);
            }
        }
        Some(brought)
    }

    /// The partitions the commits that wrote the versions above `earlier`
    /// up to `later` added a file to, in the order of the bytes they are
    /// named by; `None` when one of those commits is not remembered.
    pub(super) fn list_partitions_added_between(
        &self,
        earlier: u64,
        later: u64,
    ) -> Option<BTreeSet<Arc<[u8]>>> {
        let commits = self.list_commits_between(earlier, later)?;
        Some(
            commits
                .iter()
                .flat_map(|commit| &commit.changes)
                .filter(|change| !change.added.is_empty())
                .map(|change| Arc::clone(&change.partition))
                .collect(),
        )
    }

    /// The commits that wrote the versions above `earlier` up to `later`,
    /// oldest first; `None` when one of them is not remembered.
    fn list_commits_between(&self, earlier: u64, later: u64) -> Option<Vec<&CommittedChanges>> {
        if earlier == later {
            return Some(Vec::new());
        }
        let oldest = self.commits.front()?.version;
        let newest = self.commits.back()?.version;
        if oldest > earlier + 1 || newest < later {
            return None;
        }
        Some(
            self.commits
                .iter()
                .filter(|commit| commit.version > earlier && commit.version <= later)
                .collect(),
        )
    }
}

/// Take `leaving` out of `index` and put `joining` in.
fn apply_change(index: &mut DeltaIndex, leaving: &[Arc<DeltaFile>], joining: &[Arc<DeltaFile>]) {
    for file in leaving {
        index.remove(&file.facts().file().path);
    }
    for file in joining {
        index.insert(Arc::clone(file));
    }
}
