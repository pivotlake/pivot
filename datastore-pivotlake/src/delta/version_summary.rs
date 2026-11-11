//! What a Delta table version holds of its files as a whole: counted while
//! the log is replayed, and brought to the next version by the files a
//! commit removed and added, so nothing of a single file is held.

use std::sync::Arc;

use imbl::OrdMap;
use table_file_index::IndexedRecord;

use super::file_record::DeltaFile;
use crate::file_entry::PartitionValues;

/// One partition a version has a file in.
#[derive(Clone)]
pub(super) struct PartitionSummary {
    /// The tuple the log records for the partition's files; `None` for the
    /// files recorded without one.
    pub values: Option<Arc<PartitionValues>>,
    /// How many live files the partition holds.
    pub file_count: u64,
}

/// One live file as a version counts it.
pub(super) struct CountedFile<'f> {
    /// The partition the file sits in, as its index names it.
    pub partition: &'f Arc<[u8]>,
    /// The tuple the log records for the file.
    pub values: Option<&'f PartitionValues>,
    pub size: u64,
    /// The rows the file holds, when the log records them.
    pub rows: Option<u64>,
    /// Where the commit that added the file stands among the table's
    /// commits.
    pub commit_sequence: u64,
}

impl<'f> CountedFile<'f> {
    /// `file` as a version counts it.
    pub(super) fn of(file: &'f DeltaFile) -> Self {
        let facts = file.facts();
        Self {
            partition: file.partition(),
            values: file.entry.partition.as_ref(),
            size: facts.file().size,
            rows: facts.row_count(),
            commit_sequence: facts.commit_sequence(),
        }
    }
}

/// The whole-table counts of one version, and its partitions.
#[derive(Clone, Default)]
pub(super) struct VersionSummary {
    /// Every partition the version has a file in, by the bytes it is named
    /// by. Persistent: the next version's shares what a commit did not
    /// touch.
    pub partitions: OrdMap<Arc<[u8]>, PartitionSummary>,
    pub file_count: u64,
    /// The rows of the files whose log entries record a count.
    counted_rows: u64,
    /// How many live files' log entries record no row count.
    uncounted_files: u64,
    /// The sum of the files' sizes.
    pub bytes: u64,
    /// Where the latest commit that added a file stands among the table's
    /// commits; 0 for a table no file was added to.
    pub latest_commit_sequence: u64,
}

impl VersionSummary {
    /// The rows of every live file, when every file's log entry records its
    /// count.
    pub(super) fn count_rows(&self) -> Option<u64> {
        (self.uncounted_files == 0).then_some(self.counted_rows)
    }

    /// Count `file` among the version's live files.
    pub(super) fn add(&mut self, file: CountedFile<'_>) {
        self.file_count += 1;
        self.bytes += file.size;
        match file.rows {
            Some(rows) => self.counted_rows += rows,
            None => self.uncounted_files += 1,
        }
        self.latest_commit_sequence = self.latest_commit_sequence.max(file.commit_sequence);
        match self.partitions.get_mut(file.partition) {
            Some(partition) => partition.file_count += 1,
            None => {
                let partition = PartitionSummary {
                    values: file.values.cloned().map(Arc::new),
                    file_count: 1,
                };
                self.partitions
                    .insert(Arc::clone(file.partition), partition);
            }
        }
    }

    /// Take `file`, a live file of the version, out of its counts. A
    /// partition left without a file is no longer one of the version's.
    pub(super) fn remove(&mut self, file: CountedFile<'_>) {
        self.file_count -= 1;
        self.bytes -= file.size;
        match file.rows {
            Some(rows) => self.counted_rows -= rows,
            None => self.uncounted_files -= 1,
        }
        let partition = self
            .partitions
            .get_mut(file.partition)
            .expect("a live file sits in a partition of its version");
        partition.file_count -= 1;
        if partition.file_count == 0 {
            self.partitions.remove(file.partition);
        }
    }
}
