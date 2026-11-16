//! The sorted runs one worker holds for each bucket.
//!
//! A run is a sequence of batches whose rows are in sort key order from first
//! to last. [`BucketRuns`] holds the runs one bucket has collected.

use arrow_array::RecordBatch;

/// One sorted run, as batch-sized chunks.
pub type SortedRun = Vec<RecordBatch>;

/// The runs one worker holds for one bucket.
#[derive(Default)]
pub struct BucketRuns {
    runs: Vec<SortedRun>,
    row_count: usize,
    /// Rows the bucket held right after all its runs were last merged into
    /// one, which is what the next such merge is measured against.
    row_count_after_full_merge: usize,
    /// How many of the newest runs came straight from sorting pending rows
    /// and have not been part of any merge.
    unmerged_run_count: usize,
}

impl BucketRuns {
    /// Adds a run freshly sorted from pending rows.
    pub fn push_run(&mut self, run: SortedRun) {
        let rows: usize = run.iter().map(RecordBatch::num_rows).sum();
        if rows == 0 {
            return;
        }
        self.row_count += rows;
        self.runs.push(run);
        self.unmerged_run_count += 1;
    }

    /// Whether the bucket holds no rows.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Number of runs that have not been part of any merge.
    pub fn unmerged_run_count(&self) -> usize {
        self.unmerged_run_count
    }

    /// Whether the bucket has at least doubled since all its runs were last
    /// merged into one, measured against a floor of `min_rows`.
    ///
    /// Merging everything only when the row count has doubled keeps the
    /// total work proportional to the rows received: each such merge copies
    /// every row the bucket holds, and at least half of those arrived since
    /// the previous one.
    pub fn has_doubled_since_full_merge(&self, min_rows: usize) -> bool {
        self.runs.len() > 1 && self.row_count >= 2 * self.row_count_after_full_merge.max(min_rows)
    }

    /// Takes every run out, leaving the bucket empty.
    pub fn take_runs(&mut self) -> Vec<SortedRun> {
        self.row_count = 0;
        self.unmerged_run_count = 0;
        std::mem::take(&mut self.runs)
    }

    /// Takes out the runs that have not been part of any merge, leaving the
    /// older ones.
    pub fn take_unmerged_runs(&mut self) -> Vec<SortedRun> {
        let first_unmerged = self.runs.len() - self.unmerged_run_count;
        let unmerged_runs = self.runs.split_off(first_unmerged);
        self.row_count = self.runs.iter().flatten().map(RecordBatch::num_rows).sum();
        self.unmerged_run_count = 0;
        unmerged_runs
    }

    /// Adds the result of merging runs this bucket handed out.
    pub fn push_merged_run(&mut self, run: SortedRun) {
        self.push_run(run);
        self.unmerged_run_count = 0;
    }

    /// Makes `run` the bucket's only run, as the result of merging every run
    /// that [`take_runs`](Self::take_runs) handed out.
    pub fn replace_with_fully_merged(&mut self, run: SortedRun) {
        debug_assert!(self.runs.is_empty());
        self.push_merged_run(run);
        self.row_count_after_full_merge = self.row_count;
    }
}
