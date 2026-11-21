//! Which small files of a partition the merges of one commit take: the runs
//! of neighbours cheapest to merge, and the sizes and counts that bound a
//! merge and the merges written together.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use object_storage::ObjectPath;

use super::{CompactionPass, MergeSelection, index};
use crate::table_format::CompactionFile;

/// The most small files one merge takes. A merge holds rows of every input
/// file in memory at once, so its memory grows with their number. A larger
/// pile is merged in steps.
pub const MAX_SMALL_FILES_PER_MERGE: usize = 10;

/// A merge takes more than [`MAX_SMALL_FILES_PER_MERGE`] files while they
/// hold no more than these bytes and are no more than this many: files this
/// small take little memory each, and merging few of them at a time removes
/// fewer files than a busy table gains.
const MAX_BYTES_OF_A_MERGE_OF_MANY_FILES: u64 = 64 * 1024 * 1024;
const MAX_FILES_OF_A_MERGE_OF_FEW_BYTES: usize = 100;

/// The most merges of small files a round swaps into the table in one
/// commit. A commit costs the same whatever it holds, so a table that gains
/// small files quickly loses them faster when one commit swaps in several
/// merges. The merges of a commit are written at the same time, each
/// holding the rows of its inputs decoded in memory until its output is
/// encoded, so the number written together is kept small.
pub const MAX_SMALL_FILE_MERGES_PER_COMMIT: usize = 5;

/// The most bytes of inputs the merges of one commit hold between them when
/// there are several: they are written at the same time, each holding the
/// rows of its inputs decoded in memory until its output is encoded. A
/// merge alone may hold more.
pub const MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER: u64 = 100 * 1024 * 1024;

/// What merging one more file costs whatever its size, added to each file's
/// bytes wherever a merge of small files is weighed. Without it many
/// similarly small files would be refused because their bytes are few.
const FIXED_FILE_COST_BYTES: f64 = (5 * 1024 * 1024) as f64;

/// How many times the size of its largest file a run of small files must be
/// to merge, in a partition with few small files. A run of equally sized
/// files then needs this many files. The more small files the partition has,
/// the less it asks, down to [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`].
const MIN_BALANCE_RATIO: f64 = 5.0;
const MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION: f64 = 2.0;

/// The counts of small files in a partition between which the balance a merge
/// must have falls from [`MIN_BALANCE_RATIO`] to
/// [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`].
const SMALL_FILES_TO_START_LOWERING_BALANCE: f64 = 10.0;
const SMALL_FILES_OF_A_CROWDED_PARTITION: f64 = 50.0;

/// A merge of a run removes one file fewer than the run has. A run is
/// scored by the bytes it rewrites for each of its files but this many, a
/// little under two, so a run of two files counts as removing a tenth of a
/// file, and a longer run that may merge is taken before it.
const FILES_OF_A_RUN_NOT_COUNTED_AS_REMOVED: f64 = 1.9;

/// A run whose merged file would be about the size of the file before it is
/// preferred, because merging it leaves two neighbours of one size for a
/// later merge. The preference applies when the file before is larger than
/// this share of the merged file, and within this many powers of two of it;
/// a run that merges to exactly its size has its score multiplied by
/// [`BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE`].
const MIN_SHARE_OF_THE_FILE_BEFORE_A_RUN: f64 = 0.9;
const MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE: f64 = 0.5;
const BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE: f64 = 0.75;

/// How many input bytes a merge is taken to read for each byte it writes:
/// a file carries metadata and dictionaries of its own that a merged file
/// holds once, so a merged file is smaller than its inputs together.
const INPUT_BYTES_PER_MERGED_BYTE: f64 = 1.3;

/// A file below half the target size, as the selection of a merge weighs
/// it.
#[derive(Clone, Copy)]
struct SmallFile {
    size: u64,
}

/// A run of small files one merge may take, as the picking of a commit's
/// runs orders it: the cheaper to merge is the greater, and of two as cheap
/// the one that starts later in the partition's order.
struct RunCandidate {
    /// The bytes the merge rewrites for each file it removes.
    score: f64,
    first: usize,
    end: usize,
}

impl Ord for RunCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.first.cmp(&other.first))
    }
}

impl PartialOrd for RunCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RunCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RunCandidate {}

impl MergeSelection {
    /// Pick the merges one commit makes of the small files of a partition,
    /// the inputs of each, from those no merge in flight holds reserved: in
    /// a routine round the runs of neighbours cheapest to merge
    /// ([`pick_small_file_runs`](Self::pick_small_file_runs)), in a final
    /// sweep one merge of the first of them, as many as a merge takes.
    pub(super) fn next_small_merges(
        &self,
        partition_files: &index::PartitionFileLists,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<Vec<Vec<Arc<dyn CompactionFile>>>> {
        let listed: Vec<&Arc<dyn CompactionFile>> =
            partition_files.list_small_files(reserved).collect();
        if listed.len() < 2 {
            return None;
        }
        let clone_files = |files: &[&Arc<dyn CompactionFile>]| -> Vec<Arc<dyn CompactionFile>> {
            files.iter().map(|file| Arc::clone(file)).collect()
        };
        let merges: Vec<Vec<Arc<dyn CompactionFile>>> = match pass {
            CompactionPass::Routine => {
                let small_files: Vec<SmallFile> = listed
                    .iter()
                    .map(|file| SmallFile {
                        size: file.file().size,
                    })
                    .collect();
                self.pick_small_file_runs(&small_files, MAX_SMALL_FILE_MERGES_PER_COMMIT)
                    .into_iter()
                    .map(|run| clone_files(&listed[run]))
                    .collect()
            }
            CompactionPass::Final => {
                let first_files = &listed[..listed.len().min(MAX_SMALL_FILES_PER_MERGE)];
                vec![clone_files(first_files)]
            }
        };
        (!merges.is_empty()).then_some(merges)
    }

    /// The runs of neighbours among `files`, the small files of one partition
    /// in its order, that the merges of one commit take, a run each, in the
    /// order they are picked: at most `max_runs` of them, of no more than
    /// [`MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER`] between them, the
    /// first whatever it holds. None when no run may merge.
    ///
    /// Each run is the one cheapest to merge for the files it removes, of
    /// the runs that hold no file of a run picked before it. Every run of at
    /// least two neighbours whose bytes stay within the most a merge's
    /// inputs may hold is weighed, of at most
    /// [`MAX_SMALL_FILES_PER_MERGE`] files, or of at
    /// most [`MAX_FILES_OF_A_MERGE_OF_FEW_BYTES`] that hold no more than
    /// [`MAX_BYTES_OF_A_MERGE_OF_MANY_FILES`] bytes. A run may merge when it
    /// has the configured number of files and is balanced: its bytes are
    /// enough times those of its largest file, so a large file is not
    /// rewritten to absorb a few small ones. A partition with many small
    /// files asks for less balance, so it merges sooner
    /// ([`find_min_balance_ratio`]). Of the runs that may merge, the one
    /// with the fewest bytes rewritten per file removed is taken, and of
    /// equals the last in the partition's order. A run that would merge to
    /// about the size of the file before it counts as cheaper than it is.
    ///
    /// The files are gone through once, for the best run that starts at
    /// each. The best of those is picked, and the runs that start at a file
    /// are weighed again only when the best of them holds a file a picked
    /// run took.
    fn pick_small_file_runs(&self, files: &[SmallFile], max_runs: usize) -> Vec<Range<usize>> {
        let min_balance_ratio = find_min_balance_ratio(files.len());

        let mut is_taken = vec![false; files.len()];
        let mut candidates: BinaryHeap<RunCandidate> = (0..files.len())
            .filter_map(|first| self.find_best_run_from(files, first, &is_taken, min_balance_ratio))
            .collect();
        let mut runs: Vec<Range<usize>> = Vec::new();
        let mut input_bytes = 0;
        while runs.len() < max_runs {
            let Some(candidate) = candidates.pop() else {
                break;
            };
            let run = candidate.first..candidate.end;
            if is_taken[run.clone()].contains(&true) {
                // What is left of the runs that start at the file, none when
                // the file itself was taken.
                candidates.extend(self.find_best_run_from(
                    files,
                    candidate.first,
                    &is_taken,
                    min_balance_ratio,
                ));
                continue;
            }
            let run_bytes: u64 = files[run.clone()].iter().map(|file| file.size).sum();
            let is_first_run = runs.is_empty();
            if !is_first_run && input_bytes + run_bytes > MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER
            {
                break;
            }
            input_bytes += run_bytes;
            is_taken[run.clone()].fill(true);
            runs.push(run);
        }
        runs
    }

    /// The run cheapest to merge of those that start at `files[first]` and
    /// hold no taken file, if any is balanced `min_balance_ratio` times and
    /// may merge, weighed as
    /// [`pick_small_file_runs`](Self::pick_small_file_runs) says: of runs as
    /// cheap, the longest.
    fn find_best_run_from(
        &self,
        files: &[SmallFile],
        first: usize,
        is_taken: &[bool],
        min_balance_ratio: f64,
    ) -> Option<RunCandidate> {
        if is_taken[first] {
            return None;
        }
        let mut best: Option<RunCandidate> = None;
        let mut total = files[first].size;
        let mut largest = files[first].size;
        for end in first + 2..=files.len() {
            if is_taken[end - 1] {
                break;
            }
            let last = files[end - 1];
            total += last.size;
            largest = largest.max(last.size);
            let count = end - first;
            let has_too_many_files = count > MAX_FILES_OF_A_MERGE_OF_FEW_BYTES
                || (count > MAX_SMALL_FILES_PER_MERGE
                    && total > MAX_BYTES_OF_A_MERGE_OF_MANY_FILES);
            if has_too_many_files || total > self.max_merge_input_bytes {
                break;
            }
            if count < self.min_files {
                continue;
            }
            let weighed_bytes = total as f64 + FIXED_FILE_COST_BYTES * count as f64;
            let balance_ratio = weighed_bytes / (largest as f64 + FIXED_FILE_COST_BYTES);
            if balance_ratio < min_balance_ratio {
                continue;
            }

            let mut score = weighed_bytes / (count as f64 - FILES_OF_A_RUN_NOT_COUNTED_AS_REMOVED);
            let merged_size = total as f64 / INPUT_BYTES_PER_MERGED_BYTE;
            let size_before = match first {
                0 => 0.0,
                _ => files[first - 1].size as f64,
            };
            if size_before > merged_size * MIN_SHARE_OF_THE_FILE_BEFORE_A_RUN {
                let powers_of_two_apart = (merged_size / size_before).log2().abs();
                if powers_of_two_apart < MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE {
                    let closeness =
                        1.0 - powers_of_two_apart / MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE;
                    score *= 1.0
                        - (1.0 - BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE) * closeness;
                }
            }

            if best.as_ref().is_none_or(|best| score <= best.score) {
                best = Some(RunCandidate { score, first, end });
            }
        }
        best
    }
}

/// How many times the size of its largest file a run of small files must be
/// to merge, in a partition of `small_file_count` small files:
/// [`MIN_BALANCE_RATIO`] among few, falling to
/// [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`] among many.
fn find_min_balance_ratio(small_file_count: usize) -> f64 {
    let crowding = ((small_file_count as f64 - SMALL_FILES_TO_START_LOWERING_BALANCE)
        / (SMALL_FILES_OF_A_CROWDED_PARTITION - SMALL_FILES_TO_START_LOWERING_BALANCE))
        .clamp(0.0, 1.0);
    MIN_BALANCE_RATIO + (MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION - MIN_BALANCE_RATIO) * crowding
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::{
        DEFAULT_COMPACT_BYTES, DEFAULT_LAYOUT_CLIQUE_SIZE, DEFAULT_MIN_FILES_TO_MERGE,
        default_max_merge_input_bytes,
    };

    const MIB: u64 = 1024 * 1024;

    /// The selection of small-file merges of at most `max_merge_input_bytes`
    /// and at least `min_files` files, toward files of the default size.
    fn build_small_file_selection(max_merge_input_bytes: u64, min_files: usize) -> MergeSelection {
        MergeSelection {
            target_bytes: DEFAULT_COMPACT_BYTES,
            max_merge_input_bytes,
            min_files,
            layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
        }
    }

    /// The selection of small-file merges a datastore makes by default.
    fn build_default_small_file_selection() -> MergeSelection {
        build_small_file_selection(
            default_max_merge_input_bytes(DEFAULT_COMPACT_BYTES),
            DEFAULT_MIN_FILES_TO_MERGE,
        )
    }

    /// `count` neighbours of `size` bytes each, files of the table.
    fn build_files_of_size(count: usize, size: u64) -> Vec<SmallFile> {
        vec![SmallFile { size }; count]
    }

    /// The run `selection` picks among `files` for a commit of one merge.
    fn pick_one_run(selection: MergeSelection, files: &[SmallFile]) -> Option<Range<usize>> {
        selection.pick_small_file_runs(files, 1).into_iter().next()
    }

    #[test]
    fn few_equal_small_files_merge_once_there_are_five() {
        let files = build_files_of_size(5, MIB);
        let selection = build_default_small_file_selection();

        let run_of_four = pick_one_run(selection, &files[..4]);
        let run_of_five = pick_one_run(selection, &files);

        assert_eq!(run_of_four, None);
        assert_eq!(run_of_five, Some(0..5));
    }

    #[test]
    fn a_merge_takes_no_fewer_files_than_configured() {
        let files = build_files_of_size(5, MIB);
        let max_merge_input_bytes = default_max_merge_input_bytes(DEFAULT_COMPACT_BYTES);

        let allowed = pick_one_run(build_small_file_selection(max_merge_input_bytes, 5), &files);
        let waiting = pick_one_run(build_small_file_selection(max_merge_input_bytes, 6), &files);

        assert_eq!(allowed, Some(0..5));
        assert_eq!(waiting, None);
    }

    #[test]
    fn a_large_file_is_left_out_of_the_merge_of_its_small_neighbours() {
        let mut files = build_files_of_size(1, 30 * MIB);
        files.extend(build_files_of_size(9, MIB));

        let run = pick_one_run(build_default_small_file_selection(), &files);

        assert_eq!(run, Some(1..10));
    }

    #[test]
    fn a_run_merges_with_the_tiny_files_at_its_end() {
        let mut files = build_files_of_size(4, 10 * MIB);
        files.extend(build_files_of_size(3, MIB / 4));

        let run = pick_one_run(build_default_small_file_selection(), &files);

        assert_eq!(run, Some(0..7));
    }

    #[test]
    fn the_run_that_rewrites_the_fewest_bytes_per_file_removed_is_taken() {
        let mut files = build_files_of_size(5, 10 * MIB);
        files.extend(build_files_of_size(5, MIB));

        let run = pick_one_run(build_default_small_file_selection(), &files);

        assert_eq!(run, Some(5..10));
    }

    #[test]
    fn a_run_that_merges_to_the_size_of_the_file_before_it_is_preferred() {
        let build_files_after = |size_before: u64| {
            let mut files = build_files_of_size(1, size_before);
            files.extend(build_files_of_size(5, MIB));
            files.extend(build_files_of_size(1, 30 * MIB));
            files.extend(build_files_of_size(5, MIB));
            files
        };
        let selection = build_default_small_file_selection();

        let run_after_a_file_of_its_merged_size =
            pick_one_run(selection, &build_files_after(4 * MIB));
        let run_after_a_far_larger_file = pick_one_run(selection, &build_files_after(30 * MIB));

        assert_eq!(run_after_a_file_of_its_merged_size, Some(1..6));
        assert_eq!(run_after_a_far_larger_file, Some(7..12));
    }

    #[test]
    fn a_crowded_partition_merges_a_run_a_sparse_one_leaves() {
        let selection = build_small_file_selection(85 * MIB, 2);

        let run_among_few = pick_one_run(selection, &build_files_of_size(9, 20 * MIB));
        let run_among_many = pick_one_run(selection, &build_files_of_size(50, 20 * MIB));

        assert_eq!(run_among_few, None);
        assert_eq!(run_among_many, Some(46..50));
    }

    #[test]
    fn a_merge_takes_no_more_bytes_than_the_merge_target() {
        let files = build_files_of_size(60, 20 * MIB);

        let run = pick_one_run(build_small_file_selection(85 * MIB, 2), &files);

        assert_eq!(run, Some(56..60));
    }

    #[test]
    fn a_merge_of_many_bytes_takes_at_most_ten_files() {
        let files = build_files_of_size(30, 8 * MIB);

        let run = pick_one_run(build_small_file_selection(1024 * MIB, 2), &files);

        assert_eq!(run, Some(20..30));
    }

    #[test]
    fn a_merge_takes_more_than_ten_files_while_they_hold_few_bytes() {
        let files = build_files_of_size(120, MIB);

        let run = pick_one_run(build_default_small_file_selection(), &files);

        assert_eq!(run, Some(56..120));
    }

    #[test]
    fn a_merge_of_few_bytes_takes_at_most_a_hundred_files() {
        let files = build_files_of_size(150, 100 * 1024);

        let run = pick_one_run(build_default_small_file_selection(), &files);

        assert_eq!(run, Some(50..150));
    }

    #[test]
    fn the_runs_of_one_commit_are_each_the_best_of_what_the_runs_before_left() {
        let files = build_files_of_size(250, MIB / 4);

        let runs = build_small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, MAX_SMALL_FILE_MERGES_PER_COMMIT);

        assert_eq!(runs, [150..250, 50..150, 0..50]);
    }

    #[test]
    fn a_commit_takes_no_more_runs_than_it_is_asked_for() {
        let files = build_files_of_size(250, MIB / 4);

        let runs = build_small_file_selection(1300 * MIB, 2).pick_small_file_runs(&files, 2);

        assert_eq!(runs, [150..250, 50..150]);
    }

    #[test]
    fn a_run_that_would_take_a_commits_inputs_past_what_is_written_together_ends_the_picking() {
        let files = build_files_of_size(120, MIB);

        let runs = build_small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, MAX_SMALL_FILE_MERGES_PER_COMMIT);

        let only_run = 56..120;
        assert_eq!(runs, [only_run]);
    }

    #[test]
    fn a_run_picked_around_a_large_file_leaves_the_runs_on_either_side_of_it() {
        let mut files = build_files_of_size(8, MIB);
        files.extend(build_files_of_size(1, 400 * MIB));
        files.extend(build_files_of_size(9, MIB));

        let runs = build_small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, MAX_SMALL_FILE_MERGES_PER_COMMIT);

        assert_eq!(runs, [9..18, 0..8]);
    }
}
