//! Decides the order in which the rows of the input files go to the output.
//!
//! Every input file is sorted by the table's sort columns. The output is
//! therefore a sequence of *runs*: some next rows of one file, then some
//! next rows of another, and so on. [`KeyMerge::plan_next_run`] gives one
//! run at a time.
//!
//! The merge works on a few row groups at a time, which
//! [`KeyMerge::row_groups_to_merge`] gives:
//!
//! - Among the row groups that hold unmerged rows, the one that *starts
//!   first* is the one with the smallest minimum of the first sort column,
//!   by the files' statistics.
//! - The row groups that *overlap* it are those whose minimum is at or
//!   below its maximum. No other row group can hold a row that goes before
//!   the last row of the one that starts first.
//!
//! Only the files whose next unmerged row is in one of those row groups
//! take part:
//!
//! - If one file takes part, nothing overlaps its row group, and the rest
//!   of that row group is the run. No key is compared.
//! - If several take part, the merge orders their next rows by their
//!   decoded keys, which the caller adds with
//!   [`KeyMerge::add_decoded_keys`]. If the key of a file's next row is not
//!   decoded, the merge returns [`NextRun::NeedsKeys`], and the caller
//!   calls `plan_next_run` again once the key is added.
//!
//! # Ordering the next rows of several files
//!
//! Each file that takes part offers a *stretch*: its next rows whose keys
//! are decoded, up to the end of its row group and to
//! [`STRETCH_ROWS`] rows. What follows a file's stretch is not known yet,
//! only that it does not sort before the stretch's last row. So the rows
//! that can be ordered now are those at or before the smallest of the
//! stretches' last keys:
//!
//! ```text
//!   file A's stretch:   1  3  5 │ 7  9
//!   file B's stretch:   2  4  6 │            B's last key, 6, is the smallest
//!   file C's stretch:           │ 8 10 12
//!                               └ the cut: 1 2 3 4 5 6 are ordered now
//! ```
//!
//! The cut is found in each stretch by bisection, and the rows before it
//! are merged in one go. The file whose last key was the smallest gives its
//! whole stretch, so every such step orders at least one stretch. Rows with
//! equal keys go out in no particular order.
//!
//! The keys are compared as they were decoded, with the comparison and the
//! merge of sorted stretches that `ORDER BY` uses ([`RowComparer`],
//! [`ChunkedRowSorter`]). The merge holds the decoded keys only until their
//! rows are merged, which is before the read sends those rows on, so it
//! holds nothing the read does not already count.
//!
//! When a run takes the last unmerged row of a file's row group, the row
//! groups are found again from where each file's next unmerged row then is.
//!
//! The row groups' bounds are compared in arrow's row format, in which the
//! bytes of two values compare the same way the values do, with NULL first.

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, Scalar, new_null_array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{ChunkedRowSorter, OrderBy, RECORD_BATCH_SIZE, RowComparer};

use crate::types::metadata::RowGroupMetadata;
use crate::writing::error::{WriteError, WriteResult};

/// The most next rows of one file that are ordered in one go, so that one
/// merge handles about a batch of rows of each file that takes part.
const STRETCH_ROWS: u64 = RECORD_BATCH_SIZE as u64;

/// The range of one sort column's values in one row group: every value is
/// at or after `lower` and at or before `upper`. Each bound is an array of
/// one value. `None` means the bound is not known. A NULL `lower` means the
/// row group may contain NULLs, which sort first.
pub(super) struct SortKeyBounds {
    pub(super) lower: Option<ArrayRef>,
    pub(super) upper: Option<ArrayRef>,
}

impl SortKeyBounds {
    /// The bounds that the statistics of `row_group` give for its column
    /// `column`, a sort column of type `key_type`.
    ///
    /// - No statistics: both bounds unknown.
    /// - Every row NULL: both bounds NULL.
    /// - Some rows may be NULL (or the NULL count is unknown): the lower
    ///   bound is NULL.
    /// - Otherwise: the recorded minimum and maximum.
    pub(super) fn from_statistics(
        row_group: &RowGroupMetadata,
        column: usize,
        key_type: &DataType,
    ) -> WriteResult<Self> {
        let Some(statistics) = row_group.column_statistics(column) else {
            return Ok(Self {
                lower: None,
                upper: None,
            });
        };
        let cast_to_key_type = |bound: Scalar<ArrayRef>| {
            arrow_cast::cast(&bound.into_inner(), key_type).map_err(WriteError::from)
        };
        let null = || Some(new_null_array(key_type, 1));
        if statistics.null_count == Some(row_group.num_rows) {
            return Ok(Self {
                lower: null(),
                upper: null(),
            });
        }
        let lower = match statistics.null_count {
            Some(0) => statistics.min().map(cast_to_key_type).transpose()?,
            _ => null(),
        };
        Ok(Self {
            lower,
            upper: statistics.max().map(cast_to_key_type).transpose()?,
        })
    }
}

/// What the merge is told about one row group of an input file.
pub(super) struct RowGroupKeyBounds {
    pub(super) row_count: u64,
    /// The bounds of the first sort column. `None` for a table without
    /// sort columns.
    pub(super) first_key_bounds: Option<SortKeyBounds>,
}

/// The result of [`KeyMerge::plan_next_run`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum NextRun {
    /// The next `rows` rows of the output are the next `rows` unmerged rows
    /// of input file `file`. They are now recorded as merged.
    Found { file: usize, rows: u64 },
    /// The next run cannot be decided until these keys are decoded.
    NeedsKeys(Vec<NeededKey>),
    /// Every row of every file is merged.
    NoRowsLeft,
}

/// One row group of an input file: `row_group` is its place among the
/// file's row groups that hold rows, in file order.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) struct RowGroupOfFile {
    pub(super) file: usize,
    pub(super) row_group: usize,
}

/// One key the merge needs decoded: the value of sort column `sort_column`
/// (its index in the sort order) in row `row` of input file `file`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) struct NeededKey {
    pub(super) file: usize,
    pub(super) sort_column: usize,
    pub(super) row: u64,
}

/// One row group of an input file as the merge tracks it.
struct MergedRowGroup {
    /// The file row right after the row group's last row.
    rows_end: u64,
    /// The bounds of the first sort column, in the row format. `None` for
    /// a bound that is not known.
    first_key_lower: Option<Vec<u8>>,
    first_key_upper: Option<Vec<u8>>,
}

/// The decoded keys of consecutive rows of one file, as they were decoded.
struct KeyBatch {
    row_count: u64,
    /// The values of each sort column, in sort order. The columns of the
    /// same rows are decoded apart and arrive one by one.
    columns: Vec<Option<ArrayRef>>,
}

impl KeyBatch {
    /// The keys as one batch of the sort columns, once every column has
    /// arrived.
    fn build_batch(&self) -> WriteResult<Option<RecordBatch>> {
        let Some(columns) = self
            .columns
            .iter()
            .cloned()
            .collect::<Option<Vec<ArrayRef>>>()
        else {
            return Ok(None);
        };
        let fields: Vec<Field> = columns
            .iter()
            .enumerate()
            .map(|(sort_column, values)| {
                Field::new(sort_column.to_string(), values.data_type().clone(), true)
            })
            .collect();
        Ok(Some(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?))
    }
}

/// The next rows of one file whose keys are decoded: what the file offers
/// to be ordered next.
struct KeyStretch {
    file: usize,
    /// The key batches that hold the stretch's rows, in row order.
    batches: Vec<RecordBatch>,
    /// The row of the first batch that the stretch starts at.
    first_row_in_batches: usize,
    row_count: usize,
}

impl KeyStretch {
    /// Where row `row` of the stretch is in its batches: the batch's place
    /// and the row of it.
    fn locate_row(&self, row: usize) -> (usize, usize) {
        let mut row_in_batch = self.first_row_in_batches + row;
        for (place, batch) in self.batches.iter().enumerate() {
            if row_in_batch < batch.num_rows() {
                return (place, row_in_batch);
            }
            row_in_batch -= batch.num_rows();
        }
        unreachable!("a stretch's rows are all in its batches")
    }
}

/// One input file as the merge tracks it.
struct MergeInput {
    /// The file's row groups, in file order.
    row_groups: Vec<MergedRowGroup>,
    /// The first file row that is not merged yet.
    next_unmerged_row: u64,
    /// The decoded keys that hold a row that is not merged yet, by the file
    /// row of their first key.
    key_batches: BTreeMap<u64, KeyBatch>,
}

impl MergeInput {
    /// An input file with the row groups `row_groups`, in file order, none
    /// of whose rows is merged and none of whose keys is decoded.
    /// `first_key_converter` converts values of the first sort column to
    /// the row format; a table without sort columns has none, and no
    /// bounds.
    fn new(
        row_groups: Vec<RowGroupKeyBounds>,
        first_key_converter: Option<&RowConverter>,
    ) -> WriteResult<Self> {
        // A one-value bound of the first sort column, in the row format.
        let encode_first_key_bound = |bound: Option<ArrayRef>| -> WriteResult<Option<Vec<u8>>> {
            let (Some(bound), Some(converter)) = (bound, first_key_converter) else {
                return Ok(None);
            };
            let rows = converter.convert_columns(&[bound])?;
            Ok(Some(rows.row(0).as_ref().to_vec()))
        };
        let mut row_count = 0;
        let mut merged_row_groups = Vec::with_capacity(row_groups.len());
        for row_group in row_groups {
            row_count += row_group.row_count;
            let (lower, upper) = match row_group.first_key_bounds {
                Some(bounds) => (bounds.lower, bounds.upper),
                None => (None, None),
            };
            merged_row_groups.push(MergedRowGroup {
                rows_end: row_count,
                first_key_lower: encode_first_key_bound(lower)?,
                first_key_upper: encode_first_key_bound(upper)?,
            });
        }
        Ok(Self {
            row_groups: merged_row_groups,
            next_unmerged_row: 0,
            key_batches: BTreeMap::new(),
        })
    }

    /// The place, among the file's row groups, of the one that holds its
    /// next unmerged row.
    fn find_row_group_of_next_row(&self) -> usize {
        self.row_groups
            .partition_point(|row_group| row_group.rows_end <= self.next_unmerged_row)
    }

    /// The file row of the first key of the key batch that holds the key of
    /// the next unmerged row, if any of its columns was decoded.
    fn find_key_batch_of_next_row(&self) -> Option<u64> {
        let (&first_row, key_batch) = self
            .key_batches
            .range(..=self.next_unmerged_row)
            .next_back()?;
        (self.next_unmerged_row < first_row + key_batch.row_count).then_some(first_row)
    }

    /// The file's next unmerged rows whose keys are decoded in every sort
    /// column, up to file row `rows_end_limit`. The key of the next
    /// unmerged row must be decoded.
    fn collect_key_stretch(&self, file: usize, rows_end_limit: u64) -> WriteResult<KeyStretch> {
        let first_batch_row = self
            .find_key_batch_of_next_row()
            .expect("a stretch is collected after the next key was found decoded");
        let mut batches = Vec::new();
        let mut rows_end = self.next_unmerged_row;
        for (&first_row, key_batch) in self.key_batches.range(first_batch_row..rows_end_limit) {
            // The next key batch may not have arrived, or not in every
            // sort column.
            let Some(batch) = key_batch.build_batch()?.filter(|_| first_row <= rows_end) else {
                break;
            };
            batches.push(batch);
            rows_end = first_row + key_batch.row_count;
        }
        Ok(KeyStretch {
            file,
            batches,
            first_row_in_batches: (self.next_unmerged_row - first_batch_row) as usize,
            row_count: (rows_end.min(rows_end_limit) - self.next_unmerged_row) as usize,
        })
    }
}

/// What the merge takes its next rows from.
struct RowGroupsToMerge {
    /// The row group that starts first and the row groups that overlap it,
    /// ordered by where they start. For a table without sort columns, the
    /// one row group that holds the next row of the output.
    row_groups: Vec<RowGroupOfFile>,
    /// The files whose next unmerged row is in one of them, in file order.
    files: Vec<usize>,
}

/// Decides the merge order of the input files by their sort columns.
pub(super) struct KeyMerge {
    /// The order the files are sorted in: one key per sort column, in sort
    /// order, ascending with NULL first.
    order_by: Vec<OrderBy>,
    files: Vec<MergeInput>,
    /// Found again whenever a run takes a file's last unmerged row of a
    /// row group.
    to_merge: RowGroupsToMerge,
    /// The rows whose order is decided and that are not given out yet, as
    /// runs: the next so many unmerged rows of a file.
    ordered_runs: VecDeque<(usize, u64)>,
}

impl KeyMerge {
    /// A merge of input files sorted by columns of the types `key_types`,
    /// in sort order. `files` gives, for each file, what the statistics say
    /// about each of its row groups, in file order. The files are numbered
    /// in the order given. No key is decoded yet.
    pub(super) fn new(
        key_types: &[DataType],
        files: Vec<Vec<RowGroupKeyBounds>>,
    ) -> WriteResult<Self> {
        let first_key_converter = key_types
            .first()
            .map(|key_type| RowConverter::new(vec![SortField::new(key_type.clone())]))
            .transpose()?;
        let files = files
            .into_iter()
            .map(|row_groups| MergeInput::new(row_groups, first_key_converter.as_ref()))
            .collect::<WriteResult<_>>()?;
        let mut merge = Self {
            order_by: (0..key_types.len())
                .map(|sort_column| OrderBy::new(sort_column, false, true))
                .collect(),
            files,
            to_merge: RowGroupsToMerge {
                row_groups: Vec::new(),
                files: Vec::new(),
            },
            ordered_runs: VecDeque::new(),
        };
        merge.set_next_row_groups_to_merge();
        Ok(merge)
    }

    /// Add decoded keys of sort column `sort_column` of `file`: `keys[i]` is
    /// the key of file row `first_row + i`. Keys may be added in any order
    /// of their rows and of their sort columns, but the keys of the same
    /// rows must be added as arrays of the same rows in every sort column.
    /// Keys of rows that are all merged already are not kept. The others
    /// are held until their rows are merged.
    pub(super) fn add_decoded_keys(
        &mut self,
        file: usize,
        sort_column: usize,
        first_row: u64,
        keys: &ArrayRef,
    ) -> WriteResult<()> {
        let input = &mut self.files[file];
        let row_count = keys.len() as u64;
        if first_row + row_count <= input.next_unmerged_row {
            return Ok(());
        }
        let sort_column_count = self.order_by.len();
        let key_batch = input
            .key_batches
            .entry(first_row)
            .or_insert_with(|| KeyBatch {
                row_count,
                columns: vec![None; sort_column_count],
            });
        if key_batch.row_count != row_count {
            return Err(WriteError::CompactionRowCountMismatch);
        }
        key_batch.columns[sort_column] = Some(keys.clone());
        Ok(())
    }

    /// How many rows of `file` are merged. This is also the file row of its
    /// next unmerged row.
    pub(super) fn count_merged_rows(&self, file: usize) -> u64 {
        self.files[file].next_unmerged_row
    }

    /// The row groups the merge takes its next rows from, ordered by where
    /// they start: the row group that starts first and the row groups that
    /// overlap it (see the [module](self)). For a table without sort
    /// columns, the one row group that holds the next row of the output.
    ///
    /// The list changes only when a run takes a file's last unmerged row
    /// of a row group.
    pub(super) fn row_groups_to_merge(&self) -> &[RowGroupOfFile] {
        &self.to_merge.row_groups
    }

    /// Every row group of every file, in the order the merge takes its
    /// rows from them: by where they start. A row group whose start is
    /// not known may start anywhere, and comes first. Row groups that
    /// start at the same key, and all of them in a table without sort
    /// columns, keep the order of their files and of their place in
    /// the file.
    pub(super) fn list_row_groups_in_merge_order(&self) -> Vec<RowGroupOfFile> {
        let mut row_groups: Vec<RowGroupOfFile> = self
            .files
            .iter()
            .enumerate()
            .flat_map(|(file, input)| {
                (0..input.row_groups.len()).map(move |row_group| RowGroupOfFile { file, row_group })
            })
            .collect();
        row_groups
            .sort_by_key(|&row_group| self.look_up_row_group(row_group).first_key_lower.as_deref());
        row_groups
    }

    /// Set what the merge takes its next rows from, from where each
    /// file's next unmerged row is.
    fn set_next_row_groups_to_merge(&mut self) {
        let row_groups_of_next_rows = self
            .files
            .iter()
            .enumerate()
            .map(|(file, input)| RowGroupOfFile {
                file,
                row_group: input.find_row_group_of_next_row(),
            })
            // A file whose rows are all merged has no such row group.
            .filter(|row_group| row_group.row_group < self.files[row_group.file].row_groups.len());
        // Without sort columns the files are merged in turn.
        if self.order_by.is_empty() {
            let row_group_of_next_row: Vec<RowGroupOfFile> =
                row_groups_of_next_rows.take(1).collect();
            self.to_merge = RowGroupsToMerge {
                files: row_group_of_next_row
                    .iter()
                    .map(|row_group| row_group.file)
                    .collect(),
                row_groups: row_group_of_next_row,
            };
            return;
        }
        let row_groups_of_next_rows: Vec<RowGroupOfFile> = row_groups_of_next_rows.collect();
        let first_key_upper = row_groups_of_next_rows
            .iter()
            .map(|&row_group| self.look_up_row_group(row_group))
            .min_by_key(|row_group| row_group.first_key_lower.as_deref())
            .and_then(|starts_first| starts_first.first_key_upper.as_deref());
        let mut row_groups = Vec::new();
        let mut files = Vec::new();
        for RowGroupOfFile { file, row_group } in row_groups_of_next_rows {
            // An unknown bound overlaps everything.
            let overlapping_row_groups = (row_group..self.files[file].row_groups.len())
                .map(|row_group| RowGroupOfFile { file, row_group })
                .take_while(|&row_group| {
                    let lower = self.look_up_row_group(row_group).first_key_lower.as_deref();
                    match (lower, first_key_upper) {
                        (Some(lower), Some(first_key_upper)) => lower <= first_key_upper,
                        _ => true,
                    }
                });
            let listed_before = row_groups.len();
            row_groups.extend(overlapping_row_groups);
            if row_groups.len() > listed_before {
                files.push(file);
            }
        }
        row_groups
            .sort_by_key(|&row_group| self.look_up_row_group(row_group).first_key_lower.as_deref());
        self.to_merge = RowGroupsToMerge { row_groups, files };
    }

    /// What the merge tracks of `row_group`.
    fn look_up_row_group(&self, row_group: RowGroupOfFile) -> &MergedRowGroup {
        &self.files[row_group.file].row_groups[row_group.row_group]
    }

    /// Give the next run of the output, of at most `row_limit` rows, and
    /// record its rows as merged. A longer run is cut at `row_limit` rows,
    /// and the next call gives its remaining rows.
    pub(super) fn plan_next_run(&mut self, row_limit: u64) -> WriteResult<NextRun> {
        if self.ordered_runs.is_empty() {
            match self.to_merge.files[..] {
                [] => return Ok(NextRun::NoRowsLeft),
                // Nothing overlaps the file's row group, so its rows go out
                // as they are, to the end of the row group.
                [only_file] => {
                    let next_unmerged_row = self.files[only_file].next_unmerged_row;
                    self.ordered_runs.push_back((
                        only_file,
                        self.find_run_limit(only_file) - next_unmerged_row,
                    ));
                }
                _ => {
                    let needed_keys = self.list_undecoded_next_keys();
                    if !needed_keys.is_empty() {
                        return Ok(NextRun::NeedsKeys(needed_keys));
                    }
                    self.order_next_rows()?;
                }
            }
        }
        let (file, ordered_rows) = self
            .ordered_runs
            .front_mut()
            .expect("rows were just ordered");
        let file = *file;
        let rows = (*ordered_rows).min(row_limit);
        *ordered_rows -= rows;
        if *ordered_rows == 0 {
            self.ordered_runs.pop_front();
        }
        self.record_rows_merged(file, rows);
        Ok(NextRun::Found { file, rows })
    }

    /// Record that the next `rows` unmerged rows of `file` are merged. The
    /// keys of merged rows are dropped: the merge does not compare them
    /// again. If the rows include the last of their row group, the row
    /// groups the merge takes its next rows from are found again.
    fn record_rows_merged(&mut self, file: usize, rows: u64) {
        let input = &mut self.files[file];
        let row_group_rows_end = input.row_groups[input.find_row_group_of_next_row()].rows_end;
        input.next_unmerged_row += rows;
        let next_unmerged_row = input.next_unmerged_row;
        while input
            .key_batches
            .first_key_value()
            .is_some_and(|(&first_row, key_batch)| {
                first_row + key_batch.row_count <= next_unmerged_row
            })
        {
            input.key_batches.pop_first();
        }
        if next_unmerged_row >= row_group_rows_end {
            self.set_next_row_groups_to_merge();
        }
    }

    /// The file row a run of `file` must end before: the end of the row
    /// group that holds its next unmerged row. Past it, the row groups the
    /// merge takes its rows from may be others.
    fn find_run_limit(&self, file: usize) -> u64 {
        let input = &self.files[file];
        input.row_groups[input.find_row_group_of_next_row()].rows_end
    }

    /// The keys of the next unmerged rows of the files that take part
    /// which are not decoded.
    fn list_undecoded_next_keys(&self) -> Vec<NeededKey> {
        let mut needed_keys = Vec::new();
        for &file in &self.to_merge.files {
            let input = &self.files[file];
            let key_batch = input
                .find_key_batch_of_next_row()
                .map(|first_row| &input.key_batches[&first_row]);
            for sort_column in 0..self.order_by.len() {
                if key_batch.is_none_or(|key_batch| key_batch.columns[sort_column].is_none()) {
                    needed_keys.push(NeededKey {
                        file,
                        sort_column,
                        row: input.next_unmerged_row,
                    });
                }
            }
        }
        needed_keys
    }

    /// Decide the order of the next rows of the files that take part, of
    /// which there are several and whose next keys are all decoded, and
    /// keep it as runs to give out (see the [module](self) for how).
    fn order_next_rows(&mut self) -> WriteResult<()> {
        let stretches: Vec<KeyStretch> = self
            .to_merge
            .files
            .iter()
            .map(|&file| {
                let input = &self.files[file];
                let rows_end_limit = self
                    .find_run_limit(file)
                    .min(input.next_unmerged_row + STRETCH_ROWS);
                input.collect_key_stretch(file, rows_end_limit)
            })
            .collect::<WriteResult<_>>()?;

        // The stretch whose last key is the smallest. Nothing after that
        // key can be ordered yet.
        let mut limiting = 0;
        for candidate in 1..stretches.len() {
            let (candidate_stretch, limiting_stretch) =
                (&stretches[candidate], &stretches[limiting]);
            let mut comparer = RowComparer::try_new(
                &self.order_by,
                &candidate_stretch.batches,
                &limiting_stretch.batches,
            )?;
            let ordering = comparer.compare(
                candidate_stretch.locate_row(candidate_stretch.row_count - 1),
                limiting_stretch.locate_row(limiting_stretch.row_count - 1),
            );
            if ordering == Ordering::Less {
                limiting = candidate;
            }
        }

        // How many rows of each stretch are at or before that key.
        let limiting_stretch = &stretches[limiting];
        let last_key = limiting_stretch.locate_row(limiting_stretch.row_count - 1);
        let mut row_counts = Vec::with_capacity(stretches.len());
        for (place, stretch) in stretches.iter().enumerate() {
            if place == limiting {
                row_counts.push(stretch.row_count);
                continue;
            }
            let mut comparer =
                RowComparer::try_new(&self.order_by, &stretch.batches, &limiting_stretch.batches)?;
            let (mut rows_before_cut, mut rows_end) = (0, stretch.row_count);
            while rows_before_cut < rows_end {
                let middle = rows_before_cut + (rows_end - rows_before_cut) / 2;
                match comparer.compare(stretch.locate_row(middle), last_key) {
                    Ordering::Greater => rows_end = middle,
                    Ordering::Less | Ordering::Equal => rows_before_cut = middle + 1,
                }
            }
            row_counts.push(rows_before_cut);
        }

        // Merge the rows before the cut. Each stretch is in key order.
        let batches: Vec<&RecordBatch> = stretches
            .iter()
            .flat_map(|stretch| &stretch.batches)
            .collect();
        let mut file_of_batch = Vec::with_capacity(batches.len());
        let mut rows = Vec::with_capacity(row_counts.iter().sum());
        for (stretch, &row_count) in stretches.iter().zip(&row_counts) {
            let first_batch = file_of_batch.len();
            file_of_batch.extend(std::iter::repeat_n(stretch.file, stretch.batches.len()));
            rows.extend((0..row_count).map(|row| {
                let (place, row_in_batch) = stretch.locate_row(row);
                ((first_batch + place) as u32, row_in_batch as u32)
            }));
        }
        ChunkedRowSorter::try_new(&self.order_by, &batches)?
            .merge_sorted_runs(&mut rows, &row_counts);

        for (batch, _) in rows {
            let file = file_of_batch[batch as usize];
            match self.ordered_runs.back_mut() {
                Some((last_file, run_rows)) if *last_file == file => *run_rows += 1,
                _ => self.ordered_runs.push_back((file, 1)),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;

    /// One input file of the tests: its row groups, each the rows' keys, one
    /// `[first sort column, second sort column]` per row.
    type TestFile = Vec<Vec<[i32; 2]>>;

    fn one_value(value: i32) -> Option<ArrayRef> {
        Some(Arc::new(Int32Array::from(vec![value])))
    }

    /// A merge of `files` by two sort columns, with each row group's true
    /// bounds of the first as its statistics.
    fn build_merge(files: &[TestFile]) -> KeyMerge {
        let key_bounds_per_file = files
            .iter()
            .map(|file| {
                file.iter()
                    .map(|rows| RowGroupKeyBounds {
                        row_count: rows.len() as u64,
                        first_key_bounds: Some(SortKeyBounds {
                            lower: one_value(rows.iter().map(|row| row[0]).min().unwrap()),
                            upper: one_value(rows.iter().map(|row| row[0]).max().unwrap()),
                        }),
                    })
                    .collect()
            })
            .collect();
        KeyMerge::new(&[DataType::Int32, DataType::Int32], key_bounds_per_file).unwrap()
    }

    /// What planning every run of a merge produced.
    struct MergeOutcome {
        /// The runs, as `(file, rows)`.
        runs: Vec<(usize, u64)>,
        /// The `(file, sort column)` of every needed key that was decoded.
        decoded_keys: Vec<(usize, usize)>,
    }

    /// Plan every run of `merge`. Whenever the merge needs a key, give it the
    /// keys of the whole row group the needed row is in.
    fn plan_all_runs(mut merge: KeyMerge, files: &[TestFile]) -> MergeOutcome {
        let mut runs = Vec::new();
        let mut decoded_keys = Vec::new();
        loop {
            match merge.plan_next_run(u64::MAX).unwrap() {
                NextRun::NoRowsLeft => return MergeOutcome { runs, decoded_keys },
                NextRun::Found { file, rows } => runs.push((file, rows)),
                NextRun::NeedsKeys(needed_keys) => {
                    for needed_key in needed_keys {
                        let mut first_row = 0;
                        let row_group = files[needed_key.file]
                            .iter()
                            .find(|rows| {
                                first_row += rows.len() as u64;
                                needed_key.row < first_row
                            })
                            .unwrap();
                        let keys: ArrayRef = Arc::new(Int32Array::from_iter_values(
                            row_group.iter().map(|row| row[needed_key.sort_column]),
                        ));
                        merge
                            .add_decoded_keys(
                                needed_key.file,
                                needed_key.sort_column,
                                first_row - row_group.len() as u64,
                                &keys,
                            )
                            .unwrap();
                        decoded_keys.push((needed_key.file, needed_key.sort_column));
                    }
                }
            }
        }
    }

    /// The rows of `files` in the order `runs` gives them.
    fn rows_in_run_order(files: &[TestFile], runs: &[(usize, u64)]) -> Vec<[i32; 2]> {
        let rows_per_file: Vec<Vec<[i32; 2]>> = files
            .iter()
            .map(|file| file.iter().flatten().copied().collect())
            .collect();
        let mut next_row = vec![0; files.len()];
        let mut merged = Vec::new();
        for &(file, rows) in runs {
            let first = next_row[file];
            merged.extend_from_slice(&rows_per_file[file][first..first + rows as usize]);
            next_row[file] += rows as usize;
        }
        merged
    }

    #[test]
    fn row_groups_are_listed_in_the_order_they_start() {
        let files = vec![
            vec![vec![[10, 0], [20, 0]], vec![[50, 0], [60, 0]]],
            vec![vec![[30, 0], [40, 0]], vec![[45, 0], [70, 0]]],
        ];

        let row_groups = build_merge(&files).list_row_groups_in_merge_order();

        let listed: Vec<(usize, usize)> = row_groups
            .iter()
            .map(|row_group| (row_group.file, row_group.row_group))
            .collect();
        assert_eq!(listed, vec![(0, 0), (1, 0), (1, 1), (0, 1)]);
    }

    #[test]
    fn files_whose_row_groups_do_not_overlap_are_merged_without_a_key() {
        let files = vec![
            vec![vec![[5, 0], [6, 0], [7, 0]]],
            vec![vec![[1, 0], [2, 0]]],
        ];

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(build_merge(&files), &files);

        assert_eq!(runs, vec![(1, 2), (0, 3)]);
        assert_eq!(decoded_keys, vec![]);
    }

    #[test]
    fn files_whose_row_groups_overlap_are_merged_by_their_keys() {
        let files = vec![
            vec![vec![[1, 0], [4, 0], [9, 0]]],
            vec![vec![[2, 0], [3, 0], [8, 0]]],
        ];

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(build_merge(&files), &files);

        assert_eq!(runs, vec![(0, 1), (1, 2), (0, 1), (1, 1), (0, 1)]);
        assert_eq!(decoded_keys, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn rows_with_the_same_first_key_are_ordered_by_the_second() {
        let files = vec![
            vec![vec![[1, 9], [2, 7]]],
            vec![vec![[2, 4], [3, 8]]],
            vec![vec![[5, 0], [6, 0]]],
        ];

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(build_merge(&files), &files);

        assert_eq!(
            rows_in_run_order(&files, &runs),
            vec![[1, 9], [2, 4], [2, 7], [3, 8], [5, 0], [6, 0]]
        );
        assert_eq!(decoded_keys, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn rows_equal_in_every_sort_column_all_go_out() {
        let files = vec![
            vec![vec![[1, 1], [1, 1]]],
            vec![vec![[1, 1], [1, 1], [1, 1]]],
        ];

        let MergeOutcome { runs, .. } = plan_all_runs(build_merge(&files), &files);

        let count_rows_of = |file| -> u64 {
            let runs_of_file = runs.iter().filter(|run| run.0 == file);
            runs_of_file.map(|run| run.1).sum()
        };
        assert_eq!([count_rows_of(0), count_rows_of(1)], [2, 3]);
    }

    /// Three files whose keys interleave row by row, each longer than the
    /// rows of one file that are ordered in one go.
    #[test]
    fn files_longer_than_one_stretch_are_merged_in_key_order() {
        let row_count = STRETCH_ROWS as i32 * 2 + 100;
        let files: Vec<TestFile> = (0..3)
            .map(|file| vec![(0..row_count).map(|row| [row * 3 + file, 0]).collect()])
            .collect();

        let MergeOutcome { runs, .. } = plan_all_runs(build_merge(&files), &files);

        let all_rows_in_key_order: Vec<[i32; 2]> = (0..row_count * 3).map(|key| [key, 0]).collect();
        assert_eq!(rows_in_run_order(&files, &runs), all_rows_in_key_order);
    }

    /// File 0's second row group starts after file 1's row group ends, so
    /// each of the three row groups goes out whole.
    #[test]
    fn a_run_of_a_row_group_that_nothing_overlaps_ends_with_the_row_group() {
        let files = vec![
            vec![vec![[1, 0], [2, 0]], vec![[8, 0], [9, 0]]],
            vec![vec![[4, 0], [5, 0]]],
        ];

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(build_merge(&files), &files);

        assert_eq!(runs, vec![(0, 2), (1, 2), (0, 2)]);
        assert_eq!(decoded_keys, vec![]);
    }

    /// File 2's row group starts after the one that starts first ends, so
    /// it takes no part until that one is merged. Its rows still go out
    /// before the larger keys of file 1.
    #[test]
    fn a_row_group_that_starts_later_is_merged_once_the_first_is_done() {
        let files = vec![
            vec![vec![[1, 0], [5, 0]]],
            vec![vec![[3, 0], [7, 0]]],
            vec![vec![[6, 0], [6, 0]]],
        ];

        let MergeOutcome { runs, .. } = plan_all_runs(build_merge(&files), &files);

        assert_eq!(
            rows_in_run_order(&files, &runs),
            vec![[1, 0], [3, 0], [5, 0], [6, 0], [6, 0], [7, 0]]
        );
    }

    /// The row group that starts first is file 0's, with keys 1 to 5. File
    /// 1's first row group starts inside it. File 1's second row group and
    /// file 2's row group start after it ends.
    #[test]
    fn row_groups_that_start_before_the_first_ends_overlap_it() {
        let files = vec![
            vec![vec![[1, 0], [5, 0]]],
            vec![vec![[3, 0], [7, 0]], vec![[8, 0], [9, 0]]],
            vec![vec![[6, 0], [6, 0]]],
        ];
        let merge = build_merge(&files);

        let overlapping = merge.row_groups_to_merge();

        let expected = [(0, 0), (1, 0)].map(|(file, row_group)| RowGroupOfFile { file, row_group });
        assert_eq!(overlapping, expected);
    }

    /// Once file 0's row group is merged, file 1's first row group starts
    /// first, with keys 4 to 5, and file 2's row group overlaps it.
    #[test]
    fn the_first_row_group_changes_as_rows_are_merged() {
        let files = vec![
            vec![vec![[1, 0], [2, 0]]],
            vec![vec![[4, 0], [5, 0]], vec![[8, 0], [9, 0]]],
            vec![vec![[5, 0], [6, 0]]],
        ];
        let mut merge = build_merge(&files);
        let overlapping_before_any_run = merge.row_groups_to_merge().to_vec();

        let first_run = merge.plan_next_run(u64::MAX).unwrap();
        let overlapping_after_first_run = merge.row_groups_to_merge();

        let row_group_of = |(file, row_group)| RowGroupOfFile { file, row_group };
        assert_eq!(overlapping_before_any_run, [(0, 0)].map(row_group_of));
        assert_eq!(first_run, NextRun::Found { file: 0, rows: 2 });
        assert_eq!(
            overlapping_after_first_run,
            [(1, 0), (2, 0)].map(row_group_of)
        );
    }

    #[test]
    fn keys_added_out_of_row_order_are_merged_in_row_order() {
        let files = vec![
            vec![vec![[1, 0], [3, 0], [5, 0], [7, 0]]],
            vec![vec![[2, 0], [4, 0], [6, 0], [8, 0]]],
        ];
        let mut merge = build_merge(&files);
        for (file, first_row) in [(0, 2), (0, 0), (1, 2), (1, 0)] {
            let two_rows = &files[file][0][first_row..first_row + 2];
            for sort_column in [0, 1] {
                let keys: ArrayRef = Arc::new(Int32Array::from_iter_values(
                    two_rows.iter().map(|row| row[sort_column]),
                ));
                merge
                    .add_decoded_keys(file, sort_column, first_row as u64, &keys)
                    .unwrap();
            }
        }

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(merge, &files);

        let all_rows_in_key_order: Vec<[i32; 2]> = (1..=8).map(|key| [key, 0]).collect();
        assert_eq!(rows_in_run_order(&files, &runs), all_rows_in_key_order);
        assert_eq!(decoded_keys, vec![]);
    }

    #[test]
    fn a_merge_without_sort_columns_takes_the_files_in_turn() {
        let one_row_group_of = |row_count| {
            vec![RowGroupKeyBounds {
                row_count,
                first_key_bounds: None,
            }]
        };
        let mut merge = KeyMerge::new(&[], vec![one_row_group_of(3), one_row_group_of(2)]).unwrap();

        let first = merge.plan_next_run(u64::MAX).unwrap();
        let second = merge.plan_next_run(u64::MAX).unwrap();

        assert_eq!(first, NextRun::Found { file: 0, rows: 3 });
        assert_eq!(second, NextRun::Found { file: 1, rows: 2 });
        assert_eq!(merge.plan_next_run(u64::MAX).unwrap(), NextRun::NoRowsLeft);
    }

    /// A row group whose bounds are not known overlaps every other, so its
    /// keys are compared.
    #[test]
    fn a_row_group_without_bounds_is_merged_by_its_keys() {
        let files = vec![vec![vec![[1, 0], [2, 0]]], vec![vec![[5, 0], [6, 0]]]];
        let one_row_group_without_bounds = || {
            vec![RowGroupKeyBounds {
                row_count: 2,
                first_key_bounds: Some(SortKeyBounds {
                    lower: None,
                    upper: None,
                }),
            }]
        };
        let merge = KeyMerge::new(
            &[DataType::Int32, DataType::Int32],
            vec![
                one_row_group_without_bounds(),
                one_row_group_without_bounds(),
            ],
        )
        .unwrap();

        let MergeOutcome { runs, decoded_keys } = plan_all_runs(merge, &files);

        assert_eq!(runs, vec![(0, 2), (1, 2)]);
        assert_eq!(decoded_keys, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }
}
