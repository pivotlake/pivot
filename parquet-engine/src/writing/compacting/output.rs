//! What a compaction's read in merge order produces.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, UInt32Array};

use super::memory_budget::MemoryReservation;

/// The decoded values of one column in consecutive rows of one input row
/// group.
pub struct InputValues {
    pub values: ArrayRef,
    /// The index of the input row group in the table's list of row groups.
    pub row_group: u32,
    /// The row number, in that row group, of the first value.
    pub first_row: u32,
    /// What the compaction's memory budget counts for the array the values
    /// were decoded as. Every `InputValues` that is a slice of that array
    /// shares it, and the bytes are given back when the last one is
    /// dropped.
    pub memory: Arc<MemoryReservation>,
}

/// Which row of which source batch each row of a merged batch is.
pub enum MergedRows {
    /// Row `i` of the merged batch is row `rows[i]` of source batch
    /// `sources[i]`.
    Listed {
        sources: UInt32Array,
        rows: UInt32Array,
    },
    /// The merged batch is consecutive rows of one source batch.
    Consecutive {
        source: u32,
        first_row: u32,
        row_count: usize,
    },
}

/// One input file's part of a merged batch, as the merge plans it: the next
/// `rows` rows of the merged batch are the next `rows` rows of input file
/// `file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FileRun {
    pub(super) file: usize,
    pub(super) rows: u32,
}

impl MergedRows {
    /// The rows of a merged batch that consists of `runs`, as rows of
    /// `sources`.
    ///
    /// `sources` holds each file's rows of the merged batch in file order,
    /// in one or more batches, and `first_source_per_file[file]` is the
    /// index of the first source batch of `file`. A run takes its file's
    /// next rows, continuing into the file's next source batch where one
    /// ends.
    pub(super) fn from_file_runs(
        runs: &[FileRun],
        sources: &[InputValues],
        first_source_per_file: &[usize],
    ) -> Self {
        if let [source] = sources {
            return MergedRows::Consecutive {
                source: 0,
                first_row: 0,
                row_count: source.values.len(),
            };
        }
        let row_count = runs.iter().map(|run| run.rows as usize).sum();
        let mut source_of_each_row = Vec::with_capacity(row_count);
        let mut row_in_source = Vec::with_capacity(row_count);
        // For each file, the source batch its next row is in, and that
        // row's number in the batch.
        let mut next_row_per_file: Vec<(usize, u32)> = first_source_per_file
            .iter()
            .map(|&first_source| (first_source, 0))
            .collect();
        for run in runs {
            let (source, row) = &mut next_row_per_file[run.file];
            let mut rows_left_in_run = run.rows;
            while rows_left_in_run > 0 {
                let rows_left_in_source = sources[*source].values.len() as u32 - *row;
                if rows_left_in_source == 0 {
                    *source += 1;
                    *row = 0;
                    continue;
                }
                let taken_rows = rows_left_in_run.min(rows_left_in_source);
                source_of_each_row.extend(std::iter::repeat_n(*source as u32, taken_rows as usize));
                row_in_source.extend(*row..*row + taken_rows);
                *row += taken_rows;
                rows_left_in_run -= taken_rows;
            }
        }
        MergedRows::Listed {
            sources: UInt32Array::from(source_of_each_row),
            rows: UInt32Array::from(row_in_source),
        }
    }
}

/// One column of one merged batch (the next rows of the output): the
/// batches those rows were decoded as, and which row of which batch each
/// row of the merged batch is.
///
/// A merged batch is sent as one of these per column of the table, all with
/// the same `sequence`.
pub struct MergedBatchColumn {
    /// The merged batch's place among the merged batches, which are
    /// numbered from 0 in output order.
    pub sequence: usize,
    /// The table column the source batches hold.
    pub column: usize,
    /// The batches the merged batch's rows are in.
    pub sources: Arc<[InputValues]>,
    pub rows: MergedRows,
}
