//! What a compaction's read in merge order produces.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};

use crate::writing::compacting::memory_budget::MemoryReservation;

/// The decoded values of one column in consecutive rows of one input row
/// group.
pub(crate) struct InputValues {
    pub(crate) values: ArrayRef,
    /// What the compaction's memory budget counts for the array the values
    /// were decoded as. Every `InputValues` that is a slice of that array
    /// shares it, and the bytes are given back when the last one is
    /// dropped.
    pub(crate) memory: Arc<MemoryReservation>,
}

/// One input file's part of a merged batch, as the merge plans it: the next
/// `rows` rows of the merged batch are the next `rows` rows of input file
/// `file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileRun {
    pub(crate) file: usize,
    pub(crate) rows: u32,
}

/// One column of one merged batch (the next rows of the output): the values
/// those rows were decoded as, and the runs that say in which order they go
/// out.
///
/// A merged batch is sent as one of these per column of the table.
pub(crate) struct MergedBatchColumn {
    /// The output row group the merged batch is in. The output row groups
    /// are numbered from 0 in output order.
    pub(crate) row_group: usize,
    /// The merged batch's place among the merged batches of its row group.
    pub(crate) batch: usize,
    /// How many merged batches that row group consists of.
    pub(crate) row_group_batch_count: usize,
    /// The table column the sources hold.
    pub(crate) column: usize,
    /// The values of the merged batch's rows: each input file's rows of the
    /// merged batch in file order, one file after the other.
    pub(crate) sources: Vec<InputValues>,
    /// For each input file, the place in `sources` of the first values of
    /// that file.
    pub(crate) first_source_per_file: Vec<usize>,
    /// The runs the merged batch consists of, in output order. Every column
    /// of the merged batch shares them.
    pub(crate) runs: Arc<Vec<FileRun>>,
}

impl MergedBatchColumn {
    /// Which row of which source each row of the merged batch is, in output
    /// order: the source's place in `sources`, and the row's number in that
    /// source.
    ///
    /// A run takes its file's next rows, continuing into the file's next
    /// source where one ends.
    pub(crate) fn list_rows(&self) -> Vec<(u32, u32)> {
        let row_count = self.runs.iter().map(|run| run.rows as usize).sum();
        let mut rows = Vec::with_capacity(row_count);
        // For each file, the source its next row is in, and that row's
        // number in the source.
        let mut next_row_per_file: Vec<(usize, u32)> = self
            .first_source_per_file
            .iter()
            .map(|&first_source| (first_source, 0))
            .collect();
        for run in self.runs.iter() {
            let (source, row) = &mut next_row_per_file[run.file];
            let mut rows_left_in_run = run.rows;
            while rows_left_in_run > 0 {
                let rows_left_in_source = self.sources[*source].values.len() as u32 - *row;
                if rows_left_in_source == 0 {
                    *source += 1;
                    *row = 0;
                    continue;
                }
                let taken_rows = rows_left_in_run.min(rows_left_in_source);
                let source_number = *source as u32;
                rows.extend((*row..*row + taken_rows).map(|row| (source_number, row)));
                *row += taken_rows;
                rows_left_in_run -= taken_rows;
            }
        }
        rows
    }
}
