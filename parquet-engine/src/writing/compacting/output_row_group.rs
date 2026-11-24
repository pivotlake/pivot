//! The row order of one output row group, as the merge decided it.

use std::ops::Range;

use super::output::FileRun;

/// Consecutive rows of the output: at most
/// [`pull_rows`](super::MergeSortSettings::pull_rows) of them. The rows of
/// an output row group are sent on one merged batch at a time.
pub(super) struct MergedBatch {
    /// The runs the batch consists of, in output order.
    pub(super) runs: Vec<FileRun>,
    /// For each input file, the file rows the batch takes from it. Empty
    /// for a file the batch takes no row from.
    pub(super) rows_per_file: Vec<Range<u64>>,
}

impl MergedBatch {
    /// How many rows the batch holds: the rows it takes from each file.
    fn count_rows(&self) -> usize {
        let rows_per_file = self.rows_per_file.iter();
        rows_per_file
            .map(|rows| (rows.end - rows.start) as usize)
            .sum()
    }

    /// How many bytes the batch's row order takes.
    fn count_held_bytes(&self) -> usize {
        self.runs.capacity() * size_of::<FileRun>()
            + self.rows_per_file.capacity() * size_of::<Range<u64>>()
    }
}

/// One merged batch of one column of an output row group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ColumnBatch {
    pub(super) column: usize,
    /// The merged batch's place among the row group's merged batches.
    pub(super) batch: usize,
}

/// The rows of one output row group, in output order, and how far each
/// column of them has got.
pub(super) struct OutputRowGroup {
    /// How many rows the row group holds when it is full.
    pub(super) row_limit: usize,
    /// How many rows a merged batch holds when it is full.
    merged_batch_rows: usize,
    /// How many input files the merge is of.
    file_count: usize,
    /// The place of the row group's first merged batch among the merged
    /// batches of the whole output, which are numbered from 0 in output
    /// order.
    pub(super) first_sequence: usize,
    pub(super) merged_batches: Vec<MergedBatch>,
    /// The first merged batch of a column whose rows' decoding was not
    /// started yet. The columns are started in order, and each column's
    /// merged batches in order, so the decoding of everything before this
    /// was started.
    next_to_decode: ColumnBatch,
    /// For each column, how many of the merged batches, from the first,
    /// were sent on.
    sent_batches: Vec<usize>,
    /// Every column before this one was sent on in every merged batch.
    first_unsent_column: usize,
}

impl OutputRowGroup {
    /// The first output row group, with no rows yet, of a merge of
    /// `file_count` files of a table of `column_count` columns. It is full
    /// at `row_limit` rows, and its merged batches at `merged_batch_rows`.
    pub(super) fn new(
        column_count: usize,
        file_count: usize,
        row_limit: usize,
        merged_batch_rows: usize,
    ) -> Self {
        Self {
            row_limit,
            merged_batch_rows,
            file_count,
            first_sequence: 0,
            merged_batches: Vec::new(),
            next_to_decode: ColumnBatch {
                column: 0,
                batch: 0,
            },
            sent_batches: vec![0; column_count],
            first_unsent_column: 0,
        }
    }

    /// The output row group after this one, with no rows yet. Its merged
    /// batches are numbered on from this one's.
    pub(super) fn start_next(&self) -> Self {
        Self {
            first_sequence: self.first_sequence + self.merged_batches.len(),
            ..Self::new(
                self.sent_batches.len(),
                self.file_count,
                self.row_limit,
                self.merged_batch_rows,
            )
        }
    }

    /// How many rows the row group holds. Every merged batch but the last
    /// is full.
    fn count_rows(&self) -> usize {
        match self.merged_batches.split_last() {
            Some((last_batch, full_batches)) => {
                full_batches.len() * self.merged_batch_rows + last_batch.count_rows()
            }
            None => 0,
        }
    }

    /// Whether the row group holds as many rows as it can.
    pub(super) fn is_full(&self) -> bool {
        self.count_rows() == self.row_limit
    }

    /// How many rows the next run may hold: the room left in the last
    /// merged batch, or in a new one if that is full, and no more than the
    /// room left in the row group.
    pub(super) fn count_room_for_next_run(&self) -> usize {
        let rows_in_last_batch = self.merged_batches.last().map(MergedBatch::count_rows);
        let room_in_batch = match rows_in_last_batch {
            Some(rows) if rows < self.merged_batch_rows => self.merged_batch_rows - rows,
            _ => self.merged_batch_rows,
        };
        room_in_batch.min(self.row_limit - self.count_rows())
    }

    /// Record that the row group's next `rows` rows are the rows of `file`
    /// from file row `first_row` on. They go to the last merged batch, or
    /// to a new one if the last is full.
    pub(super) fn append_run(&mut self, file: usize, first_row: u64, rows: u32) {
        let is_last_batch_full = self
            .merged_batches
            .last()
            .is_none_or(|batch| batch.count_rows() == self.merged_batch_rows);
        if is_last_batch_full {
            self.merged_batches.push(MergedBatch {
                runs: Vec::new(),
                rows_per_file: vec![0..0; self.file_count],
            });
        }
        let batch = self.merged_batches.last_mut().unwrap();
        let rows_of_file = &mut batch.rows_per_file[file];
        if rows_of_file.is_empty() {
            *rows_of_file = first_row..first_row;
        }
        rows_of_file.end += rows as u64;
        match batch.runs.last_mut() {
            Some(last_run) if last_run.file == file => last_run.rows += rows,
            _ => batch.runs.push(FileRun { file, rows }),
        }
    }

    /// How many bytes the row group's row order takes.
    pub(super) fn count_held_bytes(&self) -> usize {
        self.merged_batches
            .iter()
            .map(MergedBatch::count_held_bytes)
            .sum()
    }

    /// The first merged batch of a column whose rows' decoding was not
    /// started yet, or `None` when the decoding of every merged batch of
    /// every column was started.
    pub(super) fn find_next_to_decode(&self) -> Option<ColumnBatch> {
        (self.next_to_decode.column < self.sent_batches.len()).then_some(self.next_to_decode)
    }

    /// Record that the decoding of the rows of the merged batch that
    /// [`find_next_to_decode`](Self::find_next_to_decode) gave was started.
    pub(super) fn record_decoding_started(&mut self) {
        self.next_to_decode.batch += 1;
        if self.next_to_decode.batch == self.merged_batches.len() {
            self.next_to_decode = ColumnBatch {
                column: self.next_to_decode.column + 1,
                batch: 0,
            };
        }
    }

    /// The next merged batch to send on in `column`, with its place among
    /// the row group's merged batches, or `None` when the column was sent
    /// on in every merged batch.
    pub(super) fn find_next_to_send(&self, column: usize) -> Option<(usize, &MergedBatch)> {
        let batch = self.sent_batches[column];
        self.merged_batches
            .get(batch)
            .map(|merged_batch| (batch, merged_batch))
    }

    /// Record that the merged batch that
    /// [`find_next_to_send`](Self::find_next_to_send) gave for `column` was
    /// sent on.
    pub(super) fn record_sent(&mut self, column: usize) {
        self.sent_batches[column] += 1;
        while self.first_unsent_column < self.sent_batches.len()
            && self.sent_batches[self.first_unsent_column] == self.merged_batches.len()
        {
            self.first_unsent_column += 1;
        }
    }

    /// The first merged batch, of the first column, that is still to send
    /// on, or `None` when every column was sent on in every merged batch.
    pub(super) fn find_first_unsent(&self) -> Option<ColumnBatch> {
        let column = self.first_unsent_column;
        (column < self.sent_batches.len()).then(|| ColumnBatch {
            column,
            batch: self.sent_batches[column],
        })
    }
}
