//! Decides which rows of a compaction's input files are fetched and decoded
//! next, and sends them on in merge order. It lives in the
//! [`MergeSortCompacter`](super::MergeSortCompacter) of the compaction
//! worker.
//!
//! The read orders the rows of one output row group at a time, from the
//! sort columns alone, and then gets the other columns of those rows, each
//! column on its own. Each time the compaction worker runs it, it does as
//! much of each of these steps as it can without waiting:
//!
//! 1. [`send_decoded_columns`](CompactionRead::send_decoded_columns) sends
//!    on the rows that have arrived decoded, which gives their memory back.
//! 2. [`fetch_sort_columns_of_overlapping_row_groups`](CompactionRead::fetch_sort_columns_of_overlapping_row_groups)
//!    requests the sort columns of the input row group that starts first
//!    in key order and of every row group that overlaps it.
//! 3. [`decide_row_order_of_output_row_groups`](CompactionRead::decide_row_order_of_output_row_groups)
//!    compares the decoded keys and decides the order of the next rows of
//!    the output. If the merge is missing a key, it lets the workers decode
//!    it. It goes on ahead of the rows being sent while memory has room.
//! 4. [`start_decoding_other_columns`](CompactionRead::start_decoding_other_columns)
//!    lets the workers decode, a column at a time, the rows whose order is
//!    decided.
//! 5. [`start_decoding_sort_keys_ahead`](CompactionRead::start_decoding_sort_keys_ahead)
//!    lets the workers decode sort keys further ahead than the merge has
//!    asked for, while memory has room.
//! 6. [`fetch_other_columns_ahead`](CompactionRead::fetch_other_columns_ahead)
//!    requests column chunks before their rows are to be decoded, in the
//!    order the merge will take rows from their row groups, while there
//!    is room to fetch ahead.
//!
//! The read holds to a [`MemoryBudget`], which counts the bytes actually
//! held. How many columns are decoded at once is not decided up front: a
//! column's rows are started when the budget has room for them. What the
//! output cannot go on without is done whatever the budget says, since
//! nothing else would give memory back: the keys the merge is waiting for
//! when no ordered rows are left to send, and the next merged batch of the
//! first column still to send.
//!
//! Nothing but memory limits how far ahead the read works. Rows are
//! ordered and sort keys decoded ahead while what is held is within half
//! the budget, which leaves the other half to the columns of the rows whose
//! order is decided. Chunks are fetched ahead within a quarter of the
//! budget.

use std::collections::VecDeque;

use arrow_schema::DataType;
use dispatch::{Sender, UnaryResult};

use super::decoded_values::DecodedValues;
use super::input_file::InputFile;
use super::key_merge::{KeyMerge, NextRun, RowGroupOfFile};
use super::output::{MergedBatchColumn, MergedRows};
use super::output_row_group::OutputRowGroup;
use super::read_requests::ReadRequests;
use super::settings::MergeSortSettings;
use crate::ParquetTable;
use crate::writing::error::WriteResult;

/// A compaction's read of its input files in merge order.
pub(super) struct CompactionRead {
    /// The table columns the files are sorted by, in sort order.
    sort_columns: Vec<usize>,
    /// The input files, in the table's order, with what was read of each.
    files: Vec<InputFile>,
    /// What the files send their requests and releases through, and count
    /// their memory in.
    requests: ReadRequests,
    merge: KeyMerge,
    /// Every row group of every file, in the order the merge takes its
    /// rows from them.
    row_groups_in_merge_order: Vec<RowGroupOfFile>,
    /// Every column of every row group before this place in that order
    /// was requested.
    next_row_group_to_fetch_ahead: usize,
    /// The output row group whose row order is being decided.
    row_group_being_ordered: OutputRowGroup,
    /// The output row groups whose row order is decided and that are not
    /// sent on in every column yet, oldest first.
    ordered_row_groups: VecDeque<OutputRowGroup>,
    /// For each column, whether rows of it arrived decoded, or an output
    /// row group was ordered, since its merged batches were last checked
    /// for sending.
    has_rows_to_check: Vec<bool>,
}

impl CompactionRead {
    /// Set up the read of the files of `table`, each sorted by the table
    /// columns `sort_columns`, into output row groups of
    /// `output_row_group_rows` rows. Nothing is requested yet. The files
    /// ask through `requests`, and the read holds to its memory budget.
    pub(super) fn new(
        table: &ParquetTable,
        sort_columns: Vec<usize>,
        settings: MergeSortSettings,
        output_row_group_rows: usize,
        requests: ReadRequests,
    ) -> WriteResult<Self> {
        let files = InputFile::list_files_of(table)?;
        let key_types: Vec<DataType> = sort_columns
            .iter()
            .map(|&column| table.schema().field(column).data_type().clone())
            .collect();
        let key_bounds_per_file = files
            .iter()
            .map(|file| file.read_key_bounds(table, &sort_columns, &key_types))
            .collect::<WriteResult<_>>()?;
        let column_count = table.schema().fields().len();
        let merge = KeyMerge::new(&key_types, key_bounds_per_file)?;
        Ok(Self {
            row_groups_in_merge_order: merge.list_row_groups_in_merge_order(),
            next_row_group_to_fetch_ahead: 0,
            merge,
            row_group_being_ordered: OutputRowGroup::new(
                column_count,
                files.len(),
                output_row_group_rows,
                settings.pull_rows,
            ),
            ordered_row_groups: VecDeque::new(),
            has_rows_to_check: vec![false; column_count],
            files,
            requests,
            sort_columns,
        })
    }

    /// Whether every row of every input file was sent on: the merge has no
    /// row group left to merge, and no ordered output row group is left to
    /// send. The last rows the merge orders are moved to the ordered output
    /// row groups at once, so none is left in the one being ordered.
    pub(super) fn has_sent_every_row(&self) -> bool {
        self.merge.row_groups_to_merge().is_empty() && self.ordered_row_groups.is_empty()
    }

    /// Do as much of each step of the read as can be done without waiting
    /// (see the [module](self)). Returns whether anything was done.
    ///
    /// The steps run in this order so that memory is given back first, and
    /// so that the columns of rows whose order is decided take the room in
    /// the budget before sort keys and chunks further ahead do.
    pub(super) fn run_ready_steps(
        &mut self,
        sender: &mut dyn Sender<MergedBatchColumn>,
    ) -> UnaryResult<bool> {
        let mut has_progressed = self.send_decoded_columns(sender)?;
        has_progressed |= self.fetch_sort_columns_of_overlapping_row_groups();
        has_progressed |= self.decide_row_order_of_output_row_groups()?;
        has_progressed |= self.start_decoding_other_columns();
        has_progressed |= self.start_decoding_sort_keys_ahead();
        self.requests.wake_gates_after_releases();
        has_progressed |= self.fetch_other_columns_ahead();
        if self.has_sent_every_row() {
            self.requests.memory_budget.record_read_complete();
            self.requests.close();
        }
        Ok(has_progressed)
    }

    /// Keep the decoded values that came back from the pipeline, with the
    /// file they are of. Values of a sort column are also given to the
    /// merge as keys.
    pub(super) fn store_decoded_values(&mut self, decoded: DecodedValues) -> UnaryResult<()> {
        let row_group = self.find_row_group_at(decoded.row_group);
        let file_row_group = &mut self.files[row_group.file].row_groups[row_group.row_group];
        let sort_column = self
            .sort_columns
            .iter()
            .position(|&sort_column| sort_column == decoded.column);
        if let Some(sort_column) = sort_column {
            self.merge.add_decoded_keys(
                row_group.file,
                sort_column,
                file_row_group.rows.start + decoded.first_row as u64,
                &decoded.values,
            )?;
        }
        self.has_rows_to_check[decoded.column] = true;
        file_row_group.store_decoded_values(decoded, &mut self.requests)?;
        Ok(())
    }

    /// Which row group of which file the row group at `table_row_group` in
    /// the table's list of row groups is.
    fn find_row_group_at(&self, table_row_group: usize) -> RowGroupOfFile {
        let file = self
            .files
            .partition_point(|file| file.row_groups[0].table_index <= table_row_group)
            - 1;
        let row_group = self.files[file]
            .row_groups
            .partition_point(|row_group| row_group.table_index < table_row_group);
        RowGroupOfFile { file, row_group }
    }

    /// Send on every merged batch whose rows of a column have all arrived
    /// decoded, as a [`MergedBatchColumn`], and drop the output row groups
    /// that were sent on in every column. Returns whether anything was
    /// sent on or dropped.
    ///
    /// The merged batches of a column are sent in output order, so that
    /// each input file's rows are taken in file order.
    fn send_decoded_columns(
        &mut self,
        sender: &mut dyn Sender<MergedBatchColumn>,
    ) -> UnaryResult<bool> {
        let mut has_sent = false;
        for column in 0..self.has_rows_to_check.len() {
            if !std::mem::take(&mut self.has_rows_to_check[column]) {
                continue;
            }
            for position in 0..self.ordered_row_groups.len() {
                while self.is_next_merged_batch_decoded(position, column) {
                    self.send_next_merged_batch(position, column, sender)?;
                    has_sent = true;
                }
                let row_group = &self.ordered_row_groups[position];
                if row_group.find_next_to_send(column).is_some() {
                    break;
                }
            }
        }
        Ok(self.drop_sent_row_groups() || has_sent)
    }

    /// Whether the ordered output row group at `position` has a merged
    /// batch still to send in `column`, and the rows of the next such batch
    /// have all arrived decoded.
    fn is_next_merged_batch_decoded(&self, position: usize, column: usize) -> bool {
        let Some((_, batch)) = self.ordered_row_groups[position].find_next_to_send(column) else {
            return false;
        };
        let mut rows_per_file = batch.rows_per_file.iter().enumerate();
        rows_per_file
            .all(|(file, rows)| rows.is_empty() || self.files[file].is_decoded(column, rows))
    }

    /// Send on the next merged batch of the ordered output row group at
    /// `position` in `column`: the values its rows were decoded as, with
    /// the list of which row of which of them each of its rows is.
    fn send_next_merged_batch(
        &mut self,
        position: usize,
        column: usize,
        sender: &mut dyn Sender<MergedBatchColumn>,
    ) -> UnaryResult<()> {
        let row_group = &mut self.ordered_row_groups[position];
        let (batch_index, batch) = row_group
            .find_next_to_send(column)
            .expect("a merged batch is sent after it was found decoded");
        let mut sources = Vec::new();
        let mut first_source_per_file = Vec::with_capacity(batch.rows_per_file.len());
        for (file, rows) in batch.rows_per_file.iter().enumerate() {
            first_source_per_file.push(sources.len());
            if rows.is_empty() {
                continue;
            }
            self.files[file].take_input_values(column, rows.clone(), &mut sources);
        }
        sender.send(MergedBatchColumn {
            sequence: row_group.first_sequence + batch_index,
            column,
            rows: MergedRows::from_file_runs(&batch.runs, &sources, &first_source_per_file),
            sources: sources.into(),
        })?;
        row_group.record_sent(column);
        Ok(())
    }

    /// Drop the oldest ordered output row groups that were sent on in every
    /// column. Returns whether any was dropped.
    fn drop_sent_row_groups(&mut self) -> bool {
        let mut has_dropped = false;
        while let Some(row_group) = self
            .ordered_row_groups
            .pop_front_if(|row_group| row_group.find_first_unsent().is_none())
        {
            self.requests
                .memory_budget
                .subtract(row_group.count_held_bytes());
            has_dropped = true;
        }
        has_dropped
    }

    /// Request, from disk, the sort columns of the row groups the merge
    /// takes its next rows from: the input row group that starts first in
    /// key order and every row group that overlaps it. Returns whether
    /// anything was requested.
    ///
    /// These requests are always made: the first row group cannot be merged
    /// without them. A table without sort columns has none to request.
    fn fetch_sort_columns_of_overlapping_row_groups(&mut self) -> bool {
        let mut has_requested = false;
        for &row_group in self.merge.row_groups_to_merge() {
            let file_row_group = &mut self.files[row_group.file].row_groups[row_group.row_group];
            for &column in &self.sort_columns {
                has_requested |= file_row_group.request_column(column, &mut self.requests);
            }
        }
        has_requested
    }

    /// Have the merge decide the order of the next rows of the output: at
    /// most one merged batch per call, so that the compaction worker gets
    /// back to its gates in between. Returns whether the order of any row
    /// was decided or the decoding of any key was started.
    ///
    /// While no ordered output row group waits to be sent on, the output
    /// cannot go on without this: the keys the merge needs are released to
    /// be decoded whatever the memory budget says. Ordering more than that
    /// is work ahead. It goes on only while the budget has room for work
    /// ahead, and a key it needs is released only if the budget has that
    /// room for it.
    fn decide_row_order_of_output_row_groups(&mut self) -> WriteResult<bool> {
        let is_every_row_ordered = self.merge.row_groups_to_merge().is_empty();
        let is_required = self.ordered_row_groups.is_empty();
        let has_room_ahead = self.requests.memory_budget.has_room_ahead_for(0);
        if is_every_row_ordered || !(is_required || has_room_ahead) {
            return Ok(false);
        }
        let mut has_ordered_rows = false;
        let mut has_started_decoding_keys = false;
        loop {
            let room = self.row_group_being_ordered.count_room_for_next_run();
            match self.merge.plan_next_run(room as u64)? {
                NextRun::Found { file, rows } => {
                    self.row_group_being_ordered.append_run(
                        file,
                        self.merge.count_merged_rows(file) - rows,
                        rows as u32,
                    );
                    has_ordered_rows = true;
                    if rows as usize == room {
                        break;
                    }
                }
                NextRun::NeedsKeys(needed_keys) => {
                    for needed_key in needed_keys {
                        let file = &mut self.files[needed_key.file];
                        let column = self.sort_columns[needed_key.sort_column];
                        let rows_end = needed_key.row + 1;
                        let estimated_bytes = file.estimate_bytes_to_decode(column, rows_end);
                        let memory_budget = &self.requests.memory_budget;
                        if is_required || memory_budget.has_room_ahead_for(estimated_bytes) {
                            has_started_decoding_keys |=
                                file.start_decoding_rows(column, rows_end, &mut self.requests);
                        }
                    }
                    break;
                }
                NextRun::NoRowsLeft => break,
            }
        }
        // The last output row group is kept when the merge has ordered its
        // last row, full or not.
        let is_every_row_ordered = self.merge.row_groups_to_merge().is_empty();
        let has_rows = !self.row_group_being_ordered.merged_batches.is_empty();
        if self.row_group_being_ordered.is_full() || (is_every_row_ordered && has_rows) {
            self.keep_ordered_row_group();
        }
        Ok(has_ordered_rows || has_started_decoding_keys)
    }

    /// Move the output row group whose row order was being decided to the
    /// ordered ones, and start the next.
    fn keep_ordered_row_group(&mut self) {
        let next_row_group = self.row_group_being_ordered.start_next();
        let row_group = std::mem::replace(&mut self.row_group_being_ordered, next_row_group);
        self.requests
            .memory_budget
            .add(row_group.count_held_bytes());
        self.ordered_row_groups.push_back(row_group);
        self.has_rows_to_check.fill(true);
    }

    /// Let the workers decompress and decode the rows of the ordered output
    /// row groups, fetching the chunks that hold them if they were not
    /// requested. Returns whether the decoding of any merged batch was
    /// started.
    ///
    /// An output row group is gone through a column at a time, in schema
    /// order, and a column a merged batch at a time. Each merged batch of a
    /// column is started only while the memory budget has room for what it
    /// is estimated to hold, so the budget decides how many columns are
    /// being decoded at once. The one exception always goes ahead: the next
    /// merged batch to send of the first unsent column of the oldest output
    /// row group.
    ///
    /// The sort columns are columns like the others here. Their rows were
    /// mostly released already, for the merge to compare. The rows whose
    /// order the merge decided from statistics were not, and are released
    /// here.
    fn start_decoding_other_columns(&mut self) -> bool {
        let mut has_started = false;
        for position in 0..self.ordered_row_groups.len() {
            while let Some(next) = self.ordered_row_groups[position].find_next_to_decode() {
                let row_group = &self.ordered_row_groups[position];
                let rows_end_per_file = row_group.merged_batches[next.batch]
                    .rows_per_file
                    .iter()
                    .enumerate()
                    .filter(|(_, rows)| !rows.is_empty())
                    .map(|(file, rows)| (file, rows.end));
                let estimated_bytes: usize = rows_end_per_file
                    .clone()
                    .map(|(file, rows_end)| {
                        self.files[file].estimate_bytes_to_decode(next.column, rows_end)
                    })
                    .sum();
                let is_required = position == 0 && row_group.find_first_unsent() == Some(next);
                let has_room = estimated_bytes == 0
                    || self.requests.memory_budget.has_room_for(estimated_bytes);
                if !is_required && !has_room {
                    return has_started;
                }
                for (file, rows_end) in rows_end_per_file {
                    self.files[file].start_decoding_rows(next.column, rows_end, &mut self.requests);
                }
                self.ordered_row_groups[position].record_decoding_started();
                has_started = true;
            }
        }
        has_started
    }

    /// Let the workers decompress and decode sort keys further ahead than
    /// the merge has asked for, so that it does not have to wait for them.
    /// Returns whether the decoding of any key was started.
    ///
    /// The keys of the row groups to merge are decoded a range of rows at a
    /// time, each row group in turn, while the memory budget has room for
    /// work ahead, and no further than one output row group's worth of rows
    /// past the last merged row of their file: the merge cannot use more
    /// before the next output row group is ordered.
    fn start_decoding_sort_keys_ahead(&mut self) -> bool {
        let mut has_started = false;
        let output_row_group_rows = self.row_group_being_ordered.row_limit as u64;
        loop {
            let mut has_started_range = false;
            for &row_group in self.merge.row_groups_to_merge() {
                let file = &mut self.files[row_group.file];
                let row_group_rows = file.row_groups[row_group.row_group].rows.clone();
                let rows_end = row_group_rows
                    .end
                    .min(self.merge.count_merged_rows(row_group.file) + output_row_group_rows);
                for &column in &self.sort_columns {
                    let released_rows_end = file.find_released_rows_end(column);
                    // The file's rows are released in file order, so a row
                    // group waits until the one before it is released.
                    if released_rows_end < row_group_rows.start || rows_end <= released_rows_end {
                        continue;
                    }
                    let estimated_bytes =
                        file.estimate_bytes_to_decode(column, released_rows_end + 1);
                    let memory_budget = &self.requests.memory_budget;
                    if !memory_budget.has_room_ahead_for(estimated_bytes) {
                        return has_started;
                    }
                    file.start_decoding_rows(column, released_rows_end + 1, &mut self.requests);
                    has_started_range = true;
                    has_started = true;
                }
            }
            if !has_started_range {
                return has_started;
            }
        }
    }

    /// Request, from disk, column chunks before their rows are to be
    /// decoded. Returns whether anything was requested.
    ///
    /// Every chunk is needed for certain, so this only moves its disk read
    /// earlier. The chunks are requested in the order the merge takes rows
    /// from their row groups, every column of a row group before the next
    /// row group, for as long as there is room to fetch ahead (see
    /// [`ReadRequests::has_room_to_fetch_ahead`]). They then wait,
    /// compressed, until their rows are released.
    fn fetch_other_columns_ahead(&mut self) -> bool {
        let mut has_requested = false;
        while let Some(&row_group) = self
            .row_groups_in_merge_order
            .get(self.next_row_group_to_fetch_ahead)
        {
            let file_row_group = &mut self.files[row_group.file].row_groups[row_group.row_group];
            for column in 0..file_row_group.columns.len() {
                let read = &file_row_group.columns[column];
                if read.is_requested {
                    continue;
                }
                if !self.requests.has_room_to_fetch_ahead(read.compressed_bytes) {
                    return has_requested;
                }
                has_requested |= file_row_group.request_column(column, &mut self.requests);
            }
            self.next_row_group_to_fetch_ahead += 1;
        }
        has_requested
    }
}
