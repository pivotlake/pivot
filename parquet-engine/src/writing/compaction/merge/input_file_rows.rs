//! One input file's rows, from the row groups not asked for yet to the rows
//! waiting to be merged.

use std::collections::{BTreeMap, VecDeque};

use arrow_array::RecordBatch;
use dispatch::{ChannelInputSender, OrderBy, count_rows_not_after};

use super::super::READ_AHEAD_ROWS_PER_FILE;
use super::super::input_batch::InputBatch;
use super::decode_gate::DecodeReleases;
use crate::ParquetTable;
use crate::reading::DECODE_RANGE_ROWS;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::writing::error::{WriteError, WriteResult};

/// One row group that was asked for and whose rows have not all arrived.
struct RequestedRowGroup {
    group: usize,
    row_count: usize,
    /// How many of its rows, from the first, are released to be decoded.
    rows_released: usize,
    /// How many of its rows, from the first, have arrived and been made
    /// ready to merge.
    rows_made_ready: usize,
    /// The batches that arrived ahead of the rows before them, by the row
    /// number of their first row.
    early_batches: BTreeMap<u32, InputBatch>,
}

/// A batch that is ready to merge, with its sort columns as a batch of
/// their own for the merge to compare.
struct ReadyBatch {
    rows: InputBatch,
    keys: RecordBatch,
}

/// One input file's rows. The file is sorted, so its rows are read, decoded
/// and merged in file order.
#[derive(Default)]
pub(super) struct InputFileRows {
    /// The file's row groups not asked for yet, in file order.
    unrequested_row_groups: VecDeque<usize>,
    /// The row groups asked for whose rows have not all arrived, in file
    /// order.
    requested: VecDeque<RequestedRowGroup>,
    /// The rows that have arrived, in file order, and are not merged yet.
    ready: VecDeque<ReadyBatch>,
    ready_row_count: usize,
    /// The sort columns of the file's last ready row, as a one-row batch.
    last_key: Option<RecordBatch>,
}

impl InputFileRows {
    /// Add `group` as the file's next row group.
    pub(super) fn add_row_group(&mut self, group: usize) {
        self.unrequested_row_groups.push_back(group);
    }

    /// Whether the file has row groups not asked for yet.
    pub(super) fn has_unrequested_row_groups(&self) -> bool {
        !self.unrequested_row_groups.is_empty()
    }

    /// Whether every row of the file has arrived.
    pub(super) fn has_all_rows_arrived(&self) -> bool {
        self.unrequested_row_groups.is_empty() && self.requested.is_empty()
    }

    /// Whether the file has rows to merge now.
    pub(super) fn has_ready_rows(&self) -> bool {
        !self.ready.is_empty()
    }

    /// The sort columns of the file's last ready row, as a one-row batch.
    pub(super) fn last_key(&self) -> Option<&RecordBatch> {
        self.last_key.as_ref()
    }

    /// Ask for the file's next row groups, every column, while fewer than
    /// [`READ_AHEAD_ROWS_PER_FILE`] of its rows are asked for and not
    /// released to be decoded. A row group asked for is fetched,
    /// decompressed and cut into ranges, which then wait to be released, so
    /// that the merge does not wait for a read when it gets to them.
    pub(super) fn request_row_groups(
        &mut self,
        table: &ParquetTable,
        requests: &ChannelInputSender<QueryRowGroupMetadata>,
    ) {
        loop {
            let unreleased_rows: usize = self
                .requested
                .iter()
                .map(|row_group| row_group.row_count - row_group.rows_released)
                .sum();
            if unreleased_rows >= READ_AHEAD_ROWS_PER_FILE {
                break;
            }
            let Some(group) = self.unrequested_row_groups.pop_front() else {
                break;
            };
            let row_count = table.row_groups()[group].num_rows as usize;
            let request =
                QueryRowGroupMetadata::new(table, group, RowSelection::All).bypassing_caches();
            if requests.try_send(request).is_err() {
                panic!("the request queue holds every row group of the compaction");
            }
            self.requested.push_back(RequestedRowGroup {
                group,
                row_count,
                rows_released: 0,
                rows_made_ready: 0,
                early_batches: BTreeMap::new(),
            });
        }
    }

    /// How many rows the file's next range to release holds: none when every
    /// row of the row groups asked for is released. A row group is decoded a
    /// range at a time.
    pub(super) fn count_next_range_rows(&self) -> usize {
        self.requested
            .iter()
            .find(|row_group| row_group.rows_released < row_group.row_count)
            .map_or(0, |row_group| {
                (DECODE_RANGE_ROWS as usize).min(row_group.row_count - row_group.rows_released)
            })
    }

    /// How many of the file's rows are released to be decoded and not ready
    /// to merge yet: being decoded, or arrived before rows that come before
    /// them.
    pub(super) fn count_rows_being_decoded(&self) -> usize {
        self.requested
            .iter()
            .map(|row_group| row_group.rows_released - row_group.rows_made_ready)
            .sum()
    }

    /// Release the file's next range of rows to be decoded.
    pub(super) fn release_next_range(&mut self, releases: &DecodeReleases) {
        let range_rows = self.count_next_range_rows();
        let row_group = self
            .requested
            .iter_mut()
            .find(|row_group| row_group.rows_released < row_group.row_count)
            .expect("the file has a range to release");
        row_group.rows_released += range_rows;
        releases.release_rows(row_group.group, row_group.rows_released);
    }

    /// Keep `batch`, rows of one of the file's row groups, and make ready to
    /// merge every row that now has all the rows before it. The sort columns
    /// are columns `key_columns` of the batch. Fails if the file's rows are
    /// not in the order `order_by` gives.
    pub(super) fn add_batch(
        &mut self,
        batch: InputBatch,
        key_columns: &[usize],
        order_by: &[OrderBy],
    ) -> WriteResult<()> {
        let row_group = self
            .requested
            .iter_mut()
            .find(|row_group| row_group.group == batch.row_group as usize)
            .expect("a decoded row group was asked for");
        let batch_end = batch.first_row as usize + batch.row_count();
        if (batch.first_row as usize) < row_group.rows_made_ready || batch_end > row_group.row_count
        {
            return Err(WriteError::CompactionRowCountMismatch);
        }
        row_group.early_batches.insert(batch.first_row, batch);

        // Rows are released in file order, so only the first row group still
        // arriving can have its next rows here.
        while let Some(row_group) = self.requested.front_mut() {
            while let Some(rows) = row_group
                .early_batches
                .remove(&(row_group.rows_made_ready as u32))
            {
                row_group.rows_made_ready += rows.row_count();
                self.ready_row_count += rows.row_count();
                let keys = rows.batch.project(key_columns)?;
                // A table without sort columns has no order to check.
                if !key_columns.is_empty() {
                    if let Some(last_key) = &self.last_key {
                        let follows = count_rows_not_after(
                            order_by,
                            std::slice::from_ref(last_key),
                            &keys,
                            0,
                        )? == 1;
                        if !follows {
                            return Err(WriteError::UnsortedCompactionInput);
                        }
                    }
                    self.last_key = Some(keys.slice(keys.num_rows() - 1, 1));
                }
                self.ready.push_back(ReadyBatch { rows, keys });
            }
            if row_group.rows_made_ready < row_group.row_count {
                break;
            }
            self.requested.pop_front();
        }
        Ok(())
    }

    /// Take out of the file its ready rows whose sort columns sort at or
    /// before `bound`, a one-row batch of sort columns, or all of them when
    /// there is no bound. Each batch of rows comes with its sort columns.
    pub(super) fn take_rows_not_after(
        &mut self,
        bound: Option<&RecordBatch>,
        order_by: &[OrderBy],
    ) -> WriteResult<Vec<(InputBatch, RecordBatch)>> {
        let row_count = match bound {
            None => self.ready_row_count,
            Some(bound) => {
                let ready_keys: Vec<RecordBatch> =
                    self.ready.iter().map(|ready| ready.keys.clone()).collect();
                count_rows_not_after(order_by, &ready_keys, bound, 0)?
            }
        };
        let mut taken = Vec::new();
        let mut rows_left = row_count;
        while rows_left > 0 {
            let ready = self.ready.front_mut().expect("the taken rows have arrived");
            let batch_rows = ready.rows.row_count();
            if batch_rows <= rows_left {
                rows_left -= batch_rows;
                let ready = self.ready.pop_front().unwrap();
                taken.push((ready.rows, ready.keys));
            } else {
                taken.push((
                    ready.rows.slice_rows(0, rows_left),
                    ready.keys.slice(0, rows_left),
                ));
                ready.rows = ready.rows.slice_rows(rows_left, batch_rows - rows_left);
                ready.keys = ready.keys.slice(rows_left, batch_rows - rows_left);
                rows_left = 0;
            }
        }
        self.ready_row_count -= row_count;
        Ok(taken)
    }
}
