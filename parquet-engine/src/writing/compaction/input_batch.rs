//! The rows of the input files as the compaction reads them.
//!
//! The compaction reads whole input row groups, every column. The decoder
//! appends two columns to every batch, which say which row group its rows
//! are in and which rows of it they are. A batch comes out of the decoder as
//! a [`DecodedBatch`]. When the planner receives one, it takes those two
//! columns off, reserves the memory the batch holds, and keeps the rows and
//! the reservation together as an [`InputBatch`].

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt32Array};

use super::memory_budget::{MemoryBudget, MemoryReservation};
use crate::reading::record_batch_metadata::{global_row_group, row_index};

/// Consecutive rows of one input row group, every column, as they come out
/// of the decoder: followed by the column that names their row group and
/// the column that numbers them in it.
pub(crate) struct DecodedBatch {
    batch: RecordBatch,
    /// The worker that took the batch from the decoder, as its index among
    /// the dataflow's workers.
    worker: usize,
}

impl DecodedBatch {
    /// `batch` as the decoder emitted it, taken by `worker`.
    pub(super) fn new(batch: RecordBatch, worker: usize) -> Self {
        Self { batch, worker }
    }

    pub(super) fn row_count(&self) -> usize {
        self.batch.num_rows()
    }

    /// The batch's rows, which there must be some of, without the two
    /// columns the decoder appended, and with their memory reserved from
    /// `memory_budget`.
    pub(super) fn into_input_batch(self, memory_budget: &Arc<MemoryBudget>) -> InputBatch {
        let row_group = global_row_group(&self.batch)
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("the row group column holds row group numbers")
            .value(0);
        let row_numbers = row_index(&self.batch);
        let first_row = row_numbers.value(0);
        // The compaction reads every row of a row group and refuses a table
        // with deleted rows, so a batch's rows follow one another.
        assert_eq!(
            row_numbers.value(row_numbers.len() - 1),
            first_row + row_numbers.len() as u32 - 1,
            "a decoded batch holds consecutive rows of its row group"
        );
        let column_count = self.batch.num_columns() - 2;
        let batch = self
            .batch
            .project(&(0..column_count).collect::<Vec<usize>>())
            .expect("the columns before the decoder's two are the batch's own");
        InputBatch {
            memory: memory_budget.reserve(batch.get_array_memory_size()),
            batch,
            row_group,
            first_row,
            worker: self.worker,
        }
    }
}

/// Consecutive rows of one input row group, every column, with the memory
/// reserved for them.
#[derive(Clone)]
pub(crate) struct InputBatch {
    pub(super) batch: RecordBatch,
    /// The input row group the rows are in.
    pub(super) row_group: u32,
    /// The row number, in that row group, of the batch's first row.
    pub(super) first_row: u32,
    /// The memory reserved for the batch the rows were decoded as, held for
    /// as long as any of them are in memory.
    pub(super) memory: Arc<MemoryReservation>,
    /// The worker that decoded the batch.
    pub(super) worker: usize,
}

impl InputBatch {
    pub(super) fn row_count(&self) -> usize {
        self.batch.num_rows()
    }

    /// `row_count` rows of the batch from its row `first`.
    pub(super) fn slice_rows(&self, first: usize, row_count: usize) -> Self {
        Self {
            batch: self.batch.slice(first, row_count),
            row_group: self.row_group,
            first_row: self.first_row + first as u32,
            memory: self.memory.clone(),
            worker: self.worker,
        }
    }
}
