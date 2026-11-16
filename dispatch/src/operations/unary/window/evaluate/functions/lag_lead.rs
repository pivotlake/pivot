//! `lag(x, offset)` and `lead(x, offset)`: the argument's value in the row
//! `offset` rows before or after each row, or NULL when that row is outside
//! the partition.

use arrow::compute::nullif;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::ArrowError;

use super::FunctionEvaluator;
use crate::arrays::take_chunked;
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

pub struct LagLeadEvaluator {
    /// The argument column of every batch of the sequence.
    argument_batches: Vec<ArrayRef>,
    offset: u64,
    /// `lead` reads a later row, `lag` an earlier one.
    reads_later_row: bool,
    /// Scratch space reused from batch to batch. The bounds are each row's
    /// partition end for `lead` and partition begin for `lag`.
    partition_bounds: Vec<u64>,
    source_positions: Vec<(u32, u32)>,
}

impl LagLeadEvaluator {
    pub fn new(batches: &[RecordBatch], column: usize, offset: u64, reads_later_row: bool) -> Self {
        Self {
            argument_batches: batches
                .iter()
                .map(|batch| batch.column(column).clone())
                .collect(),
            offset,
            reads_later_row,
            partition_bounds: Vec::new(),
            source_positions: Vec::new(),
        }
    }
}

impl FunctionEvaluator for LagLeadEvaluator {
    fn evaluate_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let rows = boundaries.batch_rows(batch_index);
        if self.reads_later_row {
            boundaries.partitions.fill_group_ends(
                batch_index,
                rows.clone(),
                &mut self.partition_bounds,
            );
        } else {
            boundaries.partitions.fill_group_begins(
                batch_index,
                rows.clone(),
                &mut self.partition_bounds,
            );
        }

        // Each row's source row, as a `(batch, row)` position. The source
        // rows rise with the rows, so the batch a source row is in is found
        // by moving forward from the previous one.
        let batch_first_rows = boundaries.batch_first_rows();
        let mut source_batch = None;
        let mut lacks_source = BooleanBufferBuilder::new(rows.len());
        self.source_positions.clear();
        for (row, &partition_bound) in rows.clone().zip(&self.partition_bounds) {
            let row = row as u64;
            let source_row = if self.reads_later_row {
                row.checked_add(self.offset)
                    .filter(|&source_row| source_row < partition_bound)
            } else {
                row.checked_sub(self.offset)
                    .filter(|&source_row| source_row >= partition_bound)
            };
            let Some(source_row) = source_row else {
                // Any position will do: the row's value is replaced by NULL.
                self.source_positions.push((0, 0));
                lacks_source.append(true);
                continue;
            };
            let source_row = source_row as usize;
            let mut batch = source_batch.unwrap_or_else(|| {
                batch_first_rows.partition_point(|&first_row| first_row <= source_row) - 1
            });
            while source_row >= batch_first_rows[batch + 1] {
                batch += 1;
            }
            source_batch = Some(batch);
            self.source_positions
                .push((batch as u32, (source_row - batch_first_rows[batch]) as u32));
            lacks_source.append(false);
        }

        let source_values =
            take_chunked(allocator, &self.argument_batches, &self.source_positions)?;
        let lacks_source = lacks_source.finish();
        if lacks_source.count_set_bits() == 0 {
            return Ok(source_values);
        }
        nullif(&source_values, &BooleanArray::new(lacks_source, None))
    }
}
