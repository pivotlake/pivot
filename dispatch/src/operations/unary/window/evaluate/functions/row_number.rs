//! `row_number()`: each row's position in its partition, from 1.

use arrow_array::ArrayRef;
use arrow_array::types::Int64Type;
use arrow_schema::ArrowError;

use super::FunctionEvaluator;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

#[derive(Default)]
pub struct RowNumberEvaluator {
    /// Scratch space reused from batch to batch.
    partition_begins: Vec<u64>,
}

impl FunctionEvaluator for RowNumberEvaluator {
    fn evaluate_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let rows = boundaries.batch_rows(batch_index);
        boundaries.partitions.fill_group_begins(
            batch_index,
            rows.clone(),
            &mut self.partition_begins,
        );
        let mut row_numbers = PrimitiveBuilder::<Int64Type>::with_capacity(allocator, rows.len());
        let batch_row_numbers = row_numbers.spare_mut(rows.len());
        for ((row_number, row), &partition_begin) in batch_row_numbers
            .iter_mut()
            .zip(rows)
            .zip(&self.partition_begins)
        {
            *row_number = (row as u64 - partition_begin + 1) as i64;
        }
        Ok(row_numbers.into_array(None))
    }
}
