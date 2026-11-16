//! `percent_rank()`: the rank scaled to run from 0 to 1 over the partition,
//! `(rank - 1) / (partition rows - 1)`.

use arrow_array::ArrayRef;
use arrow_array::types::Float64Type;
use arrow_schema::ArrowError;

use super::FunctionEvaluator;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

#[derive(Default)]
pub struct PercentRankEvaluator {
    /// Scratch space reused from batch to batch.
    partition_begins: Vec<u64>,
    partition_ends: Vec<u64>,
    tie_group_begins: Vec<u64>,
}

impl FunctionEvaluator for PercentRankEvaluator {
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
        boundaries
            .partitions
            .fill_group_ends(batch_index, rows.clone(), &mut self.partition_ends);
        boundaries.tie_groups.fill_group_begins(
            batch_index,
            rows.clone(),
            &mut self.tie_group_begins,
        );
        let mut percent_ranks =
            PrimitiveBuilder::<Float64Type>::with_capacity(allocator, rows.len());
        let row_bounds = self
            .partition_begins
            .iter()
            .zip(&self.partition_ends)
            .zip(&self.tie_group_begins);
        for (percent_rank, ((&partition_begin, &partition_end), &tie_group_begin)) in percent_ranks
            .spare_mut(rows.len())
            .iter_mut()
            .zip(row_bounds)
        {
            // A partition of one row has no other row to rank against.
            let rows_after_first = partition_end - partition_begin - 1;
            *percent_rank = if rows_after_first == 0 {
                0.0
            } else {
                (tie_group_begin - partition_begin) as f64 / rows_after_first as f64
            };
        }
        Ok(percent_ranks.into_array(None))
    }
}
