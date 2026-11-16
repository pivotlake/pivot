//! `rank()`: the row number of the first row of each row's tie group, so
//! tied rows share a value and the values skip after a tie.

use arrow_array::ArrayRef;
use arrow_array::types::Int64Type;
use arrow_schema::ArrowError;

use super::FunctionEvaluator;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

#[derive(Default)]
pub struct RankEvaluator {
    /// Scratch space reused from batch to batch.
    partition_begins: Vec<u64>,
    tie_group_begins: Vec<u64>,
}

impl FunctionEvaluator for RankEvaluator {
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
        boundaries.tie_groups.fill_group_begins(
            batch_index,
            rows.clone(),
            &mut self.tie_group_begins,
        );
        let mut ranks = PrimitiveBuilder::<Int64Type>::with_capacity(allocator, rows.len());
        let batch_ranks = ranks.spare_mut(rows.len());
        for ((rank, &tie_group_begin), &partition_begin) in batch_ranks
            .iter_mut()
            .zip(&self.tie_group_begins)
            .zip(&self.partition_begins)
        {
            *rank = (tie_group_begin - partition_begin + 1) as i64;
        }
        Ok(ranks.into_array(None))
    }
}
