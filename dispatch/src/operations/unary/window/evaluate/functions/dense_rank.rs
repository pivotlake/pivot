//! `dense_rank()`: how many tie groups the partition has had up to each row,
//! so tied rows share a value and the next group gets the next integer.

use arrow_array::ArrayRef;
use arrow_array::types::Int64Type;
use arrow_schema::ArrowError;

use super::FunctionEvaluator;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

pub struct DenseRankEvaluator {
    /// Per batch of the sequence, the dense rank of the row just before the
    /// batch, within that row's partition. Dense rank counts tie groups,
    /// which is not a difference of two row numbers, so each batch starts
    /// from this count.
    dense_rank_before_batch: Vec<i64>,
}

impl DenseRankEvaluator {
    pub fn new(boundaries: &SequenceBoundaries) -> Self {
        // The dense rank of a batch's last row is one more than the number
        // of tie groups that start after its partition's first row. Count
        // those from the batch's last partition start when it has one, and
        // add the batch's tie groups to the previous count when it has none.
        let tie_group_starts = boundaries.tie_groups.starts();
        let batch_count = boundaries.batch_first_rows().len() - 1;
        let mut dense_rank_before_batch = Vec::with_capacity(batch_count);
        let mut dense_rank_at_batch_end = 0;
        for batch_index in 0..batch_count {
            dense_rank_before_batch.push(dense_rank_at_batch_end);
            let rows = boundaries.batch_rows(batch_index);
            dense_rank_at_batch_end = match boundaries.partitions.last_start_in_batch(batch_index) {
                Some(partition_begin) => {
                    let later_rows = partition_begin + 1..rows.end;
                    1 + tie_group_starts
                        .slice(later_rows.start, later_rows.len())
                        .count_set_bits() as i64
                }
                None => {
                    dense_rank_at_batch_end
                        + tie_group_starts
                            .slice(rows.start, rows.len())
                            .count_set_bits() as i64
                }
            };
        }
        Self {
            dense_rank_before_batch,
        }
    }
}

impl FunctionEvaluator for DenseRankEvaluator {
    fn evaluate_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let rows = boundaries.batch_rows(batch_index);
        let starts_partition = boundaries.partitions.starts().slice(rows.start, rows.len());
        let starts_tie_group = boundaries.tie_groups.starts().slice(rows.start, rows.len());
        let mut dense_ranks = PrimitiveBuilder::<Int64Type>::with_capacity(allocator, rows.len());
        let batch_dense_ranks = dense_ranks.spare_mut(rows.len());
        let mut dense_rank = self.dense_rank_before_batch[batch_index];
        let row_flags = starts_partition.iter().zip(starts_tie_group.iter());
        for (row_dense_rank, (starts_partition, starts_tie_group)) in
            batch_dense_ranks.iter_mut().zip(row_flags)
        {
            dense_rank = if starts_partition {
                1
            } else {
                dense_rank + i64::from(starts_tie_group)
            };
            *row_dense_rank = dense_rank;
        }
        Ok(dense_ranks.into_array(None))
    }
}
