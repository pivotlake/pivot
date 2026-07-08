//! Filter operator: keeps rows the filter closure selects.
//!
//! The filter closure receives an owned `RecordBatch` and returns the batch of
//! surviving rows. Owning the batch lets the closure evaluate multiple
//! conditions progressively — shrink the batch after each condition, so later
//! (often costlier) conditions only see rows the earlier ones kept. Batches
//! where no rows survive are skipped entirely (no empty batch emitted).

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::{Unary, UnaryFactory};
use arrow_array::RecordBatch;

/// Factory that wraps a filter closure. `F` is the per-worker closure created by the
/// builder passed to [`RecordBatchOperatorSpec::filter`](crate::api::RecordBatchOperatorSpec::filter).
pub struct FilterFactory<F>(pub F);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FilterFactory<F>
where
    F: FnMut(RecordBatch) -> RecordBatch + Send + 'static,
{
    type Unary = Filter<F>;

    fn build_unary(self) -> Self::Unary {
        Filter { func: self.0 }
    }
}

pub struct Filter<F>
where
    F: FnMut(RecordBatch) -> RecordBatch + Send,
{
    func: F,
}

impl<F> Unary<RecordBatch, RecordBatch> for Filter<F>
where
    F: FnMut(RecordBatch) -> RecordBatch + Send,
{
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        output: &mut OP,
    ) -> unary::Result<()> {
        let kept = (self.func)(batch);
        if kept.num_rows() == 0 {
            return Ok(());
        }
        output.send(kept)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_unary;
    use arrow_array::{ArrayRef, BooleanArray, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn i32_col(batch: &RecordBatch) -> Vec<i32> {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn keeps_matching_rows() {
        let filter = Filter {
            func: |b: RecordBatch| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let mask = BooleanArray::from_iter(col.values().iter().map(|v| Some(*v > 3)));
                arrow::compute::filter_record_batch(&b, &mask).unwrap()
            },
        };

        let out = run_unary(filter, vec![batch(&[1, 2, 3, 4, 5])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![4, 5]);
    }

    #[test]
    fn no_matches_produces_no_output() {
        let filter = Filter {
            func: |b: RecordBatch| b.slice(0, 0),
        };

        let out = run_unary(filter, vec![batch(&[1, 2, 3])]);

        assert_eq!(out.len(), 0);
    }

    #[test]
    fn all_match() {
        let filter = Filter {
            func: |b: RecordBatch| b,
        };

        let out = run_unary(filter, vec![batch(&[10, 20, 30])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![10, 20, 30]);
    }

    #[test]
    fn multiple_batches() {
        let filter = Filter {
            func: |b: RecordBatch| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let mask = BooleanArray::from_iter(col.values().iter().map(|v| Some(*v % 2 == 0)));
                arrow::compute::filter_record_batch(&b, &mask).unwrap()
            },
        };

        let out = run_unary(filter, vec![batch(&[1, 2, 3]), batch(&[4, 5, 6])]);

        assert_eq!(out.len(), 2);
        assert_eq!(i32_col(&out[0]), vec![2]);
        assert_eq!(i32_col(&out[1]), vec![4, 6]);
    }

    #[test]
    fn empty_batch() {
        let filter = Filter {
            func: |b: RecordBatch| b,
        };

        let out = run_unary(filter, vec![batch(&[])]);

        assert_eq!(out.len(), 0);
    }
}
