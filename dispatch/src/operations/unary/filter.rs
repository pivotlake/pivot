//! Filter operator: keeps rows matching a boolean mask.
//!
//! The filter closure receives a `&RecordBatch` and returns a [`BooleanArray`] mask.
//! Rows where the mask is `true` are kept; the rest are dropped. Batches where no
//! rows match are skipped entirely (no empty batch emitted).

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::{Unary, UnaryFactory};
use crate::scan::trailing_metadata_columns;
use arrow::compute::{FilterBuilder, filter_record_batch};
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, BooleanArray, Int32Array, RecordBatch, RunArray};
use arrow_schema::ArrowError;
use std::sync::Arc;

/// Factory that wraps a filter closure. `F` is the per-worker closure created by the
/// builder passed to [`RecordBatchOperatorSpec::filter`](crate::api::RecordBatchOperatorSpec::filter).
pub struct FilterFactory<F>(pub F);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FilterFactory<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send + 'static,
{
    type Unary = Filter<F>;

    fn build_unary(self) -> Self::Unary {
        Filter { func: self.0 }
    }
}

pub struct Filter<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send,
{
    func: F,
}

impl<F> Unary<RecordBatch, RecordBatch> for Filter<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send,
{
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        output: &mut OP,
    ) -> unary::Result<()> {
        let mask = (self.func)(&batch);
        debug_assert_eq!(mask.len(), batch.num_rows());

        let kept = mask.true_count();
        if kept == 0 {
            return Ok(());
        }
        output.send(filter_batch(&batch, &mask, kept)?)?;
        Ok(())
    }
}

/// Filter a batch by `mask`, sidestepping arrow's run-end filter for the
/// trailing row-group-metadata pair a metadata-emitting scan appends: arrow
/// filters a run-end-encoded array by walking every LOGICAL row in a scalar
/// loop, so the single-run row-group column would cost a whole per-row pass
/// per batch. That column is one run covering the batch, so its filtered form
/// is just the same value over `kept` rows, rebuilt in constant time.
fn filter_batch(
    batch: &RecordBatch,
    mask: &BooleanArray,
    kept: usize,
) -> Result<RecordBatch, ArrowError> {
    if trailing_metadata_columns(&batch.schema()) == 0 {
        return filter_record_batch(batch, mask);
    }
    let row_group_column_idx = batch.num_columns() - 2;
    let Some(single_run) = rebuild_single_run(batch.column(row_group_column_idx), kept) else {
        return filter_record_batch(batch, mask);
    };

    let predicate = FilterBuilder::new(mask).optimize().build();
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(idx, column)| {
            if idx == row_group_column_idx {
                Ok(single_run.clone())
            } else {
                predicate.filter(column)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(batch.schema(), columns)
}

/// The single-run run-end-encoded `array` re-spanned over `kept` rows, or
/// `None` when the array is not the expected one-run shape (then the caller
/// falls back to arrow's generic filter).
fn rebuild_single_run(array: &ArrayRef, kept: usize) -> Option<ArrayRef> {
    let run_array = array.as_any().downcast_ref::<RunArray<Int32Type>>()?;
    let run_ends = run_array.run_ends();
    if run_ends.get_start_physical_index() != run_ends.get_end_physical_index() {
        return None;
    }
    let value = run_array
        .values()
        .slice(run_ends.get_start_physical_index(), 1);
    let rebuilt = RunArray::try_new(&Int32Array::from(vec![kept as i32]), value.as_ref()).ok()?;
    Some(Arc::new(rebuilt) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_unary;
    use arrow_array::{ArrayRef, Int32Array};
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
            func: |b: &RecordBatch| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                BooleanArray::from_iter(col.values().iter().map(|v| Some(*v > 3)))
            },
        };

        let out = run_unary(filter, vec![batch(&[1, 2, 3, 4, 5])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![4, 5]);
    }

    #[test]
    fn no_matches_produces_no_output() {
        let filter = Filter {
            func: |_: &RecordBatch| BooleanArray::from(vec![false, false, false]),
        };

        let out = run_unary(filter, vec![batch(&[1, 2, 3])]);

        assert_eq!(out.len(), 0);
    }

    #[test]
    fn all_match() {
        let filter = Filter {
            func: |b: &RecordBatch| BooleanArray::from(vec![true; b.num_rows()]),
        };

        let out = run_unary(filter, vec![batch(&[10, 20, 30])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![10, 20, 30]);
    }

    #[test]
    fn multiple_batches() {
        let filter = Filter {
            func: |b: &RecordBatch| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                BooleanArray::from_iter(col.values().iter().map(|v| Some(*v % 2 == 0)))
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
            func: |b: &RecordBatch| BooleanArray::from(vec![true; b.num_rows()]),
        };

        let out = run_unary(filter, vec![batch(&[])]);

        assert_eq!(out.len(), 0);
    }

    /// A batch tagging `values` with the trailing metadata pair a
    /// metadata-emitting scan appends: a single-run row-group column of
    /// `group` and a dense row-index column.
    fn metadata_batch(values: &[i32], group: u32) -> RecordBatch {
        let run_ends = Int32Array::from(vec![values.len() as i32]);
        let groups = arrow_array::UInt32Array::from(vec![group]);
        let run_array = RunArray::<Int32Type>::try_new(&run_ends, &groups).unwrap();
        let row_idxs = arrow_array::UInt32Array::from_iter_values(0..values.len() as u32);
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int32, false),
            Field::new(
                crate::scan::ROW_GROUP_IDX_FIELD,
                arrow_array::Array::data_type(&run_array).clone(),
                false,
            ),
            Field::new(crate::scan::ROW_IDX_FIELD, DataType::UInt32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(values.to_vec())),
                Arc::new(run_array),
                Arc::new(row_idxs),
            ],
        )
        .unwrap()
    }

    #[test]
    fn metadata_columns_survive_filtering_with_the_run_respanned() {
        let filter = Filter {
            func: |b: &RecordBatch| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                BooleanArray::from_iter(col.values().iter().map(|v| Some(*v > 2)))
            },
        };

        let out = run_unary(filter, vec![metadata_batch(&[1, 2, 3, 4], 9)]);

        assert_eq!(i32_col(&out[0]), vec![3, 4]);
        let groups = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .unwrap();
        assert_eq!(arrow_array::Array::len(groups), 2);
        assert_eq!(groups.run_ends().values(), &[2]);
        let row_idxs = out[0]
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap();
        assert_eq!(row_idxs.values().as_ref(), &[2, 3]);
    }
}
