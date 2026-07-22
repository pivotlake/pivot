//! Filter operator: keeps rows the filter closure selects.
//!
//! The filter closure receives an owned `RecordBatch` and returns the batch
//! together with an optional mask of surviving rows. Owning the batch lets
//! the closure evaluate multiple conditions progressively - shrink the batch
//! after each condition, so later (often costlier) conditions only see rows
//! the earlier ones kept (the operator'"'"'s [`SlabAllocator`] is passed in for
//! that, see [`take`](crate::arrays::take::take)).
//! A returned mask is applied by appending the surviving rows to a
//! [`BatchAccumulator`], which coalesces survivors across input batches and
//! emits full-size batches, so a selective filter'"'"'s downstream sees a few
//! large batches instead of a runt batch per input.

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::BatchAccumulator;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::{Unary, UnaryFactory};
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;

/// The row positions a filter mask keeps: true bits, with a null mask entry
/// dropping the row (Arrow's filter semantics).
fn kept_bits(mask: &BooleanArray) -> BooleanBuffer {
    match mask.nulls() {
        None => mask.values().clone(),
        Some(nulls) => mask.values() & nulls.inner(),
    }
}

/// Collect the ascending kept-row positions of `mask` into `out`.
///
/// Extending a `Vec` straight from the bit iterator pays a capacity check
/// per element (the iterator has no usable size hint); reserving the exact
/// popcount up front and writing raw skips that.
pub fn collect_selected_indices(mask: &BooleanArray, out: &mut Vec<u32>) {
    let bits = kept_bits(mask);
    out.clear();
    let kept = bits.count_set_bits();
    out.reserve(kept);
    // SAFETY: `reserve(kept)` guarantees capacity for every set bit, and the
    // iterator yields exactly `kept` indices.
    unsafe {
        let ptr = out.as_mut_ptr();
        let mut len = 0;
        for index in bits.set_indices_u32() {
            ptr.add(len).write(index);
            len += 1;
        }
        out.set_len(len);
    }
}

/// How a filter closure describes the surviving rows of the input batch.
/// `Indices` refers to the ascending row positions the closure wrote into
/// the scratch vec it was handed. A closure holding a boolean mask converts
/// it with [`collect_selected_indices`].
pub enum RowSelection {
    /// Every row survives; the batch is passed through untouched.
    All,
    /// The rows at the scratch indices survive.
    Indices,
}

/// Factory that wraps a filter closure. `F` is the per-worker closure created by the
/// builder passed to [`RecordBatchOperatorSpec::filter`](crate::api::RecordBatchOperatorSpec::filter).
pub struct FilterFactory<F>(pub F);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FilterFactory<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send + 'static,
{
    type Unary = Filter<F>;

    fn build_unary(self) -> Self::Unary {
        Filter {
            func: self.0,
            allocator: SlabAllocator::new(false),
            accumulator: None,
            selection: Vec::new(),
        }
    }
}

pub struct Filter<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send,
{
    func: F,
    allocator: SlabAllocator,
    /// Coalesces surviving rows across batches. Created on the first batch
    /// because that is when a schema first exists: dispatch pipelines are
    /// schema-agnostic by design (batches carry their schema; specs carry
    /// none), so the factory has nothing to construct this from.
    accumulator: Option<BatchAccumulator>,
    /// Scratch handed to the closure for the selected row positions.
    selection: Vec<u32>,
}

impl<F> Filter<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send,
{
    /// Apply the selection sitting in `self.selection` to `batch`: send whole
    /// surviving batches directly, coalesce partial ones through the
    /// accumulator.
    fn deliver_selected_rows<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        output: &mut OP,
    ) -> unary::Result<()> {
        let kept = self.selection.len();
        if kept == 0 {
            return Ok(());
        }
        if kept == batch.num_rows() && batch.num_rows() >= RECORD_BATCH_SIZE / 2 {
            output.send(batch)?;
            return Ok(());
        }
        if batch.num_columns() == 0 {
            let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(kept));
            let selected = RecordBatch::try_new_with_options(batch.schema(), vec![], &options)
                .map_err(unary::Error::from)?;
            output.send(selected)?;
            return Ok(());
        }
        let accumulator = self
            .accumulator
            .get_or_insert_with(|| BatchAccumulator::new(batch.schema(), &mut self.allocator));
        accumulator.append(&batch, &self.selection);
        if accumulator.should_emit() {
            output.send(accumulator.take_batch(&mut self.allocator)?)?;
        }
        Ok(())
    }
}

impl<F> Unary<RecordBatch, RecordBatch> for Filter<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send,
{
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        output: &mut OP,
    ) -> unary::Result<()> {
        self.selection.clear();
        let selection = (self.func)(&batch, &mut self.allocator, &mut self.selection);
        if batch.num_rows() == 0 {
            return Ok(());
        }
        match selection {
            // Pass-through, so flows that depend on prompt delivery (e.g. a
            // downstream LIMIT cancelling upstream) see every batch as soon
            // as it exists.
            RowSelection::All => output.send(batch)?,
            RowSelection::Indices => self.deliver_selected_rows(batch, output)?,
        }
        Ok(())
    }

    fn finish<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        if let Some(accumulator) = &mut self.accumulator
            && !accumulator.is_empty()
        {
            output.send(accumulator.take_batch(&mut self.allocator)?)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::test_utils::run_unary_to_completion;
    use arrow_array::{ArrayRef, BooleanArray, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn filter_operator<F>(func: F) -> Filter<F>
    where
        F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send + 'static,
    {
        FilterFactory(func).build_unary()
    }

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
        init_test_free_pool(4);
        let filter = filter_operator(
            |b: &RecordBatch, _: &mut SlabAllocator, indices: &mut Vec<u32>| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let mask = BooleanArray::from_iter(col.values().iter().map(|v| Some(*v > 3)));
                collect_selected_indices(&mask, indices);
                RowSelection::Indices
            },
        );

        let out = run_unary_to_completion(filter, vec![batch(&[1, 2, 3, 4, 5])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![4, 5]);
    }

    #[test]
    fn no_matches_produces_no_output() {
        init_test_free_pool(4);
        let filter = filter_operator(
            |_: &RecordBatch, _: &mut SlabAllocator, out: &mut Vec<u32>| {
                out.clear();
                RowSelection::Indices
            },
        );

        let out = run_unary_to_completion(filter, vec![batch(&[1, 2, 3])]);

        assert_eq!(out.len(), 0);
    }

    #[test]
    fn all_match() {
        init_test_free_pool(4);
        let filter = filter_operator(|_: &RecordBatch, _: &mut SlabAllocator, _: &mut Vec<u32>| {
            RowSelection::All
        });

        let out = run_unary_to_completion(filter, vec![batch(&[10, 20, 30])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![10, 20, 30]);
    }

    #[test]
    fn multiple_batches() {
        init_test_free_pool(4);
        let filter = filter_operator(
            |b: &RecordBatch, _: &mut SlabAllocator, indices: &mut Vec<u32>| {
                let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let mask = BooleanArray::from_iter(col.values().iter().map(|v| Some(*v % 2 == 0)));
                collect_selected_indices(&mask, indices);
                RowSelection::Indices
            },
        );

        let out = run_unary_to_completion(filter, vec![batch(&[1, 2, 3]), batch(&[4, 5, 6])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![2, 4, 6]);
    }

    #[test]
    fn empty_batch() {
        init_test_free_pool(4);
        let filter = filter_operator(|_: &RecordBatch, _: &mut SlabAllocator, _: &mut Vec<u32>| {
            RowSelection::All
        });

        let out = run_unary_to_completion(filter, vec![batch(&[])]);

        assert_eq!(out.len(), 0);
    }
}
