//! Filter operator: keeps rows the filter closure selects.
//!
//! The filter closure receives the input batch and writes the positions of
//! the surviving rows into a scratch vec, or reports that every row survives
//! ([`RowSelection`]). The closure may evaluate several conditions
//! progressively, shrinking a working copy of the batch after each one so
//! later (often costlier) conditions only see rows the earlier ones kept; the
//! operator's [`SlabAllocator`] is passed in for that, see
//! [`take`](crate::arrays::take).
//! A selection is applied by appending the surviving rows to a
//! [`BatchAccumulator`], which coalesces survivors across input batches and
//! emits full-size batches, so a selective filter's downstream sees a few
//! large batches instead of a runt batch per input. Whether to coalesce at all
//! is the consumer's call, carried as a [`RowDelivery`]: an operator that acts
//! on early rows needs them as soon as they are selected, and for it each
//! input batch's survivors are gathered out of that batch and sent at once.

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::BatchAccumulator;
use crate::arrays::take;
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

/// When a filter hands its surviving rows to the operator below it.
///
/// The choice belongs to that operator, not to the filter: it is about
/// whether anything downstream can act on rows before the input ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowDelivery {
    /// Coalesce survivors across input batches and emit full-size batches, so
    /// the consumer pays its per-batch costs once per full batch rather than
    /// once per runt. For an operator that must see the whole input anyway (a
    /// group-by, a join, a sort), holding rows back costs nothing.
    Coalesced,
    /// Emit each input batch's survivors as soon as they are selected. An
    /// operator that acts on early rows needs them early: a LIMIT cancels its
    /// input once it has enough, and a Top-N publishes the boundary that
    /// prunes row groups from the scan. Rows held back to fill a batch delay
    /// that decision, and under a selective filter the delay is long enough to
    /// read far more of the table than the query needs.
    Immediate,
}

/// Factory that wraps a filter closure. `F` is the per-worker closure created by the
/// builder passed to [`RecordBatchOperatorSpec::filter`](crate::api::RecordBatchOperatorSpec::filter).
pub struct FilterFactory<F>(pub F, pub RowDelivery);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FilterFactory<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send + 'static,
{
    type Unary = Filter<F>;

    fn build_unary(self) -> Self::Unary {
        Filter {
            func: self.0,
            delivery: self.1,
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
    /// Whether surviving rows wait for a full batch or go downstream at once.
    delivery: RowDelivery,
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
    /// surviving batches directly, gather partial ones out of the batch when
    /// delivery is immediate, and coalesce them through the accumulator
    /// otherwise.
    fn deliver_selected_rows(
        &mut self,
        batch: RecordBatch,
        output: &mut dyn Sender<RecordBatch>,
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
        // An immediately delivered batch is consumed as soon as it is sent, so
        // its survivors are gathered straight out of the input: the gather is
        // sized to the selection and a view column keeps sharing the input's
        // data buffers, which are released moments later regardless. The
        // accumulator would decide per batch whether to copy the values, a
        // cost that only pays off for an accumulation that outlives its
        // inputs, and would hand out full-capacity slabs for a batch of a few
        // rows.
        if self.delivery == RowDelivery::Immediate {
            let columns = batch
                .columns()
                .iter()
                .map(|column| take(&mut self.allocator, column, &self.selection))
                .collect::<Result<Vec<_>, _>>()
                .map_err(unary::Error::from)?;
            let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(kept));
            let selected = RecordBatch::try_new_with_options(batch.schema(), columns, &options)
                .map_err(unary::Error::from)?;
            output.send(selected)?;
            return Ok(());
        }
        // The accumulator coalesces into a fixed schema, but consecutive batches
        // need not share one: a variant column shredded differently file to file
        // arrives as a different struct type each time. When the schema changes,
        // flush what is held and start a fresh accumulator, so unlike batches are
        // emitted separately rather than concatenated into a type mismatch.
        if let Some(accumulator) = &mut self.accumulator
            && accumulator.schema() != &batch.schema()
        {
            if !accumulator.is_empty() {
                output.send(accumulator.take_batch(&mut self.allocator)?)?;
            }
            self.accumulator = None;
        }
        let accumulator = self
            .accumulator
            .get_or_insert_with(|| BatchAccumulator::new(batch.schema(), &mut self.allocator));
        accumulator.append_batch_by_indices_flushing::<unary::Error>(
            &batch,
            &self.selection,
            &mut self.allocator,
            &mut |full| output.send(full).map_err(Into::into),
        )?;
        Ok(())
    }
}

impl<F> Unary<RecordBatch, RecordBatch> for Filter<F>
where
    F: FnMut(&RecordBatch, &mut SlabAllocator, &mut Vec<u32>) -> RowSelection + Send,
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        output: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
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

    fn finish(&mut self, output: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
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
        FilterFactory(func, RowDelivery::Coalesced).build_unary()
    }

    fn keep_first_row(
        _: &RecordBatch,
        _: &mut SlabAllocator,
        indices: &mut Vec<u32>,
    ) -> RowSelection {
        indices.clear();
        indices.push(0);
        RowSelection::Indices
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
    fn oversized_input_batch_survivors_all_emitted() {
        // An input batch far larger than the accumulator's capacity (join and
        // group-by outputs can exceed one output batch) must flow through in
        // full: the accumulator splits the append itself, flushing as it
        // fills, instead of writing past its slabs.
        init_test_free_pool(16);
        let n = 40_000i32;
        let values: Vec<i32> = (0..n).collect();
        let filter = filter_operator(
            |b: &RecordBatch, _: &mut SlabAllocator, indices: &mut Vec<u32>| {
                indices.clear();
                // Keep all but every 1000th row, so the survivors are a
                // gathered (indices) selection rather than a pass-through.
                indices.extend((0..b.num_rows() as u32).filter(|i| i % 1000 != 0));
                RowSelection::Indices
            },
        );

        let out = run_unary_to_completion(filter, vec![batch(&values)]);

        let got: Vec<i32> = out.iter().flat_map(i32_col).collect();
        let expected: Vec<i32> = (0..n).filter(|v| v % 1000 != 0).collect();
        assert_eq!(got, expected);
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

    /// A one-column batch whose column is a struct of `fields`, all rows zero.
    /// Stands in for a variant column, which is a struct whose shredded shape
    /// (and so arrow type) differs file to file.
    fn struct_batch(fields: Vec<Field>) -> RecordBatch {
        use arrow_array::StructArray;
        use arrow_schema::Fields;
        let arrays: Vec<ArrayRef> = fields
            .iter()
            .map(|_| Arc::new(Int32Array::from(vec![0, 0, 0])) as ArrayRef)
            .collect();
        let fields = Fields::from(fields);
        let column = Arc::new(StructArray::new(fields.clone(), arrays, None)) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Struct(fields),
            false,
        )]));
        RecordBatch::try_new(schema, vec![column]).unwrap()
    }

    #[test]
    fn survivors_of_unlike_schema_are_not_coalesced() {
        // Partially-selected batches of different struct shape must not be
        // coalesced into one accumulation: their columns are of different types
        // and would fail to concatenate. This is what a variant column shredded
        // differently across files produces.
        init_test_free_pool(4);
        let one = struct_batch(vec![Field::new("a", DataType::Int32, true)]);
        let two = struct_batch(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]);

        // Keep one row from each, so both take the coalescing (accumulator) path
        // rather than passing through whole.
        let filter = filter_operator(
            |_: &RecordBatch, _: &mut SlabAllocator, indices: &mut Vec<u32>| {
                indices.clear();
                indices.push(0);
                RowSelection::Indices
            },
        );

        let out = run_unary_to_completion(filter, vec![one, two]);

        assert_eq!(out.len(), 2, "unlike batches come out separately");
        assert_eq!(out[0].num_rows(), 1);
        assert_eq!(out[1].num_rows(), 1);
        assert_ne!(out[0].schema(), out[1].schema());
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

    /// A downstream LIMIT or Top-N acts on the rows it has been handed, so a
    /// selective filter under one must not sit on them waiting for a full
    /// batch: it emits per input batch.
    #[test]
    fn immediate_delivery_emits_per_input_batch() {
        init_test_free_pool(4);
        let filter = FilterFactory(keep_first_row, RowDelivery::Immediate).build_unary();

        let out = run_unary_to_completion(filter, vec![batch(&[1, 2, 3]), batch(&[4, 5, 6])]);

        assert_eq!(out.len(), 2);
        assert_eq!(i32_col(&out[0]), vec![1]);
        assert_eq!(i32_col(&out[1]), vec![4]);
    }

    /// An immediately delivered batch is consumed at once, so its strings keep
    /// pointing at the input's data buffers instead of being copied.
    #[test]
    fn immediate_delivery_shares_the_input_buffers() {
        use arrow_array::StringViewArray;
        use arrow_array::cast::AsArray;
        init_test_free_pool(4);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let col: ArrayRef = Arc::new(StringViewArray::from(vec![
            "a value longer than twelve bytes";
            20
        ]));
        let input = RecordBatch::try_new(schema, vec![col]).unwrap();
        let source_buffer = input.column(0).as_string_view().data_buffers()[0].as_ptr();
        let filter = FilterFactory(keep_first_row, RowDelivery::Immediate).build_unary();

        let out = run_unary_to_completion(filter, vec![input]);

        let emitted = out[0].column(0).as_string_view();
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted.data_buffers()[0].as_ptr(), source_buffer);
    }

    /// Immediate delivery never coalesces, so batches of unlike schema come
    /// out one by one without any flush between them.
    #[test]
    fn immediate_delivery_emits_unlike_schemas_separately() {
        init_test_free_pool(4);
        let one = struct_batch(vec![Field::new("a", DataType::Int32, true)]);
        let two = struct_batch(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]);
        let filter = FilterFactory(keep_first_row, RowDelivery::Immediate).build_unary();

        let out = run_unary_to_completion(filter, vec![one, two]);

        assert_eq!(out.len(), 2);
        assert_ne!(out[0].schema(), out[1].schema());
    }

    /// The same filter under a group-by or a join hands over one coalesced
    /// batch instead, since nothing downstream can act before the input ends.
    #[test]
    fn coalesced_delivery_emits_one_batch() {
        init_test_free_pool(4);
        let filter = FilterFactory(keep_first_row, RowDelivery::Coalesced).build_unary();

        let out = run_unary_to_completion(filter, vec![batch(&[1, 2, 3]), batch(&[4, 5, 6])]);

        assert_eq!(out.len(), 1);
        assert_eq!(i32_col(&out[0]), vec![1, 4]);
    }

    /// A filter keeping a row per input batch must not hand downstream a batch
    /// that references every input's data buffers: that would keep each
    /// source's decompressed page alive for as long as the coalesced batch is.
    #[test]
    fn a_selective_filter_emits_no_source_buffer() {
        use arrow_array::StringViewArray;
        use arrow_array::cast::AsArray;
        init_test_free_pool(4);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let string_batch = |value: &str| {
            let col: ArrayRef = Arc::new(StringViewArray::from(vec![value; 20]));
            RecordBatch::try_new(schema.clone(), vec![col]).unwrap()
        };
        let inputs = vec![
            string_batch("a value longer than twelve bytes"),
            string_batch("another value longer than twelve"),
        ];
        let source_buffers: Vec<*const u8> = inputs
            .iter()
            .flat_map(|batch| batch.column(0).as_string_view().data_buffers().to_vec())
            .map(|buffer| buffer.as_ptr())
            .collect();
        let filter = filter_operator(keep_first_row);

        let out = run_unary_to_completion(filter, inputs);

        assert_eq!(out.len(), 1);
        let emitted = out[0].column(0).as_string_view();
        assert_eq!(emitted.len(), 2);
        assert!(
            emitted
                .data_buffers()
                .iter()
                .all(|buffer| !source_buffers.contains(&buffer.as_ptr()))
        );
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
