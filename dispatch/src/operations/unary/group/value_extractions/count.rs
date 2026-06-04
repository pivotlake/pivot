//! `COUNT(*)` value extraction: every row contributes 1, no column is read.

use crate::arrays::OutputPrimitiveBuilder;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::aggregations::{Count, GroupAggSlot};
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::value_extractions::{ValueColumns, ValueExtractor};
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};

/// A [`ValueExtractor`] for `COUNT(*)`: a single [`Count`] slot that increments
/// once per row, independent of any input column. Pairs with any key extractor
/// to form `GROUP BY key, COUNT(*)`.
pub struct CountValueExtractor;

impl ValueExtractor for CountValueExtractor {
    type Value = Count;
    type Reader<'b> = ();
    type Columns = CountColumns;

    #[inline(always)]
    fn make_reader<'b>(_batch: &'b RecordBatch, _value_slots: &[GroupAggSlot]) {}

    #[inline(always)]
    fn value(_reader: &(), _idx: usize) -> Count {
        Count::single()
    }

    #[inline(always)]
    fn sort_key(value: &Count, _slot: usize) -> i64 {
        value.value as i64
    }
}

/// Emits the single `UInt64` count column into an engine slab buffer.
pub struct CountColumns(OutputPrimitiveBuilder<UInt64Type>);

impl ValueColumns for CountColumns {
    type Value = Count;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self(OutputPrimitiveBuilder::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &Count) {
        self.0.push(value.value as u64);
    }

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![Field::new("value", DataType::UInt64, false)];
        (fields, vec![self.0.into_array()])
    }
}
