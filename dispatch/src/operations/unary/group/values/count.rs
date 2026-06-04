//! `COUNT(*)` value extraction: every row contributes 1, no column is read.

use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::values::{GroupAggSlot, ValueColumns, ValueExtractor};
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};

/// A simple counting aggregation that sits inline in a hash-table entry and
/// holds the running count for a key.
#[derive(Default, Copy, Clone)]
pub struct Count {
    pub value: usize,
}

impl Count {
    /// Create a count with a specific initial value.
    pub fn new(size: usize) -> Self {
        Self { value: size }
    }
}

impl Value for Count {
    #[inline]
    fn single() -> Self {
        Self { value: 1 }
    }

    fn merge(mut self, v: Self) -> Self {
        self.value += v.value;
        self
    }
}

/// A [`ValueExtractor`] for `COUNT(*)`: a single [`Count`] slot that increments
/// once per row, independent of any input column. Pairs with any key extractor
/// to form `GROUP BY key, COUNT(*)`.
pub struct CountValueExtractor;

impl ValueExtractor for CountValueExtractor {
    type Value = Count;
    type Reader<'b> = ();
    type Columns = CountColumn;

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
pub struct CountColumn(PrimitiveBuilder<UInt64Type>);

impl ValueColumns for CountColumn {
    type Value = Count;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self(PrimitiveBuilder::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &Count) {
        self.0.push(&(value.value as u64), 1);
    }

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![Field::new("value", DataType::UInt64, false)];
        (fields, vec![self.0.into_array(None)])
    }
}
