//! `COUNT(*)` value extraction: every row contributes 1, no column is read.

use crate::operations::unary::group::aggregations::{Count, GroupAggSlot};
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::value_extractions::{ValueColumns, ValueExtractor};
use arrow_array::builder::UInt64Builder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

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

/// Emits the single `UInt64` count column.
pub struct CountColumns(UInt64Builder);

impl ValueColumns for CountColumns {
    type Value = Count;

    fn with_capacity(rows: usize) -> Self {
        Self(UInt64Builder::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &Count) {
        self.0.append_value(value.value as u64);
    }

    fn finish(mut self) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![Field::new("value", DataType::UInt64, false)];
        let columns: Vec<ArrayRef> = vec![Arc::new(self.0.finish())];
        (fields, columns)
    }
}
