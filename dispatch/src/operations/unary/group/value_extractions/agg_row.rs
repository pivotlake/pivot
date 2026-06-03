//! Multi-slot `COUNT(*)`/`COUNT(col)`/`SUM(col)` value extraction.
//!
//! Each of the `N` output slots accumulates into an `i64` ([`AggRow<N>`]). The
//! per-slot [`GroupAggKind`] only matters here, during extraction: a count slot
//! contributes `1`, a sum slot contributes the (widened) column value. `N` is
//! monomorphised per query arity so each hash-table entry is exactly as wide as
//! the query needs.

use crate::operations::unary::group::aggregations::{AggRow, GroupAggKind, GroupAggSlot};
use crate::operations::unary::group::value_extractions::{ValueColumns, ValueExtractor};
use arrow_array::builder::Int64Builder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// Per-slot reader: how to produce slot `s`'s contribution for a given row.
enum SlotReader<'b> {
    /// `COUNT(*)` or `COUNT(non-null col)` — contributes 1.
    One,
    SumI16(&'b PrimitiveArray<Int16Type>),
    SumI32(&'b PrimitiveArray<Int32Type>),
    SumI64(&'b PrimitiveArray<Int64Type>),
}

impl SlotReader<'_> {
    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        match self {
            SlotReader::One => 1,
            SlotReader::SumI16(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotReader::SumI32(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotReader::SumI64(a) => unsafe { a.value_unchecked(idx) },
        }
    }
}

/// Per-batch reader: one [`SlotReader`] per output slot.
pub struct AggRowReader<'b> {
    slots: Vec<SlotReader<'b>>,
}

/// A [`ValueExtractor`] producing an [`AggRow<N>`] of `N` count/sum slots.
pub struct AggRowValueExtractor<const N: usize>;

impl<const N: usize> ValueExtractor for AggRowValueExtractor<N> {
    type Value = AggRow<N>;
    type Reader<'b> = AggRowReader<'b>;
    type Columns = AggRowColumns<N>;

    fn make_reader<'b>(batch: &'b RecordBatch, value_slots: &[GroupAggSlot]) -> AggRowReader<'b> {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let slots = value_slots
            .iter()
            .map(|slot| match slot.kind {
                GroupAggKind::CountStar | GroupAggKind::Count => SlotReader::One,
                GroupAggKind::Sum => {
                    let col = batch.column(slot.column);
                    match col.data_type() {
                        DataType::Int16 => SlotReader::SumI16(col.as_primitive::<Int16Type>()),
                        DataType::Int32 => SlotReader::SumI32(col.as_primitive::<Int32Type>()),
                        DataType::Int64 => SlotReader::SumI64(col.as_primitive::<Int64Type>()),
                        other => panic!("grouped SUM: unsupported column type {other:?}"),
                    }
                }
            })
            .collect();
        AggRowReader { slots }
    }

    #[inline(always)]
    fn value(reader: &AggRowReader<'_>, idx: usize) -> AggRow<N> {
        let mut out = [0i64; N];
        for (s, slot) in reader.slots.iter().enumerate() {
            out[s] = slot.at(idx);
        }
        AggRow(out)
    }

    #[inline(always)]
    fn sort_key(value: &AggRow<N>, slot: usize) -> i64 {
        value.0[slot]
    }
}

/// Emits one `Int64` column per slot (`v0`, `v1`, …).
pub struct AggRowColumns<const N: usize> {
    builders: Vec<Int64Builder>,
}

impl<const N: usize> ValueColumns for AggRowColumns<N> {
    type Value = AggRow<N>;

    fn with_capacity(rows: usize) -> Self {
        Self {
            builders: (0..N).map(|_| Int64Builder::with_capacity(rows)).collect(),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &AggRow<N>) {
        for (b, &v) in self.builders.iter_mut().zip(value.0.iter()) {
            b.append_value(v);
        }
    }

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, mut b) in self.builders.into_iter().enumerate() {
            fields.push(Field::new(format!("v{s}"), DataType::Int64, false));
            columns.push(Arc::new(b.finish()));
        }
        (fields, columns)
    }
}
