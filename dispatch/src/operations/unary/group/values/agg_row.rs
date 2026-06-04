//! Multi-slot `COUNT(*)`/`COUNT(col)`/`SUM(col)` value extraction.
//!
//! Each of the `N` output slots accumulates into an `i64` ([`AggRow<N>`]). The
//! per-slot [`GroupAggKind`] only matters here, during extraction: a count slot
//! contributes `1`, a sum slot contributes the (widened) column value. `N` is
//! monomorphised per query arity so each hash-table entry is exactly as wide as
//! the query needs.

use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::values::{
    GroupAggKind, GroupAggSlot, ValueColumns, ValueExtractor,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::{DataType, Field};

/// A group-by aggregate value: `N` `i64` accumulator slots merged elementwise.
///
/// Every slot (count or sum) accumulates into an `i64`, so the merge and the
/// output emission are uniform — the per-slot [`GroupAggKind`] only matters
/// during the consume phase (whether a row contributes `1` or a column value).
/// `N` is monomorphised per query arity so the value is exactly as wide as the
/// query needs (it sits inline in every hash-table entry, of which there can be
/// very many, so width matters).
#[derive(Copy, Clone)]
pub struct AggRow<const N: usize>(pub [i64; N]);

impl<const N: usize> Default for AggRow<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> Value for AggRow<N> {
    #[inline]
    fn single() -> Self {
        // Not used for multi-aggregate grouping (values are built per row from
        // the slot configuration); provided only to satisfy the trait.
        Self([0; N])
    }

    #[inline]
    fn merge(mut self, v: Self) -> Self {
        for i in 0..N {
            self.0[i] += v.0[i];
        }
        self
    }
}

/// The per-row input to a single output slot: the constant `1` for a `COUNT`,
/// or the value read from a typed column for a `SUM`. One is resolved per slot
/// per batch; [`at`](SlotValueReader::at) yields that slot's contribution for a row.
/// (The running total lives in [`AggRow`], not here.)
enum SlotValueReader<'b> {
    /// `COUNT(*)` or `COUNT(non-null col)` — contributes 1.
    One,
    SumI16(&'b PrimitiveArray<Int16Type>),
    SumI32(&'b PrimitiveArray<Int32Type>),
    SumI64(&'b PrimitiveArray<Int64Type>),
}

impl SlotValueReader<'_> {
    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        match self {
            SlotValueReader::One => 1,
            SlotValueReader::SumI16(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotValueReader::SumI32(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotValueReader::SumI64(a) => unsafe { a.value_unchecked(idx) },
        }
    }
}

/// Per-batch reader: one [`SlotValueReader`] per output slot, sized to `N` so the
/// per-row [`value`](AggRowValueExtractor::value) loop unrolls.
pub struct AggRowReader<'b, const N: usize> {
    slots: [SlotValueReader<'b>; N],
}

/// A [`ValueExtractor`] producing an [`AggRow<N>`] of `N` count/sum slots.
pub struct AggRowValueExtractor<const N: usize>;

impl<const N: usize> ValueExtractor for AggRowValueExtractor<N> {
    type Value = AggRow<N>;
    type Reader<'b> = AggRowReader<'b, N>;
    type Columns = AggRowColumns<N>;

    fn make_reader<'b>(batch: &'b RecordBatch, value_slots: &[GroupAggSlot]) -> AggRowReader<'b, N> {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let slots = std::array::from_fn(|s| match value_slots[s].kind {
            GroupAggKind::CountStar | GroupAggKind::Count => SlotValueReader::One,
            GroupAggKind::Sum => {
                let col = batch.column(value_slots[s].column);
                match col.data_type() {
                    DataType::Int16 => SlotValueReader::SumI16(col.as_primitive::<Int16Type>()),
                    DataType::Int32 => SlotValueReader::SumI32(col.as_primitive::<Int32Type>()),
                    DataType::Int64 => SlotValueReader::SumI64(col.as_primitive::<Int64Type>()),
                    other => panic!("grouped SUM: unsupported column type {other:?}"),
                }
            }
        });
        AggRowReader { slots }
    }

    #[inline(always)]
    fn value(reader: &AggRowReader<'_, N>, idx: usize) -> AggRow<N> {
        let mut out = [0i64; N];
        for s in 0..N {
            out[s] = reader.slots[s].at(idx);
        }
        AggRow(out)
    }

    #[inline(always)]
    fn sort_key(value: &AggRow<N>, slot: usize) -> i64 {
        value.0[slot]
    }
}

/// Emits one `Int64` column per slot (`v0`, `v1`, …), each into an engine slab.
pub struct AggRowColumns<const N: usize> {
    cols: [PrimitiveBuilder<Int64Type>; N],
}

impl<const N: usize> ValueColumns for AggRowColumns<N> {
    type Value = AggRow<N>;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            cols: std::array::from_fn(|_| PrimitiveBuilder::with_capacity(allocator, rows)),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &AggRow<N>) {
        for s in 0..N {
            self.cols[s].push(&value.0[s], 1);
        }
    }

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, c) in self.cols.into_iter().enumerate() {
            fields.push(Field::new(format!("v{s}"), DataType::Int64, false));
            columns.push(c.into_array(None));
        }
        (fields, columns)
    }
}
