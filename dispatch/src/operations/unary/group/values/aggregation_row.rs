//! The runtime-dispatched value extractor — the fallback for aggregate
//! signatures we haven't specialised. (See [`compiled`](super::compiled) for the
//! monomorphised path; the planner sends a query here only when no compiled
//! shape matches.)
//!
//! Each of the `N` output slots accumulates into an [`Accumulator`] `A`
//! ([`AggregationRow<N, A>`]) — `i64` by default, `i128` when a sum's column is
//! wide enough to overflow `i64`. A per-slot enum resolves the per-row
//! contribution at runtime from the slot's [`AggregationKind`] / column type: a
//! count slot contributes `1`, a sum slot the (widened) column value. `N` is
//! monomorphised per query arity so each hash-table entry is exactly as wide as
//! the query needs.

use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::values::accumulator::Accumulator;
use crate::operations::unary::group::values::aggregate::{Aggregate, Count, Sum};
use crate::operations::unary::group::values::{
    AggregationKind, AggregationSlot, ValueColumns, ValueExtractor,
};
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};

/// A group-by aggregate value: `N` accumulator slots merged elementwise.
///
/// Every slot (count or sum) accumulates into the same width `A`, so the merge
/// and the output emission are uniform — the per-slot [`AggregationKind`] only
/// matters during the consume phase (whether a row contributes `1` or a column
/// value). `N` is monomorphised per query arity and `A` per width so the value
/// is exactly as wide as the query needs (it sits inline in every hash-table
/// entry, of which there can be very many, so width matters).
pub struct AggregationRow<const N: usize, A: Accumulator = i64>(pub [A; N]);

impl<const N: usize, A: Accumulator> Copy for AggregationRow<N, A> {}
impl<const N: usize, A: Accumulator> Clone for AggregationRow<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize, A: Accumulator> Default for AggregationRow<N, A> {
    fn default() -> Self {
        Self([A::default(); N])
    }
}

impl<const N: usize, A: Accumulator> Value for AggregationRow<N, A> {
    #[inline]
    fn merge(mut self, v: Self) -> Self {
        for i in 0..N {
            self.0[i] += v.0[i];
        }
        self
    }
}

/// The per-row input to a single output slot, resolved at runtime from the slot
/// kind / column type. This is the fallback's dispatch table over the shared
/// [`Aggregate`] ops — the actual per-row logic lives in [`Count`]/[`Sum`], and
/// [`at`](SlotValueReader::at) just delegates. (The running total lives in
/// [`AggregationRow`], not here.)
enum SlotValueReader<'b> {
    Count,
    Sum16(<Sum<Int16Type> as Aggregate>::Reader<'b>),
    Sum32(<Sum<Int32Type> as Aggregate>::Reader<'b>),
    Sum64(<Sum<Int64Type> as Aggregate>::Reader<'b>),
}

impl SlotValueReader<'_> {
    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        match self {
            SlotValueReader::Count => Count::contribution(&(), idx),
            SlotValueReader::Sum16(r) => Sum::<Int16Type>::contribution(r, idx),
            SlotValueReader::Sum32(r) => Sum::<Int32Type>::contribution(r, idx),
            SlotValueReader::Sum64(r) => Sum::<Int64Type>::contribution(r, idx),
        }
    }
}

/// Per-batch reader: one [`SlotValueReader`] per output slot, sized to `N` so the
/// per-row [`value`](AggregationRowValueExtractor::value) loop unrolls.
pub struct AggregationRowReader<'b, const N: usize> {
    slots: [SlotValueReader<'b>; N],
}

/// A [`ValueExtractor`] producing an [`AggregationRow<N, A>`] of `N` count/sum slots.
pub struct AggregationRowValueExtractor<const N: usize, A: Accumulator = i64>(
    std::marker::PhantomData<A>,
);

impl<const N: usize, A: Accumulator> ValueExtractor for AggregationRowValueExtractor<N, A> {
    type Value = AggregationRow<N, A>;
    type Reader<'b> = AggregationRowReader<'b, N>;
    type Columns = AggregationRowColumns<N, A>;
    type SortKey = A;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        value_slots: &[AggregationSlot],
    ) -> AggregationRowReader<'b, N> {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let slots = std::array::from_fn(|s| {
            let slot = value_slots[s];
            match slot.kind {
                AggregationKind::CountStar | AggregationKind::Count => SlotValueReader::Count,
                AggregationKind::Sum => match batch.column(slot.column).data_type() {
                    DataType::Int16 => {
                        SlotValueReader::Sum16(Sum::<Int16Type>::make_reader(batch, slot.column))
                    }
                    DataType::Int32 => {
                        SlotValueReader::Sum32(Sum::<Int32Type>::make_reader(batch, slot.column))
                    }
                    DataType::Int64 => {
                        SlotValueReader::Sum64(Sum::<Int64Type>::make_reader(batch, slot.column))
                    }
                    other => panic!("grouped SUM: unsupported column type {other:?}"),
                },
            }
        });
        AggregationRowReader { slots }
    }

    #[inline(always)]
    fn value(reader: &AggregationRowReader<'_, N>, idx: usize) -> AggregationRow<N, A> {
        AggregationRow(std::array::from_fn(|s| A::from(reader.slots[s].at(idx))))
    }

    #[inline(always)]
    fn sort_key(value: &AggregationRow<N, A>, slot: usize) -> A {
        value.0[slot]
    }
}

/// Emits one column per slot (`v0`, `v1`, …) of the accumulator's Arrow type
/// (`Int64` for `i64`, `Decimal128(38, 0)` for `i128`), each into an engine slab.
pub struct AggregationRowColumns<const N: usize, A: Accumulator = i64> {
    cols: [PrimitiveBuilder<A::Arrow>; N],
}

impl<const N: usize, A: Accumulator> ValueColumns for AggregationRowColumns<N, A> {
    type Value = AggregationRow<N, A>;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            cols: std::array::from_fn(|_| PrimitiveBuilder::with_capacity(allocator, rows)),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &AggregationRow<N, A>) {
        for s in 0..N {
            self.cols[s].push(&value.0[s], 1);
        }
    }

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, c) in self.cols.into_iter().enumerate() {
            let array = A::finalize(c.into_array(None));
            fields.push(Field::new(format!("v{s}"), array.data_type().clone(), false));
            columns.push(array);
        }
        (fields, columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use arrow_schema::Schema;
    use std::sync::Arc;

    /// A grouped `SUM` over a 64-bit column whose group total exceeds `i64::MAX`
    /// must accumulate in `i128` (the wide path) without overflow — the bug the
    /// old fixed-`i64` `AggregationRow` had. Three `i64::MAX` values sum to past
    /// `i64::MAX`.
    #[test]
    fn wide_grouped_sum_uses_i128_without_overflow() {
        let col = Int64Array::from(vec![i64::MAX, i64::MAX, i64::MAX]);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("u", DataType::Int64, false)])),
            vec![Arc::new(col)],
        )
        .unwrap();
        let slots = vec![AggregationSlot::new(AggregationKind::Sum, 0)];

        type V = AggregationRowValueExtractor<1, i128>;
        let reader = V::make_reader(&batch, &slots);
        let mut acc = AggregationRow::<1, i128>::default();
        for i in 0..3 {
            acc = acc.merge(V::value(&reader, i));
        }
        assert_eq!(acc.0[0], 3 * i64::MAX as i128);
    }
}
