//! The per-batch value reader, shared by every aggregation value type.
//!
//! Reading a row is the same for all of them — a `COUNT` slot contributes `1`, a
//! `SUM`/`MIN`/`MAX` slot the (widened) column value — they differ only in how
//! the read cells later *combine*. So the read lives here once; the value types
//! ([`Mono`](super::Mono) etc.) call [`RowReader::read`] and wrap the result.

use super::accumulator::Accumulator;
use super::aggregate::{Aggregate, Count, Sum};
use super::row::AggregationRow;
use super::{AggregationKind, AggregationSlot};
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;

/// How one slot reads its per-row contribution, resolved once (from the slot kind
/// and column type) into a small enum dispatched per row. The actual per-row logic
/// lives in the [`Count`]/[`Sum`] ops; this just selects which.
enum SlotReader<'b> {
    Count,
    Sum16(<Sum<Int16Type> as Aggregate>::Reader<'b>),
    Sum32(<Sum<Int32Type> as Aggregate>::Reader<'b>),
    Sum64(<Sum<Int64Type> as Aggregate>::Reader<'b>),
}

impl SlotReader<'_> {
    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        match self {
            SlotReader::Count => Count::contribution(&(), idx),
            SlotReader::Sum16(r) => Sum::<Int16Type>::contribution(r, idx),
            SlotReader::Sum32(r) => Sum::<Int32Type>::contribution(r, idx),
            SlotReader::Sum64(r) => Sum::<Int64Type>::contribution(r, idx),
        }
    }
}

/// One [`SlotReader`] per output slot, sized to `N` so [`read`](RowReader::read)
/// unrolls.
pub struct RowReader<'b, const N: usize> {
    slots: [SlotReader<'b>; N],
}

impl<'b, const N: usize> RowReader<'b, N> {
    /// Bind `batch`'s value columns for the configured `slots`.
    pub fn new(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self {
        assert_eq!(slots.len(), N, "slot count must match N");
        let slots = std::array::from_fn(|s| {
            let slot = slots[s];
            match slot.kind {
                AggregationKind::CountStar | AggregationKind::Count => SlotReader::Count,
                // SUM/MIN/MAX all read the same per-row value (the widened column
                // value); only how they combine differs, which is the value type's
                // concern, not the reader's.
                AggregationKind::Sum | AggregationKind::Min | AggregationKind::Max => {
                    match batch.column(slot.column).data_type() {
                        DataType::Int16 => {
                            SlotReader::Sum16(Sum::<Int16Type>::make_reader(batch, slot.column))
                        }
                        DataType::Int32 => {
                            SlotReader::Sum32(Sum::<Int32Type>::make_reader(batch, slot.column))
                        }
                        DataType::Int64 => {
                            SlotReader::Sum64(Sum::<Int64Type>::make_reader(batch, slot.column))
                        }
                        other => panic!("grouped SUM/MIN/MAX: unsupported column type {other:?}"),
                    }
                }
            }
        });
        RowReader { slots }
    }

    /// Read row `idx` into a fresh accumulator row of width `A`.
    #[inline(always)]
    pub fn read<A: Accumulator>(&self, idx: usize) -> AggregationRow<N, A> {
        AggregationRow(std::array::from_fn(|s| A::from(self.slots[s].at(idx))))
    }
}
