//! [`DynamicMixed`] — the fallback aggregation value for a heterogeneous
//! signature (e.g. `COUNT(*), MIN(x)`), folding each slot on its runtime kind.

use super::accumulator::Accumulator;
use super::columns::RowColumns;
use super::reader::RowReader;
use super::row::AggregationRow;
use super::{AggregationSlot, AggregationValue};
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// `N` cells of width `A`, each folded by its own slot kind. Used when the
/// signature mixes additive and extreme slots, so there's no single [`CellFold`]
/// — the per-slot kinds ride in the [`MergeConfig`](AggregationValue::MergeConfig)
/// and [`merge`](AggregationValue::merge) dispatches on them.
///
/// [`CellFold`]: super::CellFold
pub struct DynamicMixed<const N: usize, A: Accumulator = i64> {
    row: AggregationRow<N, A>,
}

impl<const N: usize, A: Accumulator> Copy for DynamicMixed<N, A> {}
impl<const N: usize, A: Accumulator> Clone for DynamicMixed<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Accumulator> Default for DynamicMixed<N, A> {
    fn default() -> Self {
        Self {
            row: AggregationRow::default(),
        }
    }
}

impl<const N: usize, A: Accumulator> AggregationValue for DynamicMixed<N, A> {
    type Reader<'b> = RowReader<'b, N>;
    type MergeConfig = Arc<[AggregationSlot]>;
    type Columns = RowColumns<N, A>;
    type SortKey = A;

    fn merge_config(slots: &[AggregationSlot]) -> Arc<[AggregationSlot]> {
        Arc::from(slots)
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> RowReader<'b, N> {
        RowReader::new(batch, slots)
    }

    #[inline(always)]
    fn value(reader: &RowReader<'_, N>, idx: usize) -> Self {
        Self {
            row: reader.read(idx),
        }
    }

    #[inline(always)]
    fn merge(self, other: Self, slots: &Arc<[AggregationSlot]>) -> Self {
        Self {
            row: AggregationRow(std::array::from_fn(|s| {
                slots[s].kind.combine(self.row.0[s], other.row.0[s])
            })),
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> A {
        self.row.0[slot]
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> RowColumns<N, A> {
        RowColumns::with_capacity(allocator, rows)
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut RowColumns<N, A>) {
        cols.push_row(&self.row);
    }

    fn finish_columns(cols: RowColumns<N, A>) -> (Vec<Field>, Vec<ArrayRef>) {
        cols.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::group::values::{AggregationKind, AggregationSlot};
    use arrow_array::{Int32Array, Int64Array};
    use arrow_schema::{DataType, Schema};

    /// A heterogeneous signature — COUNT(*) + MIN(c) — folds each slot by its kind.
    #[test]
    fn mixed_count_and_min() {
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![5i32, 2, 9]))],
        )
        .unwrap();
        let slots: Arc<[AggregationSlot]> = Arc::from(
            [
                AggregationSlot::new(AggregationKind::CountStar, 0),
                AggregationSlot::new(AggregationKind::Min, 0),
            ]
            .as_slice(),
        );
        let r = DynamicMixed::<2>::make_reader(&b, &slots);
        let mut acc = DynamicMixed::<2>::value(&r, 0);
        for i in 1..3 {
            acc = acc.update_from_reader(&r, i, &slots);
        }
        assert_eq!(acc.sort_key(0), 3); // count
        assert_eq!(acc.sort_key(1), 2); // min
    }

    /// A SUM over a 64-bit column whose total exceeds `i64::MAX` accumulates in i128.
    #[test]
    fn wide_sum_uses_i128() {
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("u", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![
                i64::MAX,
                i64::MAX,
                i64::MAX,
            ]))],
        )
        .unwrap();
        let slots: Arc<[AggregationSlot]> =
            Arc::from([AggregationSlot::new(AggregationKind::Sum, 0)].as_slice());
        let r = DynamicMixed::<1, i128>::make_reader(&b, &slots);
        let acc = DynamicMixed::<1, i128>::value(&r, 0)
            .merge(DynamicMixed::<1, i128>::value(&r, 1), &slots)
            .merge(DynamicMixed::<1, i128>::value(&r, 2), &slots);
        assert_eq!(acc.sort_key(0), 3 * i64::MAX as i128);
    }
}
