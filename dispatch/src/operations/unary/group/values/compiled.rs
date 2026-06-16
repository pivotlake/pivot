//! [`CompiledMixed`] — a fixed aggregate signature monomorphised over a tuple of
//! [`Aggregate`] ops, so reading and folding are straight-line with no per-row
//! dispatch.
//!
//! Adding capability is cheap: a new aggregate is one [`Aggregate`] impl; a new
//! query shape is one tuple type (e.g. `CompiledMixed<(Count, Sum<Int16Type>), 2>`)
//! selected in the planner. Anything not specialised falls back to
//! [`Mono`](super::Mono) / [`DynamicMixed`](super::DynamicMixed).

use super::aggregate::Aggregate;
use super::cell::Cell;
use super::columns::RowColumns;
use super::row::AggregationRow;
use super::{AggregationSlot, AggregationValue};
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;

/// `N` cells of width `A` filled and folded by a fixed op tuple `Ops` (`N` is the
/// tuple arity; stable Rust can't derive it, so the planner passes both).
pub struct CompiledMixed<Ops, const N: usize, A: Cell = i64> {
    row: AggregationRow<N, A>,
    _ops: PhantomData<Ops>,
}

impl<Ops, const N: usize, A: Cell> Copy for CompiledMixed<Ops, N, A> {}
impl<Ops, const N: usize, A: Cell> Clone for CompiledMixed<Ops, N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Ops, const N: usize, A: Cell> Default for CompiledMixed<Ops, N, A> {
    fn default() -> Self {
        Self {
            row: AggregationRow::default(),
            _ops: PhantomData,
        }
    }
}

/// Implements [`AggregationValue`] for a `CompiledMixed<(Op0, Op1, …), N, A>` of a
/// given arity. `$idx` are the tuple field indices, which also index the slots.
macro_rules! impl_compiled {
    ($n:literal; $($Op:ident $idx:tt),+) => {
        impl<Acc: Cell, $($Op: Aggregate + Send + Sync + 'static),+> AggregationValue
            for CompiledMixed<($($Op,)+), $n, Acc>
        {
            type Reader<'b> = ($($Op::Reader<'b>,)+);
            type MergeConfig = ();
            type Columns = RowColumns<$n, Acc>;
            type SortKey = Acc;

            #[inline(always)]
            fn merge_config(_slots: &[AggregationSlot]) {}

            #[inline(always)]
            fn make_reader<'b>(
                batch: &'b RecordBatch,
                slots: &[AggregationSlot],
            ) -> Self::Reader<'b> {
                assert_eq!(slots.len(), $n, "slot count must match compiled arity");
                ($( $Op::make_reader(batch, slots[$idx].column), )+)
            }

            #[inline(always)]
            fn value(reader: &Self::Reader<'_>, idx: usize) -> Self {
                Self {
                    row: AggregationRow([$( Acc::from($Op::contribution(&reader.$idx, idx)), )+]),
                    _ops: PhantomData,
                }
            }

            #[inline(always)]
            fn merge(self, other: Self, _cfg: &()) -> Self {
                // Each op's KIND is a const, so the combine folds to a straight
                // `+`/`min`/`max` per slot — no runtime branch.
                Self {
                    row: AggregationRow([$(
                        $Op::KIND.combine(self.row.0[$idx], other.row.0[$idx]),
                    )+]),
                    _ops: PhantomData,
                }
            }

            #[inline(always)]
            fn sort_key(&self, slot: usize) -> Acc {
                self.row.0[slot]
            }

            fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> RowColumns<$n, Acc> {
                RowColumns::with_capacity(allocator, rows)
            }

            #[inline(always)]
            fn push_to(&self, cols: &mut RowColumns<$n, Acc>) {
                cols.push_row(&self.row);
            }

            fn finish_columns(cols: RowColumns<$n, Acc>) -> (Vec<Field>, Vec<ArrayRef>) {
                cols.finish()
            }
        }
    };
}

impl_compiled!(1; A 0);
impl_compiled!(2; A 0, B 1);
impl_compiled!(3; A 0, B 1, C 2);
impl_compiled!(4; A 0, B 1, C 2, D 3);
impl_compiled!(5; A 0, B 1, C 2, D 3, E 4);
impl_compiled!(6; A 0, B 1, C 2, D 3, E 4, F 5);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::group::values::aggregate::{Count, Sum};
    use crate::operations::unary::group::values::{AggregationKind, AggregationSlot};
    use arrow_array::Int16Array;
    use arrow_array::types::Int16Type;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    /// `COUNT(*), SUM(i16), SUM(i16), COUNT` (q32's shape) reads straight-line and
    /// folds elementwise additively.
    #[test]
    fn count_sum_sum_count() {
        let a = Int16Array::from(vec![1i16, 2, 3]);
        let b = Int16Array::from(vec![10i16, 20, 30]);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int16, false),
            Field::new("b", DataType::Int16, false),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)]).unwrap();
        let slots = vec![
            AggregationSlot::new(AggregationKind::CountStar, 0),
            AggregationSlot::new(AggregationKind::Sum, 0),
            AggregationSlot::new(AggregationKind::Sum, 1),
            AggregationSlot::new(AggregationKind::Count, 0),
        ];

        type V = CompiledMixed<(Count, Sum<Int16Type>, Sum<Int16Type>, Count), 4>;
        let reader = V::make_reader(&batch, &slots);
        // Fold rows 0 and 1 into one group: counts add, sums add.
        let acc = V::value(&reader, 0).merge(V::value(&reader, 1), &());
        assert_eq!(acc.row.0, [2, 3, 30, 2]);
        assert_eq!(acc.sort_key(2), 30);
    }
}
