//! Compiled (monomorphised) GROUP BY aggregations.
//!
//! [`AggregationRowValueExtractor`](super::AggregationRowValueExtractor) dispatches each slot
//! through a runtime enum (`SlotValueReader`) on every row. These specialise a
//! *fixed* aggregate signature into straight-line code instead: each output slot
//! is a zero-sized [`Aggregate`] op, and the value extractor is monomorphised
//! over a tuple of them (and the accumulator width `A`), so `value()` is just
//! typed column reads with no per-row branch.
//!
//! Adding capability is cheap:
//! - a new aggregate function (e.g. `MIN`) is one `impl Aggregate`;
//! - a new query shape is one tuple type (e.g.
//!   `Compiled<(Count, Sum<Int16Type>, Count)>`), selected in the planner.
//!
//! The enum [`AggregationRowValueExtractor`](super::AggregationRowValueExtractor) stays as the
//! fallback for any signature we haven't compiled, so arbitrary queries still
//! run (just with the per-row dispatch).

use std::marker::PhantomData;

use arrow_array::RecordBatch;

use crate::operations::unary::group::values::accumulator::Accumulator;
use crate::operations::unary::group::values::aggregate::Aggregate;
use crate::operations::unary::group::values::aggregation_row::{
    AggregationRow, AggregationRowColumns,
};
use crate::operations::unary::group::values::{AggregationSlot, ValueExtractor};

/// A [`ValueExtractor`] monomorphised over a tuple of [`Aggregate`] ops — one per
/// output slot — and the accumulator width `A`. `value()` is straight-line typed
/// reads with no per-row dispatch.
pub struct Compiled<Ops, A: Accumulator = i64>(PhantomData<(Ops, A)>);

/// Implements [`ValueExtractor`] for a `Compiled<(Op0, Op1, …), A>` tuple of a
/// given arity. `$idx` are the tuple field indices, which also index `value_slots`.
macro_rules! impl_compiled {
    ($n:literal; $($Op:ident $idx:tt),+) => {
        impl<Acc: Accumulator, $($Op: Aggregate + Send + 'static),+> ValueExtractor for Compiled<($($Op,)+), Acc> {
            type Value = AggregationRow<$n, Acc>;
            type Reader<'b> = ($($Op::Reader<'b>,)+);
            type Columns = AggregationRowColumns<$n, Acc>;
            type SortKey = Acc;

            #[inline(always)]
            fn make_reader<'b>(
                batch: &'b RecordBatch,
                value_slots: &[AggregationSlot],
            ) -> Self::Reader<'b> {
                assert_eq!(value_slots.len(), $n, "slot count must match compiled arity");
                ($( $Op::make_reader(batch, value_slots[$idx].column), )+)
            }

            #[inline(always)]
            fn value(reader: &Self::Reader<'_>, idx: usize) -> AggregationRow<$n, Acc> {
                AggregationRow([$( Acc::from($Op::contribution(&reader.$idx, idx)), )+])
            }

            #[inline(always)]
            fn sort_key(value: &AggregationRow<$n, Acc>, slot: usize) -> Acc {
                value.0[slot]
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
    use crate::operations::unary::group::values::AggregationKind;
    use crate::operations::unary::group::values::aggregate::{Count, Sum};
    use arrow_array::Int16Array;
    use arrow_array::types::Int16Type;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    /// A mixed multi-aggregate signature: `COUNT(*), SUM(i16), SUM(i16), COUNT`
    /// — the straight-line value matches what the enum extractor would produce.
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

        type Agg = Compiled<(Count, Sum<Int16Type>, Sum<Int16Type>, Count)>;
        let reader = Agg::make_reader(&batch, &slots);
        assert_eq!(Agg::value(&reader, 0).0, [1, 1, 10, 1]);
        assert_eq!(Agg::value(&reader, 1).0, [1, 2, 20, 1]);
        assert_eq!(Agg::value(&reader, 2).0, [1, 3, 30, 1]);
    }
}
