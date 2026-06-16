//! [`Mono`] — a homogeneous aggregation value: every cell folds the same way.

use super::cell::Cell;
use super::columns::RowColumns;
use super::fold::CellFold;
use super::reader::RowReader;
use super::row::AggregationRow;
use super::{AggregationSlot, AggregationValue};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `N` cells of width `A`, all folded by `F` ([`Add`](super::Add)/[`Min`](super::Min)/
/// [`Max`](super::Max)). The common all-`SUM`/`COUNT` query is `Mono<Add>`; a
/// pure-`MIN`/`MAX` query is `Mono<Min>`/`Mono<Max>`. The fold is branch-free and
/// carries no config — mixed signatures use [`DynamicMixed`](super::DynamicMixed).
pub struct Mono<F, const N: usize, A: Cell = i64> {
    row: AggregationRow<N, A>,
    _fold: PhantomData<F>,
}

// Manual impls so `F` needn't be Copy/Clone/Default (it's a zero-sized marker).
impl<F, const N: usize, A: Cell> Copy for Mono<F, N, A> {}
impl<F, const N: usize, A: Cell> Clone for Mono<F, N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F, const N: usize, A: Cell> Default for Mono<F, N, A> {
    fn default() -> Self {
        Self {
            row: AggregationRow::default(),
            _fold: PhantomData,
        }
    }
}

impl<F: CellFold<A>, const N: usize, A: Cell> AggregationValue for Mono<F, N, A> {
    type Reader<'b> = RowReader<'b, N>;
    type MergeConfig = ();
    type Columns = RowColumns<N, A>;
    type SortKey = A;

    #[inline(always)]
    fn merge_config(_slots: &[AggregationSlot], _arena: &Arc<SharedArena>) {}

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> RowReader<'b, N> {
        RowReader::new(batch, slots)
    }

    #[inline(always)]
    fn value(reader: &RowReader<'_, N>, idx: usize, _arena: &mut WorkerArena) -> Self {
        Self {
            row: reader.read(idx),
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn merge(self, other: Self, _cfg: &()) -> Self {
        Self {
            row: self.row.zip(other.row, F::fold),
            _fold: PhantomData,
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

    fn finish_columns(
        cols: RowColumns<N, A>,
        _arena: &Arc<SharedArena>,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        cols.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::values::{Add, AggregationKind, AggregationSlot, Min};
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Schema};
    use std::sync::Arc;

    fn int_batch(vals: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vals.to_vec()))],
        )
        .unwrap()
    }

    // Numeric values ignore the arena, but `value`/`update_from_reader` take one.
    fn test_arena() -> WorkerArena {
        init_test_free_pool(8);
        WorkerArena::new(SharedArena::new(8))
    }

    #[test]
    fn mono_add_sums_the_column() {
        let b = int_batch(&[5, 2, 9]);
        let mut a = test_arena();
        let slots = [AggregationSlot::new(AggregationKind::Sum, 0)];
        let r = Mono::<Add, 1>::make_reader(&b, &slots);
        let acc = Mono::<Add, 1>::value(&r, 0, &mut a)
            .update_from_reader(&r, 1, &mut a, &())
            .update_from_reader(&r, 2, &mut a, &());
        assert_eq!(acc.sort_key(0), 16);
    }

    #[test]
    fn mono_min_keeps_the_smallest() {
        let b = int_batch(&[5, 2, 9]);
        let mut a = test_arena();
        let slots = [AggregationSlot::new(AggregationKind::Min, 0)];
        let r = Mono::<Min, 1>::make_reader(&b, &slots);
        let mut acc = Mono::<Min, 1>::value(&r, 0, &mut a);
        for i in 1..3 {
            acc = acc.update_from_reader(&r, i, &mut a, &());
        }
        assert_eq!(acc.sort_key(0), 2);
    }
}
