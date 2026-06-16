//! [`Dynamic`] — the numeric fallback for a heterogeneous *numeric* signature
//! (e.g. `COUNT(*), MIN(x), SUM(y)`), folding each slot by its runtime kind.
//!
//! Generic over the accumulator width `A` (`i64` narrow / `i128` wide), so a
//! numeric mix stays as narrow as a `Compiled` shape — no wider cells. Its
//! per-slot reader is numeric only (`COUNT` or an integer column, by width), so
//! it has no string arm and no `unreachable!`. Strings never come here: a string
//! extreme is always a fixed [`Compiled`](super::Compiled) tuple.

use super::super::cell::Numeric;
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// One slot's numeric read: a `COUNT` (`One`) or an integer column by width.
/// Public only because it surfaces in [`Dynamic`]'s `Reader` associated type.
pub enum NumInput<'b> {
    One,
    Col16(&'b PrimitiveArray<Int16Type>),
    Col32(&'b PrimitiveArray<Int32Type>),
    Col64(&'b PrimitiveArray<Int64Type>),
}

impl<'b> NumInput<'b> {
    fn new(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        use super::super::AggregationKind::*;
        match slot.kind {
            CountStar | Count => NumInput::One,
            Sum | Min | Max => {
                let col = batch.column(slot.column);
                match col.data_type() {
                    DataType::Int16 => NumInput::Col16(col.as_primitive::<Int16Type>()),
                    DataType::Int32 => NumInput::Col32(col.as_primitive::<Int32Type>()),
                    DataType::Int64 => NumInput::Col64(col.as_primitive::<Int64Type>()),
                    other => panic!("numeric aggregate over unsupported column type {other:?}"),
                }
            }
        }
    }

    #[inline(always)]
    fn read<A: Numeric>(&self, idx: usize) -> A {
        match self {
            NumInput::One => A::from(1),
            NumInput::Col16(a) => A::from(unsafe { a.value_unchecked(idx) } as i64),
            NumInput::Col32(a) => A::from(unsafe { a.value_unchecked(idx) } as i64),
            NumInput::Col64(a) => A::from(unsafe { a.value_unchecked(idx) }),
        }
    }
}

/// `N` numeric cells of width `A`, each folded by its own slot kind.
pub struct Dynamic<const N: usize, A: Numeric = i64> {
    cells: [A; N],
}

impl<const N: usize, A: Numeric> Copy for Dynamic<N, A> {}
impl<const N: usize, A: Numeric> Clone for Dynamic<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Numeric> Default for Dynamic<N, A> {
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

impl<const N: usize, A: Numeric> AggregationValue for Dynamic<N, A> {
    type Reader<'b> = [NumInput<'b>; N];
    type MergeConfig = Arc<[AggregationSlot]>;
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;

    fn merge_config(
        slots: &[AggregationSlot],
        _arena: &Arc<SharedArena>,
    ) -> Arc<[AggregationSlot]> {
        Arc::from(slots)
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [NumInput<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| NumInput::new(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[NumInput<'_>; N], idx: usize, _arena: &mut WorkerArena) -> Self {
        Self {
            cells: std::array::from_fn(|s| reader[s].read::<A>(idx)),
        }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &[NumInput<'_>; N],
        idx: usize,
        _arena: &mut WorkerArena,
        slots: &Arc<[AggregationSlot]>,
    ) -> Self {
        // Indexes three same-length arrays (cells / reader / slots) by slot.
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            self.cells[s] = slots[s]
                .kind
                .combine(self.cells[s], reader[s].read::<A>(idx));
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, slots: &Arc<[AggregationSlot]>) -> Self {
        Self {
            cells: std::array::from_fn(|s| slots[s].kind.combine(self.cells[s], other.cells[s])),
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        self.cells[slot].into()
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> [SlabColumn<A>; N] {
        std::array::from_fn(|_| SlabColumn::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut [SlabColumn<A>; N]) {
        for (col, cell) in cols.iter_mut().zip(self.cells.iter()) {
            col.push(*cell);
        }
    }

    fn finish_columns(
        cols: [SlabColumn<A>; N],
        _arena: &Arc<SharedArena>,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let (f, a) = A::finish(&format!("v{s}"), col);
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
