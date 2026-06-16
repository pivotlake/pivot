//! [`Dynamic`] — the numeric fallback for a heterogeneous *numeric* signature
//! (e.g. `COUNT(*), MIN(x), SUM(y)`), folding each slot on its runtime kind.
//!
//! One monomorph per width `A`. Strings don't go here — a string `MIN`/`MAX` is
//! homogeneous ([`Mono`](super::Mono)) or, mixed with other aggregates, a fixed
//! [`Compiled`](super::Compiled) shape; `Dynamic` keeps the per-slot dispatch to a
//! single numeric `combine`.

use super::super::cell::NumericCell;
use super::super::read::SlotReader;
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// `N` numeric cells of width `A`, each folded by its own slot kind.
pub struct Dynamic<const N: usize, A: NumericCell = i64> {
    cells: [A; N],
}

impl<const N: usize, A: NumericCell> Copy for Dynamic<N, A> {}
impl<const N: usize, A: NumericCell> Clone for Dynamic<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: NumericCell> Default for Dynamic<N, A> {
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

impl<const N: usize, A: NumericCell> AggregationValue for Dynamic<N, A> {
    type Reader<'b> = [SlotReader<'b>; N];
    type MergeConfig = Arc<[AggregationSlot]>;
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;

    fn merge_config(
        slots: &[AggregationSlot],
        _arena: &Arc<SharedArena>,
    ) -> Arc<[AggregationSlot]> {
        Arc::from(slots)
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [SlotReader<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| SlotReader::new(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[SlotReader<'_>; N], idx: usize, _arena: &mut WorkerArena) -> Self {
        Self {
            cells: std::array::from_fn(|s| reader[s].read_num::<A>(idx)),
        }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &[SlotReader<'_>; N],
        idx: usize,
        _arena: &mut WorkerArena,
        slots: &Arc<[AggregationSlot]>,
    ) -> Self {
        // Indexes three same-length arrays (cells / reader / slots) by slot.
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            self.cells[s] = slots[s]
                .kind
                .combine(self.cells[s], reader[s].read_num::<A>(idx));
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
        self.cells[slot].to_i128()
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
