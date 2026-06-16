//! [`Mono`] — `N` slots sharing one [`Fold`], the homogeneous fast path.
//!
//! Slots share the fold `F` and width `A` but read independently (a `COUNT` slot
//! vs a `SUM` slot), so the reader is a per-slot [`SlotReader`]. One monomorph per
//! `(F, N, A)` covers every count/sum mix (`Mono<Add>`), every all-`MIN` /
//! all-`MAX`, and — with `A = u128` carrying an `ArenaKey` — homogeneous string
//! `MIN`/`MAX` (`Mono<StrMin, N, u128>`).

use super::super::cell::Cell;
use super::super::fold::Fold;
use super::super::read::SlotReader;
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `N` cells of width `A`, all folded by `F`.
pub struct Mono<F, const N: usize, A: Cell = i64> {
    cells: [A; N],
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
            cells: [A::default(); N],
            _fold: PhantomData,
        }
    }
}

impl<F: Fold<A> + Send + Sync + 'static, const N: usize, A: Cell> AggregationValue
    for Mono<F, N, A>
{
    type Reader<'b> = [SlotReader<'b>; N];
    type MergeConfig = F::Cfg;
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;

    #[inline(always)]
    fn merge_config(_slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> F::Cfg {
        F::cfg(arena)
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [SlotReader<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| SlotReader::new(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[SlotReader<'_>; N], idx: usize, arena: &mut WorkerArena) -> Self {
        Self {
            cells: std::array::from_fn(|s| F::seed(&reader[s], idx, arena)),
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &[SlotReader<'_>; N],
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &F::Cfg,
    ) -> Self {
        for (cell, slot) in self.cells.iter_mut().zip(reader.iter()) {
            *cell = F::update(*cell, slot, idx, arena, cfg);
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, cfg: &F::Cfg) -> Self {
        Self {
            cells: std::array::from_fn(|s| F::combine(self.cells[s], other.cells[s], cfg)),
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        F::sort_key(self.cells[slot])
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
        arena: &Arc<SharedArena>,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let (f, a) = F::finish(&format!("v{s}"), col, arena);
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
