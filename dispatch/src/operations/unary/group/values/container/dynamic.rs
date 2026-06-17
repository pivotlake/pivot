//! [`Dynamic`] — the runtime-signature value: each slot folded by its own
//! [`Aggregation`] op, over a uniform cell width `A`.
//!
//! Where [`Compiled`](super::Compiled) names its ops in the type (so it never
//! branches), `Dynamic` resolves them at runtime: `make_reader` binds each slot
//! to a [`BoundSlot`] — one flat variant per (op, column width) — and the fold
//! then calls that op's [`seed`](Aggregation::seed)/[`update`](Aggregation::update)
//! directly. The op owns the fold; `Dynamic` only selects it. The enum is
//! exhaustive, so the dispatch holds no `unreachable!` and re-implements nothing;
//! the per-width variants are the explicit cost of resolving the column type at
//! runtime rather than in the type (as `Compiled` does).
//!
//! Generic over the accumulator width `A` (`i64` narrow / `i128` wide). A string
//! extreme rides the *wide* (`i128`) instantiation: its `ArenaKey` is a 128-bit
//! `StringView` header, stored in the cell via [`StringCell`] (the `i64` arms are
//! the fail-out — the planner always widens a string signature to `i128`).

use super::super::aggregation::Aggregation;
use super::super::cell::{Numeric, StringCell};
use super::super::{
    AggregationKind, AggregationSlot, AggregationValue, Count, Max, Min, StrMax, StrMin, Sum,
};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// One slot's bound op and downcast input for a batch: the runtime counterpart
/// to a `Compiled` tuple element, one variant per (op, column width). Built once
/// per batch by [`bind`](BoundSlot::bind).
pub enum BoundSlot<'b> {
    Count,
    SumI16(&'b PrimitiveArray<Int16Type>),
    SumI32(&'b PrimitiveArray<Int32Type>),
    SumI64(&'b PrimitiveArray<Int64Type>),
    MinI16(&'b PrimitiveArray<Int16Type>),
    MinI32(&'b PrimitiveArray<Int32Type>),
    MinI64(&'b PrimitiveArray<Int64Type>),
    MaxI16(&'b PrimitiveArray<Int16Type>),
    MaxI32(&'b PrimitiveArray<Int32Type>),
    MaxI64(&'b PrimitiveArray<Int64Type>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> BoundSlot<'b> {
    fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        // Downcast the slot's column to one of the three integer widths, tagging
        // it with the op (`$i16`/`$i32`/`$i64` are the matching variants).
        macro_rules! by_width {
            ($i16:ident, $i32:ident, $i64:ident) => {{
                let col = batch.column(slot.column);
                match col.data_type() {
                    DataType::Int16 => BoundSlot::$i16(col.as_primitive::<Int16Type>()),
                    DataType::Int32 => BoundSlot::$i32(col.as_primitive::<Int32Type>()),
                    DataType::Int64 => BoundSlot::$i64(col.as_primitive::<Int64Type>()),
                    other => panic!("numeric aggregate over unsupported column type {other:?}"),
                }
            }};
        }
        use AggregationKind::*;
        match slot.kind {
            CountStar | Count => BoundSlot::Count,
            Sum => by_width!(SumI16, SumI32, SumI64),
            Min => by_width!(MinI16, MinI32, MinI64),
            Max => by_width!(MaxI16, MaxI32, MaxI64),
            StrMin => BoundSlot::StrMin(batch.column(slot.column).as_string_view()),
            StrMax => BoundSlot::StrMax(batch.column(slot.column).as_string_view()),
        }
    }
}

/// `N` cells of width `A`, each folded by its own slot op.
pub struct Dynamic<const N: usize, A: Numeric + StringCell = i64> {
    cells: [A; N],
}

impl<const N: usize, A: Numeric + StringCell> Copy for Dynamic<N, A> {}
impl<const N: usize, A: Numeric + StringCell> Clone for Dynamic<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Numeric + StringCell> Default for Dynamic<N, A> {
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

impl<const N: usize, A: Numeric + StringCell> AggregationValue for Dynamic<N, A> {
    type Reader<'b> = [BoundSlot<'b>; N];
    /// The per-slot kinds (which op merges each cell) and the value arena (which a
    /// string extreme resolves its `ArenaKey`s through during the partition merge).
    type MergeConfig = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;

    fn merge_config(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self::MergeConfig {
        (Arc::from(slots), arena.clone())
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [BoundSlot<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| BoundSlot::bind(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[BoundSlot<'_>; N], idx: usize, arena: &mut WorkerArena) -> Self {
        Self {
            cells: std::array::from_fn(|s| match &reader[s] {
                BoundSlot::Count => Count::<A>::seed(&(), idx, arena),
                BoundSlot::SumI16(a) => Sum::<Int16Type, A>::seed(a, idx, arena),
                BoundSlot::SumI32(a) => Sum::<Int32Type, A>::seed(a, idx, arena),
                BoundSlot::SumI64(a) => Sum::<Int64Type, A>::seed(a, idx, arena),
                BoundSlot::MinI16(a) => Min::<Int16Type, A>::seed(a, idx, arena),
                BoundSlot::MinI32(a) => Min::<Int32Type, A>::seed(a, idx, arena),
                BoundSlot::MinI64(a) => Min::<Int64Type, A>::seed(a, idx, arena),
                BoundSlot::MaxI16(a) => Max::<Int16Type, A>::seed(a, idx, arena),
                BoundSlot::MaxI32(a) => Max::<Int32Type, A>::seed(a, idx, arena),
                BoundSlot::MaxI64(a) => Max::<Int64Type, A>::seed(a, idx, arena),
                BoundSlot::StrMin(a) => A::from_key(StrMin::seed(a, idx, arena)),
                BoundSlot::StrMax(a) => A::from_key(StrMax::seed(a, idx, arena)),
            }),
        }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &[BoundSlot<'_>; N],
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::MergeConfig,
    ) -> Self {
        let (_, shared) = cfg;
        // Indexes the parallel cells / reader arrays by slot.
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let c = self.cells[s];
            self.cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::update(c, &(), idx, arena, &()),
                BoundSlot::SumI16(a) => Sum::<Int16Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::SumI32(a) => Sum::<Int32Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::SumI64(a) => Sum::<Int64Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MinI16(a) => Min::<Int16Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MinI32(a) => Min::<Int32Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MinI64(a) => Min::<Int64Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MaxI16(a) => Max::<Int16Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MaxI32(a) => Max::<Int32Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::MaxI64(a) => Max::<Int64Type, A>::update(c, a, idx, arena, &()),
                BoundSlot::StrMin(a) => {
                    A::from_key(StrMin::update(c.into_key(), a, idx, arena, shared))
                }
                BoundSlot::StrMax(a) => {
                    A::from_key(StrMax::update(c.into_key(), a, idx, arena, shared))
                }
            };
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, cfg: &Self::MergeConfig) -> Self {
        let (slots, shared) = cfg;
        Self {
            cells: std::array::from_fn(|s| {
                let (a, b) = (self.cells[s], other.cells[s]);
                // Combine two finished cells via each op's own `merge` — the same
                // ops the consume path folds with. The partition merge has no
                // column reader, but a numeric op's `merge` is width-independent
                // (it folds two already-materialised `A`s), so the integer ops are
                // named at an arbitrary width; a string extreme resolves both keys
                // through the value arena.
                match slots[s].kind {
                    AggregationKind::CountStar | AggregationKind::Count => {
                        Count::<A>::merge(a, b, &())
                    }
                    AggregationKind::Sum => Sum::<Int64Type, A>::merge(a, b, &()),
                    AggregationKind::Min => Min::<Int64Type, A>::merge(a, b, &()),
                    AggregationKind::Max => Max::<Int64Type, A>::merge(a, b, &()),
                    AggregationKind::StrMin => {
                        A::from_key(StrMin::merge(a.into_key(), b.into_key(), shared))
                    }
                    AggregationKind::StrMax => {
                        A::from_key(StrMax::merge(a.into_key(), b.into_key(), shared))
                    }
                }
            }),
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // Numeric cells widen to their `ORDER BY` key. A string extreme never
        // feeds a top-k (the planner doesn't push one), so its raw view bits here
        // are inert — never compared.
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
        arena: &Arc<SharedArena>,
        cfg: &Self::MergeConfig,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, _) = cfg;
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let name = format!("v{s}");
            let (f, a) = if slots[s].kind.is_string_extreme() {
                A::finish_strings(&name, col, arena)
            } else {
                A::finish(&name, col)
            };
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
