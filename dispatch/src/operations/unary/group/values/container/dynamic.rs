//! [`Dynamic`] — the runtime-signature value: `N` cells of a uniform width `A`,
//! each folded by its slot's op.
//!
//! Where [`Compiled`](super::Compiled) names its `(`[`Read`]`, `[`Fold`]`)` slots
//! in the type, `Dynamic` resolves them at runtime: [`make_reader`](AggregationValue::make_reader)
//! binds each slot to a [`BoundSlot`] (one variant *per op*, the column width a
//! [`NumReader`] payload — so the variants are `ops`, not `ops × widths`), and the
//! fold drives every slot through the *same* `Op::<A>::update(cell, read, arena,
//! cfg)`. The cell is `A` in and `A` out for every op — a string extreme's
//! `ArenaKey` is just the 128 bits of `A` (`= i128`), viewed as a key *inside*
//! [`StrMin`]/[`StrMax`] (via [`StringCell`]), so the container never reinterprets
//! and never branches string-vs-int.
//!
//! Generic over `A` (`i64` narrow / `i128` wide). A string extreme rides the wide
//! (`i128`) instantiation; the `i64` [`StringCell`] arms are the fail-out (the
//! planner always widens a string signature to `i128`).

use super::super::aggregation::{Fold, FoldAcc};
use super::super::cell::{Numeric, StringCell};
use super::super::read::{IntRead, Read, StrRead};
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

/// An integer column bound at one of the three widths, read as `i64`. The width
/// is a *payload* here, not a cross-product with the op: adding a width is one
/// more variant in this enum, shared by every numeric op.
pub enum NumReader<'b> {
    I16(&'b PrimitiveArray<Int16Type>),
    I32(&'b PrimitiveArray<Int32Type>),
    I64(&'b PrimitiveArray<Int64Type>),
}

impl<'b> NumReader<'b> {
    pub(crate) fn bind(batch: &'b RecordBatch, column: usize) -> Self {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Int16 => NumReader::I16(col.as_primitive::<Int16Type>()),
            DataType::Int32 => NumReader::I32(col.as_primitive::<Int32Type>()),
            DataType::Int64 => NumReader::I64(col.as_primitive::<Int64Type>()),
            other => panic!("numeric aggregate over unsupported column type {other:?}"),
        }
    }
    #[inline(always)]
    pub(crate) fn read(&self, idx: usize) -> i64 {
        match self {
            NumReader::I16(a) => IntRead::<Int16Type>::read(a, idx),
            NumReader::I32(a) => IntRead::<Int32Type>::read(a, idx),
            NumReader::I64(a) => IntRead::<Int64Type>::read(a, idx),
        }
    }
}

/// One slot's bound reader for a batch — one variant *per op*, the numeric column
/// width carried inside [`NumReader`]. Built once per batch by [`bind`](BoundSlot::bind).
pub enum BoundSlot<'b> {
    Count,
    Sum(NumReader<'b>),
    Min(NumReader<'b>),
    Max(NumReader<'b>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> BoundSlot<'b> {
    fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match slot.kind {
            CountStar | Count => BoundSlot::Count,
            Sum => BoundSlot::Sum(NumReader::bind(batch, slot.column)),
            Min => BoundSlot::Min(NumReader::bind(batch, slot.column)),
            Max => BoundSlot::Max(NumReader::bind(batch, slot.column)),
            StrMin => BoundSlot::StrMin(StrRead::bind(batch, slot.column)),
            StrMax => BoundSlot::StrMax(StrRead::bind(batch, slot.column)),
        }
    }
}

/// `N` cells of width `A`, each folded by its slot op.
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
    /// The per-slot kinds (which op merges/renders each cell) and the value arena
    /// (which a string extreme resolves its keys through).
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
        // A plain loop, not `std::array::from_fn`: the per-slot match is large, so
        // as a `from_fn` closure it exceeds the inline threshold and is emitted
        // out-of-line through the `Wrapped`/try-trait machinery — measured at ~40%
        // of a q09 merge regression. The loop keeps the op `seed`s inlined.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::seed((), arena),
                BoundSlot::Sum(r) => Sum::<A>::seed(r.read(idx), arena),
                BoundSlot::Min(r) => Min::<A>::seed(r.read(idx), arena),
                BoundSlot::Max(r) => Max::<A>::seed(r.read(idx), arena),
                BoundSlot::StrMin(a) => StrMin::<A>::seed(StrRead::read(a, idx), arena),
                BoundSlot::StrMax(a) => StrMax::<A>::seed(StrRead::read(a, idx), arena),
            };
        }
        Self { cells }
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
        // Every arm is `Op::<A>::update(c, <read>, arena, <cfg>)`: cell `A` in, `A`
        // out, for numeric and string alike. The read value and the cfg differ by
        // op (each declares its own `Read::Val` / `Fold::Cfg`); the container does
        // not — no reinterpret, no string branch.
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let c = self.cells[s];
            self.cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::update(c, (), arena, &()),
                BoundSlot::Sum(r) => Sum::<A>::update(c, r.read(idx), arena, &()),
                BoundSlot::Min(r) => Min::<A>::update(c, r.read(idx), arena, &()),
                BoundSlot::Max(r) => Max::<A>::update(c, r.read(idx), arena, &()),
                BoundSlot::StrMin(a) => StrMin::<A>::update(c, StrRead::read(a, idx), arena, shared),
                BoundSlot::StrMax(a) => StrMax::<A>::update(c, StrRead::read(a, idx), arena, shared),
            };
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, cfg: &Self::MergeConfig) -> Self {
        let (slots, shared) = cfg;
        // A plain loop, not `std::array::from_fn`, for the same inlining reason as
        // `value` — this runs per matched entry in the partition merge, the hottest
        // path for a high-cardinality `COUNT(DISTINCT)`. Each slot combines via its
        // op's own `merge` — same `Op::<A>::merge(a, b, cfg)` shape, no reader.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let (a, b) = (self.cells[s], other.cells[s]);
            cells[s] = match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b, &()),
                AggregationKind::Sum => Sum::<A>::merge(a, b, &()),
                AggregationKind::Min => Min::<A>::merge(a, b, &()),
                AggregationKind::Max => Max::<A>::merge(a, b, &()),
                AggregationKind::StrMin => StrMin::<A>::merge(a, b, shared),
                AggregationKind::StrMax => StrMax::<A>::merge(a, b, shared),
            };
        }
        Self { cells }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // Numeric cells widen to their `ORDER BY` key. A string extreme never feeds
        // a top-k (the planner doesn't push one), so its raw bits here are inert.
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
            // Each op renders its own column (numeric → its width's Arrow type,
            // string → `Utf8View`) — same `Op::<A>::finish` shape.
            let (f, a) = match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<A>::finish(&name, col, arena)
                }
                AggregationKind::Sum => Sum::<A>::finish(&name, col, arena),
                AggregationKind::Min => Min::<A>::finish(&name, col, arena),
                AggregationKind::Max => Max::<A>::finish(&name, col, arena),
                AggregationKind::StrMin => StrMin::<A>::finish(&name, col, arena),
                AggregationKind::StrMax => StrMax::<A>::finish(&name, col, arena),
            };
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
