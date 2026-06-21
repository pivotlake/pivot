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

use super::super::cell::{Numeric, StringCell};
use super::super::fold::{Fold, FoldAcc};
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
///
/// `ONLY_ADDITIVE` is a fast-path promise: when `true`, every slot is guaranteed
/// (by the planner) to be a `COUNT` or `SUM`, so the per-slot folds (`update`,
/// `merge`) drop their `Min`/`Max`/`StrMin`/`StrMax` arms to `unreachable!()`. With
/// the non-additive arms gone the compiler collapses the per-slot dispatch and the
/// `StringCell`/`i128` machinery to a branch-free additive loop — recovering the
/// hand-written `Mono` codegen from this same generic container. The two-level
/// `COUNT(DISTINCT)` path (all additive) sets it; everything else leaves it `false`.
pub struct Dynamic<const N: usize, A: Numeric + StringCell = i64, const ONLY_ADDITIVE: bool = false>
{
    cells: [A; N],
}

impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool> Copy
    for Dynamic<N, A, ONLY_ADDITIVE>
{
}
impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool> Clone
    for Dynamic<N, A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool> Default
    for Dynamic<N, A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool> AggregationValue
    for Dynamic<N, A, ONLY_ADDITIVE>
{
    type Reader<'b> = [BoundSlot<'b>; N];
    /// The per-slot kinds (which op merges/renders each cell) and the value arena
    /// (which a string extreme resolves its keys through). It spawns a per-worker
    /// [`WorkerArena`] via [`SharedContext::worker`].
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;
    /// A runtime signature may carry a string extreme, so it always takes a real
    /// per-worker [`WorkerArena`] (its numeric arms thread a throwaway `&mut ()`).
    type WorkerContext = WorkerArena;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [BoundSlot<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| BoundSlot::bind(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[BoundSlot<'_>; N], idx: usize, wc: &mut WorkerArena) -> Self {
        // A plain loop, not `std::array::from_fn`: the per-slot match is large, so
        // as a `from_fn` closure it exceeds the inline threshold and is emitted
        // out-of-line through the `Wrapped`/try-trait machinery — measured at ~40%
        // of a q09 merge regression. The loop keeps the op `seed`s inlined.
        // Numeric arms take a throwaway `&mut ()` (their `Arena` is `()`); only a
        // string arm touches the real `WorkerArena`.
        let mut na = ();
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            // `ONLY_ADDITIVE` prunes each non-additive arm *in its body* — `if
            // ONLY_ADDITIVE { unreachable!() } else { .. }` const-folds to a bare
            // `unreachable!()` arm. A `_ if ONLY_ADDITIVE` guard arm instead lowers to
            // a worse Count/Sum dispatch — measured ~2.3B more in `consume_window`.
            cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::seed((), &mut na),
                BoundSlot::Sum(r) => Sum::<A>::seed(r.read(idx), &mut na),
                BoundSlot::Min(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Min::<A>::seed(r.read(idx), &mut na)
                    }
                }
                BoundSlot::Max(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Max::<A>::seed(r.read(idx), &mut na)
                    }
                }
                BoundSlot::StrMin(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMin::<A>::seed(StrRead::read(a, idx), wc)
                    }
                }
                BoundSlot::StrMax(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMax::<A>::seed(StrRead::read(a, idx), wc)
                    }
                }
            };
        }
        Self { cells }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &[BoundSlot<'_>; N],
        idx: usize,
        wc: &mut WorkerArena,
        ctx: &Self::SharedContext,
    ) -> Self {
        let (_, shared) = ctx;
        let mut na = ();
        // Every arm is `Op::<A>::update(c, <read>, arena, <cfg>)`: cell `A` in, `A`
        // out, for numeric and string alike. Numeric arms take a throwaway
        // `&mut ()`; only a string arm folds into the real `WorkerArena`.
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let c = self.cells[s];
            // Per-arm body pruning under `ONLY_ADDITIVE` (see `value` for why this
            // beats a `_ if ONLY_ADDITIVE` guard arm).
            self.cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::update(c, (), &mut na, &()),
                BoundSlot::Sum(r) => Sum::<A>::update(c, r.read(idx), &mut na, &()),
                BoundSlot::Min(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Min::<A>::update(c, r.read(idx), &mut na, &())
                    }
                }
                BoundSlot::Max(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Max::<A>::update(c, r.read(idx), &mut na, &())
                    }
                }
                BoundSlot::StrMin(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMin::<A>::update(c, StrRead::read(a, idx), wc, shared)
                    }
                }
                BoundSlot::StrMax(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMax::<A>::update(c, StrRead::read(a, idx), wc, shared)
                    }
                }
            };
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, ctx: &Self::SharedContext) -> Self {
        let (slots, shared) = ctx;
        // A plain loop, not `std::array::from_fn`, for the same inlining reason as
        // `value` — this runs per matched entry in the partition merge, the hottest
        // path for a high-cardinality `COUNT(DISTINCT)`. Each slot combines via its
        // op's own `merge` — same `Op::<A>::merge(a, b, cfg)` shape, no reader.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let (a, b) = (self.cells[s], other.cells[s]);
            cells[s] = if ONLY_ADDITIVE {
                // All-additive: `Count` and `Sum` both merge by `+`, so skip the
                // per-slot `slots[s].kind` load and Count/Sum branch entirely — a
                // pure add, identical to Mono's branch-free merge.
                a + b
            } else {
                match slots[s].kind {
                    AggregationKind::CountStar | AggregationKind::Count => {
                        Count::<A>::merge(a, b, &())
                    }
                    AggregationKind::Sum => Sum::<A>::merge(a, b, &()),
                    AggregationKind::Min => Min::<A>::merge(a, b, &()),
                    AggregationKind::Max => Max::<A>::merge(a, b, &()),
                    AggregationKind::StrMin => StrMin::<A>::merge(a, b, shared),
                    AggregationKind::StrMax => StrMax::<A>::merge(a, b, shared),
                }
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
        ctx: &Self::SharedContext,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = ctx;
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let name = format!("v{s}");
            // Each op renders its own column (numeric → its width's Arrow type via
            // `&()`, string → `Utf8View` resolved through the arena).
            let (f, a) = match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<A>::finish(&name, col, &())
                }
                AggregationKind::Sum => Sum::<A>::finish(&name, col, &()),
                AggregationKind::Min => Min::<A>::finish(&name, col, &()),
                AggregationKind::Max => Max::<A>::finish(&name, col, &()),
                AggregationKind::StrMin => StrMin::<A>::finish(&name, col, arena),
                AggregationKind::StrMax => StrMax::<A>::finish(&name, col, arena),
            };
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
