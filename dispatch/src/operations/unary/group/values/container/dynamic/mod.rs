//! [`Dynamic`] — the runtime-signature value: `N` cells of a uniform width `A`,
//! each folded by its slot's op.
//!
//! Where [`Compiled`](super::Compiled) names its `(`[`Read`]`, `[`Fold`]`)` slots
//! in the type, `Dynamic` resolves them at runtime: [`make_reader`](OwnedValue::make_reader)
//! binds each slot to a [`BoundSlot`] (one variant *per (op, width family)* — the
//! column is downcast into an [`I64Reader`]/[`F64Reader`]/[`U128Reader`] payload,
//! see [`readers`]), and the fold drives every slot through the *same*
//! `Op::<A>::update(cell, read, …)`. The cell is `A` in and `A` out for every op — a
//! string extreme's `ArenaKey`, a float's raw `f64` bits, and a wide `i128` partial
//! are all just the bits of `A` (`= i128`), viewed by the op via
//! [`StringCell`]/[`F64Cell`]/[`WideCell`], so the container never reinterprets and
//! never branches on the value's family.
//!
//! Generic over `A` (`i64` narrow / `i128` wide) and `ONLY_ADDITIVE` (the
//! branch-free additive fast path). A string extreme, a float value, and a re-read
//! wide partial all ride the wide (`i128`) instantiation; the `i64` arms of those
//! cell traits are the fail-out (the planner always widens such a signature). This
//! is the runtime container for every non-`Compiled` signature — numeric, string, or
//! mixed.

mod readers;

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::fold::Fold;
use super::super::read::{Read, StrRead};
use super::super::{
    AggregationKind, AggregationSlot, Count, F64Max, F64Min, F64Sum, Max, Min, OwnedValue, StrMax,
    StrMin, Sum, U128Max, U128Min, U128Sum, ValueColumns,
};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

pub use readers::{F64Reader, I64Reader, U128Reader};

/// One slot's bound reader for a batch — one variant *per (op, width family)*, the
/// column downcast into its [`readers`] payload. Built once per batch by
/// [`bind`](BoundSlot::bind). The `I64`/`F64`/`U128` prefixes name the value width
/// the op folds (`i64` / `f64` / `i128`), so the container's arm reads exactly that
/// type with no per-row `is_float` branch.
pub enum BoundSlot<'b> {
    Count,
    I64Sum(I64Reader<'b>),
    I64Min(I64Reader<'b>),
    I64Max(I64Reader<'b>),
    F64Sum(F64Reader<'b>),
    F64Min(F64Reader<'b>),
    F64Max(F64Reader<'b>),
    U128Sum(U128Reader<'b>),
    U128Min(U128Reader<'b>),
    U128Max(U128Reader<'b>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> BoundSlot<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match slot.kind {
            CountStar | Count => BoundSlot::Count,
            Sum => Self::bind_numeric(
                batch,
                slot.column,
                BoundSlot::I64Sum,
                BoundSlot::F64Sum,
                BoundSlot::U128Sum,
            ),
            Min => Self::bind_extreme(
                batch,
                slot.column,
                BoundSlot::StrMin,
                BoundSlot::I64Min,
                BoundSlot::F64Min,
                BoundSlot::U128Min,
            ),
            Max => Self::bind_extreme(
                batch,
                slot.column,
                BoundSlot::StrMax,
                BoundSlot::I64Max,
                BoundSlot::F64Max,
                BoundSlot::U128Max,
            ),
        }
    }

    /// Pick the reader for a `SUM` column by its width family — an integer column
    /// reads as `i64`, a float as `f64`, a `Decimal128` (a re-read wide partial) as
    /// `i128` — and wrap it in the caller's matching op variant.
    fn bind_numeric(
        batch: &'b RecordBatch,
        column: usize,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        match batch.column(column).data_type() {
            DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                on_i64(I64Reader::bind(batch, column))
            }
            DataType::Float32 | DataType::Float64 => on_f64(F64Reader::bind(batch, column)),
            DataType::Decimal128(_, _) => on_u128(U128Reader::bind(batch, column)),
            other => panic!("numeric aggregate over unsupported column type {other:?}"),
        }
    }

    /// A `MIN`/`MAX` picks its op by column type: a `Utf8` column reads the string
    /// extreme (the raw `&str`, folded through the value arena); any other column is
    /// numeric ([`bind_numeric`](BoundSlot::bind_numeric)).
    fn bind_extreme(
        batch: &'b RecordBatch,
        column: usize,
        on_str: fn(&'b StringViewArray) -> Self,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        if let DataType::Utf8View = batch.column(column).data_type() {
            on_str(StrRead::bind(batch, column))
        } else {
            Self::bind_numeric(batch, column, on_i64, on_f64, on_u128)
        }
    }
}

/// Seed one slot's cell from row `idx`, a new group's first value. This is the
/// per-slot fold shared by the const-arity [`Dynamic`] and the runtime-arity
/// [`Variable`](super::Variable) containers, so every runtime signature folds
/// identically whichever holds it. Only a string arm touches the `WorkerArena`;
/// the numeric `seed`s take their value directly.
///
/// `ONLY_ADDITIVE` prunes each non-additive arm *in its body*: `if
/// ONLY_ADDITIVE { unreachable!() } else { .. }` const-folds to a bare
/// `unreachable!()` arm. A `_ if ONLY_ADDITIVE` guard arm instead lowers
/// to a worse Count/Sum dispatch (measured ~2.3B more in `consume_window`).
/// `Count` and the integer/wide `SUM` arms carry no guard; they are the
/// additive ops the fast path keeps.
#[inline(always)]
pub(in super::super) fn seed_slot<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ONLY_ADDITIVE: bool,
>(
    slot: &BoundSlot<'_>,
    idx: usize,
    wc: &mut WorkerArena,
) -> A {
    match slot {
        BoundSlot::Count => Count::<A>::seed(()),
        BoundSlot::I64Sum(r) => Sum::<A>::seed(r.read(idx)),
        BoundSlot::U128Sum(r) => U128Sum::<A>::seed(r.read(idx)),
        BoundSlot::I64Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::I64Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                Max::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Sum(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Sum::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Max::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::U128Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                U128Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::U128Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                U128Max::<A>::seed(r.read(idx))
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
    }
}

/// Fold row `idx` into one slot's existing cell. A numeric arm folds its own cell
/// (no context); a string arm folds into the real `WorkerArena` and resolves the
/// current extreme through `shared`. Per-arm `ONLY_ADDITIVE` pruning as in
/// [`seed_slot`].
#[inline(always)]
pub(in super::super) fn update_slot<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ONLY_ADDITIVE: bool,
>(
    cell: A,
    slot: &BoundSlot<'_>,
    idx: usize,
    wc: &mut WorkerArena,
    shared: &Arc<SharedArena>,
) -> A {
    match slot {
        BoundSlot::Count => Count::<A>::update(cell, ()),
        BoundSlot::I64Sum(r) => Sum::<A>::update(cell, r.read(idx)),
        BoundSlot::U128Sum(r) => U128Sum::<A>::update(cell, r.read(idx)),
        BoundSlot::I64Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::I64Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Sum(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Sum::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                F64Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::U128Min(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                U128Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::U128Max(r) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                U128Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::StrMin(a) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                StrMin::<A>::update(cell, StrRead::read(a, idx), wc, shared)
            }
        }
        BoundSlot::StrMax(a) => {
            if ONLY_ADDITIVE {
                unreachable!()
            } else {
                StrMax::<A>::update(cell, StrRead::read(a, idx), wc, shared)
            }
        }
    }
}

/// Combine one slot's two finished partial cells, for the partition merge and
/// the radix fold. The reader is gone by the merge, so the value family is read
/// off the slot's declared `output_type`: a `Utf8View` extreme, a floating
/// `SUM`/`MIN`/`MAX`, else integer/wide. A wide (`i128`) re-read merges
/// identically to the narrow integer one (both accumulate in `A = i128`), so it
/// needs no separate arm. Each check lives in its own arm, so `Count` and the
/// integer paths pay for none of them. (The caller's all-additive fast path, a
/// branch-free `a + b`, skips this dispatch entirely.)
#[inline(always)]
pub(in super::super) fn merge_slot<A: IntCell + StringCell + F64Cell + WideCell>(
    a: A,
    b: A,
    slot: &AggregationSlot,
    shared: &Arc<SharedArena>,
) -> A {
    let ty = &slot.output_type;
    match slot.kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b),
        AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::merge(a, b),
        AggregationKind::Sum => Sum::<A>::merge(a, b),
        AggregationKind::Min if *ty == DataType::Utf8View => StrMin::<A>::merge(a, b, shared),
        AggregationKind::Min if ty.is_floating() => F64Min::<A>::merge(a, b),
        AggregationKind::Min => Min::<A>::merge(a, b),
        AggregationKind::Max if *ty == DataType::Utf8View => StrMax::<A>::merge(a, b, shared),
        AggregationKind::Max if ty.is_floating() => F64Max::<A>::merge(a, b),
        AggregationKind::Max => Max::<A>::merge(a, b),
    }
}

/// Render one slot's finished output column. Called once per output column (not
/// per row), so the per-slot kind dispatch is irrelevant. Each op renders its own
/// column, dispatched by kind + declared `output_type` as in [`merge_slot`]
/// (numeric to its width's Arrow type, float to `Float64`, string to `Utf8View`
/// resolved through the arena, wide re-read to `Decimal128` via the integer arm).
pub(in super::super) fn finish_slot<A: IntCell + StringCell + F64Cell + WideCell>(
    name: &str,
    slot: &AggregationSlot,
    col: SlabColumn<A>,
    arena: &Arc<SharedArena>,
) -> (Field, ArrayRef) {
    let ty = &slot.output_type;
    match slot.kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::finish(name, col),
        AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::finish(name, col),
        AggregationKind::Sum => Sum::<A>::finish(name, col),
        AggregationKind::Min if *ty == DataType::Utf8View => StrMin::<A>::finish(name, col, arena),
        AggregationKind::Min if ty.is_floating() => F64Min::<A>::finish(name, col),
        AggregationKind::Min => Min::<A>::finish(name, col),
        AggregationKind::Max if *ty == DataType::Utf8View => StrMax::<A>::finish(name, col, arena),
        AggregationKind::Max if ty.is_floating() => F64Max::<A>::finish(name, col),
        AggregationKind::Max => Max::<A>::finish(name, col),
    }
}

/// `N` cells of width `A`, each folded by its slot op.
///
/// `ONLY_ADDITIVE` is a fast-path promise: when `true`, every slot is guaranteed
/// (by the planner) to be a `COUNT` or an integer/wide `SUM`, so `merge` drops the
/// per-slot kind dispatch to a branch-free `a + b`, and `value`/`update` prune their
/// non-additive arms to `unreachable!()`. An all-additive grouped aggregate (e.g. a
/// low-card `pair (int, int)` aggregate) sets it; a string extreme, a float value, or
/// a `Min`/`Max` leaves it `false`. Worth ~1.5-2% on a low-card grouped aggregate.
pub struct Dynamic<
    const N: usize,
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
> {
    cells: [A; N],
}

impl<const N: usize, A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Copy
    for Dynamic<N, A, ONLY_ADDITIVE>
{
}
impl<const N: usize, A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Clone
    for Dynamic<N, A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    Default for Dynamic<N, A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

/// The output-column builders for a [`Dynamic`] signature: `N` [`SlabColumn`]s of
/// width `A`. The value-side counterpart to a key extractor's
/// [`KeyColumns`](crate::operations::unary::group::keys::KeyColumns). The
/// `ONLY_ADDITIVE` flag is carried only to bind these columns one-to-one to their
/// [`Dynamic`] value; the output path renders by slot kind regardless.
pub struct DynamicColumns<
    const N: usize,
    A: IntCell + StringCell + F64Cell + WideCell,
    const ONLY_ADDITIVE: bool,
> {
    cols: [SlabColumn<A>; N],
}

impl<const N: usize, A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    OwnedValue for Dynamic<N, A, ONLY_ADDITIVE>
{
    type Reader<'b> = [BoundSlot<'b>; N];
    /// The per-slot kinds (which op merges/renders each cell) and the value arena
    /// (which a string extreme resolves its keys through). It spawns a per-worker
    /// [`WorkerArena`] via [`SharedContext::worker`](super::super::SharedContext::worker).
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type Columns = DynamicColumns<N, A, ONLY_ADDITIVE>;
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
        // out-of-line through the `Wrapped`/try-trait machinery. The loop keeps
        // the op `seed`s inlined.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            cells[s] = seed_slot::<A, ONLY_ADDITIVE>(&reader[s], idx, wc);
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
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            self.cells[s] =
                update_slot::<A, ONLY_ADDITIVE>(self.cells[s], &reader[s], idx, wc, shared);
        }
        self
    }

    #[inline(always)]
    fn merge(self, other: Self, ctx: &Self::SharedContext) -> Self {
        let (slots, shared) = ctx;
        // A plain loop, not `std::array::from_fn`, for the same inlining reason as
        // `value` — this runs per matched entry in the partition merge, the hottest
        // path for a high-cardinality `COUNT(DISTINCT)`.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let (a, b) = (self.cells[s], other.cells[s]);
            cells[s] = if ONLY_ADDITIVE {
                // All-additive: `Count` and every `Sum` merge by `+`, so skip the
                // per-slot `slots[s].kind` load and dispatch entirely — a branch-free
                // add, the additive fast path the partition merge wants.
                a + b
            } else {
                merge_slot(a, b, &slots[s], shared)
            };
        }
        Self { cells }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // Integer cells widen to their `ORDER BY` key. A string extreme never feeds
        // a top-k (the planner doesn't push one), so its raw bits here are inert.
        self.cells[slot].into()
    }
}

impl<const N: usize, A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    ValueColumns for DynamicColumns<N, A, ONLY_ADDITIVE>
{
    type Value = Dynamic<N, A, ONLY_ADDITIVE>;
    type Context = (Arc<[AggregationSlot]>, Arc<SharedArena>);

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _context: &Self::Context) -> Self {
        Self {
            cols: std::array::from_fn(|_| SlabColumn::with_capacity(allocator, rows)),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Self::Value) {
        for (col, cell) in self.cols.iter_mut().zip(value.cells.iter()) {
            col.push(*cell);
        }
    }

    #[inline(always)]
    fn push_stored(&mut self, stored: &Self::Value) {
        self.push(stored);
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = context;
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in self.cols.into_iter().enumerate() {
            let (f, a) = finish_slot(&format!("v{s}"), &slots[s], col, arena);
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
