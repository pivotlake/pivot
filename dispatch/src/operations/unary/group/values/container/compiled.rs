//! [`Compiled`] — a fixed aggregate signature monomorphised over a tuple of
//! per-slot ops, straight-line with no per-row dispatch.
//!
//! A slot is one `(`[`Read`]`, `[`Fold`]`)` pair — named [`Pair`], aliased to
//! [`CountSlot`]/[`SumSlot`]/… — a whole op with its own input array and its own
//! cell, so a `Compiled` mixes integer families freely (a `SUM` beside a `MAX`),
//! each reading its *own typed array* with no per-row dispatch. It is
//! **numeric-only**: every op's [`WorkerContext`](super::super::fold::FoldAcc::WorkerContext)
//! and [`SharedContext`](super::super::fold::FoldAcc::SharedContext) is `()`, so
//! both its value contexts are concretely `()` — consume threads `&mut ()` (free)
//! and merge/finish thread `&()`. A signature carrying a string extreme — which
//! needs a real `WorkerArena` to store winners — takes the
//! [`Dynamic`](super::Dynamic) path instead.
//!
//! There is no per-slot trait and no plumbing trait: a slot's behaviour *is* its
//! [`Read`] plus its [`Fold`]/[`FoldAcc`], so `impl_compiled!` emits the whole
//! [`AggregationValue`] impl for each arity directly, calling those — `R::read` to
//! pull the value, `F::seed`/`update`/`merge`/`finish` to fold it — unrolled over
//! the tuple, with the reader/config/column shapes as literal tuples. The lone
//! [`OpTuple`] trait carries a single associated type (the cell tuple), because a
//! `Compiled<Ops>` struct declared once over a generic `Ops` has to name its one
//! stored field's type somehow; it holds no behaviour.
//!
//! `Pair<R, F>` is only a nominal tag so a slot alias names a single type and the
//! `Compiled<…>` signature stays shallow (a bare nested `(R, F)` tuple sends the
//! monomorphisation collector into a loop through the top-k heap). The planner
//! instantiates the tuple it needs; numeric runtime signatures (shape not known
//! until plan time) fall back to [`Dynamic`](super::Dynamic).

use super::super::cell::Cell;
use super::super::fold::{Count, Fold, FoldAcc, Max, Min, Sum};
use super::super::read::{IntRead, NoRead, Read};
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;

/// A `(Read, Fold)` pair as one *nominal* type — a slot. A bare `(R, F)` tuple
/// would do, but nesting tuples inside a `Compiled<…>` signature blows the
/// monomorphisation collector when the value flows through the top-k heap; a named
/// struct keeps the type shallow. It's a pure type tag — never instantiated as a
/// value (a `Compiled` stores the cells, not the pairs).
pub struct Pair<R, F>(PhantomData<(R, F)>);

/// Slot aliases — a [`Pair`] named by what it computes, so signatures read
/// `Compiled<(SumSlot<Int32Type>, CountSlot)>` instead of the raw pairs.
///
/// `Count` defaults its accumulator to `i64` (a count never exceeds the row
/// count); a numeric extreme/sum defaults to `i64` too (pass `i128` for a wide
/// sum). A string extreme always rides `i128`, since its cell holds a 128-bit
/// `ArenaKey`.
pub type CountSlot<A = i64> = Pair<NoRead, Count<A>>;
/// `SUM(col: T)` accumulating in `A` — `SumSlot<T, i128>` is the wide sum.
pub type SumSlot<T, A = i64> = Pair<IntRead<T>, Sum<A>>;
/// `MIN(col: T)` over an integer column, accumulating in `A`.
pub type MinSlot<T, A = i64> = Pair<IntRead<T>, Min<A>>;
/// `MAX(col: T)` over an integer column, accumulating in `A`.
pub type MaxSlot<T, A = i64> = Pair<IntRead<T>, Max<A>>;
// (No `StrMinSlot`/`StrMaxSlot`: `Compiled` is numeric-only — a string extreme.s
// `WorkerContext` is `WorkerArena`, not `()` — so a string signature uses `Dynamic`.)

/// What a tuple of slots stores: the parallel tuple of accumulator cells, e.g.
/// `(i64,)` or `(i128, i64)`. The *only* thing [`Compiled`] needs from `Ops` that
/// it can't write inline — the struct holds one `Ops::Accs` field, and a struct
/// declared once over a generic `Ops` has to name that field's type through an
/// associated type. Every other shape (the reader, the configs, the columns) is a
/// literal tuple written directly in the macro-generated [`AggregationValue`]
/// impl, so there is no behaviour here — just the cell type.
pub trait OpTuple: Send + Sync + 'static {
    /// The per-slot accumulator cells.
    type Accs: Cell;
}

/// A fixed aggregate signature: the slots named by the op tuple `Ops`, each its
/// own cell. The running cells are the only state — their tuple type is `Ops::Accs`.
pub struct Compiled<Ops: OpTuple> {
    accs: Ops::Accs,
}

impl<Ops: OpTuple> Copy for Compiled<Ops> {}
impl<Ops: OpTuple> Clone for Compiled<Ops> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Ops: OpTuple> Default for Compiled<Ops> {
    fn default() -> Self {
        Self {
            accs: Ops::Accs::default(),
        }
    }
}

/// One per arity: the [`OpTuple`] cell type for a tuple of slots `(Pair<R, F>, …)`,
/// and the whole [`AggregationValue`] impl for the `Compiled` over it. Each method
/// unrolls the obvious per-slot call — `R::read` to pull the value,
/// `F::seed`/`update`/`merge`/`finish` to fold it — and writes the
/// reader / config / column shapes as literal tuples. The lifetime-free
/// `F::Acc`/`SharedContext`/`merge`/… come from `FoldAcc`; only `seed`/`update` (which take
/// the read value) need the `for<'b> Fold<…>` bound a borrowed `&str` forces.
macro_rules! impl_compiled {
    ($($R:ident $F:ident $idx:tt),+) => {
        impl<$($R, $F),+> OpTuple for ($(Pair<$R, $F>,)+)
        where
            $($R: Read, $F: FoldAcc + for<'b> Fold<$R::Val<'b>>,)+
        {
            type Accs = ($($F::Acc,)+);
        }

        // `Compiled` is numeric-only — every op's `WorkerContext`/`SharedContext`
        // is `()` — so both its contexts are concretely `()`: consume threads
        // `&mut ()` (free) and merge/finish thread `&()`. A signature with a string
        // extreme takes the `Dynamic` path instead, whose contexts are a real
        // `WorkerArena` / `(slots, Arc<SharedArena>)`.
        impl<$($R, $F),+> AggregationValue for Compiled<($(Pair<$R, $F>,)+)>
        where
            $($R: Read, $F: FoldAcc<SharedContext = (), WorkerContext = ()> + for<'b> Fold<$R::Val<'b>>, $F::Acc: Into<i128>,)+
        {
            type Reader<'b> = ($($R::Input<'b>,)+);
            type SharedContext = ();
            type Columns = ($(SlabColumn<$F::Acc>,)+);
            type SortKey = i128;
            type WorkerContext = ();

            fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
                debug_assert_eq!(slots.len(), [$($idx),+].len(), "slot count must match the tuple arity");
                ($($R::bind(batch, slots[$idx].column),)+)
            }

            #[inline(always)]
            fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut ()) -> Self {
                Self { accs: ($($F::seed($R::read(&reader.$idx, idx), wc),)+) }
            }

            #[inline(always)]
            fn update_from_reader(
                self,
                reader: &Self::Reader<'_>,
                idx: usize,
                wc: &mut (),
                ctx: &(),
            ) -> Self {
                Self {
                    accs: ($($F::update(self.accs.$idx, $R::read(&reader.$idx, idx), wc, ctx),)+),
                }
            }

            #[inline(always)]
            fn merge(self, other: Self, ctx: &()) -> Self {
                Self { accs: ($($F::merge(self.accs.$idx, other.accs.$idx, ctx),)+) }
            }

            #[inline(always)]
            fn sort_key(&self, slot: usize) -> i128 {
                // Widen the cell directly — a numeric extreme/sum/count to its
                // `ORDER BY` key; a string extreme's raw bits are inert (the
                // planner never pushes a top-k onto one), same as `Dynamic`.
                match slot {
                    $($idx => self.accs.$idx.into(),)+
                    _ => unreachable!("sort_key slot {slot} out of range"),
                }
            }

            fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns {
                ($(SlabColumn::<$F::Acc>::with_capacity(allocator, rows),)+)
            }

            #[inline(always)]
            fn push_to(&self, cols: &mut Self::Columns) {
                $(cols.$idx.push(self.accs.$idx);)+
            }

            fn finish_columns(
                cols: Self::Columns,
                ctx: &(),
            ) -> (Vec<Field>, Vec<ArrayRef>) {
                let mut fields = Vec::new();
                let mut arrays = Vec::new();
                $(
                    let (f, a) = $F::finish(&format!("v{}", $idx), cols.$idx, ctx);
                    fields.push(f);
                    arrays.push(a);
                )+
                (fields, arrays)
            }
        }
    };
}

impl_compiled!(R0 F0 0);
impl_compiled!(R0 F0 0, R1 F1 1);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3, R4 F4 4);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3, R4 F4 4, R5 F5 5);
