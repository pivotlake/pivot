//! [`Slot`] — one `(`[`Read`]`, `[`Fold`]`)` pair, the unit a container folds.
//!
//! A *blanket* impl bundles any read with any fold whose value it produces, so a
//! container sees a single per-slot interface (bind → seed/update → merge →
//! finish) and never learns whether the slot is a string or an integer: every
//! method is the same `F::_(R::read(input, idx), …)`. The `(R, F)` product is
//! formed here, once, by the blanket impl — not enumerated per (op, width).

use super::super::aggregation::{Count, Fold, FoldAcc, Max, Min, StrMax, StrMin, Sum};
use super::super::cell::Cell;
use super::super::read::{IntRead, NoRead, Read, StrRead};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// A `(Read, Fold)` pair as one *nominal* type. A bare `(R, F)` tuple would do,
/// but nesting tuples inside a `Compiled<…>` signature blows the monomorphisation
/// collector when that value flows through the top-k heap; a named struct keeps
/// the type shallow.
pub struct Pair<R, F>(PhantomData<R>, PhantomData<F>);

/// Slot aliases — a [`Pair`] named by what it computes, so signatures read
/// `Compiled<(SumSlot<Int32Type>, CountSlot)>` instead of the raw pairs.
pub type CountSlot<A = i64> = Pair<NoRead, Count<A>>;
/// `SUM(col: T)` accumulating in `A` (`SumSlot<T, i128>` is the wide sum).
pub type SumSlot<T, A = i64> = Pair<IntRead<T>, Sum<A>>;
pub type MinSlot<T, A = i64> = Pair<IntRead<T>, Min<A>>;
pub type MaxSlot<T, A = i64> = Pair<IntRead<T>, Max<A>>;
pub type StrMinSlot<A = i128> = Pair<StrRead, StrMin<A>>;
pub type StrMaxSlot<A = i128> = Pair<StrRead, StrMax<A>>;

/// The bundled per-slot interface a container drives. One impl: the blanket over
/// `(R, F)` below. A container never names `Read`/`Fold` directly — it folds
/// `Slot`s, string and integer alike.
pub trait Slot: Send + Sync + 'static {
    type Acc: Cell;
    type Input<'b>;
    type Cfg: Clone + Send + Sync + 'static;

    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;
    fn bind(batch: &RecordBatch, column: usize) -> Self::Input<'_>;
    fn seed(input: &Self::Input<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Acc;
    fn update(
        acc: Self::Acc,
        input: &Self::Input<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::Cfg,
    ) -> Self::Acc;
    fn merge(a: Self::Acc, b: Self::Acc, cfg: &Self::Cfg) -> Self::Acc;
    fn sort_key(acc: Self::Acc) -> i128;
    fn finish(
        name: &str,
        col: SlabColumn<Self::Acc>,
        arena: &Arc<SharedArena>,
    ) -> (Field, ArrayRef);
}

/// Any read paired with any fold over the value it yields. `Acc`/`Cfg`/`merge`/
/// `finish` come from [`FoldAcc`] — lifetime-free, so naming them never touches
/// the `for<'b> Fold<…>` bound a borrowed read value forces. Only `seed`/`update`
/// (which actually consume the read value) go through [`Fold`].
impl<R, F> Slot for Pair<R, F>
where
    R: Read,
    F: FoldAcc + for<'b> Fold<R::Val<'b>>,
{
    type Acc = F::Acc;
    type Input<'b> = R::Input<'b>;
    type Cfg = F::Cfg;

    #[inline(always)]
    fn cfg(arena: &Arc<SharedArena>) -> F::Cfg {
        F::cfg(arena)
    }
    #[inline(always)]
    fn bind(batch: &RecordBatch, column: usize) -> R::Input<'_> {
        R::bind(batch, column)
    }
    #[inline(always)]
    fn seed(input: &R::Input<'_>, idx: usize, arena: &mut WorkerArena) -> F::Acc {
        F::seed(R::read(input, idx), arena)
    }
    #[inline(always)]
    fn update(
        acc: F::Acc,
        input: &R::Input<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &F::Cfg,
    ) -> F::Acc {
        F::update(acc, R::read(input, idx), arena, cfg)
    }
    #[inline(always)]
    fn merge(a: F::Acc, b: F::Acc, cfg: &F::Cfg) -> F::Acc {
        F::merge(a, b, cfg)
    }
    #[inline(always)]
    fn sort_key(acc: F::Acc) -> i128 {
        F::sort_key(acc)
    }
    fn finish(name: &str, col: SlabColumn<F::Acc>, arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        F::finish(name, col, arena)
    }
}
