//! [`Slot`] — one `(`[`Read`]`, `[`Fold`]`)` pair, the unit a container folds.
//!
//! A *blanket* impl bundles any read with any fold whose value it produces, so a
//! container sees a single per-slot interface (bind → seed/update → merge →
//! finish) and never learns whether the slot is a string or an integer: every
//! method is the same `F::_(R::read(input, idx), …)`. The `(R, F)` product is
//! formed here, once, by the blanket impl — not enumerated per (op, width).

use super::super::aggregation::Fold;
use super::super::cell::Cell;
use super::super::read::Read;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

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

/// Any read paired with any fold over the value it yields. `Acc`/`Cfg` are impl
/// params constrained by the higher-ranked bound, which also *asserts* they don't
/// depend on the read's lifetime (the `Acc = Acc, Cfg = Cfg` equalities hold
/// `for<'b>`), so a container can name them lifetime-free.
impl<R, F, Acc, Cfg> Slot for (R, F)
where
    R: Read,
    Acc: Cell,
    Cfg: Clone + Send + Sync + 'static,
    F: for<'b> Fold<R::Val<'b>, Acc = Acc, Cfg = Cfg>,
{
    type Acc = Acc;
    type Input<'b> = R::Input<'b>;
    type Cfg = Cfg;

    #[inline(always)]
    fn cfg(arena: &Arc<SharedArena>) -> Cfg {
        <F as Fold<R::Val<'static>>>::cfg(arena)
    }
    #[inline(always)]
    fn bind(batch: &RecordBatch, column: usize) -> R::Input<'_> {
        R::bind(batch, column)
    }
    #[inline(always)]
    fn seed(input: &R::Input<'_>, idx: usize, arena: &mut WorkerArena) -> Acc {
        F::seed(R::read(input, idx), arena)
    }
    #[inline(always)]
    fn update(
        acc: Acc,
        input: &R::Input<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Cfg,
    ) -> Acc {
        F::update(acc, R::read(input, idx), arena, cfg)
    }
    #[inline(always)]
    fn merge(a: Acc, b: Acc, cfg: &Cfg) -> Acc {
        <F as Fold<R::Val<'static>>>::merge(a, b, cfg)
    }
    #[inline(always)]
    fn sort_key(acc: Acc) -> i128 {
        <F as Fold<R::Val<'static>>>::sort_key(acc)
    }
    fn finish(name: &str, col: SlabColumn<Acc>, arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        <F as Fold<R::Val<'static>>>::finish(name, col, arena)
    }
}
