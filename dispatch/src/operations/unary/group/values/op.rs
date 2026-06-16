//! An **op atom** — one slot's aggregate, a [`Read`] paired with a [`Fold`].
//!
//! [`Op<R, F, A>`](Op) is the only implementor of [`Aggregate`]: it pairs a typed
//! read with a fold over cell `A`. The named aggregates are aliases:
//!
//! ```text
//!   Count    = Op<One,    Add,    i64 >
//!   Sum<T>   = Op<Col<T>, Add,    i64 >
//!   Min<T>   = Op<Col<T>, NumMin, i64 >
//!   Max<T>   = Op<Col<T>, NumMax, i64 >
//!   StrMin   = Op<Str,    StrMin, u128>
//!   StrMax   = Op<Str,    StrMax, u128>
//! ```
//!
//! Atoms are what [`Compiled`](super::container::Compiled)'s tuple elements (and
//! [`Dynamic`](super::container::Dynamic)'s enum) are. Unlike
//! [`Mono`](super::container::Mono) — which carries one fold + a per-slot
//! [`SlotReader`](super::read::SlotReader) — an atom's read is a concrete type, so
//! a `Compiled` shape is straight-line with no per-row dispatch. A wide `SUM`
//! uses `Op<Col<Int64Type>, Add, u128>` directly.

use super::cell::Cell;
use super::fold::Fold;
use super::read::Read;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// One slot's complete aggregate behaviour. The seed/update/merge split mirrors
/// the [`probe_fold`](crate::operations::unary::group::hashtables) phases and
/// [`Fold`]'s; an atom seeds and updates from its *typed* reader (string slots in
/// a mixed `Compiled` persist eagerly — the lazy path is
/// [`Mono`](super::container::Mono)'s, for the common homogeneous string case).
pub trait Aggregate {
    type Acc: Cell;
    type Reader<'b>;
    type Cfg: Clone + Send + Sync + 'static;

    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;
    fn make_reader(batch: &RecordBatch, column: usize) -> Self::Reader<'_>;
    fn seed(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Acc;
    fn update(
        acc: Self::Acc,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::Cfg,
    ) -> Self::Acc;
    fn merge(acc: Self::Acc, incoming: Self::Acc, cfg: &Self::Cfg) -> Self::Acc;
    fn sort_key(acc: Self::Acc) -> i128;
    fn finish(
        name: &str,
        col: SlabColumn<Self::Acc>,
        arena: &Arc<SharedArena>,
    ) -> (Field, ArrayRef);
}

/// A [`Read`] `R` paired with a [`Fold`] `F` over cell `A` — the atom.
pub struct Op<R, F, A>(PhantomData<(R, F, A)>);

impl<A: Cell, R: Read<A>, F: Fold<A>> Aggregate for Op<R, F, A> {
    type Acc = A;
    type Reader<'b> = R::Reader<'b>;
    type Cfg = F::Cfg;

    #[inline(always)]
    fn cfg(arena: &Arc<SharedArena>) -> F::Cfg {
        F::cfg(arena)
    }
    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> R::Reader<'_> {
        R::make_reader(batch, column)
    }
    #[inline(always)]
    fn seed(reader: &R::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> A {
        R::read(reader, idx, arena)
    }
    #[inline(always)]
    fn update(
        acc: A,
        reader: &R::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &F::Cfg,
    ) -> A {
        F::combine(acc, R::read(reader, idx, arena), cfg)
    }
    #[inline(always)]
    fn merge(acc: A, incoming: A, cfg: &F::Cfg) -> A {
        F::combine(acc, incoming, cfg)
    }
    #[inline(always)]
    fn sort_key(acc: A) -> i128 {
        F::sort_key(acc)
    }
    fn finish(name: &str, col: SlabColumn<A>, arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        F::finish(name, col, arena)
    }
}

use super::fold::{
    Add, Max as MaxFold, Min as MinFold, StrMax as StrMaxFold, StrMin as StrMinFold,
};
use super::read::{Col, One, Str};

/// `COUNT` — `1` per row, summed.
pub type Count = Op<One, Add, i64>;
/// `SUM` over an integer column, narrow accumulator.
pub type Sum<T> = Op<Col<T>, Add, i64>;
/// `MIN` over an integer column.
pub type Min<T> = Op<Col<T>, MinFold, i64>;
/// `MAX` over an integer column.
pub type Max<T> = Op<Col<T>, MaxFold, i64>;
/// `SUM` over a 64-bit column, wide accumulator.
pub type WideSum<T> = Op<Col<T>, Add, u128>;
/// `MIN` over a string column.
pub type StrMin = Op<Str, StrMinFold, u128>;
/// `MAX` over a string column.
pub type StrMax = Op<Str, StrMaxFold, u128>;

/// Bound alias for an op usable as a `Compiled`/`Dynamic` slot.
pub trait SlotOp: Aggregate + Send + Sync + 'static {}
impl<T: Aggregate + Send + Sync + 'static> SlotOp for T {}
