//! [`StrMin<A>`](StrMin) / [`StrMax<A>`](StrMax) — string extremes, folding the
//! `&str` their [`Read`](super::super::read::StrRead) *borrows* from the column.
//!
//! The accumulator is the plain cell width `A`: an [`ArenaKey`] is just a 128-bit
//! value, so a string extreme rides the same `A` (`= i128`) cell a numeric slot
//! uses — no separate cell type, no container reinterpret. Viewing those 128 bits
//! as a key is *this op's* business, via [`StringCell`] (the identity for `i128`,
//! the fail-out for `i64`). Because the fold holds the real `&str`, it compares
//! *before* persisting, so only a winner ever touches the value arena (lazy).

use super::{Fold, FoldAcc};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::StringCell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `MIN` over a string column, the winning key held in cell width `A`.
pub struct StrMin<A = i128>(PhantomData<A>);
/// `MAX` over a string column.
pub struct StrMax<A = i128>(PhantomData<A>);

macro_rules! str_extreme {
    ($Op:ident, $wins:tt) => {
        // Lifetime-free: a container names `StrMin`'s `Acc`/`Cfg`/`merge` without
        // touching the borrowed-`&str` `Fold` bound below.
        impl<A: StringCell> FoldAcc for $Op<A> {
            type Acc = A;
            type Cfg = Arc<SharedArena>;

            #[inline(always)]
            fn cfg(arena: &Arc<SharedArena>) -> Arc<SharedArena> {
                arena.clone()
            }
            #[inline(always)]
            fn merge(a: A, b: A, cfg: &Arc<SharedArena>) -> A {
                if b.into_key().resolve(cfg) $wins a.into_key().resolve(cfg) {
                    b
                } else {
                    a
                }
            }
            fn finish(
                name: &str,
                col: SlabColumn<A>,
                arena: &Arc<SharedArena>,
            ) -> (Field, ArrayRef) {
                A::finish(name, col, arena)
            }
        }

        impl<'b, A: StringCell> Fold<&'b str> for $Op<A> {
            #[inline(always)]
            fn seed(v: &str, arena: &mut WorkerArena) -> A {
                A::from_key(arena.push(v))
            }
            #[inline(always)]
            fn update(acc: A, v: &str, arena: &mut WorkerArena, cfg: &Arc<SharedArena>) -> A {
                // Raw bytes vs the current extreme (the cell viewed as a key);
                // persist only a winner.
                if v.as_bytes() $wins acc.into_key().resolve(cfg) {
                    A::from_key(arena.push(v))
                } else {
                    acc
                }
            }
        }
    };
}

str_extreme!(StrMin, <);
str_extreme!(StrMax, >);
