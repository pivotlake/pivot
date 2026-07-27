//! [`StrMin<A>`](StrMin) / [`StrMax<A>`](StrMax) — string extremes over the `&str`
//! their [`Read`](super::super::read::StrRead) *borrows* from the column.
//!
//! Unlike the numeric ops these are *not* [`Fold`](super::Fold)s: they need a value
//! arena (a per-worker [`WorkerArena`] to persist winners, the shared
//! [`SharedArena`] to resolve them) and they hold a borrowed `&str`, so they carry
//! context a numeric fold never does. They expose plain inherent methods, called
//! directly from the runtime [`Variable`](super::super::container::Variable)
//! container's string arms — the only place a string extreme appears (there is no
//! `Compiled` string slot).
//!
//! The accumulator is the plain cell width `A`: an `ArenaKey` is just a 128-bit
//! value, so a string extreme rides the same `A` (`= i128`) cell a numeric slot
//! uses — no separate cell type, no container reinterpret. Viewing those 128 bits
//! as a key is *this op's* business, via [`StringCell`] (the identity for `i128`,
//! the fail-out for `i64`). Because the op holds the real `&str`, it compares
//! *before* persisting, so only a winner ever touches the value arena (lazy).

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
        impl<A: StringCell> $Op<A> {
            /// Materialise a new group's cell from a row's string (persisted into
            /// the per-worker arena).
            #[inline(always)]
            pub fn seed(v: &str, wc: &mut WorkerArena) -> A {
                A::from_key(wc.push(v))
            }
            /// Fold a row's string into an existing cell: compare the raw bytes vs
            /// the current extreme (the cell viewed as a key) and persist only a
            /// winner.
            #[inline(always)]
            pub fn update(acc: A, v: &str, wc: &mut WorkerArena, ctx: &Arc<SharedArena>) -> A {
                if v.as_bytes() $wins acc.into_key().resolve(ctx) {
                    A::from_key(wc.push(v))
                } else {
                    acc
                }
            }
            /// Combine two finished partials, keeping the winner (resolved through
            /// the shared arena).
            #[inline(always)]
            pub fn merge(a: A, b: A, ctx: &Arc<SharedArena>) -> A {
                if b.into_key().resolve(ctx) $wins a.into_key().resolve(ctx) {
                    b
                } else {
                    a
                }
            }
            /// Render a finished column of keys as a `Utf8View` array over the arena.
            pub fn finish(name: &str, col: SlabColumn<A>, ctx: &Arc<SharedArena>) -> (Field, ArrayRef) {
                A::finish(name, col, ctx)
            }
        }
    };
}

str_extreme!(StrMin, <);
str_extreme!(StrMax, >);
