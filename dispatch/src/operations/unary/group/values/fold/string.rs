//! [`StrMin<A>`](StrMin) / [`StrMax<A>`](StrMax) — string extremes over the `&str`
//! their [`Read`](super::super::read::StrRead) *borrows* from the column.
//!
//! These operations need arenas and borrowed strings, so they do not implement
//! the context-free numeric [`Fold`](super::Fold) trait. Runtime aggregation
//! calls their inherent methods directly.
//!
//! The `i128` cell contains an `ArenaKey`. Updates compare the borrowed input
//! before persisting it, so losing candidates consume no arena space.

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
