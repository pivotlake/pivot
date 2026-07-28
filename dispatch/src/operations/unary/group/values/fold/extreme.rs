//! [`Min<A>`](Min) / [`Max<A>`](Max) — integer extremes, folding the `i64` their
//! [`Read`](super::super::read) yields. They fold exactly as [`Sum`](super::Sum)
//! reads and differ only in the keep (`Ord::min` / `Ord::max`).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::IntCell;
use arrow_array::ArrayRef;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;
use std::marker::PhantomData;

/// `MIN`, accumulating in width `A`.
pub struct Min<A = i64>(PhantomData<A>);
/// `MAX`, accumulating in width `A`.
pub struct Max<A = i64>(PhantomData<A>);

macro_rules! int_extreme {
    ($Op:ident, $keep:ident, $identity:ident) => {
        impl<A: IntCell> Fold for $Op<A> {
            type Val = i64;
            type Acc = A;

            /// The width's own extreme, so any folded value wins against it.
            #[inline(always)]
            fn empty() -> A {
                A::$identity
            }
            #[inline(always)]
            fn seed(v: i64) -> A {
                A::from(v)
            }
            #[inline(always)]
            fn update(acc: A, v: i64) -> A {
                Ord::$keep(acc, A::from(v))
            }
            #[inline(always)]
            fn merge(a: A, b: A) -> A {
                Ord::$keep(a, b)
            }
            fn finish(
                name: &str,
                col: SlabColumn<A>,
                nulls: Option<NullBuffer>,
            ) -> (Field, ArrayRef) {
                A::finish(name, col, nulls)
            }
        }
    };
}

int_extreme!(Min, min, MAX_VALUE);
int_extreme!(Max, max, MIN_VALUE);
