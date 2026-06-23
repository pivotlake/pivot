//! [`Min<A>`](Min) / [`Max<A>`](Max) — integer extremes, folding the `i64` their
//! [`Read`](super::super::read) yields. They fold exactly as [`Sum`](super::Sum)
//! reads and differ only in the keep (`Ord::min` / `Ord::max`).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::Numeric;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// `MIN`, accumulating in width `A`.
pub struct Min<A = i64>(PhantomData<A>);
/// `MAX`, accumulating in width `A`.
pub struct Max<A = i64>(PhantomData<A>);

macro_rules! int_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: Numeric> Fold for $Op<A> {
            type Val = i64;
            type Acc = A;

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
            fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
                A::finish(name, col)
            }
        }
    };
}

int_extreme!(Min, min);
int_extreme!(Max, max);
