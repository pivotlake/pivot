//! [`Min<A>`](Min) / [`Max<A>`](Max) — integer extremes, folding the `i64` their
//! [`Read`](super::super::read) yields. They fold exactly as [`Sum`](super::Sum)
//! reads and differ only in the keep (`Ord::min` / `Ord::max`).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::Numeric;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `MIN`, accumulating in width `A`.
pub struct Min<A = i64>(PhantomData<fn() -> A>);
/// `MAX`, accumulating in width `A`.
pub struct Max<A = i64>(PhantomData<fn() -> A>);

macro_rules! int_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: Numeric> Fold<i64> for $Op<A> {
            type Acc = A;
            type Cfg = ();

            #[inline(always)]
            fn cfg(_arena: &Arc<SharedArena>) {}

            #[inline(always)]
            fn seed(v: i64, _arena: &mut WorkerArena) -> A {
                A::from(v)
            }
            #[inline(always)]
            fn update(acc: A, v: i64, _arena: &mut WorkerArena, _cfg: &()) -> A {
                Ord::$keep(acc, A::from(v))
            }
            #[inline(always)]
            fn merge(a: A, b: A, _cfg: &()) -> A {
                Ord::$keep(a, b)
            }
            #[inline(always)]
            fn sort_key(acc: A) -> i128 {
                acc.into()
            }
            fn finish(
                name: &str,
                col: SlabColumn<A>,
                _arena: &Arc<SharedArena>,
            ) -> (Field, ArrayRef) {
                A::finish(name, col)
            }
        }
    };
}

int_extreme!(Min, min);
int_extreme!(Max, max);
