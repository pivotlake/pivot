//! [`Sum<A>`](Sum) — `SUM`, folding the `i64` its [`Read`](super::super::read)
//! yields, accumulating in `A`. `WideSum` is the same op with an `i128`
//! accumulator, for a 64-bit column whose total can overflow `i64`. The *column*
//! width lives in the read, so there is one `Sum` per accumulator width, not per
//! column type.

use super::{Fold, FoldAcc};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::Numeric;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `SUM` accumulating in width `A` (`i64` by default).
pub struct Sum<A = i64>(PhantomData<A>);

/// `SUM` accumulating in `i128` — the planner picks this for a 64-bit column.
pub type WideSum = Sum<i128>;

impl<A: Numeric> FoldAcc for Sum<A> {
    type Acc = A;
    type Cfg = ();

    #[inline(always)]
    fn cfg(_arena: &Arc<SharedArena>) {}
    #[inline(always)]
    fn merge(a: A, b: A, _cfg: &()) -> A {
        a + b
    }
    #[inline(always)]
    fn sort_key(acc: A) -> i128 {
        acc.into()
    }
    fn finish(name: &str, col: SlabColumn<A>, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        A::finish(name, col)
    }
}

impl<A: Numeric> Fold<i64> for Sum<A> {
    #[inline(always)]
    fn seed(v: i64, _arena: &mut WorkerArena) -> A {
        A::from(v)
    }
    #[inline(always)]
    fn update(acc: A, v: i64, _arena: &mut WorkerArena, _cfg: &()) -> A {
        acc + A::from(v)
    }
}
