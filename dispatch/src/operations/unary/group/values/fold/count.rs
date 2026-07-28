//! [`Count`] — `COUNT(*)` / `COUNT(col)`: `+1` per row. Folds `()` — it reads no
//! column (its [`Read`](super::super::read::NoRead) yields nothing).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::IntCell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// Counts rows in accumulator width `A`.
///
/// Runtime aggregation uses one width for all slots, so `A` may be `i128` even
/// though a compiled count normally uses `i64`.
pub struct Count<A = i64>(PhantomData<A>);

impl<A: IntCell> Fold for Count<A> {
    type Val = ();
    type Acc = A;

    #[inline(always)]
    fn seed(_v: ()) -> A {
        A::from(1)
    }
    #[inline(always)]
    fn update(acc: A, _v: ()) -> A {
        acc + A::from(1)
    }
    #[inline(always)]
    fn merge(a: A, b: A) -> A {
        a + b
    }
    fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
        A::finish(name, col)
    }
}
