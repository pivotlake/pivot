//! [`Count`] — `COUNT(*)` / `COUNT(col)`: `+1` per row. Folds `()` — it reads no
//! column (its [`Read`](super::super::read::NoRead) yields nothing).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::IntCell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// Counts rows, in accumulator width `A` (`i64` by default — a count never
/// exceeds the row count). The width is generic so the runtime
/// [`Variable`](crate::operations::unary::group::values::container::Variable) value,
/// whose cells are a uniform width, can hold a `Count` slot in the same cell its
/// numeric extremes use; `Compiled` instantiates the default `Count<i64>`.
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
