//! [`Count`] is `COUNT(*)` / `COUNT(col)`: `+1` per counted row. Folds `()`; a
//! `COUNT(*)` pairs it with [`NoRead`](super::super::read::NoRead) (every row
//! counts), a `COUNT(col)` with [`ValidRead`](super::super::read::ValidRead)
//! (NULL rows contribute nothing).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::IntCell;
use arrow_array::ArrayRef;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;
use std::marker::PhantomData;

/// Counts rows, in accumulator width `A` (`i64` by default — a count never
/// exceeds the row count). The width is generic so the runtime
/// [`Dynamic`](crate::operations::unary::group::values::container::Dynamic) value,
/// whose cells are a uniform width, can hold a `Count` slot in the same cell its
/// numeric extremes use; `Compiled` instantiates the default `Count<i64>`.
pub struct Count<A = i64>(PhantomData<A>);

impl<A: IntCell> Fold for Count<A> {
    type Val = ();
    type Acc = A;

    /// A count is `0` for a group with no counted rows, never SQL NULL.
    const ALWAYS_SEEN: bool = true;

    #[inline(always)]
    fn empty() -> A {
        A::from(0)
    }
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
    fn finish(name: &str, col: SlabColumn<A>, nulls: Option<NullBuffer>) -> (Field, ArrayRef) {
        A::finish(name, col, nulls)
    }
}
