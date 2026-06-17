//! [`Count`] — `COUNT(*)` / `COUNT(col)`: `+1` per row, reads no column.

use super::Aggregation;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::Numeric;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// Counts rows, in accumulator width `A` (`i64` by default — a count never
/// exceeds the row count). The width is generic so the runtime `Dynamic` value,
/// whose cells are a uniform width, can hold a `Count` slot in the same cell its
/// numeric extremes use; `Compiled` instantiates the default `Count<i64>`.
pub struct Count<A = i64>(PhantomData<fn() -> A>);

impl<A: Numeric> Aggregation for Count<A> {
    type Acc = A;
    type Input<'b> = ();
    type Cfg = ();

    #[inline(always)]
    fn bind(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn cfg(_arena: &Arc<SharedArena>) {}

    #[inline(always)]
    fn seed(_input: &(), _idx: usize, _arena: &mut WorkerArena) -> A {
        A::from(1)
    }
    #[inline(always)]
    fn update(acc: A, _input: &(), _idx: usize, _arena: &mut WorkerArena, _cfg: &()) -> A {
        acc + A::from(1)
    }
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
