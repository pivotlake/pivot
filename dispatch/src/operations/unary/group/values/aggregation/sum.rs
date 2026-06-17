//! [`Sum<T>`](Sum) — `SUM` over an integer column. `WideSum<T>` is the same op
//! with an `i128` accumulator, for a 64-bit column whose total can overflow `i64`.

use super::Aggregation;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::Numeric;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `SUM` over column `T`, accumulating in width `A` (`i64` by default).
pub struct Sum<T, A = i64>(PhantomData<fn() -> (T, A)>);

/// `SUM` accumulating in `i128` — the planner picks this for a 64-bit column.
pub type WideSum<T> = Sum<T, i128>;

impl<T: ArrowPrimitiveType, A: Numeric> Aggregation for Sum<T, A>
where
    T::Native: Into<i64>,
{
    type Acc = A;
    type Input<'b> = &'b PrimitiveArray<T>;
    type Cfg = ();

    #[inline(always)]
    fn bind(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn cfg(_arena: &Arc<SharedArena>) {}

    #[inline(always)]
    fn seed(input: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> A {
        A::from(unsafe { input.value_unchecked(idx) }.into())
    }
    #[inline(always)]
    fn update(
        acc: A,
        input: &&PrimitiveArray<T>,
        idx: usize,
        _arena: &mut WorkerArena,
        _cfg: &(),
    ) -> A {
        acc + A::from(unsafe { input.value_unchecked(idx) }.into())
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
