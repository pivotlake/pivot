//! [`Min<T>`](Min) / [`Max<T>`](Max) — integer extremes over a column. They read
//! the column exactly as [`Sum`](super::Sum) does and differ only in the fold
//! (`Ord::min` / `Ord::max`).

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

/// `MIN` over column `T`, accumulating in width `A`.
pub struct Min<T, A = i64>(PhantomData<fn() -> (T, A)>);
/// `MAX` over column `T`, accumulating in width `A`.
pub struct Max<T, A = i64>(PhantomData<fn() -> (T, A)>);

macro_rules! int_extreme {
    ($Op:ident, $keep:ident) => {
        impl<T: ArrowPrimitiveType, A: Numeric> Aggregation for $Op<T, A>
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
                Ord::$keep(acc, A::from(unsafe { input.value_unchecked(idx) }.into()))
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
