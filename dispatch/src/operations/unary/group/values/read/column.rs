//! [`Col<T>`](Col) — a row's integer column value, widened into the cell.
//!
//! This is the single read behind `SUM`, `MIN` and `MAX` over an integer column:
//! all three pull the same per-row value and differ only in their
//! [`Fold`](super::super::fold).

use super::Read;
use crate::operations::unary::group::arena::WorkerArena;
use crate::operations::unary::group::values::cell::NumericCell;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch};
use std::marker::PhantomData;

/// Reads column `T`'s value at the row, widened to the cell `A`.
pub struct Col<T>(PhantomData<T>);

impl<A: NumericCell, T: ArrowPrimitiveType> Read<A> for Col<T>
where
    T::Native: Into<i64>,
{
    type Reader<'b> = &'b PrimitiveArray<T>;
    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn read(reader: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> A {
        A::from_i64(unsafe { reader.value_unchecked(idx) }.into())
    }
}
