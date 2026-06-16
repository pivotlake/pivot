//! [`One`] — the `COUNT` read: contributes `1`, reads no column.

use super::Read;
use crate::operations::unary::group::arena::WorkerArena;
use crate::operations::unary::group::values::cell::NumericCell;
use arrow_array::RecordBatch;

/// Reads nothing; every row contributes `1`. Paired with [`Add`](super::super::fold::Add)
/// this is `COUNT`.
pub struct One;

impl<A: NumericCell> Read<A> for One {
    type Reader<'b> = ();
    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn read(_reader: &(), _idx: usize, _arena: &mut WorkerArena) -> A {
        A::from_i64(1)
    }
}
