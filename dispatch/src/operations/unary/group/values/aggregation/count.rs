//! [`Count`] — `COUNT(*)` / `COUNT(col)`: `+1` per row, reads no column.

use super::Aggregation;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::NumericArrow;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// Counts rows. Its accumulator is always `i64` — a count can't exceed the row
/// count.
pub struct Count;

impl Aggregation for Count {
    type Acc = i64;
    type Input<'b> = ();
    type Cfg = ();

    #[inline(always)]
    fn bind(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn cfg(_arena: &Arc<SharedArena>) {}

    #[inline(always)]
    fn seed(_input: &(), _idx: usize, _arena: &mut WorkerArena) -> i64 {
        1
    }
    #[inline(always)]
    fn update(acc: i64, _input: &(), _idx: usize, _arena: &mut WorkerArena, _cfg: &()) -> i64 {
        acc + 1
    }
    #[inline(always)]
    fn merge(a: i64, b: i64, _cfg: &()) -> i64 {
        a + b
    }
    #[inline(always)]
    fn sort_key(acc: i64) -> i128 {
        acc as i128
    }
    fn finish(name: &str, col: SlabColumn<i64>, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        i64::finish(name, col)
    }
}
