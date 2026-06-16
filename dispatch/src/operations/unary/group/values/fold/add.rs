//! [`Add`] — the additive fold (`COUNT`/`SUM`).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::NumericCell;
use crate::operations::unary::group::values::read::SlotReader;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::sync::Arc;

/// Sums contributions. With [`One`](super::super::read::One) this is `COUNT`; with
/// [`Col<T>`](super::super::read::Col), `SUM`.
pub struct Add;

impl<A: NumericCell> Fold<A> for Add {
    type Cfg = ();
    #[inline(always)]
    fn cfg(_arena: &Arc<SharedArena>) {}
    #[inline(always)]
    fn seed(slot: &SlotReader<'_>, idx: usize, _arena: &mut WorkerArena) -> A {
        slot.read_num::<A>(idx)
    }
    #[inline(always)]
    fn combine(acc: A, incoming: A, _cfg: &()) -> A {
        acc.add(incoming)
    }
    #[inline(always)]
    fn sort_key(cell: A) -> i128 {
        cell.to_i128()
    }
    fn finish(name: &str, col: SlabColumn<A>, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        A::finish(name, col)
    }
}
