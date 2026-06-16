//! [`Min`]/[`Max`] — the integer extreme folds. Paired with
//! [`Col<T>`](super::super::read::Col) they are `MIN`/`MAX` over an integer column;
//! they read the column exactly as [`Add`](super::Add)-backed `SUM` does and differ
//! only in [`combine`](Fold::combine).

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::cell::NumericCell;
use crate::operations::unary::group::values::read::SlotReader;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::sync::Arc;

/// Keeps the smallest contribution (integer `MIN`).
pub struct Min;
/// Keeps the largest contribution (integer `MAX`).
pub struct Max;

macro_rules! num_extreme {
    ($Fold:ident, $keep:ident) => {
        impl<A: NumericCell> Fold<A> for $Fold {
            type Cfg = ();
            #[inline(always)]
            fn cfg(_arena: &Arc<SharedArena>) {}
            #[inline(always)]
            fn seed(slot: &SlotReader<'_>, idx: usize, _arena: &mut WorkerArena) -> A {
                slot.read_num::<A>(idx)
            }
            #[inline(always)]
            fn combine(acc: A, incoming: A, _cfg: &()) -> A {
                acc.$keep(incoming)
            }
            #[inline(always)]
            fn sort_key(cell: A) -> i128 {
                cell.to_i128()
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

num_extreme!(Min, min);
num_extreme!(Max, max);
