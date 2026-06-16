//! [`StrMin`]/[`StrMax`] — the string extreme folds, over a `u128`
//! [`ArenaKey`](crate::operations::unary::group::ArenaKey) cell. Paired with
//! [`Str`](super::super::read::Str) they are `MIN`/`MAX` over a string column.
//!
//! `seed` persists the row's string; `update` compares the raw bytes against the
//! current cell and persists only a winner (a losing row never touches the
//! arena); `combine` picks the extreme of two already-persisted cells; `finish`
//! emits a zero-copy `Utf8View` array into the arena's ring buffers.

use super::Fold;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::values::read::{SlotReader, arena_key};
use arrow_array::{ArrayRef, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// Keeps the lexicographically smallest string (`MIN`).
pub struct StrMin;
/// Keeps the lexicographically largest string (`MAX`).
pub struct StrMax;

macro_rules! str_extreme {
    ($Fold:ident, $wins:tt) => {
        impl Fold<u128> for $Fold {
            type Cfg = Arc<SharedArena>;
            #[inline(always)]
            fn cfg(arena: &Arc<SharedArena>) -> Arc<SharedArena> {
                arena.clone()
            }
            #[inline(always)]
            fn seed(slot: &SlotReader<'_>, idx: usize, arena: &mut WorkerArena) -> u128 {
                slot.read_str(idx, arena)
            }
            #[inline(always)]
            fn update(
                acc: u128,
                slot: &SlotReader<'_>,
                idx: usize,
                arena: &mut WorkerArena,
                cfg: &Arc<SharedArena>,
            ) -> u128 {
                // Compare the raw bytes before persisting; a loser stays out of
                // the arena. (`cfg` and the worker arena's shared arena are the
                // same arena — the cell was persisted there.)
                let incoming = slot.str_at(idx);
                let current_key = arena_key(acc);
                let current = current_key.resolve(cfg);
                if incoming.as_bytes() $wins current {
                    arena.push(incoming).as_u128()
                } else {
                    acc
                }
            }
            #[inline(always)]
            fn combine(acc: u128, incoming: u128, cfg: &Arc<SharedArena>) -> u128 {
                let acc_key = arena_key(acc);
                let incoming_key = arena_key(incoming);
                if incoming_key.resolve(cfg) $wins acc_key.resolve(cfg) {
                    incoming
                } else {
                    acc
                }
            }
            #[inline(always)]
            fn sort_key(cell: u128) -> i128 {
                // Raw view bits, not lexicographic — a string extreme never feeds
                // an `ORDER BY <agg>` top-k (the planner doesn't push one).
                cell as i128
            }
            fn finish(
                name: &str,
                col: SlabColumn<u128>,
                arena: &Arc<SharedArena>,
            ) -> (Field, ArrayRef) {
                let len = col.len();
                let views = ScalarBuffer::<u128>::new(col.into_buffer(), 0, len);
                let buffers = arena.to_arrow_buffers();
                // Safety: the views are valid ArenaKeys and the arena (Arc-held in
                // each Buffer) outlives the array — same contract as string keys.
                let arr: ArrayRef =
                    Arc::new(unsafe { StringViewArray::new_unchecked(views, buffers, None) });
                (Field::new(name, DataType::Utf8View, false), arr)
            }
        }
    };
}

str_extreme!(StrMin, <);
str_extreme!(StrMax, >);
