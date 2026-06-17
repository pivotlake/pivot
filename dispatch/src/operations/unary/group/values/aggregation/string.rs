//! [`StrMin`] / [`StrMax`] — string extremes over a `StringViewArray`, with an
//! [`ArenaKey`] cell. `update` holds the real `&str`, so it compares *before*
//! persisting and only a winner ever touches the value arena (lazy). `finish`
//! emits a zero-copy `Utf8View` array into the arena's ring buffers.

use super::Aggregation;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::keys::ArenaKey;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// `MIN` over a string column.
pub struct StrMin;
/// `MAX` over a string column.
pub struct StrMax;

macro_rules! str_extreme {
    ($Op:ident, $wins:tt) => {
        impl Aggregation for $Op {
            type Acc = ArenaKey;
            type Input<'b> = &'b StringViewArray;
            type Cfg = Arc<SharedArena>;

            #[inline(always)]
            fn bind(batch: &RecordBatch, column: usize) -> &StringViewArray {
                batch.column(column).as_string_view()
            }
            #[inline(always)]
            fn cfg(arena: &Arc<SharedArena>) -> Arc<SharedArena> {
                arena.clone()
            }

            #[inline(always)]
            fn seed(input: &&StringViewArray, idx: usize, arena: &mut WorkerArena) -> ArenaKey {
                arena.push(input.value(idx))
            }
            #[inline(always)]
            fn update(
                acc: ArenaKey,
                input: &&StringViewArray,
                idx: usize,
                arena: &mut WorkerArena,
                cfg: &Arc<SharedArena>,
            ) -> ArenaKey {
                // Raw bytes vs the current extreme; persist only a winner.
                let incoming = input.value(idx);
                if incoming.as_bytes() $wins acc.resolve(cfg) {
                    arena.push(incoming)
                } else {
                    acc
                }
            }
            #[inline(always)]
            fn merge(a: ArenaKey, b: ArenaKey, cfg: &Arc<SharedArena>) -> ArenaKey {
                if b.resolve(cfg) $wins a.resolve(cfg) { b } else { a }
            }
            #[inline(always)]
            fn sort_key(acc: ArenaKey) -> i128 {
                // Raw view bits, not lexicographic — a string extreme never feeds
                // an `ORDER BY <agg>` top-k (the planner doesn't push one).
                acc.as_u128() as i128
            }
            fn finish(
                name: &str,
                col: SlabColumn<ArenaKey>,
                arena: &Arc<SharedArena>,
            ) -> (Field, ArrayRef) {
                let len = col.len();
                // `ArenaKey` is a transparent `u128`; reinterpret the slab as
                // StringView headers (zero-copy) over the arena's ring buffers.
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
