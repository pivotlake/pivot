//! [`StringExtreme`] — a homogeneous string `MIN`/`MAX` value: every slot keeps
//! the extreme string so far, persisted lazily into the worker arena.
//!
//! The cell is an [`ArenaKey`] (a `u128` StringView into the arena). A new group
//! persists its first string; an existing group persists an incoming string only
//! when it beats the current extreme — so a row that loses never touches the
//! arena (the reason consume is reader-driven). The partition merge picks the
//! extreme of two already-persisted keys, resolving them through the shared arena,
//! with no new persist. Output is a zero-copy `StringViewArray` into the arena's
//! ring buffers, exactly like the string *key* column.

use super::fold::{Max, Min};
use super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::keys::ArenaKey;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::marker::PhantomData;
use std::sync::Arc;

/// Which extreme a string slot keeps. Implemented for [`Min`]/[`Max`] (reused from
/// [`fold`](super::fold), though string extremes compare bytes rather than add).
pub trait StrFold: Send + Sync + 'static {
    /// Whether `incoming` should replace the current extreme `current`.
    fn keep_incoming(incoming: &[u8], current: &[u8]) -> bool;
}

impl StrFold for Min {
    #[inline(always)]
    fn keep_incoming(incoming: &[u8], current: &[u8]) -> bool {
        incoming < current
    }
}

impl StrFold for Max {
    #[inline(always)]
    fn keep_incoming(incoming: &[u8], current: &[u8]) -> bool {
        incoming > current
    }
}

/// `N` string-extreme cells, each the current `MIN`/`MAX` string as an [`ArenaKey`].
pub struct StringExtreme<F, const N: usize> {
    keys: [ArenaKey; N],
    _fold: PhantomData<F>,
}

impl<F, const N: usize> Copy for StringExtreme<F, N> {}
impl<F, const N: usize> Clone for StringExtreme<F, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F, const N: usize> Default for StringExtreme<F, N> {
    fn default() -> Self {
        Self {
            keys: [ArenaKey::default(); N],
            _fold: PhantomData,
        }
    }
}

/// Per-batch reader: one `StringViewArray` per slot.
pub struct StringReader<'b, const N: usize> {
    cols: [&'b StringViewArray; N],
}

impl<const N: usize> StringReader<'_, N> {
    #[inline(always)]
    fn str_at(&self, slot: usize, row: usize) -> &str {
        self.cols[slot].value(row)
    }
}

impl<F: StrFold, const N: usize> AggregationValue for StringExtreme<F, N> {
    type Reader<'b> = StringReader<'b, N>;
    type MergeConfig = Arc<SharedArena>;
    type Columns = [SlabColumn<u128>; N];
    type SortKey = u128;

    fn merge_config(_slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Arc<SharedArena> {
        arena.clone()
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> StringReader<'b, N> {
        assert_eq!(slots.len(), N, "slot count must match N");
        StringReader {
            cols: std::array::from_fn(|s| batch.column(slots[s].column).as_string_view()),
        }
    }

    #[inline(always)]
    fn value(reader: &StringReader<'_, N>, idx: usize, arena: &mut WorkerArena) -> Self {
        Self {
            keys: std::array::from_fn(|s| arena.push(reader.str_at(s, idx))),
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &StringReader<'_, N>,
        idx: usize,
        arena: &mut WorkerArena,
        _cfg: &Arc<SharedArena>,
    ) -> Self {
        let mut keys = self.keys;
        for (s, key) in keys.iter_mut().enumerate() {
            let incoming = reader.str_at(s, idx);
            // Resolve borrows the arena immutably just long enough to decide; the
            // winning string is then persisted (mutable). Losers never persist.
            let wins = F::keep_incoming(incoming.as_bytes(), key.resolve(arena.shared()));
            if wins {
                *key = arena.push(incoming);
            }
        }
        Self {
            keys,
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn merge(self, other: Self, arena: &Arc<SharedArena>) -> Self {
        Self {
            keys: std::array::from_fn(|s| {
                if F::keep_incoming(other.keys[s].resolve(arena), self.keys[s].resolve(arena)) {
                    other.keys[s]
                } else {
                    self.keys[s]
                }
            }),
            _fold: PhantomData,
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> u128 {
        // Raw view bits, not lexicographic — a string extreme never feeds an
        // `ORDER BY <agg>` top-k (the planner doesn't push one for it).
        self.keys[slot].as_u128()
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> [SlabColumn<u128>; N] {
        std::array::from_fn(|_| SlabColumn::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut [SlabColumn<u128>; N]) {
        for (s, col) in cols.iter_mut().enumerate() {
            col.push(self.keys[s].as_u128());
        }
    }

    fn finish_columns(
        cols: [SlabColumn<u128>; N],
        arena: &Arc<SharedArena>,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let buffers = arena.to_arrow_buffers();
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let len = col.len();
            let views = ScalarBuffer::<u128>::new(col.into_buffer(), 0, len);
            // Safety: the views are valid ArenaKeys and the arena (Arc-held in
            // each Buffer) outlives the array — same contract as string keys.
            let arr: ArrayRef =
                Arc::new(unsafe { StringViewArray::new_unchecked(views, buffers.clone(), None) });
            fields.push(Field::new(format!("v{s}"), DataType::Utf8View, false));
            columns.push(arr);
        }
        (fields, columns)
    }
}
