use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::arena_key::ResolvedKey;
use crate::operations::unary::group::key_extractions::{KeyColumns, KeyExtractor};
use crate::operations::unary::group::{ArenaKey, StringKey};
use ahash::RandomState;
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// A [`KeyExtractor`] for a single `StringViewArray` column.
///
/// Live keys borrow the raw `&str` from the input array; new keys are
/// persisted into the shared arena as an [`ArenaKey`] (a `u128` with the same
/// layout as Arrow's StringView). Output is zero-copy: the result
/// `StringViewArray` points directly into the arena's ring buffers.
pub struct StringKeyExtractor;

impl KeyExtractor for StringKeyExtractor {
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = StringKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Reader<'b> = &'b StringViewArray;
    type Columns = StringKeyColumns;

    fn make_reader<'b>(batch: &'b RecordBatch, key_cols: &[usize]) -> Self::Reader<'b> {
        batch
            .column(key_cols[0])
            .as_any()
            .downcast_ref::<StringViewArray>()
            .expect("string key column type mismatch")
    }

    #[inline(always)]
    fn rows(reader: &Self::Reader<'_>) -> usize {
        reader.len()
    }

    #[inline(always)]
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(unsafe { reader.value_unchecked(idx) })
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        let val = unsafe { reader.value_unchecked(idx) };
        StringKey::new(arena, val)
    }

    fn resolve_persisted(arena: &SharedArena, persisted: ArenaKey) -> ResolvedKey<'_> {
        ResolvedKey {
            key: persisted,
            arena,
        }
    }
}

/// Emits the string key column as a zero-copy `StringViewArray` whose views
/// point into the shared arena's ring buffers.
pub struct StringKeyColumns {
    views: Vec<u128>,
}

impl KeyColumns for StringKeyColumns {
    type Key = ArenaKey;

    fn with_capacity(rows: usize) -> Self {
        Self {
            views: Vec::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &ArenaKey) {
        self.views.push(key.as_u128());
    }

    fn finish(self, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        let buffers = arena.to_arrow_buffers();
        // Safety: views were built from valid ArenaKeys; SharedArena (via Arc in
        // each Buffer) keeps the ring memory alive as long as the array exists.
        let keys: ArrayRef = Arc::new(unsafe {
            StringViewArray::new_unchecked(ScalarBuffer::from(self.views), buffers, None)
        });
        let fields = vec![Field::new("key", DataType::Utf8View, false)];
        (fields, vec![keys])
    }
}
