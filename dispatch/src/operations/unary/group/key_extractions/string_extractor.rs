use crate::operations::KeyExtractor;
use crate::operations::unary::group::aggregations::{Count, GroupAggSlot};
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::arena_key::ResolvedKey;
use crate::operations::unary::group::hashtables::{Table, TableStorage, Value};
use crate::operations::unary::group::{ArenaKey, StringKey};
use ahash::RandomState;
use arrow_array::builder::UInt64Builder;
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::sync::Arc;

/// A `KeyExtractor` for a single `StringViewArray` column, counting
/// occurrences per key (`GROUP BY str_col` → `COUNT(*)`).
///
/// Live keys borrow the raw `&str` from the input array; new keys are
/// persisted into the shared arena as an `ArenaKey` (a u128 with the same
/// layout as Arrow's StringView). Output is zero-copy: the result
/// `StringViewArray` points directly into the arena's ring buffers.
pub struct StringKeyExtractor;

impl KeyExtractor for StringKeyExtractor {
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = StringKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Value = Count;
    type Reader<'b> = &'b StringViewArray;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _value_slots: &[GroupAggSlot],
    ) -> Self::Reader<'b> {
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

    #[inline(always)]
    fn value(_reader: &Self::Reader<'_>, _idx: usize) -> Count {
        Count::single()
    }

    fn resolve_persisted(arena: &SharedArena, persisted: ArenaKey) -> ResolvedKey<'_> {
        ResolvedKey {
            key: persisted,
            arena,
        }
    }

    fn create_record_batch<S: TableStorage<Self>>(
        table: Table<Self, S>,
        arena: &Arc<SharedArena>,
    ) -> Result<RecordBatch, ArrowError> {
        let mut views: Vec<u128> = Vec::with_capacity(table.len());
        let mut val_b = UInt64Builder::with_capacity(table.len());

        for entry in table.iter(0) {
            views.push(entry.key().as_u128());
            val_b.append_value(entry.value().value as u64);
        }

        let buffers = arena.to_arrow_buffers();
        // Safety: views were built from valid ArenaKeys; SharedArena (via Arc in each Buffer)
        // keeps ring memory alive as long as the StringViewArray exists.
        let keys: ArrayRef = Arc::new(unsafe {
            StringViewArray::new_unchecked(ScalarBuffer::from(views), buffers, None)
        });
        let vals: ArrayRef = Arc::new(val_b.finish());

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8View, false),
            Field::new("value", DataType::UInt64, false),
        ]));

        RecordBatch::try_new(schema, vec![keys, vals])
    }
}
