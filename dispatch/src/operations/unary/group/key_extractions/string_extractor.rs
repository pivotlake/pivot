use crate::operations::KeyExtractor;
use crate::operations::unary::group::aggregations::Count;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::arena_key::ResolvedKey;
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use crate::operations::unary::group::{ArenaKey, StringKey};
use arrow_array::builder::UInt64Builder;
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::sync::Arc;

/// A `KeyExtractor` for `StringViewArray` columns. This allows creating group bys with Strings
/// as keys.
///
/// Live keys borrow the raw `&str` from the input array and hold a mutable
/// reference to the `WorkerArena`. If a key turns out to be new (not
/// already in the hash table), `LiveKey::persist` pushes the string bytes
/// into the arena and returns an `ArenaKey` — a u128 with the same layout
/// as Arrow's StringView (inline for ≤12 bytes, otherwise a buffer
/// index + offset).
///
/// During the merge phase, persisted keys are resolved back to byte slices
/// via `ResolvedKey`, which borrows from the `SharedArena`.
///
/// Output uses zero-copy: the `StringViewArray` in the result batch points
/// directly into the arena's ring buffers (kept alive via `Arc<SharedArena>`). The u128s are also
/// copied as is
pub struct StringKeyExtractor;

impl KeyExtractor for StringKeyExtractor {
    type ArrayRef<'a> = &'a StringViewArray;
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = StringKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Value = Count;

    #[inline(always)]
    fn live_key<'a, 'b>(
        column: &Self::ArrayRef<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        let val = unsafe { column.value_unchecked(idx) };
        StringKey::new(arena, val)
    }

    fn resolve_persisted<'a>(arena: &'a SharedArena, persisted: ArenaKey) -> ResolvedKey<'a> {
        ResolvedKey {
            key: persisted,
            arena,
        }
    }

    fn downcast_column(column: &dyn Array) -> Option<&StringViewArray> {
        column.as_any().downcast_ref::<StringViewArray>()
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
