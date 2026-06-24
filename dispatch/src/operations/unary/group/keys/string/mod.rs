mod arena_key;
mod live_key;

pub use arena_key::ArenaKey;
pub use live_key::{ResolvedKey, StringKey};

use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::keys::{KeyColumns, KeyExtractor};
use ahash::RandomState;
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_buffer::{Buffer, ScalarBuffer};
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
    // Long string keys benefit from abandon (dedup-during-scan avoids copying every
    // occurrence into the arena); fixed-width keys scatter instead.
    const RADIX_ABANDON: bool = true;
    type Config = ();
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = StringKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Reader<'b> = &'b StringViewArray;
    type Columns = StringKeyColumn;
    type Scratch = ();

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
        _scratch: &'b mut (),
    ) -> Self::Reader<'b> {
        batch
            .column(key_cols[0])
            .as_any()
            .downcast_ref::<StringViewArray>()
            .expect("string key column type mismatch")
    }

    #[inline(always)]
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, state: &RandomState, hashes: &mut [u64]) {
        for (i, h) in hashes.iter_mut().enumerate() {
            *h = state.hash_one(unsafe { reader.value_unchecked(i) });
        }
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
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

/// Emits the string key column as a zero-copy `StringViewArray` whose views point
/// into the shared arena's ring buffers.
///
/// All output batches share the one `Arc<[Buffer]>` built once for the output
/// phase (see [`OutputAccumulator`](super::super::super::output::OutputAccumulator)),
/// so each batch only clones that `Arc` (a single refcount bump) rather than
/// re-wrapping every arena buffer. The downstream `concat`/`take` (arrow-select)
/// detect the shared `Arc` and reuse it, so a high-partition-count result never
/// rebuilds the buffer list, keeping the whole output path off the millions of
/// per-buffer refcount operations a naive per-batch wrap would cost.
pub struct StringKeyColumn {
    views: SlabColumn<u128>,
}

impl KeyColumns for StringKeyColumn {
    type Key = ArenaKey;
    type Config = ();

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _config: &()) -> Self {
        // The view headers (one `u128` per group) are built onto a slab, like the
        // primitive output columns; the string bytes they point at already live
        // on the shared arena's ring buffers.
        Self {
            views: SlabColumn::with_capacity(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &ArenaKey) {
        self.views.push(key.as_u128());
    }

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        output_buffers: &Arc<[Buffer]>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let len = self.views.len();
        let views = ScalarBuffer::<u128>::new(self.views.into_buffer(), 0, len);
        // Zero-copy: the views already point into the arena, and every output batch
        // shares the one `Arc<[Buffer]>` built for this output phase. Cloning the
        // `Arc` is a single refcount bump, and the downstream `concat`/`take`
        // fast-paths reuse it (see arrow-select), so the buffer list is never
        // rebuilt per batch.
        // Safety: views were built from valid ArenaKeys, and the shared buffers (via
        // their `Arc<SharedArena>`) keep the ring memory alive as long as the array.
        let keys: ArrayRef = Arc::new(unsafe {
            StringViewArray::new_unchecked(views, output_buffers.clone(), None)
        });
        let fields = vec![Field::new("key", DataType::Utf8View, false)];
        (fields, vec![keys])
    }
}
