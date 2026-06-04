use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::values::GroupAggSlot;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    BatchRowSource, DEFAULT_CAPACITY, KeyExtractor, LiveKey, MultiSlabTable, ValueExtractor,
};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

/// Adapts a [`KeyExtractor`] + [`ValueExtractor`] + key arena into a
/// [`BatchRowSource`] for [`BaseHashTable::merge_batch`]. Holding the `&mut`
/// arena here (rather than in a closure that returns a borrowed live key) keeps
/// the arena borrow from escaping while still allowing string keys to be
/// persisted on insert.
struct RowSrc<'r, 'b, K: KeyExtractor, V: ValueExtractor> {
    key_reader: &'r K::Reader<'b>,
    // Held by value (not `&`): for a `COUNT(*)` value extractor the reader is a
    // ZST, so this is a zero-size field and `value()` compiles to a constant with
    // no per-row load — what a `&V::Reader` would otherwise force on every insert.
    value_reader: V::Reader<'b>,
    arena: &'r mut WorkerArena,
}

impl<K: KeyExtractor, V: ValueExtractor> BatchRowSource<K::Persisted, V::Value>
    for RowSrc<'_, '_, K, V>
{
    #[inline(always)]
    fn persisted(&mut self, i: usize) -> K::Persisted {
        K::live_key(self.key_reader, i, &mut *self.arena).persist()
    }
    #[inline(always)]
    fn key_eq(&mut self, i: usize, persisted: &K::Persisted) -> bool {
        K::live_key(self.key_reader, i, &mut *self.arena).eq_persisted(persisted)
    }
    #[inline(always)]
    fn value(&mut self, i: usize) -> V::Value {
        V::value(&self.value_reader, i)
    }
}

/// Per-worker aggregation state for the consume phase.
///
/// Holds a stack of [`MultiSlabTable`]s. Incoming arrays are merged into the
/// top table. When that table exceeds its load threshold, it is frozen
/// (pushed onto the stack) and a new, 2x-larger table is created for
/// subsequent rows. This avoids expensive in-place resizing while keeping
/// most insertions in a single hot table.
///
/// After consumption finishes, [`flush`](Self::flush) returns all tables
/// for merging in the output phase.
pub struct AggregatedTable<K: KeyExtractor, V: ValueExtractor> {
    hash_state: RandomState,
    worker_arena: WorkerArena,
    allocator: SlabAllocator,
    tables: Vec<MultiSlabTable<K, V>>,
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    /// Scratch for the batched multi-pass probe: current slot per row, and the
    /// two ping-pong selection vectors of still-unresolved row indices.
    slots: Box<[usize; RECORD_BATCH_SIZE]>,
    sel: Box<[u32; RECORD_BATCH_SIZE]>,
    sel_next: Box<[u32; RECORD_BATCH_SIZE]>,
}

impl<K: KeyExtractor, V: ValueExtractor> AggregatedTable<K, V> {
    /// Create a new aggregation state with a single small table.
    pub fn new(state: RandomState, shared_arena: Arc<SharedArena>) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = BaseHashTable::multi_slab(&mut allocator, DEFAULT_CAPACITY, 0);
        Self {
            hash_state: state,
            worker_arena: WorkerArena::new(shared_arena),
            allocator,
            tables: vec![table],
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            slots: vec![0usize; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            sel: vec![0u32; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            sel_next: vec![0u32; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    /// Allocate a new table with `size` slots, choosing single- or multi-slab
    /// based on whether the byte size fits in one buffer.
    #[inline(always)]
    fn create_new_table(&mut self) {
        // Grow by 4x rather than 2x. The frozen tables in the stack are pure
        // slack that must be allocated and zeroed every query: their cumulative
        // capacity is `final * r/(r-1)`, so 4x growth wastes ~1.33x the final
        // size vs doubling's 2x. Fewer, lower-loaded tables also mean fewer
        // sources for the merge to scan. At very large group counts this is gigabytes less
        // memory to zero each query.
        let new_size = self.tables.last().unwrap().capacity() * 4;
        self.tables
            .push(BaseHashTable::multi_slab(&mut self.allocator, new_size, 0));
    }

    /// Merge all rows of `batch` into the current table stack, reading the key
    /// columns through `K` and the per-row aggregate values through `V`.
    ///
    /// Hashes the whole batch up front, then dispatches to either the batched
    /// multi-pass probe ([`consume_batched`](Self::consume_batched)) or the
    /// scalar per-row fallback ([`consume_scalar`](Self::consume_scalared)).
    #[inline(always)]
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[GroupAggSlot],
    ) {
        let key_reader = K::make_reader(batch, key_cols);
        let value_reader = V::make_reader(batch, value_slots);
        let length = batch.num_rows();

        let mut i = 0;
        while i < length {
            self.hashes[i] = K::hash(&key_reader, i, &self.hash_state);
            i += 1;
        }

        // The active table can take the whole batch iff no probe can overflow it
        // mid-batch — only then is the batched path safe (every probe is
        // guaranteed an empty slot). The large final table, where ~all of the
        // inserts land, always meets this; only the small early tables fall
        // back to the scalar loop.
        let free = {
            let t = self.tables.last().unwrap();
            t.capacity() - t.len()
        };
        if free > length {
            self.consume_batched(length, &key_reader, value_reader);
        } else {
            self.consume_scalared(length, &key_reader, &value_reader);
        }
    }

    /// Resolve a whole batch against the active table in one batched multi-pass
    /// probe — used when the table has room for every row, so no mid-batch table
    /// switch is needed. Resolves all rows with high memory-level parallelism
    /// instead of one stalling probe chain at a time.
    #[inline(always)]
    fn consume_batched<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: V::Reader<'b>,
    ) {
        // `value_reader` is moved into `src` (held by value so a ZST value reader
        // stays zero-size); the scalar path takes it by reference instead.
        let mut src = RowSrc::<K, V> {
            key_reader,
            value_reader,
            arena: &mut self.worker_arena,
        };
        self.tables.last_mut().unwrap().merge_batch(
            length,
            &mut self.hashes[..],
            &mut self.slots[..],
            &mut self.sel[..],
            &mut self.sel_next[..],
            &mut src,
        );
        if self.tables.last().unwrap().undersized() {
            self.create_new_table();
        }
    }

    /// Resolve a batch row-by-row, switching to a fresh table when the current
    /// one overflows mid-batch — used only for the small early tables. Two-level
    /// software prefetch (DRAM→L2 far, then L2→L1 near) hides the per-row probe
    /// latency.
    #[inline(always)]
    fn consume_scalared<'b>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
    ) {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        let mut table = self.tables.last_mut().unwrap();
        let mut i = 0;
        while i < length {
            let hash = self.hashes[i];
            if i + L2_DISTANCE < length {
                table.prefetch_l2(self.hashes[i + L2_DISTANCE]);
            }
            if i + L1_DISTANCE < length {
                table.prefetch(self.hashes[i + L1_DISTANCE]);
            }

            let key = K::live_key(key_reader, i, &mut self.worker_arena);
            let value = V::value(value_reader, i);
            table.merge::<false, _>(hash, key, value);

            if table.undersized() {
                self.create_new_table();
                table = self.tables.last_mut().unwrap();
            }
            i += 1;
        }
    }

    /// Finalize this worker's aggregation: flush the arena and return all tables.
    pub fn flush(self) -> Vec<MultiSlabTable<K, V>> {
        self.worker_arena.flush();
        self.tables
    }
}
