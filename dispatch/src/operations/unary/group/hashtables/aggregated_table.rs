use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::aggregations::GroupAggSlot;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{DEFAULT_CAPACITY, KeyExtractor, MultiSlabTable};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::sync::Arc;

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
pub struct AggregatedTable<K: KeyExtractor> {
    hash_state: RandomState,
    worker_arena: WorkerArena,
    allocator: SlabAllocator,
    tables: Vec<MultiSlabTable<K>>,
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
}

impl<K: KeyExtractor> AggregatedTable<K> {
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
        }
    }

    /// Allocate a new table with `size` slots, choosing single- or multi-slab
    /// based on whether the byte size fits in one buffer.
    #[inline(always)]
    fn create_new_table(&mut self) {
        let new_size = self.tables.last().unwrap().capacity() * 2;
        self.tables
            .push(BaseHashTable::multi_slab(&mut self.allocator, new_size, 0));
    }

    /// Merge all rows of `batch` into the current table stack, reading the
    /// key columns and per-row aggregate values through the extractor.
    #[inline(always)]
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_cols: &[usize],
        value_slots: &[GroupAggSlot],
    ) {
        const PREFETCH_DISTANCE: usize = 16;
        let reader = K::make_reader(batch, key_cols, value_slots);
        let length = K::rows(&reader);

        let mut i = 0;
        while i < length {
            self.hashes[i] = K::hash(&reader, i, &self.hash_state);
            i += 1;
        }

        // We always initialize maps with at least one map, so this is safe
        let mut table = self.tables.last_mut().unwrap();

        let mut i = 0;
        while i < length {
            let hash = self.hashes[i];
            if i + PREFETCH_DISTANCE + 1 < length {
                let ph = self.hashes[i + PREFETCH_DISTANCE];
                table.prefetch(ph);
            }

            let key = K::live_key(&reader, i, &mut self.worker_arena);
            let value = K::value(&reader, i);
            table.merge::<false, _>(hash, key, value);

            if table.undersized() {
                self.create_new_table();
                table = self.tables.last_mut().unwrap();
            }
            i += 1;
        }
    }
    /// Finalize this worker's aggregation: flush the arena and return all tables.
    pub fn flush(self) -> Vec<MultiSlabTable<K>> {
        self.worker_arena.flush();
        self.tables
    }
}
