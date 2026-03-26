use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
use crate::operations::unary::group::hashtables::{
    DEFAULT_CAPACITY, KeyExtractor, MultiSlabTable, Value,
};
use ahash::RandomState;
use arrow_array::{Array, ArrayAccessor, ArrayRef};
use std::sync::Arc;

/// Per-worker aggregation state for the consume phase.
///
/// Holds a stack of [`SlabTable`]s. Incoming arrays are merged into the
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

    /// Hash every element in `col` using the shared hash state.
    #[inline(always)]
    fn compute_hashes(&mut self, col: K::ArrayRef<'_>) {
        let mut i = 0;
        let length = col.len();
        while i < length {
            self.hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            i += 1;
        }
    }

    /// Merge all rows from `array` into the current table stack.
    #[inline(always)]
    pub fn merge_array(&mut self, array: &ArrayRef) {
        const PREFETCH_DISTANCE: usize = 16;
        let array = K::downcast_column(array).unwrap();
        self.compute_hashes(array);

        // We always initialize maps with at least one map, so this is safe
        let mut table = self.tables.last_mut().unwrap();
        let length = array.len();

        let mut i = 0;

        while i < length {
            let hash = self.hashes[i];
            if i + PREFETCH_DISTANCE + 1 < length {
                let ph = self.hashes[i + PREFETCH_DISTANCE];
                table.prefetch(ph);
            }

            let key = K::live_key(&array, i, &mut self.worker_arena);
            table.merge::<false, _>(hash, key, Value::single());

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
