use std::cell::UnsafeCell;
use std::sync::{Arc, OnceLock, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use arrow_row::Row;
use arrow_array::{ArrayRef, RecordBatch};
use crate::NUM_WORKERS;
use crate::operations::binary::join::hashtable::Directory;

type Value = (u64, u64);

pub struct Barrier<T> {
    remaining: AtomicUsize,
    value: OnceLock<T>
}


impl<T: Send + Sync> Barrier<T> {
    pub fn new(num_workers: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(num_workers),
            value: OnceLock::new(),
        }
    }

    /// Call exactly once per worker. The last arrival runs `f` to produce
    /// the next phase's shared state.
    pub fn arrive(&self, f: impl FnOnce() -> T) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _ = self.value.set(f());
        }
    }

    /// Non-blocking poll. Returns Some once all workers have arrived
    /// and the last one has initialized the result.
    pub fn try_get(&self) -> Option<&T> {
        self.value.get()
    }
}



struct SettingCountsShared {
    counts_per_hash_per_worker: Arc<Vec<UnsafeCell<usize>>>,
    writing_barrier: Arc<Barrier<WritingShared>>
}

struct WritingShared {
    row_arena: Arc<UnsafeCell<Vec<Value>>>,
    // Directory into row arena
    directory: Arc<UnsafeCell<Directory>>,
}

enum BuildState {
    Hashing {
        hashes: Vec<u64>,
        total_count: Arc<AtomicUsize>,
        barrier: Arc<Barrier<SettingCountsShared>>
    },
    SettingCounts(SettingCountsShared),
    Writing(WritingShared)
}


impl BuildState {
    #[inline(always)]
    fn compute_hashes(&mut self, col: ArrayRef) {
        // let mut i = 0;
        // let length = col.len();
        // while i < length {
        //     self.hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
        //     i += 1;
        // }
    }

    pub fn insert(&mut self, batch: RecordBatch) {
        match self {
            BuildState::Hashing { hashes, .. } => {
                // hashes.push()
            },
            _ => panic!("impossible")
        }
    }

    pub fn try_build(&self) {
        match self {
            BuildState::Hashing { hashes, total_count, barrier } => {
                total_count.fetch_add(hashes.len(), Ordering::SeqCst);
                barrier.arrive(|| {
                    let count = total_count.load(Ordering::SeqCst);
                    let cells = Arc::new(Vec::with_capacity(0..count * NUM_WORKERS));
                })
            }
            BuildState::SettingCounts(_) => {}
            BuildState::Writing(_) => {}
        }
    }
}