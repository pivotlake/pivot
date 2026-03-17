/// A per-worker work-stealing pool of free buffer indices, split into zeroed and dirty pools.
///
/// Each worker thread owns two [`FreePool`] instances: one for zeroed buffers and one for dirty.
/// When a worker needs a write buffer it specifies whether it prefers zeroed buffer or not. This is
/// a preference and is not promised (the caller must check `.zeroed` on the `WriteBuffer` to
/// verify). Pop does not zero out the buffer as the caller may be able to zero out a portion
/// instead of the whole buffer.
///
/// The pop logic tries the preferred pool through all 3 tiers (local deque → injector → steal),
/// then falls back to the other pool.
///
/// This keeps the common path lock-free and NUMA-friendly because buffers tend to stay on the
/// core that faulted them in. LIFO is also importantt since it means we'll re-use buffers hot
/// in our cache.
///
/// Returning a buffer routes it back to its *home worker* (`idx % NUM_WORKERS`) and into the
/// correct pool (zeroed or dirty) based on the caller's knowledge of the buffer's state.
use crate::memory::BUFFER_SIZE;
use crate::memory::ring::RING;
use crate::num_workers;
use crate::worker::WORKER_IDX;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::cell::RefCell;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, LazyLock, Mutex};

static ZEROED_REGISTRY: LazyLock<Mutex<Vec<Stealer<usize>>>> =
    LazyLock::new(|| Mutex::new(Vec::with_capacity(num_workers())));
static DIRTY_REGISTRY: LazyLock<Mutex<Vec<Stealer<usize>>>> =
    LazyLock::new(|| Mutex::new(Vec::with_capacity(num_workers())));

static BARRIER: LazyLock<Barrier> = LazyLock::new(|| Barrier::new(num_workers()));

static ZEROED_INJECTORS: LazyLock<Vec<Injector<usize>>> =
    LazyLock::new(|| (0..num_workers()).map(|_| Injector::new()).collect());
static DIRTY_INJECTORS: LazyLock<Vec<Injector<usize>>> =
    LazyLock::new(|| (0..num_workers()).map(|_| Injector::new()).collect());

/// A work-stealing deque of free buffer indices.
///
/// Each instance holds a local LIFO deque, a snapshot of peer stealers, and a reference
/// to its corresponding global injectors. The 3-tier pop logic (local → injector → steal)
/// is written once here; the zeroed/dirty distinction is handled by having two instances
/// per worker, each wired to different statics.
struct FreePool {
    worker: Worker<usize>,
    worker_idx: usize,
    stealers: Vec<Stealer<usize>>,
    injectors: &'static [Injector<usize>],
    last_stealer_idx: AtomicUsize,
}

struct Pools {
    zeroed: FreePool,
    dirty: FreePool,
}

thread_local! {
    static LOCAL_POOLS: RefCell<Option<Pools>> = const { RefCell::new(None) };
}

impl FreePool {
    /// Try to obtain a free buffer index.
    ///
    /// 1. Pop from the local LIFO deque (cheapest – no contention).
    /// 2. Drain the per-worker injector (picks up buffers returned by other threads).
    /// 3. Steal from another worker's deque (round-robin starting from `last_stealer_idx`).
    ///
    /// Returns `None` only when every source is empty.
    fn pop(&self) -> Option<usize> {
        if let Some(idx) = self.worker.pop() {
            return Some(idx);
        }

        loop {
            match self.injectors[self.worker_idx].steal() {
                Steal::Success(idx) => return Some(idx),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }

        let start = self.last_stealer_idx.fetch_add(1, Ordering::Relaxed);
        for i in 0..self.stealers.len() {
            if let Steal::Success(idx) = self.stealers[(start + i) % self.stealers.len()].steal() {
                return Some(idx);
            }
        }

        None
    }

    /// Push a buffer index back into this pool's local deque.
    fn push(&self, idx: usize) {
        self.worker.push(idx);
    }
}

/// Initialize both free pools (zeroed + dirty) for the calling worker thread.
///
/// Must be called exactly once per worker. Each worker:
/// 1. Creates two LIFO deques and publishes their stealer handles.
/// 2. Waits at a barrier until all workers have registered.
/// 3. Snapshots the stealers lists so it can steal from any peer.
/// 4. Pre-faults and zeroes its share of ring buffers (strided by `NUM_WORKERS`),
///    then pushes them into the zeroed pool.
pub fn init_free_pool(worker_idx: usize) {
    LOCAL_POOLS.with(|p| {
        let zeroed_worker = Worker::new_lifo();
        let dirty_worker = Worker::new_lifo();
        ZEROED_REGISTRY
            .lock()
            .unwrap()
            .push(zeroed_worker.stealer());
        DIRTY_REGISTRY.lock().unwrap().push(dirty_worker.stealer());
        BARRIER.wait();
        let zeroed_stealers = ZEROED_REGISTRY.lock().unwrap().clone();
        let dirty_stealers = DIRTY_REGISTRY.lock().unwrap().clone();
        let stealer_start = (WORKER_IDX.get() + 1) % num_workers();
        *p.borrow_mut() = Some(Pools {
            zeroed: FreePool {
                worker: zeroed_worker,
                worker_idx,
                stealers: zeroed_stealers,
                injectors: &ZEROED_INJECTORS,
                last_stealer_idx: AtomicUsize::new(stealer_start),
            },
            dirty: FreePool {
                worker: dirty_worker,
                worker_idx,
                stealers: dirty_stealers,
                injectors: &DIRTY_INJECTORS,
                last_stealer_idx: AtomicUsize::new(stealer_start),
            },
        });

        // Pre-fault buffers (strided by NUM_WORKERS) so each worker faults different pages.
        // We forget the WriteBuffer to avoid the Drop impl pushing to the dirty pool,
        // then manually release the slot and push to the zeroed pool.
        for i in (worker_idx..RING.len()).step_by(num_workers()) {
            let mut write = RING.try_write(i).unwrap();
            for j in (0..BUFFER_SIZE).step_by(4096) {
                write.as_mut()[j] = 1u8;
            }
            write.zero_out();
            black_box(());
        }
    });
}

/// Return a buffer index to the pool.
///
/// Routes the index to the home worker's zeroed or dirty pool. If the current thread
/// *is* the home worker, the push goes directly into the local deque. Otherwise it is
/// placed into the home worker's injector so the owner retrieves it on its next pop.
pub fn push_free_idx(idx: usize, zeroed: bool) {
    let home_worker = idx % num_workers();

    LOCAL_POOLS.with(|l| match l.borrow().as_ref() {
        Some(pools) if pools.zeroed.worker_idx == home_worker => {
            if zeroed {
                pools.zeroed.push(idx)
            } else {
                pools.dirty.push(idx)
            }
        }
        _ => {
            if zeroed {
                ZEROED_INJECTORS[home_worker].push(idx);
            } else {
                DIRTY_INJECTORS[home_worker].push(idx);
            }
        }
    })
}

/// Pop a free buffer index for the current worker thread.
///
/// When `prefer_zeroed` is true, tries the zeroed pool first then dirty.
/// When false, tries dirty first then zeroed.
///
/// Panics if called from a thread that has not been initialized via [`init_free_pool`]
/// or [`init_test_free_pool`].
pub fn pop_free_idx(prefer_zeroed: bool) -> Option<usize> {
    LOCAL_POOLS.with(|l| {
        let pools = l.borrow();
        let pools = pools
            .as_ref()
            .expect("Cannot allocate free index from nonworker thread");
        let (first, second) = if prefer_zeroed {
            (&pools.zeroed, &pools.dirty)
        } else {
            (&pools.dirty, &pools.zeroed)
        };
        first.pop().or_else(|| second.pop())
    })
}

pub fn pop_dirty_idx() -> Option<usize> {
    LOCAL_POOLS.with(|l| {
        let pools = l.borrow();
        let pools = pools
            .as_ref()
            .expect("Cannot allocate free index from nonworker thread");
        pools.dirty.pop()
    })
}

/// Drain any stale values from the worker-0 injectors.
///
/// Called at the start of each test to prevent cross-test contamination via the
/// global injector queues (which persist across tests in the same process).
#[cfg(test)]
fn drain_test_injectors() {
    loop {
        match ZEROED_INJECTORS[0].steal() {
            Steal::Success(_) => continue,
            Steal::Empty => break,
            Steal::Retry => continue,
        }
    }
    loop {
        match DIRTY_INJECTORS[0].steal() {
            Steal::Success(_) => continue,
            Steal::Empty => break,
            Steal::Retry => continue,
        }
    }
}

/// Initialize minimal free pools for the current test thread.
///
/// Allocates `count` unique buffer indices from a global atomic counter so parallel
/// tests never share the same Ring slot. All indices are pushed to the dirty pool
/// (callers that prefer zeroed will fall back to dirty transparently).
#[cfg(test)]
pub fn init_test_free_pool(count: usize) {
    use std::sync::atomic::AtomicUsize;
    static NEXT_IDX: AtomicUsize = AtomicUsize::new(0);

    // Unit tests don't call init(), so ensure NUM_WORKERS is set.
    crate::NUM_WORKERS.get_or_init(|| 1);

    drain_test_injectors();

    LOCAL_POOLS.with(|p| {
        let start = NEXT_IDX.fetch_add(count, Ordering::Relaxed);
        let dirty = Worker::new_lifo();
        for i in start..start + count {
            dirty.push(i);
        }
        *p.borrow_mut() = Some(Pools {
            zeroed: FreePool {
                worker: Worker::new_lifo(),
                worker_idx: 0,
                stealers: vec![],
                injectors: &ZEROED_INJECTORS,
                last_stealer_idx: Default::default(),
            },
            dirty: FreePool {
                worker: dirty,
                worker_idx: 0,
                stealers: vec![],
                injectors: &DIRTY_INJECTORS,
                last_stealer_idx: Default::default(),
            },
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that check injector state or assert pool emptiness,
    /// preventing cross-test contamination through the global injector queues.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Two indices that both route to worker 0 (home_worker = idx % NUM_WORKERS).
    fn home_indices() -> (usize, usize) {
        (0, num_workers())
    }

    #[test]
    fn pop_returns_none_when_empty() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);

        assert_eq!(pop_free_idx(true), None);
        assert_eq!(pop_free_idx(false), None);
    }

    #[test]
    fn push_dirty_then_pop_dirty() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        push_free_idx(0, false);

        assert_eq!(pop_free_idx(false), Some(0));
    }

    #[test]
    fn push_zeroed_then_pop_zeroed() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        push_free_idx(0, true);

        assert_eq!(pop_free_idx(true), Some(0));
    }

    #[test]
    fn prefer_zeroed_picks_zeroed_over_dirty() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        let (dirty_idx, zeroed_idx) = home_indices();
        push_free_idx(dirty_idx, false);
        push_free_idx(zeroed_idx, true);

        assert_eq!(pop_free_idx(true), Some(zeroed_idx));
    }

    #[test]
    fn prefer_dirty_picks_dirty_over_zeroed() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        let (dirty_idx, zeroed_idx) = home_indices();
        push_free_idx(zeroed_idx, true);
        push_free_idx(dirty_idx, false);

        assert_eq!(pop_free_idx(false), Some(dirty_idx));
    }

    #[test]
    fn prefer_zeroed_falls_back_to_dirty() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        push_free_idx(0, false);

        assert_eq!(pop_free_idx(true), Some(0));
    }

    #[test]
    fn prefer_dirty_falls_back_to_zeroed() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        push_free_idx(0, true);

        assert_eq!(pop_free_idx(false), Some(0));
    }

    #[test]
    fn pop_drains_both_pools() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);
        let (a, b) = home_indices();
        push_free_idx(a, false);
        push_free_idx(b, true);

        let mut results = vec![pop_free_idx(false).unwrap(), pop_free_idx(false).unwrap()];
        results.sort();

        assert_eq!(results, vec![a, b]);
        assert_eq!(pop_free_idx(false), None);
    }

    #[test]
    fn cross_thread_push_lands_in_injector() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);

        // Push from a thread that has no pool — hits the `_ =>` branch,
        // routing to DIRTY_INJECTORS[0] (since 0 % NUM_WORKERS == 0).
        std::thread::spawn(|| push_free_idx(0, false))
            .join()
            .unwrap();

        assert_eq!(pop_free_idx(false), Some(0));
    }

    #[test]
    fn cross_thread_push_zeroed_lands_in_injector() {
        let _lock = TEST_LOCK.lock().unwrap();
        init_test_free_pool(0);

        std::thread::spawn(|| push_free_idx(0, true))
            .join()
            .unwrap();

        assert_eq!(pop_free_idx(true), Some(0));
    }
}
