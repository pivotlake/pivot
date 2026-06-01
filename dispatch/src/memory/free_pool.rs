//! Work-stealing pool of free buffer indices.
//!
//! Each [`crate::memory::MemoryContext`] owns one or two [`FreePool`]s (one for
//! zeroed buffers, one for dirty). A pool offers three tiers of pop, from
//! cheapest to most expensive:
//!
//! 1. local LIFO deque (no contention, hot in cache),
//! 2. per-worker injector (picks up indices returned by *other* workers), and
//! 3. work-stealing from sibling workers' deques.
//!
//! Pools belonging to the same `MemoryContextFactory` share their injectors and
//! stealers, so a buffer pushed by one worker is reachable by every other
//! worker in the same group.
//!
//! On push, the index is routed to its *home worker* (`idx % NUM_WORKERS`).
//! Pushing from the home worker hits the local deque; pushing from any other
//! worker hits the home worker's injector.

use crate::worker::WORKER_IDX;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

/// Builds a coordinated set of [`FreePool`]s — one per worker — that share
/// injectors and stealer lists.
///
/// Each factory holds Arc clones of the shared state; calling
/// [`create_pool`](Self::create_pool) on the calling worker thread registers
/// that worker's stealer and waits on a barrier so every pool sees the final
/// stealer list.
#[derive(Clone)]
pub struct PoolFactory {
    registry: Arc<Mutex<Vec<Stealer<usize>>>>,
    injectors: Arc<Vec<Injector<usize>>>,
    barrier: Arc<Barrier>,
}

impl PoolFactory {
    /// Returns `count` factories, all sharing the same injectors, stealer
    /// registry, and registration barrier. Pass one factory to each worker.
    pub fn create_many(count: usize) -> Vec<PoolFactory> {
        let registry = Arc::new(Mutex::new(Vec::new()));
        let injectors: Arc<Vec<_>> = Arc::new((0..count).map(|_| Injector::new()).collect());
        let barrier = Arc::new(Barrier::new(count));
        (0..count)
            .map(|_| Self {
                registry: registry.clone(),
                injectors: injectors.clone(),
                barrier: barrier.clone(),
            })
            .collect()
    }

    /// Register this worker's local deque and return its [`FreePool`].
    ///
    /// Blocks on the shared barrier until every sibling factory has also
    /// called `create_pool`, so the snapshot of stealers is complete.
    pub fn create_pool(&self) -> FreePool {
        let local = Worker::new_lifo();
        self.registry.lock().unwrap().push(local.stealer());
        self.barrier.wait();
        let stealers = self.registry.lock().unwrap().clone();
        FreePool {
            worker: local,
            stealers,
            injectors: self.injectors.clone(),
            last_stealer_idx: AtomicUsize::new(0),
        }
    }
}

/// A single worker's view of the free pool: a private LIFO deque, peer
/// stealer handles, and a shared per-worker injector array.
pub struct FreePool {
    worker: Worker<usize>,
    stealers: Vec<Stealer<usize>>,
    injectors: Arc<Vec<Injector<usize>>>,
    last_stealer_idx: AtomicUsize,
}

impl FreePool {
    /// Try to obtain a free buffer index.
    ///
    /// 1. Pop from the local LIFO deque (cheapest – no contention).
    /// 2. Drain the per-worker injector (buffers returned by other threads).
    /// 3. If `steal`, steal from another worker's deque (round-robin from `last_stealer_idx`).
    ///
    /// Callers pass `steal=false` for opportunistic background work (e.g. idle-time
    /// dirty-buffer cleanup) so they don't pull buffers off peer workers that may
    /// want them. Real allocation paths pass `steal=true`.
    ///
    /// Returns `None` only when every consulted source is empty.
    pub fn pop(&self, steal: bool) -> Option<usize> {
        if let Some(idx) = self.worker.pop() {
            return Some(idx);
        }

        loop {
            match self.injectors[WORKER_IDX.get()].steal() {
                Steal::Success(idx) => return Some(idx),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }

        if !steal {
            return None;
        }

        let start = self.last_stealer_idx.fetch_add(1, Ordering::Relaxed);
        for i in 0..self.stealers.len() {
            if let Steal::Success(idx) = self.stealers[(start + i) % self.stealers.len()].steal() {
                return Some(idx);
            }
        }

        None
    }

    /// Return a buffer index. If this worker is `idx`'s home worker, the index
    /// lands in the local deque; otherwise it lands in the home worker's
    /// injector, where the home worker can pick it up on its next `pop`.
    pub(crate) fn push(&self, idx: usize) {
        let home_worker = idx % self.injectors.len();
        if WORKER_IDX.get() == home_worker {
            self.worker.push(idx)
        } else {
            self.injectors[home_worker].push(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests at the [`FreePool`] / [`PoolFactory`] level: routing within a
    //! single pool, cross-worker routing through the shared injectors, and
    //! work-stealing from sibling deques.
    //!
    //! Anything that mixes a *zeroed* and a *dirty* pool — preference,
    //! fallback — lives in [`crate::memory::context`]'s tests instead, because
    //! that pairing is a `MemoryContext` concept, not a `FreePool` one.

    use super::*;
    use crate::worker::NUM_WORKERS;

    /// Build `count` `FreePool`s wired to a shared injector array and stealer
    /// list, without the registration barrier (so a test thread can hold every
    /// pool at once). Returns the pools in worker-index order.
    fn build_pools(count: usize) -> Vec<FreePool> {
        let injectors: Arc<Vec<_>> = Arc::new((0..count).map(|_| Injector::new()).collect());
        let workers: Vec<Worker<usize>> = (0..count).map(|_| Worker::new_lifo()).collect();
        let stealers: Vec<_> = workers.iter().map(|w| w.stealer()).collect();
        workers
            .into_iter()
            .map(|w| FreePool {
                worker: w,
                stealers: stealers.clone(),
                injectors: injectors.clone(),
                last_stealer_idx: AtomicUsize::new(0),
            })
            .collect()
    }

    /// Configure this thread to act as worker `idx` of `count`.
    fn act_as_worker(idx: usize, count: usize) {
        WORKER_IDX.set(idx);
        NUM_WORKERS.set(count);
    }

    #[test]
    fn empty_pool_returns_none() {
        // Setup
        act_as_worker(0, 1);
        let pool = build_pools(1).pop().unwrap();

        // Execute
        let popped = pool.pop(true);

        // Assert
        assert_eq!(popped, None);
    }

    #[test]
    fn local_push_returns_via_local_deque() {
        // Setup: home-worker push goes to the local LIFO deque.
        act_as_worker(0, 1);
        let pool = build_pools(1).pop().unwrap();

        // Execute
        pool.push(0);
        let popped = pool.pop(true);

        // Assert
        assert_eq!(popped, Some(0));
    }

    #[test]
    fn local_deque_is_lifo() {
        // Setup
        act_as_worker(0, 1);
        let pool = build_pools(1).pop().unwrap();

        // Execute
        pool.push(0);
        pool.push(1);
        let order = [pool.pop(true), pool.pop(true), pool.pop(true)];

        // Assert: last in, first out — cache-friendly reuse.
        assert_eq!(order, [Some(1), Some(0), None]);
    }

    #[test]
    fn cross_worker_push_routes_through_home_injector() {
        // Setup: two pools sharing injectors; this thread is worker 0.
        act_as_worker(0, 2);
        let mut pools = build_pools(2).into_iter();
        let pool0 = pools.next().unwrap();
        let pool1 = pools.next().unwrap();

        // Execute: from worker 0, push idx 1 (home worker = 1).
        pool0.push(1);

        // Assert: pool0's local deque stays empty; pool1 picks it up via its
        // own injector slot (worker 1 popping while acting as worker 1).
        act_as_worker(1, 2);
        assert_eq!(pool1.pop(true), Some(1));
    }

    #[test]
    fn pop_steals_from_sibling_when_local_and_injector_empty() {
        // Setup: pool1 has an index in its local deque; pool0 is empty.
        act_as_worker(0, 2);
        let mut pools = build_pools(2).into_iter();
        let pool0 = pools.next().unwrap();
        let pool1 = pools.next().unwrap();
        act_as_worker(1, 2);
        pool1.push(1);
        act_as_worker(0, 2);

        // Execute
        let popped = pool0.pop(true);

        // Assert: pool0 steals from pool1's deque.
        assert_eq!(popped, Some(1));
    }

    #[test]
    fn create_pool_via_factory_for_single_worker() {
        // Setup: exercise the factory's barrier path (1-worker barrier returns
        // immediately, so we can do this on a single thread).
        act_as_worker(0, 1);
        let factory = PoolFactory::create_many(1).pop().unwrap();
        let pool = factory.create_pool();

        // Execute
        pool.push(7);
        let popped = pool.pop(true);

        // Assert
        assert_eq!(popped, Some(7));
    }
}
