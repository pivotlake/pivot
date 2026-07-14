//! Work-stealing pool of free buffer indices.
//!
//! Each `MemoryContext` owns one or two [`FreePool`]s (one for
//! zeroed buffers, one for dirty). A pool offers three tiers of pop, from
//! cheapest to most expensive:
//!
//! 1. local LIFO deque (no contention, hot in cache),
//! 2. per-worker injector (picks up indices returned by *other* workers), and
//! 3. work-stealing from same-node sibling workers' deques.
//!
//! Pools belonging to the same `MemoryContextFactory` batch share one global
//! injector array (so a buffer released *anywhere* can be routed back), but a
//! worker's stealer list covers only its own NUMA node's siblings: acquiring a
//! slot is what places memory, so it must stay node-local (see
//! [`RingLayout`](super::RingLayout)).
//!
//! On push, the index is routed to its *home worker*
//! ([`RingLayout::home_worker`](super::RingLayout::home_worker), always a
//! worker on the slot's own node). Pushing from the home worker hits the local
//! deque; pushing from any other worker (any node) hits the home worker's
//! injector.

use crate::memory::RingLayout;
use crate::worker::WORKER_IDX;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

/// Builds a coordinated set of [`FreePool`]s — one per worker — that share
/// a global injector array and per-node stealer lists.
///
/// Each factory holds Arc clones of the shared state; calling
/// [`create_pool`](Self::create_pool) on the calling worker thread registers
/// that worker's stealer and waits on its node's barrier so every same-node
/// pool sees the final stealer list.
pub struct PoolFactory {
    /// This worker's node's stealer registry (same-node siblings only).
    registry: Arc<Mutex<Vec<Stealer<usize>>>>,
    /// One injector per worker across all nodes, indexed by global worker index.
    injectors: Arc<Vec<Injector<usize>>>,
    /// Registration barrier for this worker's node.
    barrier: Arc<Barrier>,
    layout: RingLayout,
}

impl PoolFactory {
    /// Returns one factory per worker (global worker order), sharing a global
    /// injector array, with stealer registries and registration barriers per
    /// node. Pass one factory to each worker.
    pub fn create_for_layout(layout: RingLayout) -> Vec<PoolFactory> {
        let topology = layout.topology();
        let injectors: Arc<Vec<_>> = Arc::new(
            (0..topology.total_workers())
                .map(|_| Injector::new())
                .collect(),
        );
        let node_registries: Vec<_> = (0..topology.node_count)
            .map(|_| Arc::new(Mutex::new(Vec::new())))
            .collect();
        let node_barriers: Vec<_> = (0..topology.node_count)
            .map(|_| Arc::new(Barrier::new(topology.workers_per_node)))
            .collect();
        (0..topology.total_workers())
            .map(|worker| {
                let node = topology.node_of_worker(worker);
                Self {
                    registry: node_registries[node].clone(),
                    injectors: injectors.clone(),
                    barrier: node_barriers[node].clone(),
                    layout,
                }
            })
            .collect()
    }

    /// Register this worker's local deque and return its [`FreePool`].
    ///
    /// Blocks on the node's barrier until every same-node sibling factory has
    /// also called `create_pool`, so the snapshot of stealers is complete.
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
            layout: self.layout,
        }
    }
}

/// One worker's view of the free pool. It holds a private LIFO deque,
/// same-node peer stealer handles, and the shared per-worker injector array.
pub struct FreePool {
    worker: Worker<usize>,
    stealers: Vec<Stealer<usize>>,
    injectors: Arc<Vec<Injector<usize>>>,
    last_stealer_idx: AtomicUsize,
    layout: RingLayout,
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

        // The `is_empty` pre-checks below (injector and stealers) keep the
        // all-empty scan to plain loads: `steal()` pins a crossbeam epoch and
        // CASes even when it finds nothing, which idle workers would otherwise
        // pay on every wakeup.
        let own_injector = &self.injectors[WORKER_IDX.get()];
        while !own_injector.is_empty() {
            match own_injector.steal() {
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
            let stealer = &self.stealers[(start + i) % self.stealers.len()];
            if stealer.is_empty() {
                continue;
            }
            if let Steal::Success(idx) = stealer.steal() {
                return Some(idx);
            }
        }

        None
    }

    /// Return a buffer index. If this worker is `idx`'s home worker, the index
    /// lands in the local deque; otherwise it lands in the home worker's
    /// injector, where the home worker can pick it up on its next `pop`. The
    /// home worker is always on the slot's own node, so a buffer released on
    /// another node finds its way back without the releaser ever *acquiring*
    /// remote memory.
    pub(crate) fn push(&self, idx: usize) {
        let home_worker = self.layout.home_worker(idx);
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
                layout: RingLayout::single_node(count, 128),
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

    /// One pool per worker of a 2-node topology with 1 worker per node and 2
    /// slots per node (built via the factory so stealer lists are per node).
    fn build_two_node_pools() -> Vec<FreePool> {
        let layout = RingLayout::new(
            crate::numa::Topology {
                workers_per_node: 1,
                node_count: 2,
            },
            2,
        );
        PoolFactory::create_for_layout(layout)
            .into_iter()
            .map(|factory| factory.create_pool())
            .collect()
    }

    #[test]
    fn a_slot_released_on_another_node_routes_back_to_its_home_worker() {
        let pools = build_two_node_pools();

        // Slot 2 lives in node 1's region; released from node 0's worker it
        // must land with node 1's worker, not the releaser.
        act_as_worker(0, 2);
        pools[0].push(2);

        act_as_worker(1, 2);
        assert_eq!(pools[1].pop(false), Some(2));
    }

    #[test]
    fn a_worker_never_steals_a_free_slot_from_another_node() {
        let pools = build_two_node_pools();
        act_as_worker(1, 2);
        pools[1].push(3);

        act_as_worker(0, 2);

        assert_eq!(pools[0].pop(true), None);
    }

    #[test]
    fn create_pool_via_factory_for_single_worker() {
        // Setup: exercise the factory's barrier path (1-worker barrier returns
        // immediately, so we can do this on a single thread).
        act_as_worker(0, 1);
        let factory = PoolFactory::create_for_layout(RingLayout::single_node(1, 128))
            .pop()
            .unwrap();
        let pool = factory.create_pool();

        // Execute
        pool.push(7);
        let popped = pool.pop(true);

        // Assert
        assert_eq!(popped, Some(7));
    }
}
