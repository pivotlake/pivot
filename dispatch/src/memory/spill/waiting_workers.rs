//! The workers of one node that are waiting for a spilled buffer.
//!
//! A worker that needs a slot and finds every spilled buffer on its node still
//! being written has nothing to do until one of those writes lands, and the
//! write may be another worker's, so its landing shows up on that worker's
//! ring and not on the waiting one's. The waiting worker records itself here
//! and blocks in its ring wait; whoever makes a buffer evictable, or takes the
//! buffers away for good, wakes every worker recorded.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct WaitingWorkers {
    /// The node's workers, as global worker indices.
    node_workers: Range<usize>,
    /// Whether each of the node's workers is waiting, in worker order.
    is_waiting: Box<[AtomicBool]>,
    /// Which worker the next wake starts from, so that no worker is always the
    /// first to be woken and to find the buffer.
    first_worker_to_wake: AtomicUsize,
}

impl WaitingWorkers {
    pub fn new(node_workers: Range<usize>) -> Self {
        Self {
            is_waiting: node_workers
                .clone()
                .map(|_| AtomicBool::new(false))
                .collect(),
            node_workers,
            first_worker_to_wake: AtomicUsize::new(0),
        }
    }

    /// Ask for `worker` to be woken the next time a buffer may have become
    /// evictable. The caller must look for a buffer once more after
    /// registering: one that became evictable between its failed look and the
    /// registration would otherwise go unseen forever. Whoever makes a buffer
    /// evictable does so before calling [`wake_all`](Self::wake_all), so a
    /// second look that still finds nothing ran before that call, which will
    /// then see this registration.
    pub fn register(&self, worker: usize) {
        self.is_waiting[worker - self.node_workers.start].store(true, Ordering::SeqCst);
    }

    /// Drop a registration whose second look settled the matter, so that
    /// `worker` is not woken for nothing.
    pub fn deregister(&self, worker: usize) {
        self.is_waiting[worker - self.node_workers.start].store(false, Ordering::SeqCst);
    }

    /// Wake every registered worker, interrupting its ring wait, and forget
    /// the registrations. The workers race for whatever became evictable.
    pub fn wake_all(&self) {
        let workers = self.is_waiting.len();
        let first = self.first_worker_to_wake.fetch_add(1, Ordering::Relaxed);
        for offset in 0..workers {
            let local_idx = (first + offset) % workers;
            if self.is_waiting[local_idx].swap(false, Ordering::SeqCst) {
                crate::waker::waker_set().notify_worker(self.node_workers.start + local_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::waker::worker_waker;

    #[test]
    fn a_registered_worker_is_woken_once() {
        init_test_free_pool(0);
        let waiting = WaitingWorkers::new(0..1);
        let before = worker_waker().ring_wake_count();

        waiting.register(0);
        waiting.wake_all();
        waiting.wake_all();

        assert_eq!(worker_waker().ring_wake_count(), before + 1);
    }

    #[test]
    fn a_deregistered_worker_is_not_woken() {
        init_test_free_pool(0);
        let waiting = WaitingWorkers::new(0..1);
        let before = worker_waker().ring_wake_count();

        waiting.register(0);
        waiting.deregister(0);
        waiting.wake_all();

        assert_eq!(worker_waker().ring_wake_count(), before);
    }
}
