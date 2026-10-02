//! Lock-free gathering of one value from each worker.
//!
//! Every worker publishes into its own set-once slot. The final arrival sees
//! all published values, runs the completion work, and wakes the worker pool.

use crate::waker::waker_set;
use crate::worker::{WORKER_IDX, dataflow_worker_idx};
use std::cell::UnsafeCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ValueSlot<T>(UnsafeCell<Option<T>>);

impl<T> ValueSlot<T> {
    fn new(value: T) -> Self {
        Self(UnsafeCell::new(Some(value)))
    }

    fn take(&self) -> T {
        // The barrier gathers only after every producer has stopped touching
        // its slot, and only the final arrival can take the values.
        unsafe { &mut *self.0.get() }
            .take()
            .expect("each gathered value is taken at most once")
    }
}

// Each slot has one producer, and its value is only read or taken after the
// final arrival. There is no concurrent access to the `UnsafeCell`.
unsafe impl<T: Send> Sync for ValueSlot<T> {}

/// Gathers exactly one value from each worker and elects the final arrival to
/// complete the operation.
pub struct GatherBarrier<T> {
    values: Box<[OnceLock<ValueSlot<T>>]>,
    remaining: AtomicUsize,
}

impl<T> GatherBarrier<T> {
    /// Creates a barrier expecting one arrival from each of `worker_count`
    /// workers.
    pub fn new(worker_count: usize) -> Self {
        assert!(
            worker_count > 0,
            "a gather barrier needs at least one worker"
        );
        Self {
            values: (0..worker_count).map(|_| OnceLock::new()).collect(),
            remaining: AtomicUsize::new(worker_count),
        }
    }

    /// Publishes `value` for the calling worker.
    ///
    /// The final worker to arrive invokes `on_complete` with every value in
    /// worker order, then wakes all workers. It receives the closure's result;
    /// earlier arrivals return `None` without waiting.
    ///
    /// Panics if the calling thread has no valid worker index or that worker has
    /// already arrived.
    pub fn arrive<R>(&self, value: T, on_complete: impl FnOnce(Vec<T>) -> R) -> Option<R> {
        let worker = dataflow_worker_idx(WORKER_IDX.get(), self.values.len());
        self.arrive_at(worker, value, on_complete)
    }

    /// Publishes `value` in an explicitly selected slot.
    ///
    /// This is useful for node-local barriers, whose slots are indexed by a
    /// worker's position within its NUMA node instead of its global worker ID.
    pub fn arrive_at<R>(
        &self,
        worker_index: usize,
        value: T,
        on_complete: impl FnOnce(Vec<T>) -> R,
    ) -> Option<R> {
        assert!(
            worker_index < self.values.len(),
            "worker {worker_index} is outside this gather barrier"
        );
        assert!(
            self.values[worker_index].set(ValueSlot::new(value)).is_ok(),
            "worker {worker_index} arrived at a gather barrier twice"
        );
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return None;
        }

        let values = self
            .values
            .iter()
            .map(|value| {
                value
                    .get()
                    .expect("the completion worker observes every gathered value")
                    .take()
            })
            .collect();
        let result = on_complete(values);
        waker_set().notify_all();
        Some(result)
    }

    /// Returns the number of workers expected by this barrier.
    pub fn worker_count(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waker::{WakerSet, WorkerWaker, init_waker_set};
    use crate::worker::WORKER_IDX;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn final_arrival_completes_with_values_in_worker_order() {
        let waker = Arc::new(WorkerWaker::new(3));
        let wakers = WakerSet::new(vec![waker], 3);
        let barrier = Arc::new(GatherBarrier::new(3));

        let arrivals = [2, 0, 1].map(|worker_index| {
            let barrier = barrier.clone();
            let wakers = wakers.clone();
            thread::spawn(move || {
                WORKER_IDX.set(worker_index);
                init_waker_set(wakers);
                barrier.arrive(worker_index * 10, |values| values)
            })
        });
        let completions: Vec<Vec<usize>> = arrivals
            .into_iter()
            .filter_map(|arrival| arrival.join().unwrap())
            .collect();

        assert_eq!(completions, vec![vec![0, 10, 20]]);
    }

    #[test]
    fn final_arrival_wakes_parked_workers() {
        let waker = Arc::new(WorkerWaker::new(1));
        let wakers = WakerSet::new(vec![waker.clone()], 1);
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
        let parked_worker = thread::spawn(move || {
            waker.register(0);
            let wake_count = waker.wake_count();
            ready_sender.send(()).unwrap();
            waker.wait_if_unchanged(wake_count, 0) > wake_count
        });
        ready_receiver.recv().unwrap();
        WORKER_IDX.set(0);
        init_waker_set(wakers);
        let barrier = GatherBarrier::new(1);

        let completion = barrier.arrive(42, |values| values[0]);

        assert_eq!(completion, Some(42));
        assert!(parked_worker.join().unwrap());
    }

    #[test]
    fn final_arrival_can_take_ownership_of_values() {
        let waker = Arc::new(WorkerWaker::new(1));
        init_waker_set(WakerSet::new(vec![waker], 1));
        let barrier = GatherBarrier::new(2);

        WORKER_IDX.set(0);
        let first = barrier.arrive(String::from("first"), |_| unreachable!());
        WORKER_IDX.set(1);
        let second = barrier.arrive(String::from("second"), |values| values);

        assert!(first.is_none());
        assert_eq!(second.unwrap(), ["first", "second"]);
    }
}
