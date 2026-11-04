//! What a stage hands the workers so each can make its own part of it.
//!
//! A dataflow runs on every worker, and every stage needs one part per
//! worker: a channel endpoint, a unary factory, an operator chain. Making
//! those parts on the thread compiling the plan costs it a loop over the
//! pool per stage, so a wide pool spends more of a small query's time
//! compiling than running it. A [`WorkerMaker`] moves that loop onto the
//! workers: the compiling thread sets up what the stage's parts share, and
//! each worker makes its own part when it builds the dataflow.
//!
//! A closure of the worker index is a maker. [`WorkerSlots`] is the maker of
//! parts made ahead of time, for stages off a query's path, and [`HandOut`]
//! shares one value with every worker while letting the last of them own it.

use std::sync::Mutex;

/// Makes one worker's part of a stage, on that worker's thread.
pub trait WorkerMaker<T>: Send + Sync + 'static {
    fn for_worker(&self, worker_idx: usize) -> T;
}

impl<T, F> WorkerMaker<T> for F
where
    T: 'static,
    F: Fn(usize) -> T + Send + Sync + 'static,
{
    fn for_worker(&self, worker_idx: usize) -> T {
        self(worker_idx)
    }
}

impl<T: 'static> WorkerMaker<T> for Box<dyn WorkerMaker<T>> {
    fn for_worker(&self, worker_idx: usize) -> T {
        (**self).for_worker(worker_idx)
    }
}

/// Parts made ahead of time, one per worker in worker order, each taken by
/// its worker when it builds.
pub struct WorkerSlots<T> {
    slots: Box<[Mutex<Option<T>>]>,
}

impl<T> WorkerSlots<T> {
    pub fn new(parts: impl IntoIterator<Item = T>) -> Self {
        Self {
            slots: parts
                .into_iter()
                .map(|part| Mutex::new(Some(part)))
                .collect(),
        }
    }
}

/// The parts in worker order, for a caller that builds every worker's part
/// itself rather than handing the slots to the pool.
impl<T> IntoIterator for WorkerSlots<T> {
    type Item = T;
    type IntoIter = std::iter::FilterMap<
        std::vec::IntoIter<Mutex<Option<T>>>,
        fn(Mutex<Option<T>>) -> Option<T>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.slots
            .into_vec()
            .into_iter()
            .filter_map(|slot| slot.into_inner().expect("a slot's lock is never poisoned"))
    }
}

impl<T: Send + 'static> WorkerMaker<T> for WorkerSlots<T> {
    fn for_worker(&self, worker_idx: usize) -> T {
        let slot = self.slots.get(worker_idx).unwrap_or_else(|| {
            panic!(
                "worker {worker_idx} has no part of a stage made for {} workers",
                self.slots.len()
            )
        });
        slot.lock()
            .expect("a slot's lock is never poisoned")
            .take()
            .unwrap_or_else(|| panic!("worker {worker_idx} took its part of a stage twice"))
    }
}

/// One value handed to every worker: a clone to each but the last to ask,
/// which gets the original. For a channel's sending end, this is what keeps
/// the channel closing once every worker's operator has dropped its sender:
/// a maker lives as long as the dataflow's launch, so a sender it kept would
/// hold the channel open past the last send, and a receiver waiting for the
/// channel to close would wait forever.
pub struct HandOut<T> {
    state: Mutex<HandOutState<T>>,
}

struct HandOutState<T> {
    original: Option<T>,
    workers_left: usize,
}

impl<T: Clone> HandOut<T> {
    /// `value`, to be handed to `worker_count` workers.
    pub fn new(value: T, worker_count: usize) -> Self {
        Self {
            state: Mutex::new(HandOutState {
                original: Some(value),
                workers_left: worker_count,
            }),
        }
    }

    /// This worker's copy: the original if it is the last worker to ask.
    pub fn take(&self) -> T {
        let mut state = self
            .state
            .lock()
            .expect("a hand-out's lock is never poisoned");
        assert!(
            state.workers_left > 0,
            "more workers asked for a hand-out than it was made for"
        );
        state.workers_left -= 1;
        if state.workers_left == 0 {
            state
                .original
                .take()
                .expect("the original leaves with the last worker")
        } else {
            state
                .original
                .as_ref()
                .expect("the original leaves with the last worker")
                .clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closure_maker_hands_each_worker_its_index() {
        let maker = |worker_idx: usize| worker_idx * 10;

        let parts: Vec<usize> = (0..3).map(|worker| maker.for_worker(worker)).collect();

        assert_eq!(parts, vec![0, 10, 20]);
    }

    #[test]
    fn slots_hand_each_worker_its_own_part_once() {
        let slots = WorkerSlots::new(vec!["a", "b"]);

        let second = slots.for_worker(1);
        let first = slots.for_worker(0);

        assert_eq!((first, second), ("a", "b"));
    }

    #[test]
    fn a_hand_out_gives_the_last_worker_the_original() {
        let (sender, receiver) = std::sync::mpsc::channel::<()>();
        let hand_out = HandOut::new(sender, 2);

        let first = hand_out.take();
        let open_with_one_left = receiver.try_recv();
        let last = hand_out.take();
        drop(first);
        drop(last);
        let closed = receiver.try_recv();

        assert_eq!(
            open_with_one_left,
            Err(std::sync::mpsc::TryRecvError::Empty)
        );
        assert_eq!(closed, Err(std::sync::mpsc::TryRecvError::Disconnected));
    }

    #[test]
    #[should_panic(expected = "took its part of a stage twice")]
    fn a_worker_cannot_take_its_slot_twice() {
        let slots = WorkerSlots::new(vec![1]);
        slots.for_worker(0);

        slots.for_worker(0);
    }
}
