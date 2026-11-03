//! What a stage hands the workers so each can make its own piece of it.
//!
//! A dataflow runs on every worker, and every stage needs one piece per
//! worker: a channel endpoint, a unary factory, an operator chain. Making
//! those pieces on the thread compiling the plan costs it a loop over the
//! pool per stage, so a wide pool spends more of a small query's time
//! compiling than running it. A [`WorkerMaker`] moves that loop onto the
//! workers: the compiling thread sets up what the stage's pieces share, and
//! each worker makes its own piece when it builds the dataflow.

use std::sync::{Arc, Mutex};

/// Makes one worker's piece of a stage, on that worker's thread.
pub trait WorkerMaker<T>: Send + Sync + 'static {
    fn make(&self, worker_idx: usize) -> T;
}

/// A maker of `T` from a closure of the worker index.
pub struct FnMaker<F, T> {
    make: F,
    _piece: std::marker::PhantomData<fn() -> T>,
}

impl<T, F> WorkerMaker<T> for FnMaker<F, T>
where
    T: 'static,
    F: Fn(usize) -> T + Send + Sync + 'static,
{
    fn make(&self, worker_idx: usize) -> T {
        (self.make)(worker_idx)
    }
}

/// Make each worker's piece with `make`, called on the worker with its index.
pub fn per_worker<T, F>(make: F) -> FnMaker<F, T>
where
    T: 'static,
    F: Fn(usize) -> T + Send + Sync + 'static,
{
    FnMaker {
        make,
        _piece: std::marker::PhantomData,
    }
}

/// Pieces made ahead of time, one per worker in worker order, each taken by
/// its worker when it builds. For stages whose pieces are cheap to make up
/// front, or that are not on a query's path.
pub struct WorkerSlots<T> {
    slots: Box<[Mutex<Option<T>>]>,
}

impl<T> WorkerSlots<T> {
    pub fn new(pieces: impl IntoIterator<Item = T>) -> Self {
        Self {
            slots: pieces
                .into_iter()
                .map(|piece| Mutex::new(Some(piece)))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl<T: Send + 'static> WorkerMaker<T> for WorkerSlots<T> {
    fn make(&self, worker_idx: usize) -> T {
        let slot = self.slots.get(worker_idx).unwrap_or_else(|| {
            panic!(
                "worker {worker_idx} has no piece of a stage made for {} workers",
                self.slots.len()
            )
        });
        slot.lock()
            .expect("a slot's lock is never poisoned")
            .take()
            .unwrap_or_else(|| panic!("worker {worker_idx} took its piece of a stage twice"))
    }
}

/// Two makers from one that makes a pair, for a stage whose two pieces for a
/// worker must come from the same making: whichever half the worker asks
/// for first makes the pair and keeps the other half for it.
pub fn split<A, B, M>(
    worker_count: usize,
    make_pair: M,
) -> (SplitHalf<A, B, M, A>, SplitHalf<A, B, M, B>)
where
    A: Send + 'static,
    B: Send + 'static,
    M: WorkerMaker<(A, B)>,
{
    let shared = Arc::new(SplitState {
        make_pair,
        kept_a: (0..worker_count).map(|_| Mutex::new(None)).collect(),
        kept_b: (0..worker_count).map(|_| Mutex::new(None)).collect(),
    });
    (
        SplitHalf {
            shared: shared.clone(),
            take: take_a,
        },
        SplitHalf {
            shared,
            take: take_b,
        },
    )
}

struct SplitState<A, B, M> {
    make_pair: M,
    kept_a: Box<[Mutex<Option<A>>]>,
    kept_b: Box<[Mutex<Option<B>>]>,
}

fn take_kept<T>(kept: &[Mutex<Option<T>>], worker_idx: usize) -> Option<T> {
    kept[worker_idx]
        .lock()
        .expect("a kept half's lock is never poisoned")
        .take()
}

fn keep<T>(kept: &[Mutex<Option<T>>], worker_idx: usize, piece: T) {
    *kept[worker_idx]
        .lock()
        .expect("a kept half's lock is never poisoned") = Some(piece);
}

fn take_a<A, B, M: WorkerMaker<(A, B)>>(state: &SplitState<A, B, M>, worker_idx: usize) -> A {
    if let Some(a) = take_kept(&state.kept_a, worker_idx) {
        return a;
    }
    let (a, b) = state.make_pair.make(worker_idx);
    keep(&state.kept_b, worker_idx, b);
    a
}

fn take_b<A, B, M: WorkerMaker<(A, B)>>(state: &SplitState<A, B, M>, worker_idx: usize) -> B {
    if let Some(b) = take_kept(&state.kept_b, worker_idx) {
        return b;
    }
    let (a, b) = state.make_pair.make(worker_idx);
    keep(&state.kept_a, worker_idx, a);
    b
}

/// One half of a [`split`] pair maker: the `T` half of an `(A, B)` pair.
pub struct SplitHalf<A, B, M, T> {
    shared: Arc<SplitState<A, B, M>>,
    take: fn(&SplitState<A, B, M>, usize) -> T,
}

impl<A, B, M, T> WorkerMaker<T> for SplitHalf<A, B, M, T>
where
    A: Send + 'static,
    B: Send + 'static,
    M: WorkerMaker<(A, B)>,
    T: 'static,
{
    fn make(&self, worker_idx: usize) -> T {
        (self.take)(&self.shared, worker_idx)
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
    remaining: usize,
}

impl<T: Clone> HandOut<T> {
    /// `value`, to be handed to `worker_count` workers.
    pub fn new(value: T, worker_count: usize) -> Self {
        Self {
            state: Mutex::new(HandOutState {
                original: Some(value),
                remaining: worker_count,
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
            state.remaining > 0,
            "more workers asked for a hand-out than it was made for"
        );
        state.remaining -= 1;
        let original = state
            .original
            .as_ref()
            .expect("the original leaves with the last worker");
        if state.remaining == 0 {
            state.original.take().expect("checked above")
        } else {
            original.clone()
        }
    }
}

/// What a stage accepts for its per-worker pieces: a maker, or the pieces
/// themselves, made ahead of time. The piece is an associated type, so a
/// stage's signature pins it down from the argument alone.
pub trait IntoWorkerMaker {
    type Piece;
    type Maker: WorkerMaker<Self::Piece>;
    fn into_worker_maker(self) -> Self::Maker;
}

/// Implements [`IntoWorkerMaker`] for a maker type as the identity.
#[macro_export]
macro_rules! maker_into_worker_maker {
    ($maker:ty, $piece:ty, [$($generics:tt)*], [$($bounds:tt)*]) => {
        impl<$($generics)*> $crate::IntoWorkerMaker for $maker
        where
            $($bounds)*
        {
            type Piece = $piece;
            type Maker = Self;
            fn into_worker_maker(self) -> Self {
                self
            }
        }
    };
}

maker_into_worker_maker!(
    FnMaker<F, T>,
    T,
    [T, F],
    [T: 'static, F: Fn(usize) -> T + Send + Sync + 'static]
);
maker_into_worker_maker!(WorkerSlots<T>, T, [T], [T: Send + 'static]);
maker_into_worker_maker!(Box<dyn WorkerMaker<T>>, T, [T], [T: 'static]);
maker_into_worker_maker!(
    SplitHalf<A, B, M, T>,
    T,
    [A, B, M, T],
    [A: Send + 'static, B: Send + 'static, M: WorkerMaker<(A, B)>, T: 'static]
);

impl<T> From<Vec<T>> for WorkerSlots<T> {
    fn from(pieces: Vec<T>) -> Self {
        WorkerSlots::new(pieces)
    }
}

impl<T: Send + 'static> IntoWorkerMaker for Vec<T> {
    type Piece = T;
    type Maker = WorkerSlots<T>;
    fn into_worker_maker(self) -> WorkerSlots<T> {
        WorkerSlots::new(self)
    }
}

impl<T: 'static> WorkerMaker<T> for Box<dyn WorkerMaker<T>> {
    fn make(&self, worker_idx: usize) -> T {
        (**self).make(worker_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closure_maker_hands_each_worker_its_index() {
        let maker = per_worker(|worker_idx| worker_idx * 10);

        let pieces: Vec<usize> = (0..3).map(|worker| maker.make(worker)).collect();

        assert_eq!(pieces, vec![0, 10, 20]);
    }

    #[test]
    fn slots_hand_each_worker_its_own_piece_once() {
        let slots = WorkerSlots::new(vec!["a", "b"]);

        let second = slots.make(1);
        let first = slots.make(0);

        assert_eq!((first, second), ("a", "b"));
    }

    #[test]
    fn a_split_pair_is_made_once_per_worker_whichever_half_is_asked_first() {
        let (lefts, rights) = split(2, per_worker(|worker_idx| (worker_idx, worker_idx + 100)));

        let right_first = rights.make(1);
        let left_after = lefts.make(1);
        let left_first = lefts.make(0);
        let right_after = rights.make(0);

        assert_eq!((left_first, right_after), (0, 100));
        assert_eq!((left_after, right_first), (1, 101));
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
    #[should_panic(expected = "took its piece of a stage twice")]
    fn a_worker_cannot_take_its_slot_twice() {
        let slots = WorkerSlots::new(vec![1]);
        slots.make(0);

        slots.make(0);
    }
}
