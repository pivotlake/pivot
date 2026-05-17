//! Work-stealing channel for passing messages between dataflow operators.
//!
//! Each worker gets its own LIFO deque for the channel. The producer (upstream operator)
//! pushes into the local deque; the consumer (downstream operator) pops from it.  When a
//! worker's local deque is empty it steals from a peer's deque, which naturally balances
//! load across workers without any central coordination.
//!
//! Messages flowing through these channels (e.g. `RecordBatch`) will continue processing on
//! a different worker thread than the one that produced it. This means the message must not be
//! logically local to a worker; for example, while `CompressedPage` *is* `Send`, it is critical to
//! return it to the worker that sent out, since it needs to be sent to the Decoder of that row group.
//! Therefore it should *not* be in a stealable channel (but in `return_to_worker`)

use crate::operations::channels;
use crate::operations::channels::{ChannelFactory, Receiver, Sender};
use crate::worker::worker_waker;
use crossbeam_deque::{Stealer, Worker};
use std::rc::Rc;

/// Factory for building one worker's stealable channel endpoint.
///
/// Holds the local [`Worker`] deque and the [`Stealer`] handles for every other worker.
/// [`stealable()`] creates one factory per worker; each is consumed by [`ChannelFactory::build`]
/// to produce the sender/receiver pair for that worker.
pub struct StealableChannelFactory<T: Send> {
    worker: Worker<T>,
    stealers: Vec<Stealer<T>>,
}

impl<T: Send> StealableChannelFactory<T> {
    pub fn new(worker: Worker<T>, stealers: Vec<Stealer<T>>) -> StealableChannelFactory<T> {
        StealableChannelFactory { worker, stealers }
    }
}

impl<T: Send + 'static> ChannelFactory<T> for StealableChannelFactory<T> {
    type Sender = Rc<Worker<T>>;
    type Receiver = StealableReceiver<T>;

    fn build(self) -> (Rc<Worker<T>>, StealableReceiver<T>) {
        let worker = Rc::new(self.worker);
        (
            worker.clone(),
            StealableReceiver::new(worker, self.stealers),
        )
    }
}

impl<O> Sender<O> for Rc<Worker<O>> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.push(item);
        // Wake any parked peer so they can steal the freshly-pushed work.
        worker_waker().notify();
        Ok(())
    }
}

/// Receiving end of a stealable channel.
///
/// Shares an `Rc<Worker<I>>` with the sender (both live on the same worker thread).
/// [`Receiver::try_recv`] pops from the local deque; [`Receiver::steal`] tries each peer's deque in order
/// when the local one is empty, enabling cross-worker load balancing.
pub struct StealableReceiver<I> {
    worker: Rc<Worker<I>>,
    stealers: Vec<Stealer<I>>,
}

impl<I> StealableReceiver<I> {
    pub fn new(worker: Rc<Worker<I>>, stealers: Vec<Stealer<I>>) -> Self {
        Self { worker, stealers }
    }
}

impl<I> Receiver<I> for StealableReceiver<I> {
    fn is_empty(&self) -> bool {
        self.worker.is_empty()
    }

    /// Pop a message from the local deque (LIFO).
    fn try_recv(&self) -> Option<I> {
        self.worker.pop()
    }

    /// Try to steal a message from a peer worker's deque.
    /// Iterates through all peer stealers, retrying on contention.
    fn steal(&self) -> Option<I> {
        use crossbeam_deque::Steal;
        for stealer in &self.stealers {
            loop {
                match stealer.steal() {
                    Steal::Success(item) => return Some(item),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }
}

/// Create one [`StealableChannelFactory`] per worker, each wired to steal from all peers.
///
/// Returns an iterator of factories — one per worker. Each factory's stealer list
/// contains every other worker's deque, so any worker can steal from any other.
pub fn stealable<T: Send>(count: usize) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    let workers: Vec<_> = (0..count)
        .map(|_| Worker::new_lifo())
        .collect();

    let workers_with_stealers = workers
        .iter()
        .enumerate()
        .map(|(i, _)| {
            workers
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, w)| w.stealer())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    workers
        .into_iter()
        .zip(workers_with_stealers)
        .map(|(w, s)| StealableChannelFactory {
            worker: w,
            stealers: s,
        })
}
