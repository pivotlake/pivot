//! Work-stealing channel for passing messages between dataflow operators.
//!
//! Each worker gets its own LIFO deque for the channel. The producer (upstream operator)
//! pushes into the local deque; the consumer (downstream operator) pops from it.  When a
//! worker's local deque is empty it steals from a *same-NUMA-node* peer's deque, which
//! balances load across the node's workers without any central coordination. Stealing
//! never crosses nodes: the messages are typically backed by ring memory the producer's
//! node owns, so a cross-node steal would drag every downstream access of that item to
//! remote memory. Cross-node balance comes from the work's *source* instead (a shared
//! injector of row groups / items), where claiming an item moves only metadata and all
//! the memory it touches is then allocated locally.
//!
//! Messages flowing through these channels (e.g. `RecordBatch`) will continue processing on
//! a different worker thread than the one that produced it. This means the message must not be
//! logically local to a worker; for example, while `CompressedPage` *is* `Send`, it is critical to
//! return it to the worker that sent out, since it needs to be sent to the Decoder of that row group.
//! Therefore it should *not* be in a stealable channel (but in `return_to_worker`)

use crate::numa::Topology;
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
    ///
    /// The `is_empty` pre-check keeps the (very common) all-empty scan cheap:
    /// it is a couple of plain loads, while `steal()` pins a crossbeam epoch
    /// and CASes even when it finds nothing. Idle workers re-run this scan on
    /// every wakeup, so without the pre-check the pool burns a large share of
    /// its cycles in epoch bookkeeping just discovering there is no work.
    fn steal(&self) -> Option<I> {
        use crossbeam_deque::Steal;
        for stealer in &self.stealers {
            if stealer.is_empty() {
                continue;
            }
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

/// Create one [`StealableChannelFactory`] per worker, each wired to steal from
/// its same-node peers only (see the module docs for why stealing never
/// crosses NUMA nodes).
pub fn stealable<T: Send>(
    topology: Topology,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    let workers: Vec<_> = (0..topology.total_workers())
        .map(|_| Worker::new_lifo())
        .collect();

    let workers_with_stealers = workers
        .iter()
        .enumerate()
        .map(|(i, _)| {
            topology
                .node_siblings(i)
                .filter(|&j| j != i)
                .map(|j| workers[j].stealer())
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

#[cfg(test)]
mod tests {
    use super::*;

    type Endpoint<T> = (
        <StealableChannelFactory<T> as ChannelFactory<T>>::Sender,
        StealableReceiver<T>,
    );

    fn two_node_endpoints<T: Send + 'static>() -> Vec<Endpoint<T>> {
        crate::worker::install_test_worker_waker();
        stealable::<T>(Topology {
            workers_per_node: 2,
            node_count: 2,
        })
        .into_iter()
        .map(ChannelFactory::build)
        .collect()
    }

    #[test]
    fn same_node_sibling_can_steal() {
        let mut endpoints = two_node_endpoints::<i64>();

        endpoints[0].0.send(7).unwrap();

        assert_eq!(endpoints[1].1.steal(), Some(7));
    }

    #[test]
    fn other_node_worker_cannot_steal() {
        let mut endpoints = two_node_endpoints::<i64>();

        endpoints[0].0.send(7).unwrap();

        assert_eq!(endpoints[2].1.steal(), None);
        assert_eq!(endpoints[3].1.steal(), None);
    }
}
