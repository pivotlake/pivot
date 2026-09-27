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
use crate::waker::worker_waker;
use crossbeam_deque::{Stealer, Worker};
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

/// Factory for building one worker's stealable channel endpoint.
///
/// Holds this worker's place in a *shared* slice of every worker's
/// [`Stealer`]. The worker's own deque is created only when the endpoint is
/// built, on the worker itself, which then publishes its stealer in its slot:
/// a deque preallocates its buffer, and building every worker's deque up
/// front would put all of those allocations on the thread compiling the
/// query, one after another, instead of spreading them over the workers.
pub struct StealableChannelFactory<T: Send> {
    new_worker: fn() -> Worker<T>,
    stealers: Arc<[OnceLock<Stealer<T>>]>,
    worker_idx: usize,
    topology: Topology,
}

impl<T: Send + 'static> ChannelFactory<T> for StealableChannelFactory<T> {
    type Sender = Rc<Worker<T>>;
    type Receiver = StealableReceiver<T>;

    fn build(self) -> (Rc<Worker<T>>, StealableReceiver<T>) {
        let worker = (self.new_worker)();
        self.stealers[self.worker_idx]
            .set(worker.stealer())
            .unwrap_or_else(|_| panic!("worker {} built its channel twice", self.worker_idx));
        let worker = Rc::new(worker);
        (
            worker.clone(),
            StealableReceiver {
                worker,
                stealers: self.stealers,
                siblings: self.topology.node_siblings(self.worker_idx),
                worker_idx: self.worker_idx,
            },
        )
    }
}

impl<O> Sender<O> for Rc<Worker<O>> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.push(item);
        // One new item needs one thief: wake a single parked same-node peer.
        // Waking more would just herd workers into finding nothing (the send
        // path must stay cheap; see WorkerWaker).
        worker_waker().notify_one();
        Ok(())
    }
}

/// Receiving end of a stealable channel.
///
/// Shares an `Rc<Worker<I>>` with the sender (both live on the same worker thread).
/// [`Receiver::try_recv`] pops from the local deque; [`Receiver::steal`] tries its
/// same-node peers' deques in order when the local one is empty, enabling
/// cross-worker load balancing within the node.
pub struct StealableReceiver<I> {
    worker: Rc<Worker<I>>,
    /// Every worker's stealer, shared by all receivers of the stage. A slot is
    /// empty until its worker has built its endpoint, and a worker that has
    /// not built it yet has nothing queued to steal.
    stealers: Arc<[OnceLock<Stealer<I>>]>,
    /// The slice of `stealers` this worker may steal from (its node's workers).
    siblings: std::ops::Range<usize>,
    /// This worker's own index in `stealers`, skipped when stealing.
    worker_idx: usize,
}

impl<I> Receiver<I> for StealableReceiver<I> {
    fn is_empty(&self) -> bool {
        self.worker.is_empty()
    }

    /// Pop a message from the local deque (newest first, or oldest first for
    /// a [`stealable_fifo`] channel).
    fn try_recv(&self) -> Option<I> {
        self.worker.pop()
    }

    /// Try to steal a message from a same-node peer worker's deque.
    /// Iterates through the node's stealers, retrying on contention.
    ///
    /// The `is_empty` pre-check keeps the (very common) all-empty scan cheap:
    /// it is a couple of plain loads, while `steal()` pins a crossbeam epoch
    /// and CASes even when it finds nothing. Idle workers re-run this scan on
    /// every wakeup, so without the pre-check the pool burns a large share of
    /// its cycles in epoch bookkeeping just discovering there is no work.
    fn steal(&self) -> Option<I> {
        use crossbeam_deque::Steal;
        for peer in self.siblings.clone() {
            if peer == self.worker_idx {
                continue;
            }
            let Some(stealer) = self.stealers[peer].get() else {
                continue;
            };
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
/// crosses NUMA nodes). The local worker pops its newest message first.
pub fn stealable<T: Send>(
    topology: Topology,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    stealable_with(topology, Worker::new_lifo)
}

/// Like [`stealable`], but the local worker pops its oldest message first,
/// for messages that form a sequence the worker should follow in order.
/// Thieves take the oldest message in both flavours.
pub fn stealable_fifo<T: Send>(
    topology: Topology,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    stealable_with(topology, Worker::new_fifo)
}

fn stealable_with<T: Send>(
    topology: Topology,
    new_worker: fn() -> Worker<T>,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    let stealers: Arc<[OnceLock<Stealer<T>>]> = (0..topology.total_workers())
        .map(|_| OnceLock::new())
        .collect();

    (0..topology.total_workers()).map(move |worker_idx| StealableChannelFactory {
        new_worker,
        stealers: stealers.clone(),
        worker_idx,
        topology,
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
        crate::waker::install_test_worker_waker();
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
    fn a_fifo_channel_hands_the_owner_its_oldest_message() {
        crate::waker::install_test_worker_waker();
        let mut endpoints: Vec<Endpoint<i64>> = stealable_fifo::<i64>(Topology {
            workers_per_node: 2,
            node_count: 1,
        })
        .into_iter()
        .map(ChannelFactory::build)
        .collect();
        endpoints[0].0.send(1).unwrap();
        endpoints[0].0.send(2).unwrap();

        let own = endpoints[0].1.try_recv();

        assert_eq!(own, Some(1));
    }

    #[test]
    fn a_thief_skips_a_peer_that_has_not_built_its_endpoint() {
        crate::waker::install_test_worker_waker();
        let mut factories: Vec<_> = stealable::<i64>(Topology {
            workers_per_node: 2,
            node_count: 1,
        })
        .into_iter()
        .collect();
        let _unbuilt_peer = factories.pop().unwrap();
        let (_, receiver) = factories.pop().unwrap().build();

        let stolen = receiver.steal();

        assert_eq!(stolen, None);
    }

    #[test]
    fn other_node_worker_cannot_steal() {
        let mut endpoints = two_node_endpoints::<i64>();

        endpoints[0].0.send(7).unwrap();

        assert_eq!(endpoints[2].1.steal(), None);
        assert_eq!(endpoints[3].1.steal(), None);
    }
}
