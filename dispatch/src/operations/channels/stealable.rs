//! Work-stealing channel for passing messages between dataflow operators.
//!
//! Each worker gets its own LIFO deque for the channel. The producer (upstream operator)
//! pushes into the local deque; the consumer (downstream operator) pops from it.  When a
//! worker's local deque is empty it steals from a *same-NUMA-node* peer's deque, which
//! balances load across the node's workers without any central coordination. Stealing
//! stays inside the node while the node has work: the messages are typically backed by
//! ring memory the producer's node owns, so a cross-node steal drags every downstream
//! access of that item to remote memory. Cross-node balance comes from the work's
//! *source* instead (a shared injector of row groups / items), where claiming an item
//! moves only metadata and all the memory it touches is then allocated locally. Only
//! once every same-node peer is out of work does a thief look at the other nodes: at
//! the tail of a phase one node routinely finishes ahead of the other, and a node's
//! worth of cores idling until the last owner drains its deque costs far more than
//! the remote reads.
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
/// The deque itself is created by [`build`](ChannelFactory::build), on the
/// worker that owns it, and its [`Stealer`] published into the slot the
/// factory holds in the *shared* stealer table. Creating the deques on the
/// workers, rather than on the thread that plans the query, keeps that
/// thread's work per stage independent of the worker count: a pool of
/// hundreds of workers would otherwise spend most of a query's compile time
/// allocating deques one after another.
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
        let worker = Rc::new((self.new_worker)());
        self.stealers[self.worker_idx]
            .set(worker.stealer())
            .unwrap_or_else(|_| panic!("stealable channel endpoint built twice"));
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
    /// empty until its worker has built its endpoint; a deque that does not
    /// exist yet holds nothing to steal.
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

    /// Try to steal a message from a peer worker's deque: the same-node peers
    /// first, and the other nodes' only when none of them holds anything.
    fn steal(&self) -> Option<I> {
        self.steal_from(self.siblings.clone())
            .or_else(|| self.steal_from(0..self.siblings.start))
            .or_else(|| self.steal_from(self.siblings.end..self.stealers.len()))
    }
}

impl<I> StealableReceiver<I> {
    /// Iterates through the given workers' stealers, retrying on contention.
    ///
    /// The `is_empty` pre-check keeps the (very common) all-empty scan cheap:
    /// it is a couple of plain loads, while `steal()` pins a crossbeam epoch
    /// and CASes even when it finds nothing. Idle workers re-run this scan on
    /// every wakeup, so without the pre-check the pool burns a large share of
    /// its cycles in epoch bookkeeping just discovering there is no work.
    fn steal_from(&self, peers: std::ops::Range<usize>) -> Option<I> {
        use crossbeam_deque::Steal;
        for peer in peers {
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
    fn a_sibling_that_has_not_built_its_endpoint_is_skipped() {
        crate::waker::install_test_worker_waker();
        let mut factories = stealable::<i64>(Topology {
            workers_per_node: 2,
            node_count: 1,
        })
        .into_iter();
        let (mut sender, receiver) = factories.next().unwrap().build();
        let unbuilt_sibling = factories.next().unwrap();
        sender.send(7).unwrap();

        let stolen_before = receiver.steal();
        let (_, sibling_receiver) = unbuilt_sibling.build();
        let stolen_after = sibling_receiver.steal();

        assert_eq!(stolen_before, None);
        assert_eq!(stolen_after, Some(7));
    }

    #[test]
    fn other_node_worker_steals_only_once_its_own_node_is_empty() {
        let mut endpoints = two_node_endpoints::<i64>();
        endpoints[0].0.send(7).unwrap();
        endpoints[3].0.send(8).unwrap();

        let from_own_node = endpoints[2].1.steal();
        let from_other_node = endpoints[2].1.steal();

        assert_eq!(from_own_node, Some(8));
        assert_eq!(from_other_node, Some(7));
    }
}
