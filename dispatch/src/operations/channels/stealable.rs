//! Stealable channels: each worker owns a deque, and idle workers steal from
//! same-node peers.
//!
//! The channel's locality follows the NUMA layout of the dispatch pool. A
//! message pushed on one node stays on that node: a thief only scans the
//! deques of its own node's workers, so the buffers it takes were allocated
//! and filled by a core sharing its memory controller and last-level cache.
//! Work moves across nodes only through the scan split and the pipeline
//! breakers that partition by hash, never through stealing.
//!
//! Note that this should be used for messages that are logically independent of the
//! worker that sent them; for example, it's fine for a RecordBatch to be processed on
//! any worker. In contrast, there are some messages that are `Send`, but are still
//! logically local to a worker; for example, while `CompressedPage` *is* `Send`, it is critical to
//! return it to the worker that sent out, since it needs to be sent to the Decoder of that row group.
//! Therefore it should *not* be in a stealable channel (but in `return_to_worker`)

use crate::api::WorkerSlots;
use crate::numa::Topology;
use crate::operations::channels;
use crate::operations::channels::{ChannelFactory, Receiver, Sender};
use crate::waker::worker_waker;
use crossbeam_deque::{Stealer, Worker};
use std::rc::Rc;
use std::sync::Arc;

/// The order a deque hands its owning worker its own messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    Lifo,
    Fifo,
}

impl Order {
    fn new_deque<T>(self) -> Worker<T> {
        match self {
            Order::Lifo => Worker::new_lifo(),
            Order::Fifo => Worker::new_fifo(),
        }
    }
}

/// Factory for building one worker's stealable channel endpoint.
///
/// Holds the local [`Worker`] deque plus the *shared* stealers over every
/// worker's deque and this worker's place among them.
pub struct StealableChannelFactory<T: Send> {
    deque: Worker<T>,
    stealers: Arc<[Stealer<T>]>,
    worker_idx: usize,
    topology: Topology,
}

impl<T: Send + 'static> ChannelFactory<T> for StealableChannelFactory<T> {
    type Sender = StealableSender<T>;
    type Receiver = StealableReceiver<T>;

    fn build(self) -> (StealableSender<T>, StealableReceiver<T>) {
        let deque = Rc::new(self.deque);
        (
            StealableSender(deque.clone()),
            StealableReceiver {
                deque,
                stealers: self.stealers,
                siblings: self.topology.node_siblings(self.worker_idx),
                worker_idx: self.worker_idx,
            },
        )
    }
}

/// The producing end: pushes onto the worker's own deque.
pub struct StealableSender<T>(Rc<Worker<T>>);

impl<O> Sender<O> for StealableSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.0.push(item);
        // One new item needs one thief: wake a single parked same-node peer.
        // Waking more would just herd workers into finding nothing (the send
        // path must stay cheap; see WorkerWaker).
        worker_waker().notify_one();
        Ok(())
    }
}

/// Receiving end of a stealable channel.
///
/// Shares the deque with the sender (both live on the same worker thread).
/// [`Receiver::try_recv`] pops from the local deque; [`Receiver::steal`] tries its
/// same-node peers' deques in order when the local one is empty, enabling
/// cross-worker load balancing within the node.
pub struct StealableReceiver<I> {
    deque: Rc<Worker<I>>,
    /// Every worker's stealer, shared by all receivers of the stage.
    stealers: Arc<[Stealer<I>]>,
    /// The slice of the stealers this worker may steal from (its node's workers).
    siblings: std::ops::Range<usize>,
    /// This worker's own index among the stealers, skipped when stealing.
    worker_idx: usize,
}

impl<I> Receiver<I> for StealableReceiver<I> {
    fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    /// Pop a message from the local deque (newest first, or oldest first for
    /// a [`stealable_fifo`] channel).
    fn try_recv(&self) -> Option<I> {
        self.deque.pop()
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
            let stealer = &self.stealers[peer];
            if stealer.is_empty() {
                continue;
            }
            loop {
                match stealer.steal() {
                    Steal::Success(item) => return Some(item),
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
        }
        None
    }
}

/// A stealable stage: the deques are made with the stage, on the thread
/// compiling the dataflow, and every worker takes its own when it builds.
/// Making each deque on its worker instead moves the same work into the
/// workers' build, on the query's critical path, where the compiling thread
/// does it while the workers are still finishing the previous query; it also
/// needs the stealers published to peers before any of them may steal.
pub type StealableStage<T> = WorkerSlots<StealableChannelFactory<T>>;

/// A stealable channel over `topology`'s workers, each wired to steal from
/// its same-node peers only (see the module docs for why stealing never
/// crosses NUMA nodes). The local worker pops its newest message first.
pub fn stealable<T: Send + 'static>(topology: Topology) -> StealableStage<T> {
    stealable_with(topology, Order::Lifo)
}

/// Like [`stealable`], but the local worker pops its oldest message first,
/// for messages that form a sequence the worker should follow in order.
/// Thieves take the oldest message in both flavours.
pub fn stealable_fifo<T: Send + 'static>(topology: Topology) -> StealableStage<T> {
    stealable_with(topology, Order::Fifo)
}

fn stealable_with<T: Send + 'static>(topology: Topology, order: Order) -> StealableStage<T> {
    let deques: Vec<Worker<T>> = (0..topology.total_workers())
        .map(|_| order.new_deque())
        .collect();
    let stealers: Arc<[Stealer<T>]> = deques.iter().map(Worker::stealer).collect();
    WorkerSlots::new(deques.into_iter().enumerate().map(|(worker_idx, deque)| {
        StealableChannelFactory {
            deque,
            stealers: stealers.clone(),
            worker_idx,
            topology,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::WorkerMaker;

    type Endpoint<T> = (StealableSender<T>, StealableReceiver<T>);

    fn two_node_endpoints<T: Send + 'static>() -> Vec<Endpoint<T>> {
        crate::waker::install_test_worker_waker();
        let stage = stealable::<T>(Topology {
            workers_per_node: 2,
            node_count: 2,
        });
        (0..4).map(|worker| stage.make(worker).build()).collect()
    }

    #[test]
    fn a_worker_steals_only_from_its_node() {
        let mut endpoints = two_node_endpoints::<i64>();
        endpoints[0].0.send(1).unwrap();

        assert_eq!(endpoints[1].1.steal(), Some(1));
        assert_eq!(endpoints[2].1.steal(), None);
        assert_eq!(endpoints[3].1.steal(), None);
    }

    #[test]
    fn a_fifo_channel_hands_the_owner_its_oldest_message() {
        crate::waker::install_test_worker_waker();
        let stage = stealable_fifo::<i64>(Topology {
            workers_per_node: 2,
            node_count: 1,
        });
        let mut endpoints: Vec<Endpoint<i64>> =
            (0..2).map(|worker| stage.make(worker).build()).collect();
        endpoints[0].0.send(1).unwrap();
        endpoints[0].0.send(2).unwrap();

        assert_eq!(endpoints[0].1.try_recv(), Some(1));
        assert_eq!(endpoints[0].1.try_recv(), Some(2));
    }
}
