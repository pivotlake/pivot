//! A shared work queue for each NUMA node.
//!
//! Every message names its preferred node. Producers route directly to that
//! node's queue, and workers only poll the queue for their own node. This keeps
//! memory-heavy work near its input while still balancing it across all local
//! workers.

use std::sync::Arc;

use crossbeam_deque::{Injector, Steal};

use crate::Topology;
use crate::waker::waker_set;

use super::{ChannelFactory, Receiver, Result, Sender};

/// Supplies the NUMA node on which a message should be processed.
pub trait NodeIdOutput {
    fn node_id(&self) -> usize;
}

pub struct NodeWorkQueueChannelFactory<T: NodeIdOutput + Send> {
    queues: Arc<[Injector<T>]>,
    node_id: usize,
}

impl<T: NodeIdOutput + Send + 'static> ChannelFactory<T> for NodeWorkQueueChannelFactory<T> {
    type Sender = NodeWorkQueueSender<T>;
    type Receiver = NodeWorkQueueReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            NodeWorkQueueSender {
                queues: self.queues.clone(),
            },
            NodeWorkQueueReceiver {
                queues: self.queues,
                node_id: self.node_id,
            },
        )
    }
}

pub struct NodeWorkQueueSender<T> {
    queues: Arc<[Injector<T>]>,
}

impl<T: NodeIdOutput> Sender<T> for NodeWorkQueueSender<T> {
    fn send(&mut self, item: T) -> Result<()> {
        let node_id = item.node_id();
        let queue = self
            .queues
            .get(node_id)
            .unwrap_or_else(|| panic!("NUMA node {node_id} is outside the worker topology"));
        queue.push(item);
        waker_set().notify_one_near(node_id);
        Ok(())
    }
}

pub struct NodeWorkQueueReceiver<T> {
    queues: Arc<[Injector<T>]>,
    node_id: usize,
}

impl<T> Receiver<T> for NodeWorkQueueReceiver<T> {
    fn is_empty(&self) -> bool {
        self.queues[self.node_id].is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        loop {
            match self.queues[self.node_id].steal() {
                Steal::Success(item) => return Some(item),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }

    fn steal(&self) -> Option<T> {
        None
    }
}

/// Creates one endpoint per worker, backed by one queue per NUMA node.
pub fn node_work_queue<T: NodeIdOutput + Send>(
    topology: Topology,
) -> impl IntoIterator<Item = NodeWorkQueueChannelFactory<T>> {
    let queues: Arc<[Injector<T>]> = (0..topology.node_count).map(|_| Injector::new()).collect();
    (0..topology.total_workers()).map(move |worker_id| NodeWorkQueueChannelFactory {
        queues: queues.clone(),
        node_id: topology.node_of_worker(worker_id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waker::{WakerSet, WorkerWaker, init_waker_set};

    struct Work(usize);

    impl NodeIdOutput for Work {
        fn node_id(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn workers_only_receive_work_for_their_node() {
        let topology = Topology {
            workers_per_node: 1,
            node_count: 2,
        };
        let wakers: Vec<_> = (0..2).map(|_| Arc::new(WorkerWaker::new(1))).collect();
        init_waker_set(WakerSet::new(wakers, 1));
        let mut endpoints: Vec<_> = node_work_queue(topology)
            .into_iter()
            .map(ChannelFactory::build)
            .collect();

        endpoints[0].0.send(Work(1)).unwrap();

        assert!(endpoints[0].1.try_recv().is_none());
        assert_eq!(endpoints[1].1.try_recv().unwrap().0, 1);
    }
}
