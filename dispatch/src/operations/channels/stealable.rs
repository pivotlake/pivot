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
//!
//! A deque allocates its buffer when created, and a dataflow creates one per
//! worker per stealable stage on the thread compiling it, so on a large pool
//! those allocations were most of a small query's compile. Instead, the deques
//! of a finished channel go back to a pool, by message type, and the next
//! channel of that type takes them from there.

use crate::numa::Topology;
use crate::operations::channels;
use crate::operations::channels::{ChannelFactory, Receiver, Sender};
use crate::waker::worker_waker;
use crossbeam_deque::{Stealer, Worker};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, LazyLock, Mutex};

/// The order a deque hands its owning worker its own messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// The deques of finished channels, keyed by message type and order, each
/// entry a `Vec<Worker<T>>` for its type.
type DequePool = Mutex<HashMap<(TypeId, Order), Box<dyn Any + Send>>>;

static DEQUE_POOL: LazyLock<DequePool> = LazyLock::new(Default::default);

/// At most this many deques are kept per message type and order: several
/// dataflows' worth on a large pool, so a burst of concurrent queries does not
/// park memory for good once it passes.
const MAX_POOLED_DEQUES_PER_SHAPE: usize = 4096;

/// `count` deques for messages of type `T`, from the pool while it has them
/// and freshly created for the rest.
fn take_deques<T: Send + 'static>(order: Order, count: usize) -> Vec<Worker<T>> {
    let mut deques = {
        let mut pool = DEQUE_POOL.lock().unwrap();
        match pool.get_mut(&(TypeId::of::<T>(), order)) {
            Some(entry) => {
                let pooled = entry
                    .downcast_mut::<Vec<Worker<T>>>()
                    .expect("a pool entry holds the deques of the type and order it is keyed by");
                let keep = pooled.len().saturating_sub(count);
                pooled.split_off(keep)
            }
            None => Vec::new(),
        }
    };
    let missing = count - deques.len();
    deques.extend((0..missing).map(|_| order.new_deque()));
    deques
}

fn return_deques<T: Send + 'static>(order: Order, deques: Vec<Worker<T>>) {
    if deques.is_empty() {
        return;
    }
    let mut pool = DEQUE_POOL.lock().unwrap();
    let pooled = pool
        .entry((TypeId::of::<T>(), order))
        .or_insert_with(|| Box::new(Vec::<Worker<T>>::new()))
        .downcast_mut::<Vec<Worker<T>>>()
        .expect("a pool entry holds the deques of the type and order it is keyed by");
    let room = MAX_POOLED_DEQUES_PER_SHAPE.saturating_sub(pooled.len());
    pooled.extend(deques.into_iter().take(room));
}

#[cfg(test)]
fn pooled_deque_count<T: Send + 'static>(order: Order) -> usize {
    let pool = DEQUE_POOL.lock().unwrap();
    pool.get(&(TypeId::of::<T>(), order)).map_or(0, |entry| {
        entry
            .downcast_ref::<Vec<Worker<T>>>()
            .expect("a pool entry holds the deques of the type and order it is keyed by")
            .len()
    })
}

/// What one channel's workers share: every worker's stealer, and a slot per
/// worker for its deque once that worker's endpoints are gone. The deques go
/// back to the pool together when the last worker lets go of this, since a
/// stealer can reach a peer's deque only while this is alive, so a pooled
/// deque is never one a peer could still steal from.
struct StageDeques<T: Send + 'static> {
    stealers: Box<[Stealer<T>]>,
    order: Order,
    returned: Box<[Mutex<Option<Worker<T>>>]>,
}

impl<T: Send + 'static> Drop for StageDeques<T> {
    fn drop(&mut self) {
        let deques = self
            .returned
            .iter()
            .filter_map(|slot| slot.lock().unwrap().take())
            .collect();
        return_deques(self.order, deques);
    }
}

/// One worker's deque, shared by that worker's sender and receiver. When the
/// last of the two drops, a drained deque goes into the stage's slot for the
/// pool; one still holding messages belongs to a cancelled dataflow, and
/// drops with them rather than carry them into the next dataflow's channel.
struct DequeOwner<T: Send + 'static> {
    deque: Option<Worker<T>>,
    stage: Arc<StageDeques<T>>,
    worker_idx: usize,
}

impl<T: Send + 'static> DequeOwner<T> {
    fn deque(&self) -> &Worker<T> {
        self.deque
            .as_ref()
            .expect("the deque leaves its owner only when the owner drops")
    }
}

impl<T: Send + 'static> Drop for DequeOwner<T> {
    fn drop(&mut self) {
        let Some(deque) = self.deque.take() else {
            return;
        };
        if deque.is_empty() {
            *self.stage.returned[self.worker_idx].lock().unwrap() = Some(deque);
        }
    }
}

pub struct StealableChannelFactory<T: Send + 'static> {
    deque: Worker<T>,
    stage: Arc<StageDeques<T>>,
    worker_idx: usize,
    topology: Topology,
}

impl<T: Send + 'static> ChannelFactory<T> for StealableChannelFactory<T> {
    type Sender = StealableSender<T>;
    type Receiver = StealableReceiver<T>;

    fn build(self) -> (StealableSender<T>, StealableReceiver<T>) {
        let owner = Rc::new(DequeOwner {
            deque: Some(self.deque),
            stage: self.stage,
            worker_idx: self.worker_idx,
        });
        (
            StealableSender(owner.clone()),
            StealableReceiver {
                owner,
                siblings: self.topology.node_siblings(self.worker_idx),
                worker_idx: self.worker_idx,
            },
        )
    }
}

/// The producing end: pushes onto the worker's own deque.
pub struct StealableSender<T: Send + 'static>(Rc<DequeOwner<T>>);

impl<O: Send + 'static> Sender<O> for StealableSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.0.deque().push(item);
        worker_waker().notify_one();
        Ok(())
    }
}

pub struct StealableReceiver<I: Send + 'static> {
    owner: Rc<DequeOwner<I>>,
    siblings: std::ops::Range<usize>,
    worker_idx: usize,
}

impl<I: Send + 'static> Receiver<I> for StealableReceiver<I> {
    fn is_empty(&self) -> bool {
        self.owner.deque().is_empty()
    }

    fn try_recv(&self) -> Option<I> {
        self.owner.deque().pop()
    }

    fn steal(&self) -> Option<I> {
        use crossbeam_deque::Steal;
        let stealers = &self.owner.stage.stealers;
        for peer in self.siblings.clone() {
            if peer == self.worker_idx {
                continue;
            }
            let stealer = &stealers[peer];
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

/// Create one [`StealableChannelFactory`] per worker. The local worker pops its
/// newest message first.
pub fn stealable<T: Send + 'static>(
    topology: Topology,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    stealable_with(topology, Order::Lifo)
}

/// Like [`stealable`], but the local worker pops its oldest message first,
/// for messages that form a sequence the worker should follow in order.
/// Thieves take the oldest message in both flavours.
pub fn stealable_fifo<T: Send + 'static>(
    topology: Topology,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    stealable_with(topology, Order::Fifo)
}

fn stealable_with<T: Send + 'static>(
    topology: Topology,
    order: Order,
) -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    let deques = take_deques::<T>(order, topology.total_workers());
    let stage = Arc::new(StageDeques {
        stealers: deques.iter().map(Worker::stealer).collect(),
        order,
        returned: deques.iter().map(|_| Mutex::new(None)).collect(),
    });

    deques
        .into_iter()
        .enumerate()
        .map(move |(worker_idx, deque)| StealableChannelFactory {
            deque,
            stage: stage.clone(),
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
    fn other_node_worker_cannot_steal() {
        let mut endpoints = two_node_endpoints::<i64>();

        endpoints[0].0.send(7).unwrap();

        assert_eq!(endpoints[2].1.steal(), None);
        assert_eq!(endpoints[3].1.steal(), None);
    }

    /// Each test pools under its own message type, so tests sharing the
    /// process-wide pool do not see each other's deques.
    #[test]
    fn a_finished_channel_pools_its_drained_deques_for_the_next_channel() {
        struct Message;
        let endpoints = two_node_endpoints::<Message>();

        drop(endpoints);
        let pooled_after_drop = pooled_deque_count::<Message>(Order::Lifo);
        let _next = two_node_endpoints::<Message>();
        let pooled_after_reuse = pooled_deque_count::<Message>(Order::Lifo);

        assert_eq!(pooled_after_drop, 4);
        assert_eq!(pooled_after_reuse, 0);
    }

    #[test]
    fn a_deque_still_holding_messages_is_not_pooled() {
        struct Message;
        let mut endpoints = two_node_endpoints::<Message>();
        endpoints[0].0.send(Message).unwrap();

        drop(endpoints);

        assert_eq!(pooled_deque_count::<Message>(Order::Lifo), 3);
    }
}
