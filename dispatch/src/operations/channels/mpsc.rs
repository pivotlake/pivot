//! Multi-producer, single-consumer channel with an atomic item count.
//!
//! Wraps `std::sync::mpsc` with an [`AtomicUsize`] counter so the receiver can
//! check [`is_empty`](Receiver::is_empty) without blocking. Used for the final
//! output channel in [`collect`](crate::api::RecordBatchOperatorSpec::collect)
//! and internally by the [`return_to_worker`](super::return_to_worker) and
//! [`fan_in`](mod@super::fan_in) channels.

use crate::operations::channels;
use crate::operations::channels::{Receiver, Sender};
use crate::waker::waker_set;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::sync::{Arc, LazyLock, Mutex};

/// Sending end of an mpsc channel. Cloneable — each clone shares the same atomic count.
pub struct MpscSender<T> {
    inner: mpsc::Sender<T>,
    count: Arc<AtomicUsize>,
    /// The global index of the worker that reads this channel, when the reader
    /// is a worker: each send wakes that worker's node so a parked receiver
    /// (possibly on another NUMA node than the sender) picks the item up.
    /// `None` when the receiver is not a worker (the final output channel,
    /// whose consumer blocks on the inner mpsc directly).
    receiving_worker: Option<usize>,
}

impl<T> Clone for MpscSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            count: self.count.clone(),
            receiving_worker: self.receiving_worker,
        }
    }
}

impl<T> MpscSender<T> {
    /// Send without requiring `&mut`, for callers that route through a shared
    /// sender slice (see `return_to_worker`). The inner mpsc sender is `&self`
    /// already; the [`Sender`] trait's `&mut` is just its calling convention.
    pub fn send_ref(&self, item: T) -> channels::Result<()> {
        self.inner
            .send(item)
            .map_err(|_| channels::Error::MpscSendError)?;
        // The item must be inside the inner mpsc *before* the count becomes
        // visible to the receiver.  Release here pairs with the Acquire load
        // in `is_empty`: if the receiver's Acquire sees count > 0, the
        // corresponding inner.send is guaranteed to have completed, so
        // `try_recv` will find the item.
        self.count.fetch_add(1, Ordering::Release);
        // Wake the receiving worker (which may be parked) so it picks up the
        // freshly-sent item. `return_to_worker` routes cross-worker messages
        // through this channel, so without a notify the target worker can sit
        // parked while its mpsc has work waiting.
        if let Some(worker) = self.receiving_worker {
            waker_set().notify_worker(worker);
        }
        Ok(())
    }
}

impl<T> Sender<T> for MpscSender<T> {
    fn send(&mut self, item: T) -> channels::Result<()> {
        self.send_ref(item)
    }
}

pub struct MpscReceiver<O> {
    inner: mpsc::Receiver<O>,
    count: Arc<AtomicUsize>,
}

impl<T> MpscReceiver<T> {
    pub fn new(inner: mpsc::Receiver<T>, count: Arc<AtomicUsize>) -> Self {
        Self { inner, count }
    }

    pub fn into_parts(self) -> (mpsc::Receiver<T>, Arc<AtomicUsize>) {
        (self.inner, self.count)
    }
}

impl<T> From<MpscReceiver<T>> for mpsc::Receiver<T> {
    fn from(value: MpscReceiver<T>) -> Self {
        value.inner
    }
}

impl<O> Receiver<O> for MpscReceiver<O> {
    /// Acquire pairs with the Release in `send`: if we see 0, no send has
    /// completed yet, so `try_recv` would also return None.  This is used by
    /// the finishing protocol in `UnaryOperator::try_finish` to decide whether
    /// all input has been drained.
    fn is_empty(&self) -> bool {
        self.count.load(Ordering::Acquire) == 0
    }

    fn try_recv(&self) -> Option<O> {
        match self.inner.try_recv().ok() {
            Some(s) => {
                // Relaxed is fine: the decrement is always on the same thread
                // that calls is_empty, so no cross-thread visibility concern.
                self.count.fetch_sub(1, Ordering::Relaxed);
                Some(s)
            }
            None => None,
        }
    }

    fn steal(&self) -> Option<O> {
        None
    }
}

/// An mpsc channel whose receiver is not a worker (e.g. the final output
/// channel drained by the query's caller): sends bump the count but wake nobody.
pub fn mpsc_channel<T>() -> (MpscSender<T>, MpscReceiver<T>) {
    let (tx, rx) = channel();
    let count = Arc::new(AtomicUsize::default());
    (
        MpscSender {
            inner: tx,
            count: count.clone(),
            receiving_worker: None,
        },
        MpscReceiver { inner: rx, count },
    )
}

/// Build an mpsc channel read by the worker with global index
/// `receiving_worker`; every send wakes that worker's node.
pub fn mpsc_channel_to<T>(receiving_worker: usize) -> (MpscSender<T>, MpscReceiver<T>) {
    let (tx, rx) = mpsc_channel();
    (
        MpscSender {
            receiving_worker: Some(receiving_worker),
            ..tx
        },
        rx,
    )
}

/// The channels of finished [`WorkerChannelSet`]s, keyed by message type,
/// each entry a `Vec<(MpscSender<T>, MpscReceiver<T>)>` for its type.
type ChannelPool = Mutex<HashMap<TypeId, Box<dyn Any + Send>>>;

static CHANNEL_POOL: LazyLock<ChannelPool> = LazyLock::new(Default::default);

/// At most this many channels are kept per message type: several dataflows'
/// worth on a large pool, so a burst of concurrent queries does not park
/// memory for good once it passes.
const MAX_POOLED_CHANNELS_PER_TYPE: usize = 4096;

type ChannelPair<T> = (MpscSender<T>, MpscReceiver<T>);

/// `count` channels, the one at index `i` read by worker `i`: from the pool
/// while it has them and freshly created for the rest.
fn take_channels<T: Send + 'static>(count: usize) -> Vec<ChannelPair<T>> {
    let mut pairs = {
        let mut pool = CHANNEL_POOL.lock().unwrap();
        match pool.get_mut(&TypeId::of::<T>()) {
            Some(entry) => {
                let pooled = entry
                    .downcast_mut::<Vec<ChannelPair<T>>>()
                    .expect("a pool entry holds the channels of the type it is keyed by");
                let keep = pooled.len().saturating_sub(count);
                pooled.split_off(keep)
            }
            None => Vec::new(),
        }
    };
    for (worker, (sender, _)) in pairs.iter_mut().enumerate() {
        sender.receiving_worker = Some(worker);
    }
    let missing = count - pairs.len();
    pairs.extend((pairs.len()..pairs.len() + missing).map(|worker| mpsc_channel_to::<T>(worker)));
    pairs
}

fn return_channels<T: Send + 'static>(pairs: Vec<ChannelPair<T>>) {
    if pairs.is_empty() {
        return;
    }
    let mut pool = CHANNEL_POOL.lock().unwrap();
    let pooled = pool
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Box::new(Vec::<ChannelPair<T>>::new()))
        .downcast_mut::<Vec<ChannelPair<T>>>()
        .expect("a pool entry holds the channels of the type it is keyed by");
    let room = MAX_POOLED_CHANNELS_PER_TYPE.saturating_sub(pooled.len());
    pooled.extend(pairs.into_iter().take(room));
}

#[cfg(test)]
fn pooled_channel_count<T: Send + 'static>() -> usize {
    let pool = CHANNEL_POOL.lock().unwrap();
    pool.get(&TypeId::of::<T>()).map_or(0, |entry| {
        entry
            .downcast_ref::<Vec<ChannelPair<T>>>()
            .expect("a pool entry holds the channels of the type it is keyed by")
            .len()
    })
}

/// One channel per worker, each read by that worker, with every sending end
/// kept together so an operator on any worker can address any receiver.
///
/// Creating a channel allocates its state, and a dataflow creates one per
/// worker for every stage routed this way, on the thread compiling it. So
/// when a set's last user lets go of it, after every receiver has come back,
/// its drained channels go to a pool, and the next set of that message type
/// takes them from there. A channel still holding messages belongs to a
/// cancelled dataflow, and drops with them.
pub struct WorkerChannelSet<T: Send + 'static> {
    senders: Vec<MpscSender<T>>,
    returned: Box<[Mutex<Option<MpscReceiver<T>>>]>,
}

impl<T: Send + 'static> WorkerChannelSet<T> {
    /// A set of `count` channels and each one's receiving end, in worker order.
    pub fn new(count: usize) -> (Arc<Self>, Vec<WorkerChannelReceiver<T>>) {
        let (senders, receivers): (Vec<_>, Vec<_>) = take_channels::<T>(count).into_iter().unzip();
        let set = Arc::new(Self {
            senders,
            returned: receivers.iter().map(|_| Mutex::new(None)).collect(),
        });
        let receivers = receivers
            .into_iter()
            .enumerate()
            .map(|(worker_idx, receiver)| WorkerChannelReceiver {
                receiver: Some(receiver),
                set: set.clone(),
                worker_idx,
            })
            .collect();
        (set, receivers)
    }

    /// Put `item` on the channel worker `worker` reads.
    pub fn send_to(&self, worker: usize, item: T) -> channels::Result<()> {
        self.senders[worker].send_ref(item)
    }
}

impl<T: Send + 'static> Drop for WorkerChannelSet<T> {
    fn drop(&mut self) {
        let pairs = std::mem::take(&mut self.senders)
            .into_iter()
            .zip(self.returned.iter())
            .filter_map(|(sender, slot)| {
                let receiver = slot.lock().unwrap().take()?;
                receiver.is_empty().then_some((sender, receiver))
            })
            .collect();
        return_channels(pairs);
    }
}

/// A worker's receiving end of a [`WorkerChannelSet`] channel; goes back to
/// the set when dropped.
pub struct WorkerChannelReceiver<T: Send + 'static> {
    receiver: Option<MpscReceiver<T>>,
    set: Arc<WorkerChannelSet<T>>,
    worker_idx: usize,
}

impl<T: Send + 'static> WorkerChannelReceiver<T> {
    fn receiver(&self) -> &MpscReceiver<T> {
        self.receiver
            .as_ref()
            .expect("the receiver leaves only when this drops")
    }
}

impl<T: Send + 'static> Receiver<T> for WorkerChannelReceiver<T> {
    fn is_empty(&self) -> bool {
        self.receiver().is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        self.receiver().try_recv()
    }

    fn steal(&self) -> Option<T> {
        None
    }
}

impl<T: Send + 'static> Drop for WorkerChannelReceiver<T> {
    fn drop(&mut self) {
        if let Some(receiver) = self.receiver.take() {
            *self.set.returned[self.worker_idx].lock().unwrap() = Some(receiver);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test pools under its own message type, so tests sharing the
    /// process-wide pool do not see each other's channels.
    #[test]
    fn a_finished_set_pools_its_drained_channels_for_the_next_set() {
        struct Message;
        crate::waker::install_test_worker_waker();
        let (set, receivers) = WorkerChannelSet::<Message>::new(3);

        drop(receivers);
        drop(set);
        let pooled_after_drop = pooled_channel_count::<Message>();
        let _next = WorkerChannelSet::<Message>::new(3);
        let pooled_after_reuse = pooled_channel_count::<Message>();

        assert_eq!(pooled_after_drop, 3);
        assert_eq!(pooled_after_reuse, 0);
    }

    #[test]
    fn a_channel_still_holding_a_message_is_not_pooled() {
        struct Message;
        crate::waker::install_test_worker_waker();
        let (set, receivers) = WorkerChannelSet::<Message>::new(3);
        set.send_to(0, Message).unwrap();

        drop(receivers);
        drop(set);

        assert_eq!(pooled_channel_count::<Message>(), 2);
    }

    #[test]
    fn a_reused_channel_delivers_to_its_new_worker() {
        struct Message(u8);
        crate::waker::install_test_worker_waker();
        let (set, receivers) = WorkerChannelSet::<Message>::new(2);
        drop(receivers);
        drop(set);
        let (set, receivers) = WorkerChannelSet::<Message>::new(2);

        set.send_to(0, Message(7)).unwrap();

        assert!(receivers[1].is_empty());
        assert_eq!(receivers[0].try_recv().map(|m| m.0), Some(7));
    }
}
