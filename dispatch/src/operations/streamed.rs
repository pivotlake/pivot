//! Work-stealing source fed incrementally by an external producer.
//!
//! The open-ended counterpart of [`in_memory`](crate::operations::in_memory)'s
//! injector source: items arrive while the dataflow runs (e.g. a protocol
//! stream parsed into batches), so emptiness does not mean exhaustion. The
//! source reports itself drained only once the producer has closed it and the
//! queue is empty; until then workers that find nothing park and every push
//! wakes one.
//!
//! The queue is a bounded [`ArrayQueue`], which enforces the capacity itself:
//! [`ChannelInputSender::try_send`] refuses items when it is full, and the
//! `on_claim` callback fires each time a worker takes an item, so an async
//! producer can sleep on its own primitive and retry when space opens up.

use crate::operations::channels::{Receiver, RootChannelFactory};
use crate::waker::{WakerSet, waker_set};
use crossbeam_queue::ArrayQueue;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

struct ChannelInputShared<T> {
    queue: ArrayQueue<T>,
    /// Set once by the producer; after this, an empty queue means exhausted.
    closed: AtomicBool,
    /// Fired on every claim so the producer can wake and refill.
    on_claim: Box<dyn Fn() + Send + Sync>,
}

/// Producer half of a [`channel_input`](crate::channel_input) source. One
/// producer feeds the dataflow and closes the channel when its stream ends
/// (dropping the sender closes it too).
pub struct ChannelInputSender<T> {
    shared: Arc<ChannelInputShared<T>>,
    wakers: WakerSet,
}

/// A rejected [`try_send`](ChannelInputSender::try_send): the queue is at
/// capacity, and the item is handed back for the producer to retry.
pub struct ChannelInputFull<T>(pub T);

impl<T: Send> ChannelInputSender<T> {
    /// Queue `item` unless the channel is at capacity, waking a parked
    /// worker. On `Err` the caller should wait for a claim (see `on_claim`)
    /// and retry.
    pub fn try_send(&self, item: T) -> Result<(), ChannelInputFull<T>> {
        match self.shared.queue.push(item) {
            Ok(()) => {
                // One item wants one worker; the pool-wide broadcast is
                // reserved for close, where every parked worker must re-run
                // its finish check. The producer is off-pool, so no node is
                // nearer than any other.
                self.wakers.notify_one_near(0);
                Ok(())
            }
            Err(item) => Err(ChannelInputFull(item)),
        }
    }
}

impl<T> ChannelInputSender<T> {
    /// Mark the stream complete: once the queue drains, the source reports
    /// exhausted and the dataflow's sources can finish. Idempotent.
    pub fn close(&self) {
        if !self.shared.closed.swap(true, Ordering::Release) {
            self.wakers.notify_all();
        }
    }
}

impl<T> Drop for ChannelInputSender<T> {
    fn drop(&mut self) {
        self.close();
    }
}

/// Factory handing each worker a [`ChannelSource`] over the shared queue.
pub struct ChannelSourceFactory<T> {
    shared: Arc<ChannelInputShared<T>>,
}

// Manual `Clone`: a derive would demand `T: Clone` for the shared `Arc`.
impl<T> Clone for ChannelSourceFactory<T> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl<T: Send> ChannelSourceFactory<T> {
    /// Build the producer and per-worker-source halves. `wakers` must be the
    /// worker pool's waker set, so pushes reach parked workers.
    pub(crate) fn new(
        capacity: usize,
        on_claim: Box<dyn Fn() + Send + Sync>,
        wakers: WakerSet,
    ) -> (ChannelInputSender<T>, Self) {
        let shared = Arc::new(ChannelInputShared {
            queue: ArrayQueue::new(capacity),
            closed: AtomicBool::new(false),
            on_claim,
        });
        (
            ChannelInputSender {
                shared: shared.clone(),
                wakers,
            },
            Self { shared },
        )
    }
}

impl<T: Send + 'static> RootChannelFactory<T> for ChannelSourceFactory<T> {
    type Receiver = ChannelSource<T>;

    fn build(self) -> Self::Receiver {
        ChannelSource {
            shared: self.shared,
        }
    }
}

/// A [`Receiver`] over the shared queue. Like the in-memory injector source,
/// work is only ever taken from the shared queue (no per-worker local queue),
/// so items load balance across the pool.
pub struct ChannelSource<T> {
    shared: Arc<ChannelInputShared<T>>,
}

impl<T> ChannelSource<T> {
    /// Bookkeeping after taking one item: free the producer's slot, and once
    /// the closed queue drains, broadcast-wake the pool so workers parked at
    /// their finish check can sign off (the streamed analogue of the in-memory
    /// source's last-claim wake).
    fn after_claim(&self) {
        (self.shared.on_claim)();
        if self.shared.closed.load(Ordering::Acquire) && self.shared.queue.is_empty() {
            waker_set().notify_all();
        }
    }
}

impl<T: Send> Receiver<T> for ChannelSource<T> {
    /// Empty means *exhausted* here: the finish protocol treats an empty root
    /// receiver as done, so an open channel must read as non-empty even while
    /// the queue is momentarily drained. `closed` is loaded first (Acquire
    /// pairing with the producer's Release) so a close racing a final push is
    /// seen with that push already visible.
    fn is_empty(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire) && self.shared.queue.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        let item = self.shared.queue.pop()?;
        self.after_claim();
        Some(item)
    }

    fn steal(&self) -> Option<T> {
        self.try_recv()
    }
}
