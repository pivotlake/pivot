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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::channel;

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

/// One worker's end of a channel that a single worker reads: the reader's
/// receiver, or nothing for every other worker, whose operator then finds no
/// input. The other workers get no channel of their own, so a channel read by
/// one of many workers costs one allocation, not one per worker.
pub struct SingleReaderReceiver<O>(Option<MpscReceiver<O>>);

impl<O> SingleReaderReceiver<O> {
    /// The reading worker's end.
    pub fn reader(receiver: MpscReceiver<O>) -> Self {
        Self(Some(receiver))
    }

    /// Any other worker's end, which never receives.
    pub fn other() -> Self {
        Self(None)
    }
}

impl<O> Receiver<O> for SingleReaderReceiver<O> {
    fn is_empty(&self) -> bool {
        self.0.as_ref().is_none_or(|receiver| receiver.is_empty())
    }

    fn try_recv(&self) -> Option<O> {
        self.0.as_ref()?.try_recv()
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
