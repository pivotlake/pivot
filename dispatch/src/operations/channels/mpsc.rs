//! Multi-producer, single-consumer channel with an atomic item count.
//!
//! Wraps `std::sync::mpsc` with an [`AtomicUsize`] counter so the receiver can
//! check [`is_empty`](Receiver::is_empty) without blocking. Used for the final
//! output channel in [`collect`](crate::api::RecordBatchOperatorSpec::collect)
//! and internally by the [`return_to_worker`](super::return_to_worker) and
//! [`fan_in`](mod@super::fan_in) channels.

use crate::operations::channels;
use crate::operations::channels::{Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::channel;

/// Sending end of an mpsc channel. Cloneable — each clone shares the same atomic count.
pub struct MpscSender<T> {
    inner: mpsc::Sender<T>,
    count: Arc<AtomicUsize>,
}

impl<T> Clone for MpscSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            count: self.count.clone(),
        }
    }
}

impl<T> MpscSender<T> {
    /// Send without requiring `&mut`, for callers that route through a shared
    /// sender slice (see `return_to_worker`). The inner mpsc sender is `&self`
    /// already; the [`Sender`] trait's `&mut` is just its calling convention.
    ///
    /// Nobody is woken. A channel whose receiver is a worker, which may be
    /// parked, wakes that worker after the send.
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

/// An mpsc channel whose sends bump the count but wake nobody, as suits a
/// receiver that is not a worker (e.g. the final output channel drained by
/// the query's caller).
pub fn mpsc_channel<T>() -> (MpscSender<T>, MpscReceiver<T>) {
    let (tx, rx) = channel();
    let count = Arc::new(AtomicUsize::default());
    (
        MpscSender {
            inner: tx,
            count: count.clone(),
        },
        MpscReceiver { inner: rx, count },
    )
}
