//! Worker-affinity channel that routes messages back to a specific worker.
//!
//! Some messages must be processed by the worker that owns related state. For example,
//! a `DecompressedPage` must return to
//! the worker whose decoder holds the corresponding row group context.
//!
//! [`WorkerAwareSender`] inspects each message's [`worker_id`](WorkerIdOutput::worker_id)
//! and sends it to that worker's dedicated mpsc channel. Each worker's receiver sees
//! only messages destined for it.

use crate::Identifier;
use crate::operations::channels;
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel_to};
use crate::operations::channels::{ChannelFactory, Sender};
use std::sync::{Arc, OnceLock};

/// Factory for building a return-to-worker channel.
///
/// This worker's channel is created by [`build`](ChannelFactory::build), on
/// the worker thread, and its sender published into the *shared* slot array
/// every built [`WorkerAwareSender`] routes through. A message can only be
/// addressed to a worker that has already built the stage, since the worker
/// id it carries comes from that worker's own operators.
pub struct ReturnToWorkerMpscFactory<T> {
    senders: Arc<[OnceLock<MpscSender<T>>]>,
    worker: usize,
}

impl<T: 'static + Send + WorkerIdOutput> ChannelFactory<T> for ReturnToWorkerMpscFactory<T> {
    type Sender = WorkerAwareSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        let (sender, receiver) = mpsc_channel_to::<T>(self.worker);
        if self.senders[self.worker].set(sender).is_err() {
            panic!("a worker's return channel was published twice");
        }
        (WorkerAwareSender::new(self.senders), receiver)
    }
}

/// Implemented by message types that know which worker they belong to.
pub trait WorkerIdOutput: 'static {
    fn worker_id(&self) -> Identifier;
}

/// A sender that routes each message to a specific worker's mpsc channel
/// based on [`WorkerIdOutput::worker_id`].
pub struct WorkerAwareSender<O> {
    senders: Arc<[OnceLock<MpscSender<O>>]>,
}

impl<O> WorkerAwareSender<O> {
    pub fn new(senders: Arc<[OnceLock<MpscSender<O>>]>) -> Self {
        Self { senders }
    }
}

impl<O: WorkerIdOutput> Sender<O> for WorkerAwareSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        let worker_idx = item.worker_id();
        self.senders[worker_idx]
            .get()
            .expect("a message returns only to a worker that has built the stage")
            .send_ref(item)?;
        Ok(())
    }
}

/// Create one [`ReturnToWorkerMpscFactory`] per worker.
///
/// Sets up N mpsc channels (one per worker). Each factory holds the shared
/// sender slice (for routing) and its own receiver.
pub fn return_to_worker_mpsc<T: 'static + Send + WorkerIdOutput>(
    count: usize,
) -> impl IntoIterator<Item = ReturnToWorkerMpscFactory<T>> {
    let senders: Arc<[OnceLock<MpscSender<T>>]> = (0..count).map(|_| OnceLock::new()).collect();

    (0..count).map(move |worker| ReturnToWorkerMpscFactory {
        senders: senders.clone(),
        worker,
    })
}
