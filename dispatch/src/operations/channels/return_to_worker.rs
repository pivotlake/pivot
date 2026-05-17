//! Worker-affinity channel that routes messages back to a specific worker.
//!
//! Some messages must be processed by the worker that owns related state. For example,
//! a [`DecompressedPage`](crate::operations::parquet::DecompressedPage) must return to
//! the worker whose decoder holds the corresponding row group context.
//!
//! [`WorkerAwareSender`] inspects each message's [`worker_id`](WorkerIdOutput::worker_id)
//! and sends it to that worker's dedicated mpsc channel. Each worker's receiver sees
//! only messages destined for it.

use crate::operations::channels;
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel};
use crate::operations::channels::{ChannelFactory, Sender};
use crate::{Identifier};

/// Factory for building a return-to-worker channel.
///
/// Holds all workers' mpsc senders (so the built [`WorkerAwareSender`] can route to any
/// worker) and this worker's receiver.
pub struct ReturnToWorkerMpscFactory<T> {
    senders: Vec<MpscSender<T>>,
    receiver: MpscReceiver<T>,
}

impl<T: 'static + Send + WorkerIdOutput> ChannelFactory<T> for ReturnToWorkerMpscFactory<T> {
    type Sender = WorkerAwareSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (WorkerAwareSender::new(self.senders), self.receiver)
    }
}

/// Implemented by message types that know which worker they belong to.
pub trait WorkerIdOutput: 'static {
    fn worker_id(&self) -> Identifier;
}

/// A sender that routes each message to a specific worker's mpsc channel
/// based on [`WorkerIdOutput::worker_id`].
pub struct WorkerAwareSender<O> {
    senders: Vec<MpscSender<O>>,
}

impl<O> WorkerAwareSender<O> {
    pub fn new(senders: Vec<MpscSender<O>>) -> Self {
        Self { senders }
    }
}

impl<O: WorkerIdOutput> Sender<O> for WorkerAwareSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        let worker_idx = item.worker_id();
        self.senders[worker_idx].send(item)?;
        Ok(())
    }
}

/// Create one [`ReturnToWorkerMpscFactory`] per worker.
///
/// Sets up N mpsc channels (one per worker). Each factory holds all N senders
/// (for routing) and its own receiver.
pub fn return_to_worker_mpsc<T: 'static + Send + WorkerIdOutput>(count: usize)
-> impl IntoIterator<Item = ReturnToWorkerMpscFactory<T>> {
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..count)
        .map(|_| mpsc_channel::<T>())
        .unzip();

    receivers.into_iter().map(move |rx| {
        let txs: Vec<_> = senders.to_vec();
        ReturnToWorkerMpscFactory {
            senders: txs,
            receiver: rx,
        }
    })
}
