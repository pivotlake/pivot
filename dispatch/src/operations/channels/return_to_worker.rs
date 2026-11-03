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
use crate::operations::channels::mpsc::{WorkerChannelReceiver, WorkerChannelSet};
use crate::operations::channels::{ChannelFactory, Sender};
use std::sync::Arc;

pub struct ReturnToWorkerMpscFactory<T: Send + 'static> {
    set: Arc<WorkerChannelSet<T>>,
    receiver: WorkerChannelReceiver<T>,
}

impl<T: 'static + Send + WorkerIdOutput> ChannelFactory<T> for ReturnToWorkerMpscFactory<T> {
    type Sender = WorkerAwareSender<T>;
    type Receiver = WorkerChannelReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (WorkerAwareSender::new(self.set), self.receiver)
    }
}

pub trait WorkerIdOutput: 'static {
    fn worker_id(&self) -> Identifier;
}

pub struct WorkerAwareSender<O: Send + 'static> {
    set: Arc<WorkerChannelSet<O>>,
}

impl<O: Send + 'static> WorkerAwareSender<O> {
    pub fn new(set: Arc<WorkerChannelSet<O>>) -> Self {
        Self { set }
    }
}

impl<O: Send + WorkerIdOutput> Sender<O> for WorkerAwareSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.set.send_to(item.worker_id(), item)
    }
}

pub fn return_to_worker_mpsc<T: 'static + Send + WorkerIdOutput>(
    count: usize,
) -> impl IntoIterator<Item = ReturnToWorkerMpscFactory<T>> {
    let (set, receivers) = WorkerChannelSet::new(count);

    receivers
        .into_iter()
        .map(move |receiver| ReturnToWorkerMpscFactory {
            set: set.clone(),
            receiver,
        })
}
