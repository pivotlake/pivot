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
/// Every worker's sender lives in a slot of a slice shared by all the
/// workers' routing senders. A worker creates its own channel when it builds
/// its end, on the worker, rather than the thread compiling the query
/// creating one per worker up front; a message only returns to a worker
/// that has built its dataflow.
pub struct ReturnToWorkerMpscFactory<T> {
    senders: Arc<[OnceLock<MpscSender<T>>]>,
    worker_idx: usize,
}

impl<T: 'static + Send + WorkerIdOutput> ChannelFactory<T> for ReturnToWorkerMpscFactory<T> {
    type Sender = WorkerAwareSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        let (sender, receiver) = mpsc_channel_to::<T>(self.worker_idx);
        self.senders[self.worker_idx]
            .set(sender)
            .unwrap_or_else(|_| panic!("worker {} built its channel twice", self.worker_idx));
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
            .expect("a message returns to a worker that has built its dataflow")
            .send_ref(item)?;
        Ok(())
    }
}

/// Create one [`ReturnToWorkerMpscFactory`] per worker.
///
/// Each worker's mpsc channel is created when that worker builds its end
/// (see [`ReturnToWorkerMpscFactory`]); every factory shares the slots the
/// routing senders read.
pub fn return_to_worker_mpsc<T: 'static + Send + WorkerIdOutput>(
    count: usize,
) -> impl IntoIterator<Item = ReturnToWorkerMpscFactory<T>> {
    let senders: Arc<[OnceLock<MpscSender<T>>]> = (0..count).map(|_| OnceLock::new()).collect();
    (0..count).map(move |worker_idx| ReturnToWorkerMpscFactory {
        senders: senders.clone(),
        worker_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::channels::Receiver;
    use crate::waker::{WakerSet, WorkerWaker, init_waker_set};

    struct Page(Identifier);

    impl WorkerIdOutput for Page {
        fn worker_id(&self) -> Identifier {
            self.0
        }
    }

    #[test]
    fn a_message_returns_to_the_worker_it_names() {
        init_waker_set(WakerSet::new(vec![Arc::new(WorkerWaker::new(2))], 2));
        let mut endpoints: Vec<_> = return_to_worker_mpsc::<Page>(2)
            .into_iter()
            .map(ChannelFactory::build)
            .collect();

        endpoints[0].0.send(Page(1)).unwrap();

        assert!(endpoints[0].1.try_recv().is_none());
        assert_eq!(endpoints[1].1.try_recv().unwrap().0, 1);
    }
}
