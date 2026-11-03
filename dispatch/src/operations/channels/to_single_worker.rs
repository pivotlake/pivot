//! A channel funneling every message to one fixed worker.
//!
//! Some stages want the whole stream on one worker: the Parquet write
//! indexer, for example, cuts an ordered stream into files, which only works
//! if a single consumer sees every batch. Every worker's sender routes to the
//! same target, so the target's receiver sees the whole stream; with a single
//! producing worker the stream also arrives in the order it was sent, which is
//! what lets the consumer rely on the order without re-establishing it. The
//! other workers' receivers stay empty.

use crate::Identifier;
use crate::operations::channels;
use crate::operations::channels::mpsc::{WorkerChannelReceiver, WorkerChannelSet};
use crate::operations::channels::{ChannelFactory, Sender};
use std::sync::Arc;

/// Factory for one worker's end of the funnel: the shared channel set (every
/// build routes to `target`) and this worker's receiver.
pub struct SingleWorkerMpscFactory<T: Send + 'static> {
    set: Arc<WorkerChannelSet<T>>,
    target: Identifier,
    receiver: WorkerChannelReceiver<T>,
}

impl<T: 'static + Send> ChannelFactory<T> for SingleWorkerMpscFactory<T> {
    type Sender = SingleWorkerSender<T>;
    type Receiver = WorkerChannelReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            SingleWorkerSender {
                set: self.set,
                target: self.target,
            },
            self.receiver,
        )
    }
}

/// A sender that puts every message on the `target` worker's channel.
pub struct SingleWorkerSender<O: Send + 'static> {
    set: Arc<WorkerChannelSet<O>>,
    target: Identifier,
}

impl<O: Send + 'static> Sender<O> for SingleWorkerSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.set.send_to(self.target, item)
    }
}

/// Create one [`SingleWorkerMpscFactory`] per worker, all routing to `target`.
pub fn to_single_worker_mpsc<T: 'static + Send>(
    count: usize,
    target: Identifier,
) -> impl IntoIterator<Item = SingleWorkerMpscFactory<T>> {
    let (set, receivers) = WorkerChannelSet::new(count);

    receivers
        .into_iter()
        .map(move |receiver| SingleWorkerMpscFactory {
            set: set.clone(),
            target,
            receiver,
        })
}
