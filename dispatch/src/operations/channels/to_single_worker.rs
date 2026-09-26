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
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel_to};
use crate::operations::channels::{ChannelFactory, Receiver, Sender};

/// Factory for one worker's end of the funnel: a sender to the one channel
/// the target reads, and this worker's receiver, which is that channel's for
/// the target and an empty stand-in for everyone else. Only the target's
/// channel exists: one per worker would cost a query one allocation per
/// worker per funnel for channels that never carry a message.
pub struct SingleWorkerMpscFactory<T> {
    sender: MpscSender<T>,
    receiver: SingleWorkerReceiver<T>,
}

impl<T: 'static + Send> ChannelFactory<T> for SingleWorkerMpscFactory<T> {
    type Sender = SingleWorkerSender<T>;
    type Receiver = SingleWorkerReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            SingleWorkerSender {
                sender: self.sender,
            },
            self.receiver,
        )
    }
}

/// A sender that puts every message on the target worker's mpsc channel.
pub struct SingleWorkerSender<O> {
    sender: MpscSender<O>,
}

impl<O> Sender<O> for SingleWorkerSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.sender.send_ref(item)?;
        Ok(())
    }
}

/// The target worker's receiver, or the empty stand-in every other worker
/// reads: it never holds a message, so their stage finishes at once.
pub enum SingleWorkerReceiver<T> {
    Target(MpscReceiver<T>),
    Empty,
}

impl<T> Receiver<T> for SingleWorkerReceiver<T> {
    fn is_empty(&self) -> bool {
        match self {
            Self::Target(receiver) => receiver.is_empty(),
            Self::Empty => true,
        }
    }

    fn try_recv(&self) -> Option<T> {
        match self {
            Self::Target(receiver) => receiver.try_recv(),
            Self::Empty => None,
        }
    }

    fn steal(&self) -> Option<T> {
        match self {
            Self::Target(receiver) => receiver.steal(),
            Self::Empty => None,
        }
    }
}

/// Create one [`SingleWorkerMpscFactory`] per worker, all routing to `target`.
pub fn to_single_worker_mpsc<T: 'static + Send>(
    count: usize,
    target: Identifier,
) -> impl IntoIterator<Item = SingleWorkerMpscFactory<T>> {
    assert!(target < count, "the funnel's target must name a worker");
    let (sender, receiver) = mpsc_channel_to::<T>(target);
    let mut receiver = Some(receiver);

    (0..count).map(move |worker| SingleWorkerMpscFactory {
        sender: sender.clone(),
        receiver: if worker == target {
            SingleWorkerReceiver::Target(
                receiver
                    .take()
                    .expect("the target's receiver is taken once"),
            )
        } else {
            SingleWorkerReceiver::Empty
        },
    })
}
