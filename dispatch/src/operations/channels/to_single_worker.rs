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
use crate::operations::channels::ChannelFactory;
use crate::operations::channels::mpsc::{MpscSender, SingleReaderReceiver, mpsc_channel_to};

/// Factory for one worker's end of the funnel: a sender to `target`'s
/// channel, and the receiver of that channel on `target` only.
pub struct SingleWorkerMpscFactory<T> {
    sender: MpscSender<T>,
    receiver: SingleReaderReceiver<T>,
}

impl<T: 'static + Send> ChannelFactory<T> for SingleWorkerMpscFactory<T> {
    type Sender = MpscSender<T>;
    type Receiver = SingleReaderReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (self.sender, self.receiver)
    }
}

/// Create one [`SingleWorkerMpscFactory`] per worker, all routing to `target`.
pub fn to_single_worker_mpsc<T: 'static + Send>(
    count: usize,
    target: Identifier,
) -> impl IntoIterator<Item = SingleWorkerMpscFactory<T>> {
    assert!(target < count, "the single worker must name a worker");
    let (sender, receiver) = mpsc_channel_to::<T>(target);
    let mut receiver = Some(receiver);
    (0..count)
        .map(|worker| SingleWorkerMpscFactory {
            sender: sender.clone(),
            receiver: if worker == target {
                SingleReaderReceiver::reader(
                    receiver
                        .take()
                        .expect("the target's receiver is handed out once"),
                )
            } else {
                SingleReaderReceiver::other()
            },
        })
        .collect::<Vec<_>>()
}
