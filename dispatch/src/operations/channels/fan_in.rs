//! Fan-in channel: every worker's upstream operator sends to a *single*
//! worker's receiver.
//!
//! The caller chooses the receiving worker. Its downstream operator reads the
//! real mpsc receiver; every other worker gets an empty receiver, so its sibling
//! operator finds nothing and finishes immediately.

use crate::operations::channels::ChannelFactory;
use crate::operations::channels::mpsc::{MpscSender, SingleReaderReceiver, mpsc_channel_to};

/// One worker's endpoint of a fan-in channel: a clone of the shared sender and
/// this worker's receiver, either the selected receiver or an empty stand-in.
pub struct FanInChannelFactory<T> {
    sender: MpscSender<T>,
    receiver: SingleReaderReceiver<T>,
}

impl<T: Send + 'static> ChannelFactory<T> for FanInChannelFactory<T> {
    type Sender = MpscSender<T>;
    type Receiver = SingleReaderReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (self.sender, self.receiver)
    }
}

/// Create one [`FanInChannelFactory`] per worker, with every sender feeding
/// `target`'s receiver.
pub fn fan_in<T: Send + 'static>(count: usize, target: usize) -> Vec<FanInChannelFactory<T>> {
    assert!(target < count, "fan-in target must name a worker");
    let (sender, receiver) = mpsc_channel_to::<T>(target);
    let mut receiver = Some(receiver);
    let mut factories = Vec::with_capacity(count);
    for worker in 0..count {
        let receiver = if worker == target {
            SingleReaderReceiver::reader(
                receiver.take().expect("the target receiver is taken once"),
            )
        } else {
            SingleReaderReceiver::other()
        };
        factories.push(FanInChannelFactory {
            sender: sender.clone(),
            receiver,
        });
    }
    // Drop our extra handle so the channel closes once all operators drop theirs.
    let _ = sender;
    factories
}
