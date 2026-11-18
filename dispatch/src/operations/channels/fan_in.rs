//! Fan-in channel: every worker's upstream operator sends to a *single*
//! worker's receiver.
//!
//! The caller chooses the receiving worker. Its downstream operator reads the
//! real mpsc receiver; every other worker gets an empty receiver, so its sibling
//! operator finds nothing and finishes immediately.

use crate::api::BuildContext;
use crate::operations::channels;
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel};
use crate::operations::channels::{ChannelFactory, Sender};
use crate::waker::WakerSet;

/// One worker's endpoint of a fan-in channel: a clone of the shared sender and
/// this worker's receiver, either the selected receiver or an empty stand-in.
pub struct FanInChannelFactory<T> {
    sender: MpscSender<T>,
    target: usize,
    receiver: MpscReceiver<T>,
}

impl<T: Send + 'static> ChannelFactory<T> for FanInChannelFactory<T> {
    type Sender = FanInSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self, context: &BuildContext) -> (Self::Sender, Self::Receiver) {
        (
            FanInSender {
                sender: self.sender,
                target: self.target,
                wakers: context.wakers(),
            },
            self.receiver,
        )
    }
}

/// A sender into the `target` worker's receiver.
pub struct FanInSender<T> {
    sender: MpscSender<T>,
    target: usize,
    /// Wakes the target, which may be parked.
    wakers: WakerSet,
}

impl<T> Sender<T> for FanInSender<T> {
    fn send(&mut self, item: T) -> channels::Result<()> {
        self.sender.send_ref(item)?;
        self.wakers.notify_worker(self.target);
        Ok(())
    }
}

/// Create one [`FanInChannelFactory`] per worker, with every sender feeding
/// `target`'s receiver.
pub fn fan_in<T: Send + 'static>(count: usize, target: usize) -> Vec<FanInChannelFactory<T>> {
    assert!(target < count, "fan-in target must name a worker");
    let (sender, receiver) = mpsc_channel::<T>();
    let mut receiver = Some(receiver);
    let mut factories = Vec::with_capacity(count);
    for worker in 0..count {
        let receiver = if worker == target {
            receiver.take().expect("the target receiver is taken once")
        } else {
            // Its sender is dropped immediately, so this receiver stays empty.
            let (_dropped, empty) = mpsc_channel::<T>();
            empty
        };
        factories.push(FanInChannelFactory {
            sender: sender.clone(),
            target,
            receiver,
        });
    }
    // Drop our extra handle so the channel closes once all operators drop theirs.
    let _ = sender;
    factories
}
