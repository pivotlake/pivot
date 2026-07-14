//! Fan-in channel: every worker's upstream operator sends to a *single*
//! worker's receiver.
//!
//! This is the special case of [`return_to_worker`](super::return_to_worker)
//! where every message is destined for worker 0 — used for the "send the
//! straddling remainder to one worker to combine" steps (mirroring how a
//! pipeline breaker collapses partial results). Worker 0's downstream operator
//! reads the real mpsc receiver; every other worker gets an empty receiver, so
//! its sibling operator finds nothing and finishes immediately, emitting
//! nothing.

use crate::operations::channels::ChannelFactory;
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel, mpsc_channel_to};

/// One worker's endpoint of a fan-in channel: a clone of the shared sender (all
/// pointing at worker 0's queue) and this worker's receiver — the real one for
/// worker 0, an empty stand-in for everyone else.
pub struct FanInChannelFactory<T> {
    sender: MpscSender<T>,
    receiver: MpscReceiver<T>,
}

impl<T: Send + 'static> ChannelFactory<T> for FanInChannelFactory<T> {
    type Sender = MpscSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (self.sender, self.receiver)
    }
}

/// Create one [`FanInChannelFactory`] per worker. All `count` senders feed the
/// single real receiver handed to worker 0; workers `1..count` get a private,
/// always-empty receiver.
pub fn fan_in<T: Send + 'static>(count: usize) -> Vec<FanInChannelFactory<T>> {
    let (sender, receiver) = mpsc_channel_to::<T>(0);
    let mut factories = Vec::with_capacity(count);
    factories.push(FanInChannelFactory {
        sender: sender.clone(),
        receiver,
    });
    for _ in 1..count {
        // A private channel whose sender is dropped immediately: the receiver is
        // forever empty, so this worker's downstream operator does nothing.
        let (_dropped, empty) = mpsc_channel::<T>();
        factories.push(FanInChannelFactory {
            sender: sender.clone(),
            receiver: empty,
        });
    }
    // Drop our extra handle so the channel closes once all operators drop theirs.
    let _ = sender;
    factories
}
