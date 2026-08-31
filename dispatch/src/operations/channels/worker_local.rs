//! A channel where each worker's messages stay on that worker.
//!
//! Some stages keep per-worker state that must see every message the same
//! worker produced and nothing else. The Parquet scan's predicate gate, for
//! example, tracks each row group's surviving rows; a row group is decoded
//! entirely by its claiming worker, so routing that worker's batches to its
//! own gate instance keeps the tracking lock-free and complete.

use crate::operations::channels;
use crate::operations::channels::mpsc::{MpscReceiver, MpscSender, mpsc_channel_to};
use crate::operations::channels::{ChannelFactory, Sender};

/// Factory for one worker's local loop: a sender that targets this worker's
/// own mpsc channel and the matching receiver.
pub struct WorkerLocalMpscFactory<T> {
    sender: MpscSender<T>,
    receiver: MpscReceiver<T>,
}

impl<T: 'static + Send> ChannelFactory<T> for WorkerLocalMpscFactory<T> {
    type Sender = WorkerLocalSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            WorkerLocalSender {
                sender: self.sender,
            },
            self.receiver,
        )
    }
}

/// A sender that puts every message on the producing worker's own channel.
pub struct WorkerLocalSender<O> {
    sender: MpscSender<O>,
}

impl<O> Sender<O> for WorkerLocalSender<O> {
    fn send(&mut self, item: O) -> channels::Result<()> {
        self.sender.send_ref(item)?;
        Ok(())
    }
}

/// Create one [`WorkerLocalMpscFactory`] per worker, each looping to itself.
pub fn worker_local_mpsc<T: 'static + Send>(
    count: usize,
) -> impl IntoIterator<Item = WorkerLocalMpscFactory<T>> {
    (0..count).map(|worker| {
        let (sender, receiver) = mpsc_channel_to::<T>(worker);
        WorkerLocalMpscFactory { sender, receiver }
    })
}
