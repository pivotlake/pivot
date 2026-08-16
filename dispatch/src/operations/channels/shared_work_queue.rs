//! A channel backed by one queue shared by every worker in a dataflow stage.
//!
//! [`stealable`](super::stealable()) gives each worker a local deque; another
//! worker checks that deque only during its idle-time steal pass. This channel
//! instead places every message in a common queue, so normal receive polling by
//! any worker can claim it. It is useful when one producer creates expensive
//! work that should be distributed immediately.
//!
//! The common queue introduces contention and does not preserve producer
//! locality. Callers should therefore reserve it for work large enough to
//! justify those costs.

use crate::operations::channels;
use crate::operations::channels::{ChannelFactory, Receiver, Sender};
use crate::waker::worker_waker;
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;

/// Builds one worker's sender and receiver for the shared queue.
pub struct SharedWorkQueueChannelFactory<T: Send> {
    queue: Arc<Injector<T>>,
}

impl<T: Send + 'static> ChannelFactory<T> for SharedWorkQueueChannelFactory<T> {
    type Sender = SharedWorkQueueSender<T>;
    type Receiver = SharedWorkQueueReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            SharedWorkQueueSender {
                queue: self.queue.clone(),
            },
            SharedWorkQueueReceiver { queue: self.queue },
        )
    }
}

pub struct SharedWorkQueueSender<T> {
    queue: Arc<Injector<T>>,
}

impl<T> Sender<T> for SharedWorkQueueSender<T> {
    fn send(&mut self, item: T) -> channels::Result<()> {
        self.queue.push(item);
        worker_waker().notify_one();
        Ok(())
    }
}

pub struct SharedWorkQueueReceiver<T> {
    queue: Arc<Injector<T>>,
}

impl<T> Receiver<T> for SharedWorkQueueReceiver<T> {
    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        loop {
            match self.queue.steal() {
                Steal::Success(item) => return Some(item),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }

    /// `try_recv` already reaches every message in this channel.
    fn steal(&self) -> Option<T> {
        None
    }
}

/// Creates `worker_count` endpoints backed by one queue.
pub fn shared_work_queue<T: Send>(
    worker_count: usize,
) -> impl IntoIterator<Item = SharedWorkQueueChannelFactory<T>> {
    let queue = Arc::new(Injector::new());
    (0..worker_count).map(move |_| SharedWorkQueueChannelFactory {
        queue: queue.clone(),
    })
}
