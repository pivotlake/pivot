//! A shared-injector channel: every message goes into one global queue that
//! any worker's receiver claims from.
//!
//! The [`stealable`](super::stealable) channel keeps items in the producing
//! worker's deque, where they wait until that worker consumes them or an
//! *idle* peer steals them; under load, when no worker is ever idle, a
//! stage's work therefore concentrates on whichever workers produced it. An
//! injector channel makes the queue itself shared: items are visible to every
//! worker's ordinary receive path the moment they are pushed, so consumption
//! spreads across the pool without any worker having to go idle first. That
//! is what a pipeline's drain stages want; in the Parquet write path, rows
//! only release their memory once dealt and encoded, so those stages must
//! never queue behind a single worker.
//!
//! The trade against `stealable` is a contended queue and no producer
//! locality (an item may be claimed by any worker of the dataflow, on any
//! node), so it fits stages whose per-item work dwarfs both a queue operation
//! and the cost of touching the item's memory from another core.

use crate::operations::channels;
use crate::operations::channels::{ChannelFactory, Receiver, Sender};
use crate::waker::worker_waker;
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;

/// Factory for one worker's endpoint of the shared queue.
pub struct InjectorChannelFactory<T: Send> {
    injector: Arc<Injector<T>>,
}

impl<T: Send + 'static> ChannelFactory<T> for InjectorChannelFactory<T> {
    type Sender = InjectorSender<T>;
    type Receiver = InjectorReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (
            InjectorSender {
                injector: self.injector.clone(),
            },
            InjectorReceiver {
                injector: self.injector,
            },
        )
    }
}

/// Pushes onto the shared queue and wakes one same-node consumer.
pub struct InjectorSender<T> {
    injector: Arc<Injector<T>>,
}

impl<T> Sender<T> for InjectorSender<T> {
    fn send(&mut self, item: T) -> channels::Result<()> {
        self.injector.push(item);
        // One new item needs one consumer, same discipline as the stealable
        // channel's send.
        worker_waker().notify_one();
        Ok(())
    }
}

/// Claims items off the shared queue.
pub struct InjectorReceiver<T> {
    injector: Arc<Injector<T>>,
}

impl<T> Receiver<T> for InjectorReceiver<T> {
    fn is_empty(&self) -> bool {
        self.injector.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        loop {
            match self.injector.steal() {
                Steal::Success(item) => return Some(item),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }

    /// The queue is already shared, so the idle-time steal pass has nothing
    /// beyond what [`try_recv`](Self::try_recv) reaches.
    fn steal(&self) -> Option<T> {
        None
    }
}

/// Create one [`InjectorChannelFactory`] per worker, all claiming from one
/// shared queue.
pub fn injector<T: Send>(count: usize) -> impl IntoIterator<Item = InjectorChannelFactory<T>> {
    let injector = Arc::new(Injector::new());
    (0..count).map(move |_| InjectorChannelFactory {
        injector: injector.clone(),
    })
}
