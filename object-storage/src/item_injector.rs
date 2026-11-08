//! Source stage for the object reads: a work-stealing queue that hands out
//! every requested item (a slice of an object, an object to download) exactly
//! once, to whatever worker steals it next. The twin of
//! [`crate::file_injector`], whose items are whole files.

use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::sync::Arc;

/// Builds an [`ItemInjector`] per worker, all sharing one queue of the items.
pub struct ItemInjectorFactory<T> {
    items: Arc<Injector<T>>,
}

impl<T> Clone for ItemInjectorFactory<T> {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
        }
    }
}

impl<T: Clone> ItemInjectorFactory<T> {
    pub fn new(items: &[T]) -> Self {
        let injector = Injector::new();
        for item in items {
            injector.push(item.clone());
        }
        Self {
            items: Arc::new(injector),
        }
    }
}

impl<T: Send + 'static> RootChannelFactory<T> for ItemInjectorFactory<T> {
    type Receiver = ItemInjector<T>;

    fn build(self) -> ItemInjector<T> {
        ItemInjector { items: self.items }
    }
}

pub struct ItemInjector<T> {
    items: Arc<Injector<T>>,
}

impl<T> ItemInjector<T> {
    /// Wake the pool if this claim was the one that emptied the queue, for the
    /// reason [`crate::file_injector::FileInjector`] does: a worker parked
    /// between seeing the queue non-empty and re-checking it owes its finish,
    /// and nothing else wakes it.
    fn wake_if_drained(&self) {
        if self.items.is_empty() {
            dispatch::waker::waker_set().notify_all();
        }
    }
}

impl<T> Receiver<T> for ItemInjector<T> {
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        // A single non-spinning attempt, as the file injector's: the hot loop
        // must not spin on `Steal::Retry`, and the next iteration retries.
        match self.items.steal() {
            Steal::Success(item) => {
                self.wake_if_drained();
                Some(item)
            }
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<T> {
        loop {
            match self.items.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(item) => {
                    self.wake_if_drained();
                    return Some(item);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dispatch::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};

    /// Claiming the last item has to wake the pool, on both claim paths, or a
    /// worker parked owing its finish never runs it and the read hangs.
    #[test]
    fn claiming_the_last_item_wakes_the_pool() {
        let waker = Arc::new(WorkerWaker::new(1));
        init_worker_waker(&waker);
        init_waker_set(WakerSet::new(vec![waker.clone()], 1));
        let injector = ItemInjectorFactory::new(&["a", "b"]).build();

        let before = waker.wake_count();
        injector.steal().expect("first item");
        let after_first = waker.wake_count();
        // `try_recv` is deliberately single-shot, so retry past a lost steal race.
        (0..1_000)
            .find_map(|_| injector.try_recv())
            .expect("last item");
        let after_last = waker.wake_count();

        assert_eq!(after_first, before);
        assert!(after_last > after_first);
    }
}
