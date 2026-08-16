//! Work-stealing source over an in-memory set of values.
//!
//! The in-memory counterpart of `catalog`'s Parquet row-group injector: instead
//! of pre-loading row groups from disk, it pre-loads caller-provided items of any
//! type `T` into a shared [`Injector`] queue. Every worker gets an
//! [`InjectorSource`] that steals items on demand, so a fixed `Vec<T>` fans out
//! across the whole pool exactly like a table scan distributes row groups.
//!
//! Paired with [`map_each`](crate::OperatorSpec::map_each), this is the basis of
//! a perfectly parallel job pipeline — e.g. one Parquet *page* encode per item,
//! stolen across workers.

use crate::operations::channels::{Receiver, RootChannelFactory, Sender};
use crate::operations::unary::{self, Unary};
use crate::waker::waker_set;
use crossbeam_deque::{Injector, Steal};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Factory that loads a value set into a shared work-stealing queue and hands
/// each worker an [`InjectorSource`] over it. Cheaply cloneable (the queue is
/// shared via `Arc`), so the same source feeds every worker.
pub struct InjectorSourceFactory<T> {
    items: Arc<Injector<T>>,
    /// Items not yet claimed; whoever claims the last one broadcast-wakes the
    /// pool. See [`InjectorSource::wakeup_workers_on_last_claim`].
    remaining: Arc<AtomicUsize>,
}

// `Injector<T>` is `Send + Sync` for `T: Send`, so the factory is too. (Derive
// would add an unwanted `T: Clone` bound, so implement `Clone` by hand.)
impl<T> Clone for InjectorSourceFactory<T> {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            remaining: self.remaining.clone(),
        }
    }
}

impl<T: Send> InjectorSourceFactory<T> {
    pub fn new(items: impl IntoIterator<Item = T>) -> Self {
        let injector = Injector::new();
        let mut count = 0;
        for item in items {
            injector.push(item);
            count += 1;
        }
        Self {
            items: Arc::new(injector),
            remaining: Arc::new(AtomicUsize::new(count)),
        }
    }
}

impl<T: Send + 'static> RootChannelFactory<T> for InjectorSourceFactory<T> {
    type Receiver = InjectorSource<T>;

    fn build(self) -> Self::Receiver {
        InjectorSource {
            items: self.items,
            remaining: self.remaining,
        }
    }
}

/// A [`Receiver`] that steals items from the shared queue. Work is only ever
/// consumed via [`steal`](Self::steal) (no per-worker local queue), so every
/// worker pulls from the same pool and load balances automatically.
pub struct InjectorSource<T> {
    items: Arc<Injector<T>>,
    /// Items not yet claimed, shared with every worker's source.
    remaining: Arc<AtomicUsize>,
}

impl<T> InjectorSource<T> {
    /// Count one item leaving the queue; broadcast-wake the pool when the last
    /// one goes. A worker that saw the queue non-empty at its finish check
    /// parks without decrementing the stage's sibling barrier, and the drain
    /// itself is silent: a claim sends nothing, and the claimer's downstream
    /// sends wake only its own node. Waking every node lets each such worker
    /// re-check the now-empty queue and sign off, so the dataflow can finish.
    /// Mirrors the row-group injector's last-claim wake in `catalog`.
    fn wakeup_workers_on_last_claim(&self) {
        if self.remaining.fetch_sub(1, Ordering::Relaxed) == 1 {
            waker_set().notify_all();
        }
    }
}

impl<T: Send> Receiver<T> for InjectorSource<T> {
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        // Consume from the eager `run_cpu_work` path, not only the idle `steal`
        // path: when downstream of this source does per-item IO, deferring to the
        // idle path serialises it (one item in flight at a time). A single
        // non-spinning attempt keeps the hot loop from spinning on `Steal::Retry`.
        // The `is_empty` pre-check keeps the drained-source case (every pass for
        // the rest of the query) to a plain load instead of an epoch-pinning steal.
        if self.items.is_empty() {
            return None;
        }
        match self.items.steal() {
            Steal::Success(item) => {
                self.wakeup_workers_on_last_claim();
                Some(item)
            }
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<T> {
        while !self.items.is_empty() {
            match self.items.steal() {
                Steal::Empty => return None,
                Steal::Success(item) => {
                    self.wakeup_workers_on_last_claim();
                    return Some(item);
                }
                Steal::Retry => continue,
            }
        }
        None
    }
}

/// Identity passthrough [`Unary`]: forwards each item downstream unchanged.
/// Used as the source operator over an [`InjectorSource`] — it just relays
/// stolen items into the pipeline.
pub struct Forward<T>(PhantomData<fn() -> T>);

impl<T> Default for Forward<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T> Unary<T, T> for Forward<T> {
    fn consume(
        &mut self,
        item: T,
        sender: &mut dyn Sender<T>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        sender.send(item)?;
        Ok(())
    }
}
