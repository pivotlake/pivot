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
use crossbeam_deque::{Injector, Steal};
use std::marker::PhantomData;
use std::sync::Arc;

/// Factory that loads a value set into a shared work-stealing queue and hands
/// each worker an [`InjectorSource`] over it. Cheaply cloneable (the queue is
/// shared via `Arc`), so the same source feeds every worker.
pub struct InjectorSourceFactory<T> {
    items: Arc<Injector<T>>,
}

// `Injector<T>` is `Send + Sync` for `T: Send`, so the factory is too. (Derive
// would add an unwanted `T: Clone` bound, so implement `Clone` by hand.)
impl<T> Clone for InjectorSourceFactory<T> {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
        }
    }
}

impl<T: Send> InjectorSourceFactory<T> {
    pub fn new(items: impl IntoIterator<Item = T>) -> Self {
        let injector = Injector::new();
        for item in items {
            injector.push(item);
        }
        Self {
            items: Arc::new(injector),
        }
    }
}

impl<T: Send + 'static> RootChannelFactory<T> for InjectorSourceFactory<T> {
    type Receiver = InjectorSource<T>;

    fn build(self) -> Self::Receiver {
        InjectorSource { items: self.items }
    }
}

/// A [`Receiver`] that steals items from the shared queue. Work is only ever
/// consumed via [`steal`](Self::steal) (no per-worker local queue), so every
/// worker pulls from the same pool and load balances automatically.
pub struct InjectorSource<T> {
    items: Arc<Injector<T>>,
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
        match self.items.steal() {
            Steal::Success(item) => Some(item),
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<T> {
        loop {
            return match self.items.steal() {
                Steal::Empty => None,
                Steal::Success(item) => Some(item),
                Steal::Retry => continue,
            };
        }
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
    fn consume<S: Sender<T>>(&mut self, item: T, sender: &mut S) -> unary::Result<()> {
        sender.send(item)?;
        Ok(())
    }
}
