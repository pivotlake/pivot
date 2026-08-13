//! Cross-worker delivery of cache-slot IO events, so a read can *join* another
//! worker's in-flight read of the same bytes instead of duplicating it.
//!
//! A worker that submits a read whose extent is already mapped (someone else's
//! read is filling it) registers itself as a waiter on the extent's ring slot
//! (see [`CompressedCache::register_waiter`]) and parks the request instead of
//! performing IO. Whichever worker *is* performing the IO later sends every
//! registered waiter a [`SlotIoEvent`] through this module: a per-worker
//! injector channel plus a targeted wake-up of that worker. The waiter then
//! either serves its parked request from the now-resident bytes, keeps
//! waiting, or, if the extent was abandoned, performs the read itself.
//!
//! The router is installed per worker thread (like the wakers), so separate
//! worker pools in one process never cross-talk. On threads with no router -
//! non-worker threads, unit tests - joining is simply disabled and every read
//! performs its own IO.
//!
//! [`CompressedCache::register_waiter`]: crate::memory::compressed_cache::CompressedCache::register_waiter

use crate::waker::WakerSet;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::cell::RefCell;
use std::sync::Arc;

/// What happened to a ring slot a waiter registered on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotIoOutcome {
    /// Blocks of the slot were committed. A parked request whose extent is now
    /// fully resident completes; one still missing bytes keeps waiting for its
    /// own extent's event.
    Committed,
    /// An uncommitted extent in the slot was abandoned: its owner's read failed
    /// or was cancelled before completing, and the extent was removed from its
    /// file's map. Parked requests must re-check their extent and perform the
    /// read themselves if it is the abandoned one.
    Abandoned,
}

/// One slot event, delivered to each worker that registered as a waiter.
#[derive(Clone, Copy, Debug)]
pub struct SlotIoEvent {
    pub slot_idx: usize,
    pub outcome: SlotIoOutcome,
}

/// Routes slot events to workers: one injector channel per worker, plus the
/// pool's wakers so a parked recipient wakes to drain it.
pub struct SlotEventRouter {
    senders: Box<[Sender<SlotIoEvent>]>,
    wakers: WakerSet,
}

impl SlotEventRouter {
    /// Build the router for a pool of `worker_count` workers, returning each
    /// worker's receiver in worker-index order.
    pub fn create(
        worker_count: usize,
        wakers: WakerSet,
    ) -> (Arc<Self>, Vec<Receiver<SlotIoEvent>>) {
        let (senders, receivers): (Vec<_>, Vec<_>) = (0..worker_count).map(|_| unbounded()).unzip();
        (
            Arc::new(Self {
                senders: senders.into_boxed_slice(),
                wakers,
            }),
            receivers,
        )
    }

    /// Deliver `event` to `worker`'s injector and wake it if parked. The
    /// channel is unbounded and its receiver lives for the worker's lifetime,
    /// so the send cannot block or fail while the pool is running; a send
    /// during teardown is dropped with the pool.
    pub(crate) fn send(&self, worker: usize, event: SlotIoEvent) {
        let _ = self.senders[worker].send(event);
        self.wakers.notify_worker(worker);
    }
}

thread_local! {
    /// This worker thread's router, installed at worker startup. `None` on
    /// threads that never installed one, where joining is disabled.
    static ROUTER: RefCell<Option<Arc<SlotEventRouter>>> = const { RefCell::new(None) };
}

/// Install this worker thread's slot-event router. Called at worker startup,
/// and by tests that exercise joining on plain threads.
pub fn install_worker_slot_events(router: Arc<SlotEventRouter>) {
    ROUTER.with_borrow_mut(|slot| *slot = Some(router));
}

/// Run `f` against this thread's router, or return `None` when no router is
/// installed (joining disabled on this thread).
pub(crate) fn with_router<R>(f: impl FnOnce(&SlotEventRouter) -> R) -> Option<R> {
    ROUTER.with_borrow(|router| router.as_deref().map(f))
}
