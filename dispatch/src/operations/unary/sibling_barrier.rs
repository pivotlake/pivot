//! The barrier a stage's workers pass through as each one finishes consuming.
//!
//! A worker arrives once, when its input has drained; the last arrival is the
//! one that may run the stage's `finish`, and every other worker waits for
//! that moment before running its own. Waiting workers poll, so the count they
//! poll must not be the count the arrivals modify: with hundreds of workers
//! every arrival would otherwise have to invalidate the cache line in every
//! poller, and a stage's finish would cost a long serial chain of those
//! transfers. The count and the completion flag therefore live on separate
//! cache lines, and pollers read only the flag, which is written once.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A value padded to its own cache line.
#[repr(align(128))]
struct Padded<T>(T);

pub struct SiblingBarrier {
    remaining: Padded<AtomicUsize>,
    complete: Padded<AtomicBool>,
}

impl SiblingBarrier {
    /// A barrier expecting one arrival from each of `worker_count` workers.
    pub fn new(worker_count: usize) -> Arc<Self> {
        Arc::new(Self {
            remaining: Padded(AtomicUsize::new(worker_count)),
            complete: Padded(AtomicBool::new(false)),
        })
    }

    /// Records this worker's arrival. Returns whether it was the last one; if
    /// so, the barrier is complete and every poller sees it.
    ///
    /// Acquire/Release on the count so the last arrival's `finish` sees every
    /// peer's pre-arrival writes (e.g. a shared total each worker's async
    /// completions add to), not just channel-delivered data.
    pub fn arrive(&self) -> bool {
        if self.remaining.0.fetch_sub(1, Ordering::AcqRel) != 1 {
            return false;
        }
        self.complete.0.store(true, Ordering::Release);
        true
    }

    /// Takes back the arrival that completed the barrier, for a worker that
    /// found new input after arriving. Only the last arrival may call this.
    pub fn retract(&self) {
        self.complete.0.store(false, Ordering::Relaxed);
        self.remaining.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether every worker has arrived.
    pub fn is_complete(&self) -> bool {
        self.complete.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_arrival_completes_the_barrier() {
        let barrier = SiblingBarrier::new(3);

        let first = barrier.arrive();
        let second = barrier.arrive();
        let seen_early = barrier.is_complete();
        let third = barrier.arrive();

        assert!(!first && !second && !seen_early);
        assert!(third && barrier.is_complete());
    }

    #[test]
    fn a_retracted_arrival_reopens_the_barrier() {
        let barrier = SiblingBarrier::new(1);
        barrier.arrive();

        barrier.retract();

        assert!(!barrier.is_complete());
        assert!(barrier.arrive());
    }
}
