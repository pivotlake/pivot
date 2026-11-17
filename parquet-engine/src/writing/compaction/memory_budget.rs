//! A limit on the memory a compaction's rows hold.
//!
//! A row is in memory, decoded, from the moment its range of rows is
//! released to be decoded until the output row group it goes to is encoded.
//! The planner releases no range while the [`MemoryBudget`] is full, and
//! reserves the size of every batch from it when the batch arrives,
//! decoded. The [`MemoryReservation`] is held by the batch and then by the
//! row groups that take rows from it, so it is given back when the last of
//! those row groups is encoded and the batch is gone.
//!
//! A range's rows are not counted while they are being decoded, so the
//! budget can be passed by what is being decoded when it fills: a few
//! ranges of each input file at most.
//!
//! An encoded row group is far smaller than the rows it was made from. It
//! takes a reservation of its own, for its encoded bytes, which its writer
//! gives back when the bytes are stored.
//!
//! Dropping a reservation gives its bytes back and wakes the compaction
//! worker, where the planner runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dispatch::DataFlowDispatcher;
use dispatch::waker::WakerSet;

/// How many bytes a compaction's rows may hold at a time, and how many they
/// hold now.
pub(super) struct MemoryBudget {
    reserved_bytes: AtomicUsize,
    limit_bytes: usize,
    /// How many output row groups are being encoded or written.
    row_groups_in_flight: AtomicUsize,
    /// The worker that waits for room, woken when a reservation is dropped.
    waiting_worker: usize,
    waker_set: WakerSet,
}

impl MemoryBudget {
    /// A budget of `limit_bytes` whose room `waiting_worker`, a worker of
    /// `dispatcher`, waits for.
    pub(super) fn new(
        dispatcher: &DataFlowDispatcher,
        limit_bytes: usize,
        waiting_worker: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            reserved_bytes: AtomicUsize::new(0),
            limit_bytes,
            row_groups_in_flight: AtomicUsize::new(0),
            waiting_worker,
            waker_set: dispatcher.waker_set(),
        })
    }

    /// Whether as many bytes as the budget allows are reserved.
    pub(super) fn is_full(&self) -> bool {
        self.reserved_bytes.load(Ordering::Acquire) >= self.limit_bytes
    }

    /// Take `bytes` out of the budget until the reservation is dropped.
    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Arc<MemoryReservation> {
        self.reserved_bytes.fetch_add(bytes, Ordering::Release);
        Arc::new(MemoryReservation {
            budget: self.clone(),
            bytes,
            is_of_encoded_row_group: false,
        })
    }

    /// Record that an output row group was sent to be encoded. It is in
    /// flight until the reservation of its encoded bytes is dropped.
    pub(super) fn record_row_group_sent(&self) {
        self.row_groups_in_flight.fetch_add(1, Ordering::Release);
    }

    /// Whether an output row group is being encoded or written. While one
    /// is, memory is on its way to being given back.
    pub(super) fn has_row_group_in_flight(&self) -> bool {
        self.row_groups_in_flight.load(Ordering::Acquire) > 0
    }
}

/// Bytes taken out of a [`MemoryBudget`], given back when this is dropped.
pub struct MemoryReservation {
    budget: Arc<MemoryBudget>,
    bytes: usize,
    /// Whether this is the reservation of a row group's encoded bytes, whose
    /// end is the end of that row group being in flight.
    is_of_encoded_row_group: bool,
}

impl MemoryReservation {
    /// Reserve, from the same budget, the `bytes` an output row group takes
    /// encoded. Its rows' own reservations are given back as the row group
    /// lets go of them, so this takes their place and is never refused.
    pub(crate) fn reserve_encoded_row_group(&self, bytes: usize) -> Arc<MemoryReservation> {
        self.budget
            .reserved_bytes
            .fetch_add(bytes, Ordering::Release);
        Arc::new(MemoryReservation {
            budget: self.budget.clone(),
            bytes,
            is_of_encoded_row_group: true,
        })
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::Release);
        if self.is_of_encoded_row_group {
            self.budget
                .row_groups_in_flight
                .fetch_sub(1, Ordering::Release);
        }
        self.budget
            .waker_set
            .notify_worker(self.budget.waiting_worker);
    }
}
