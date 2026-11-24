//! The memory a compaction's read holds, counted against one limit.
//!
//! Every byte the read holds is counted from the moment the compacter lets
//! it come into existence until it is gone:
//!
//! | held as | size | counted from | until |
//! |---|---|---|---|
//! | a fetched column chunk, compressed | the footer's | the request for it | each page leaves the [`DecompressGate`](super::decompress_gate::DecompressGate) |
//! | a decompressed page or dictionary page | the page header's | the page leaves the `DecompressGate` | nothing holds the page's bytes any more |
//! | decoded values | the most their type lets them own, then what the array owns | their rows are released to be decoded | the array is dropped by everything that holds it |
//! | the row order of an output row group | measured | the order is decided | the row group's last column is sent on |
//!
//! What can outlive the compacter is counted by a [`MemoryReservation`]
//! that travels with it and gives its bytes back when it is dropped:
//!
//! - A decompressed page's bytes carry one (see
//!   [`PageMemoryCounter`](super::page_memory_counter::PageMemoryCounter)).
//!   The decoded values of a string column point into its pages, so the
//!   page stays counted until the last array of them is dropped. The
//!   decoded values of a fixed-width column are copies, so the page is
//!   given back when its rows are decoded.
//! - Decoded values carry one for the bytes they own (see
//!   [`InputValues`](super::InputValues)), which is given back when the
//!   last holder of the array, the compacter or whatever it was sent to,
//!   drops it.
//!
//! Reservations are dropped on any thread, so the count is shared between
//! threads. Dropping one wakes the compaction worker, where the compacter
//! may be waiting for room.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use dispatch::waker::WakerSet;

/// How many bytes a compaction's read may hold, and how many it holds now.
pub(super) struct MemoryBudget {
    held_bytes: AtomicUsize,
    limit_bytes: usize,
    /// Whether the read has sent every row on. Once it has, nothing is
    /// held but what reservations still count.
    is_read_complete: AtomicBool,
    /// The worker the compacter runs on, woken when bytes are given back.
    compaction_worker: usize,
    waker_set: WakerSet,
}

impl MemoryBudget {
    /// A budget of `limit_bytes` with nothing held, for a compacter on
    /// `compaction_worker`, a worker that `waker_set` wakes.
    pub(super) fn new(
        waker_set: WakerSet,
        limit_bytes: usize,
        compaction_worker: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            held_bytes: AtomicUsize::new(0),
            limit_bytes,
            is_read_complete: AtomicBool::new(false),
            compaction_worker,
            waker_set,
        })
    }

    /// Count `bytes` more as held.
    pub(super) fn add(&self, bytes: usize) {
        self.held_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Count `bytes` fewer as held.
    pub(super) fn subtract(&self, bytes: usize) {
        let held_bytes = self.held_bytes.fetch_sub(bytes, Ordering::Relaxed);
        debug_assert!(bytes <= held_bytes, "more bytes given back than are held");
    }

    /// Count `bytes` more as held until the returned reservation is
    /// dropped.
    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> MemoryReservation {
        self.add(bytes);
        self.reserve_counted_bytes(bytes)
    }

    /// A reservation for `bytes` that are counted as held already. They
    /// stay counted until it is dropped.
    pub(super) fn reserve_counted_bytes(self: &Arc<Self>, bytes: usize) -> MemoryReservation {
        MemoryReservation {
            budget: self.clone(),
            bytes,
        }
    }

    /// Whether `bytes` more can be held without passing the limit.
    pub(super) fn has_room_for(&self, bytes: usize) -> bool {
        self.held_bytes.load(Ordering::Relaxed) + bytes <= self.limit_bytes
    }

    /// Whether `bytes` more can be held without passing half the limit.
    ///
    /// This is the test for work that is done ahead of the rows whose order
    /// is decided: sort keys decoded further ahead, and column chunks
    /// fetched before their rows are ordered. The other half of the limit
    /// is left for the columns of the rows whose order is decided.
    pub(super) fn has_room_ahead_for(&self, bytes: usize) -> bool {
        self.held_bytes.load(Ordering::Relaxed) + bytes <= self.limit_bytes / 2
    }

    /// Record that the read has sent every row on.
    pub(super) fn record_read_complete(&self) {
        self.is_read_complete.store(true, Ordering::Relaxed);
    }
}

impl Drop for MemoryBudget {
    /// The budget goes when its last reservation does. If the read sent
    /// every row on, every byte that was counted must have been given back
    /// by then.
    fn drop(&mut self) {
        if *self.is_read_complete.get_mut() && !std::thread::panicking() {
            debug_assert_eq!(
                *self.held_bytes.get_mut(),
                0,
                "bytes are still counted as held after the read and all it sent on are gone"
            );
        }
    }
}

/// Bytes counted against a [`MemoryBudget`], given back when this is
/// dropped.
pub struct MemoryReservation {
    budget: Arc<MemoryBudget>,
    bytes: usize,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget.subtract(self.bytes);
        self.budget
            .waker_set
            .notify_worker(self.budget.compaction_worker);
    }
}
