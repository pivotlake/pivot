//! The memory a compaction holds, counted against one limit.
//!
//! Every byte of its columns that the compaction holds is counted from the
//! moment the compacter lets it come into existence until it is stored in
//! an output file:
//!
//! | held as | size | counted from | until |
//! |---|---|---|---|
//! | a fetched column chunk, compressed | the footer's | the request for it | each page leaves the [`DecompressGate`](crate::writing::compacting::read::pipeline::decompress_gate::DecompressGate) |
//! | a decompressed page or dictionary page | the page header's | the page leaves the `DecompressGate` | nothing holds the page's bytes any more |
//! | decoded values | the most their type lets them own, then what the array owns | their rows are released to be decoded | the array is dropped by everything that holds it |
//! | the leaves of one column of a merged batch, in output order | measured | they are gathered | every leaf of the column is copied to be encoded |
//! | one leaf of an output row group, copied into one array to be encoded | what its batches hold | before the copy | the leaf is encoded |
//! | the encoded pages of a leaf | measured | the leaf is encoded | the pages are stored |
//!
//! The row order of the output row groups being read is not counted: it is
//! a few megabytes at most.
//!
//! Only the read waits for room. What the write half holds is counted when
//! it exists, whatever the limit says: its rows are already decoded, and
//! encoding them is what gives their memory back. The read sees those
//! bytes as held, so it slows down to the encoding and to the storage.
//!
//! What can outlive the compacter is counted by a [`MemoryReservation`]
//! that travels with it and gives its bytes back when it is dropped:
//!
//! - A decompressed page's bytes carry one (see
//!   [`PageMemoryCounter`](crate::writing::compacting::read::pipeline::page_memory_counter::PageMemoryCounter)).
//!   The decoded values of a string column point into its pages, so the
//!   page stays counted until the last array of them is dropped. The
//!   decoded values of a fixed-width column are copies, so the page is
//!   given back when its rows are decoded.
//! - Decoded values carry one for the bytes they own (see
//!   [`InputValues`](super::read::output::InputValues)), which is given back when the
//!   last holder of the array, the compacter or whatever it was sent to,
//!   drops it.
//! - Everything the write half passes on carries one: gathered leaves, a
//!   leaf being encoded, and encoded pages. The reservation of the encoded
//!   column chunks that are written together is handed to whoever stores
//!   them, and is dropped when their bytes are stored.
//!
//! # What goes ahead when there is no room
//!
//! The read does what the output cannot go on without even when the
//! budget has no room, since nothing else would give memory back. That
//! holds only while no output row group is *in flight*: sent on by the
//! read in every column, with its last column chunks not stored yet. A
//! row group in flight is
//! being encoded or stored, which gives its memory back without anything
//! more from the read, so the read waits for it instead. Otherwise a
//! storage that has stopped taking bytes would be fed row groups without
//! end.
//!
//! Reservations are dropped on any thread, so the count is shared between
//! threads. Dropping one wakes the compaction worker, where the compacter
//! may be waiting for room.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use dispatch::waker::WakerSet;

/// How many bytes a compaction may hold, and how many it holds now.
pub(super) struct MemoryBudget {
    held_bytes: AtomicUsize,
    limit_bytes: usize,
    /// How many output row groups the read has sent on in every column
    /// and whose last column chunks are not stored yet.
    output_row_groups_in_flight: AtomicUsize,
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
            output_row_groups_in_flight: AtomicUsize::new(0),
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
            ends_row_group_in_flight: false,
        }
    }

    /// Whether `bytes` more can be held without passing the limit.
    pub(super) fn has_room_for(&self, bytes: usize) -> bool {
        self.held_bytes.load(Ordering::Relaxed) + bytes <= self.limit_bytes
    }

    /// Whether `bytes` more can be held without passing half the limit.
    ///
    /// This is the test for ordering rows further ahead than the workers
    /// need to stay busy. The other half of the limit is left for the
    /// columns of the rows whose order is decided.
    pub(super) fn has_room_ahead_for(&self, bytes: usize) -> bool {
        self.held_bytes.load(Ordering::Relaxed) + bytes <= self.limit_bytes / 2
    }

    /// Record that the read has sent an output row group on in every
    /// column. The row group is in flight until the reservation of its
    /// last encoded column chunks is dropped.
    pub(super) fn record_output_row_group_sent(&self) {
        self.output_row_groups_in_flight
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Whether an output row group is being encoded or stored. While one
    /// is, memory is on its way to being given back.
    pub(super) fn has_output_row_group_in_flight(&self) -> bool {
        self.output_row_groups_in_flight.load(Ordering::Relaxed) > 0
    }

    /// One reservation for the encoded column chunks of an output row
    /// group that are written together, in place of `leaf_reservations`,
    /// the reservations of their encoded leaves. The bytes stay counted
    /// throughout. If the column chunks are the row group's last
    /// (`ends_row_group`), dropping it also ends the row group being in
    /// flight.
    pub(super) fn reserve_encoded_column_chunks(
        self: &Arc<Self>,
        leaf_reservations: Vec<MemoryReservation>,
        ends_row_group: bool,
    ) -> MemoryReservation {
        let mut bytes = 0;
        for mut leaf_reservation in leaf_reservations {
            bytes += std::mem::take(&mut leaf_reservation.bytes);
        }
        MemoryReservation {
            budget: self.clone(),
            bytes,
            ends_row_group_in_flight: ends_row_group,
        }
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
                "bytes are still counted as held after the compaction and all it sent on are gone"
            );
        }
    }
}

/// Bytes counted against a [`MemoryBudget`], given back when this is
/// dropped.
pub struct MemoryReservation {
    budget: Arc<MemoryBudget>,
    bytes: usize,
    /// Whether this is the reservation of an output row group's last
    /// encoded column chunks, whose end is the end of that row group being
    /// in flight.
    ends_row_group_in_flight: bool,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget.subtract(self.bytes);
        if self.ends_row_group_in_flight {
            self.budget
                .output_row_groups_in_flight
                .fetch_sub(1, Ordering::Relaxed);
        }
        self.budget
            .waker_set
            .notify_worker(self.budget.compaction_worker);
    }
}
